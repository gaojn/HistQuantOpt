"""
信号强度 → 策略IR 标定实验。

用「未来 fwd_days 日收益」反向构造已知 IC / ICIR / decay 的合成因子，
量化「因子信号强度」与「策略IR」之间的映射关系，供日后估算真实因子的
预期 IR。

⚠️ 警告：本模块构造的因子使用了未来信息（前视），仅作为标定工具，
   绝不可用于实盘或真实回测。它回答的是「假如我有一个 IC=X 的因子，
   在我这套约束下大概能做出多少 IR」。

三层 IR 口径（约束依次增强，IR 依次衰减）：
  1. ir_theory : ICIR × √(年调仓次数)            —— 纯信息论上限（无组合构建）
  2. ls_ir     : 分位多空组合的年化 IR            —— 含组合构建、无约束
  3. full_ir   : 全管线（行业/风格中性+TE+换手+long-only）  —— 由真实回测锚点标定

转换系数 TC = full_ir / ir_theory，由已跑的真实结果反推（≈0.5~0.58）。
经验法则：  年化IR ≈ ICIR × √(年调仓次数) × TC ≈ 3.9 × ICIR （5日调仓本项目约束下）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

# 默认网格（精简两轴）
DEFAULT_IC_LIST   = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12]
DEFAULT_ICIR_LIST = [0.3, 0.5, 0.8, 1.2]
DEFAULT_DECAY_LIST = [0.0, 0.5, 0.8, 0.9, 0.95]


def _pivot_adj(panel: pl.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    return (
        panel.select(["date", "code", price_col]).to_pandas()
        .pivot(index="date", columns="code", values=price_col)
        .sort_index()
    )


@dataclass
class GridPoint:
    """单组网格参数与实测结果。"""
    in_ic: float
    in_ic_std: float
    in_icir: float
    in_decay: float
    ic: float          # 实测 Spearman IC 均值
    ic_std: float      # 实测 IC 标准差
    icir: float        # 实测 ICIR
    ir_theory: float   # ICIR × √(年调仓次数)
    ls_ir: float       # 分位多空年化 IR
    ls_ann: float      # 分位多空年化收益
    autocorr: float    # 因子相邻期自相关（越高换手越低）
    turnover: float    # 多头分位成分换手率
    n_seeds: int


class SignalGridRunner:
    """
    合成信号网格标定器。

    Parameters
    ----------
    panel : pl.DataFrame
        行情面板（需含 date / code / price_col）
    fwd_days : int
        前瞻收益窗口（与策略持有期一致），默认 5
    rebal_freq : int
        IC/回测的取样间隔（非重叠），默认 5（与调仓频率一致）
    min_names : int
        某日有效股票数下限，低于则跳过该截面
    price_col : str
        用于计算未来收益的价格列，默认 "adj_close"；
        可传 "adj_vwap" 以基于未来 VWAP 涨跌幅生成因子。
    """

    def __init__(
        self,
        panel: pl.DataFrame,
        fwd_days: int = 5,
        rebal_freq: int = 5,
        min_names: int = 50,
        price_col: str = "adj_close",
    ) -> None:
        self.fwd_days = fwd_days
        self.rebal_freq = rebal_freq
        self.min_names = min_names
        self.price_col = price_col

        adj = _pivot_adj(panel, price_col)
        self.fwd_ret = adj.shift(-fwd_days) / adj - 1
        valid = self.fwd_ret.notna().sum(axis=1) > min_names
        self.gen_dates = self.fwd_ret.index[valid]
        self.rebal_dates = self.gen_dates[::rebal_freq]
        self.periods_per_year = 252.0 / fwd_days

        # 预计算每个生成日的 z-score 化未来收益（避免每组重复 pivot）
        self._zr: dict = {}
        for dt in self.gen_dates:
            r = self.fwd_ret.loc[dt].dropna()
            mu, sig = r.mean(), r.std()
            if sig < 1e-9:
                continue
            self._zr[dt] = (r.index, ((r - mu) / sig).values)

    # ------------------------------------------------------------------
    # 合成因子生成（提速版：复用预计算的 z-score 未来收益）
    # ------------------------------------------------------------------

    def gen_alpha(
        self,
        ic_mean: float,
        ic_std: float,
        decay: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        生成一组合成因子（与 pipeline.build_synthetic_alpha 同构）。

        每个截面：sig = ρ·z(fwd_ret) + √(1-ρ²)·ε，ρ ~ N(ic_mean, ic_std)
        AR(1) 平滑：f_t = decay·f_{t-1} + √(1-decay²)·sig
        """
        rng = np.random.default_rng(seed)
        rows: dict = {}
        f_prev: pd.Series | None = None

        for dt in self.gen_dates:
            if dt not in self._zr:
                continue
            idx, zr = self._zr[dt]
            rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
            eps = rng.standard_normal(len(zr))
            sig = rho * zr + np.sqrt(max(1 - rho**2, 0.0)) * eps
            s = pd.Series(sig, index=idx)
            s = (s - s.mean()) / (s.std() + 1e-10)

            if f_prev is None or decay == 0.0:
                f = s
            else:
                common = f_prev.index.intersection(s.index)
                f = s.copy()
                if len(common) > 0:
                    f[common] = (
                        decay * f_prev[common]
                        + np.sqrt(max(1 - decay**2, 0.0)) * s[common]
                    )
                f = (f - f.mean()) / (f.std() + 1e-10)
            f_prev = f
            rows[dt] = f

        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # 评估：实测 IC / 分位多空 IR / 自相关 / 换手
    # ------------------------------------------------------------------

    def evaluate(self, alpha_df: pd.DataFrame, q: float = 0.2) -> dict:
        """在非重叠调仓日上评估一组因子。"""
        ics: list[float] = []
        ls_rets: list[float] = []
        autocorrs: list[float] = []
        turns: list[float] = []
        prev_alpha: pd.Series | None = None
        prev_top: set | None = None

        for dt in self.rebal_dates:
            if dt not in alpha_df.index:
                continue
            a = alpha_df.loc[dt].dropna()
            r = self.fwd_ret.loc[dt].reindex(a.index).dropna()
            a = a.reindex(r.index)
            if len(a) < self.min_names:
                continue

            ic = spearmanr(a.values, r.values).correlation
            if np.isfinite(ic):
                ics.append(float(ic))

            k = max(int(len(a) * q), 1)
            top = a.nlargest(k).index
            bot = a.nsmallest(k).index
            ls_rets.append(float(r[top].mean() - r[bot].mean()))

            if prev_alpha is not None:
                common = a.index.intersection(prev_alpha.index)
                if len(common) > 10:
                    ac = spearmanr(a[common].values, prev_alpha[common].values).correlation
                    if np.isfinite(ac):
                        autocorrs.append(float(ac))
                turns.append(1.0 - len(set(top) & prev_top) / k)

            prev_alpha = a
            prev_top = set(top)

        ics_a = np.array(ics)
        ls_a = np.array(ls_rets)
        ic_m = float(ics_a.mean()) if len(ics_a) else 0.0
        ic_s = float(ics_a.std(ddof=1)) if len(ics_a) > 1 else 0.0
        icir = ic_m / (ic_s + 1e-12)
        ppy = self.periods_per_year

        ls_ir = (
            float(ls_a.mean() / (ls_a.std(ddof=1) + 1e-12)) * np.sqrt(ppy)
            if len(ls_a) > 1 else 0.0
        )

        return {
            "ic": ic_m,
            "ic_std": ic_s,
            "icir": icir,
            "ir_theory": icir * np.sqrt(ppy),
            "ls_ir": ls_ir,
            "ls_ann": float(ls_a.mean()) * ppy if len(ls_a) else 0.0,
            "autocorr": float(np.mean(autocorrs)) if autocorrs else np.nan,
            "turnover": float(np.mean(turns)) if turns else np.nan,
        }

    # ------------------------------------------------------------------
    # 单点（多种子平均降噪）
    # ------------------------------------------------------------------

    def run_point(
        self,
        ic_mean: float,
        ic_std: float,
        decay: float,
        seeds: tuple[int, ...] = (42, 43, 44),
        q: float = 0.2,
    ) -> dict:
        accum = []
        for sd in seeds:
            alpha = self.gen_alpha(ic_mean, ic_std, decay, sd)
            accum.append(self.evaluate(alpha, q))
        avg = pd.DataFrame(accum).mean().to_dict()
        avg.update({
            "in_ic": ic_mean,
            "in_ic_std": ic_std,
            "in_icir": ic_mean / ic_std,
            "in_decay": decay,
            "n_seeds": len(seeds),
        })
        return avg


def run_grid(
    runner: SignalGridRunner,
    ic_list: list[float] | None = None,
    icir_list: list[float] | None = None,
    decay_fixed: float = 0.8,
    decay_list: list[float] | None = None,
    ic_fixed: float = 0.08,
    icir_fixed: float = 0.8,
    seeds: tuple[int, ...] = (42, 43, 44),
    q: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    跑精简两轴网格：
      A. IC × ICIR @ decay_fixed
      B. decay 单独扫 @ (ic_fixed, icir_fixed)

    Returns
    -------
    pd.DataFrame  每行一组参数与实测指标，列含 sweep 标签
    """
    ic_list = ic_list or DEFAULT_IC_LIST
    icir_list = icir_list or DEFAULT_ICIR_LIST
    decay_list = decay_list or DEFAULT_DECAY_LIST

    rows: list[dict] = []

    # ── Sweep A: IC × ICIR ──
    n_a = len(ic_list) * len(icir_list)
    i = 0
    for ic in ic_list:
        for icir in icir_list:
            i += 1
            res = runner.run_point(ic, ic / icir, decay_fixed, seeds, q)
            res["sweep"] = "ic_icir"
            rows.append(res)
            if verbose:
                print(f"  [A {i:2d}/{n_a}] IC={ic:.2f} ICIR={icir:.1f} "
                      f"→ 实测IC={res['ic']:.4f} ICIR={res['icir']:.2f} "
                      f"LS_IR={res['ls_ir']:.2f}")

    # ── Sweep B: decay ──
    for j, dc in enumerate(decay_list, 1):
        res = runner.run_point(ic_fixed, ic_fixed / icir_fixed, dc, seeds, q)
        res["sweep"] = "decay"
        rows.append(res)
        if verbose:
            print(f"  [B {j:2d}/{len(decay_list)}] decay={dc:.2f} "
                  f"→ 实测IC={res['ic']:.4f} ICIR={res['icir']:.2f} "
                  f"LS_IR={res['ls_ir']:.2f} 自相关={res['autocorr']:.2f} "
                  f"换手={res['turnover']:.2f}")

    cols = [
        "sweep", "in_ic", "in_ic_std", "in_icir", "in_decay",
        "ic", "ic_std", "icir", "ir_theory", "ls_ir", "ls_ann",
        "autocorr", "turnover", "n_seeds",
    ]
    return pd.DataFrame(rows)[cols]
