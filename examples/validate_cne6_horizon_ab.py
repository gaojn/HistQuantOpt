"""CNE6S（短周期，hl=63）vs CNE6L（长周期，hl=252）风险模型 横向 A/B 对照。

背景：BarraCNE6 把 CNE6S/CNE6L 设计成"同一套 19 风格因子，仅 EWMA 半衰期不同"
（详见 BarraCNE6/docs/CNE6_VS_CNTR_NOTES.md §4）。本脚本验证：在低频（月度）
换仓的指数增强组合上，用估计窗口更长、更平滑的 CNE6L 风险模型，是否比用
对短期波动更敏感的 CNE6S，能取得更稳健的风险预测 / 更优的 matched-TE IR。

同回测期（2023-02~2026-05，两版风险面板已对齐）、同合成 alpha、同月度换仓节奏，
对比：
    A 组  CNE6S（短周期，data/barra_cne6/，hl=63）   —— 扫 λ
    B 组  CNE6L（长周期，data/barra_cne6_L/，hl=252）—— 扫 λ

各自构建 λ↔TE 前沿，再做 matched-TE 的 IR/MDD/换手对比 ——
而不是直接比较同一 λ 下的结果（因为 S/L 的协方差量级不同，同一 λ 对应的
实际风险厌恶程度不可比，必须先用 TE 对齐"风险预算"再比效用）。

运行：python examples/validate_cne6_horizon_ab.py
"""

from __future__ import annotations

import sys
import os
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio_optimizer.pipeline.batch_optimize import run_batch_optimize
from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester

# 基础配置可通过命令行覆盖，便于复用同一脚本跑不同换仓频率的 A/B：
#   python examples/validate_cne6_horizon_ab.py                                  → 月度
#   python examples/validate_cne6_horizon_ab.py configs/zz500_enhance_cne6_weekly.yaml → 周度
BASE_CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/zz500_enhance_cne6_horizon.yaml"
RUN_TAG = Path(BASE_CONFIG).stem.replace("zz500_enhance_cne6_", "")  # horizon | weekly | ...
INDEX = "zz500"

# (组标签, cne6_data_dir, λ 网格)
PANELS = [
    ("S_短周期(hl=63)",  None,                 [5.0, 10.0, 20.0, 40.0, 80.0]),
    ("L_长周期(hl=252)", "data/barra_cne6_L",  [5.0, 10.0, 20.0, 40.0, 80.0]),
]


def make_config(label: str, data_dir: str | None, lam: float) -> str:
    cfg = yaml.safe_load(open(BASE_CONFIG, encoding="utf-8"))
    cfg["optimizer"]["cne6_data_dir"] = data_dir
    cfg["optimizer"]["risk_aversion"] = lam
    cfg["output"]["weights"] = f"output/_hzab_{RUN_TAG}_{label}_weights.parquet"
    tmp = Path(tempfile.gettempdir()) / f"_hzab_{RUN_TAG}_{label}.yaml"
    yaml.dump(cfg, open(tmp, "w", encoding="utf-8"), allow_unicode=True)
    return str(tmp)


def load_market(start_date, end_date):
    panel = load_panel(
        start_date, end_date,
        columns=[
            "code", "date", "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "trade_status",
            "free_mv", "total_mv", "is_hs300", "is_zz500", "is_zz1000",
        ],
    )

    def to_wide(col: str) -> pd.DataFrame:
        df = (panel.select(["date", "code", col]).to_pandas()
              .pivot(index="date", columns="code", values=col).sort_index())
        df.index = pd.to_datetime(df.index)
        return df

    wides = {c: to_wide(c) for c in
             ["adj_close", "adj_vwap", "close", "limit_up", "limit_down", "trade_status"]}

    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)

    daily_ret_all = wides["adj_close"].pct_change(fill_method=None).fillna(0.0)
    w_lag = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)
    bm_ret.name = INDEX.upper()
    return wides, bm_ret


def backtest_metrics(weight_df: pd.DataFrame, wides: dict, bm_ret: pd.Series) -> dict:
    weight_df = weight_df.copy()
    weight_df.index = pd.to_datetime(weight_df.index)

    bt = RealisticBacktester(cost_buy=0.001, cost_sell=0.002, risk_free=0.02)
    result, _ = bt.run(
        weight_df=weight_df,
        adj_close=wides["adj_close"], adj_vwap=wides["adj_vwap"],
        close_raw=wides["close"],
        limit_up_df=wides["limit_up"], limit_down_df=wides["limit_down"],
        trade_status_df=wides["trade_status"],
        benchmark_ret=bm_ret, initial_value=1e8,
    )

    exc = (result.daily_ret - result.bm_ret).dropna()
    n = len(exc)
    excess_nav = result.excess_nav.dropna()
    ann_excess = excess_nav.iloc[-1] ** (252 / n) - 1
    te = exc.std() * np.sqrt(252)
    ir = ann_excess / te if te > 1e-9 else np.nan
    exc_mdd = float((excess_nav / excess_nav.cummax() - 1).min())
    return {
        "年化超额": ann_excess,
        "实际TE": te,
        "IR": ir,
        "超额MDD": exc_mdd,
        "平均换手": result.turnover.mean(),
    }


def matched_te_pick(frontier: pd.DataFrame, target_te: float) -> pd.Series:
    """在 λ↔TE 前沿上，找 TE 最接近 target_te 的一行。"""
    idx = (frontier["实际TE"] - target_te).abs().idxmin()
    return frontier.loc[idx]


def main() -> None:
    base = yaml.safe_load(open(BASE_CONFIG, encoding="utf-8"))
    from datetime import date
    sd = date.fromisoformat(base["backtest"]["start_date"])
    ed = date.fromisoformat(base["backtest"]["end_date"])
    rf = base["backtest"]["rebalance_freq"]

    print(f"\n{'='*72}\n  CNE6S vs CNE6L 横向 A/B 对照 [{RUN_TAG}]  {sd}~{ed}  "
          f"({INDEX.upper()}，每{rf}日换仓)\n{'='*72}")
    print("\n[行情] 加载 + 基准（一次，两组复用）...")
    wides, bm_ret = load_market(sd, ed)

    frontiers: dict[str, pd.DataFrame] = {}
    for label, data_dir, lam_grid in PANELS:
        rows = {}
        for lam in lam_grid:
            tag = f"{label}_λ{lam:g}"
            print(f"\n{'─'*72}\n>>> {tag}  cne6_data_dir={data_dir}  λ={lam}\n{'─'*72}")
            cfg_path = make_config(tag, data_dir, lam)
            weight_df = run_batch_optimize(cfg_path)
            rows[lam] = backtest_metrics(weight_df, wides, bm_ret)
        fr = pd.DataFrame(rows).T
        fr.index.name = "λ"
        frontiers[label] = fr

    # ── λ↔TE 前沿 ──
    print(f"\n{'='*72}\n  λ ↔ TE 前沿（两组分别扫描）\n{'='*72}")
    for label, fr in frontiers.items():
        print(f"\n[{label}]")
        print(fr[["实际TE", "年化超额", "IR", "超额MDD", "平均换手"]]
              .to_string(float_format=lambda x: f"{x:.4f}"))

    # ── matched-TE 对比：以两组 TE 范围交集的中点为目标 TE ──
    te_lo = max(fr["实际TE"].min() for fr in frontiers.values())
    te_hi = min(fr["实际TE"].max() for fr in frontiers.values())
    if te_lo < te_hi:
        target_te = (te_lo + te_hi) / 2
        print(f"\n{'='*72}\n  Matched-TE 对比  目标 TE ≈ {target_te:.4f}"
              f"（两组前沿交集 [{te_lo:.4f}, {te_hi:.4f}] 中点）\n{'='*72}")
        picks = {label: matched_te_pick(fr, target_te) for label, fr in frontiers.items()}
        cmp_df = pd.DataFrame(picks).T
        print(cmp_df[["实际TE", "年化超额", "IR", "超额MDD", "平均换手"]]
              .to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n  解读：在相近的实际跟踪误差（风险预算对齐）下，比较 IR/回撤/换手——")
        print("  IR 更高 / MDD 更浅 / 换手更低的一方，说明其风险模型对"
              "「月度换仓」这个持有期的风险预测更贴合实际。")
    else:
        print("\n  [警告] 两组 λ↔TE 前沿无重叠区间，扩大 λ 网格后重试 matched-TE 对比")


if __name__ == "__main__":
    main()
