"""
合成 Alpha 因子生成 + 单日优化 Demo。

构造方法：
    对每个截面日 t，取真实未来5日收益率 r_{t+5}，生成：
        ρ_t  ~ clip( N(IC_mean, IC_std²), -0.95, 0.95 )
        f_t  =  ρ_t · zscore(r_{t+5}) + √(1-ρ_t²) · ε
    使截面 corr(f_t, r_{t+5}) ≈ ρ_t

运行：
    cd /Users/guoguo/Desktop/HistQuantOpt
    python examples/demo_synthetic_alpha.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer

FACTOR_PATH = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")
IC_MEAN     = 0.10
IC_STD      = 0.07
SEED        = 42


# ──────────────────────────────────────────────────────────────
# 1. 生成合成 Alpha
# ──────────────────────────────────────────────────────────────

def build_synthetic_alpha(
    panel: pl.DataFrame,
    fwd_days: int  = 5,
    ic_mean: float = IC_MEAN,
    ic_std: float  = IC_STD,
    decay: float   = 0.90,
    seed: int      = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    生成全期合成 Alpha 矩阵，并返回每期实现 IC。

    decay 参数控制因子跨期持续性（AR-1 衰减系数）：
        f_t = decay · f_{t-1} + √(1-decay²) · new_signal_t

        decay = 0.00 → 每期独立，自相关≈0，日换手≈96%（纯噪声）
        decay = 0.70 → 自相关≈0.70，日换手≈30-40%
        decay = 0.90 → 自相关≈0.90，日换手≈10-15%（类动量因子）
        decay = 0.97 → 自相关≈0.97，日换手≈3-5%（长周期价值因子）

    IC 统计由 ic_mean / ic_std 控制，与 decay 独立。

    Returns
    -------
    alpha_df  : pd.DataFrame  (date × ticker)
    ic_series : pd.Series     每期实现 Pearson IC
    """
    rng = np.random.default_rng(seed)

    # adj_close 宽表
    adj = (
        panel.select(["date", "code", "adj_close"])
        .to_pandas()
        .pivot(index="date", columns="code", values="adj_close")
        .sort_index()
    )

    # 全量 ticker
    all_tickers = adj.columns.tolist()

    # 5 日未来收益率
    fwd_ret = adj.shift(-fwd_days) / adj - 1

    dates_with_fwd = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    alpha_rows: dict = {}
    ic_rows:    dict = {}
    f_prev: pd.Series | None = None   # 上期因子（全量 ticker）

    for dt in dates_with_fwd:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50:
            continue

        # ---- new_signal_t：由未来收益控制 IC ----
        mu, sigma = r.mean(), r.std()
        if sigma < 1e-8:
            continue
        z_r = (r - mu) / sigma

        rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps
        new_sig = pd.Series(new_sig, index=r.index)
        # 标准化 new_signal（截面 z-score）
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        # ---- AR-1 更新 ----
        if f_prev is None or decay == 0.0:
            f = new_sig
        else:
            # 对齐：只更新本期有数据的 ticker；上期缺失的用 new_sig 初始化
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    decay * f_prev[common] +
                    np.sqrt(max(1 - decay**2, 0)) * new_sig[common]
                )

        # 截面 z-score 标准化后存储
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f   # 保留全量（含本期有数据的所有 ticker）

        alpha_rows[dt] = f

        # 实现 IC（用 new_signal 对应的 z_r 验证，AR 混入历史后 IC 会略降）
        realized_ic = float(np.corrcoef(f[r.index].values, z_r.values)[0, 1])
        ic_rows[dt] = realized_ic

    alpha_df = pd.DataFrame(alpha_rows).T
    alpha_df.index.name = "date"

    ic_series = pd.Series(ic_rows, name="IC")
    return alpha_df, ic_series


# ──────────────────────────────────────────────────────────────
# 2. IC 统计汇总
# ──────────────────────────────────────────────────────────────

def print_ic_stats(ic: pd.Series) -> None:
    print(f"\n  {'期数':<10}: {len(ic)}")
    print(f"  {'IC Mean':<10}: {ic.mean():.4f}   (目标={IC_MEAN})")
    print(f"  {'IC Std':<10}: {ic.std():.4f}   (目标={IC_STD})")
    print(f"  {'IC IR':<10}: {ic.mean()/ic.std():.4f}")
    print(f"  {'IC>0 比例':<10}: {(ic>0).mean()*100:.1f}%")
    print(f"  {'IC 范围':<10}: [{ic.min():.4f}, {ic.max():.4f}]")

    # 简单 ASCII 直方图
    bins  = np.linspace(ic.min(), ic.max(), 13)
    hist, edges = np.histogram(ic, bins=bins)
    print(f"\n  IC 分布直方图：")
    max_h = max(hist)
    for i, (h, lo, hi) in enumerate(zip(hist, edges[:-1], edges[1:])):
        bar  = "█" * int(h / max_h * 20)
        mark = " ← 0" if lo <= 0 < hi else ""
        print(f"    [{lo:+.3f},{hi:+.3f})  {bar:<20} {h:>4}{mark}")


# ──────────────────────────────────────────────────────────────
# 3. 单日优化
# ──────────────────────────────────────────────────────────────

def run_single_opt(
    target_date: date,
    alpha_df: pd.DataFrame,
    panel: pl.DataFrame,
) -> None:
    print(f"\n{'─'*55}")
    print(f"  单日优化  {target_date}  (合成 Alpha IC≈0.1)")
    print(f"{'─'*55}")

    adapter = RealMarketAdapter()
    snapshot = adapter.build_snapshot_from_panel(
        panel=panel,
        target_date=target_date,
        index="hs300",
        portfolio_value=1e8,
    )

    barra = JYBarraFactors(
        snapshot=snapshot,
        target_date=target_date,
        factor_path=FACTOR_PATH,
        panel=panel,
    )
    style_loading = barra.style_loading

    # 对齐 alpha 到 snapshot.tickers（统一转 Timestamp 比较）
    ts = pd.Timestamp(target_date)
    if ts not in alpha_df.index:
        avail = alpha_df.index[alpha_df.index <= ts]
        target_date_alpha = avail[-1] if len(avail) > 0 else alpha_df.index[0]
    else:
        target_date_alpha = ts

    alpha_row = alpha_df.loc[target_date_alpha].reindex(snapshot.tickers).fillna(0.0).values

    config = AlphaMaxConfig(
        weight_upper=0.02,
        industry_upper=0.20,
        min_constituent_ratio=0.40,
        diversification_penalty=0.05,
        style_bound=1.0,
        max_turnover=None,
    )
    optimizer  = AlphaMaxOptimizer(config)
    result     = optimizer.optimize(alpha_row, snapshot, style_loading=style_loading)

    print(result.summary())

    exp = result.style_exposures(style_loading)
    print(f"\n  风格暴露（绝对值）：")
    for fname, val in exp.items():
        bar  = "█" * min(int(abs(val) * 10), 20)
        sign = "+" if val >= 0 else ""
        print(f"    {fname:<22} {sign}{val:+.3f}  {bar}")

    print(f"\n  前10大持仓：")
    top = result.top_holdings(10)
    for ticker, row in top.iterrows():
        ic_mark = "✓" if row.get("is_constituent", False) else " "
        ind     = str(row.get("industry", ""))[:14]
        print(f"    {ticker:<14} {row['weight_pct']:>5.3f}%  {ind:<16} {ic_mark}")

    print(f"\n  行业（前8）：")
    for ind, w in result.industry_weights().head(8).items():
        print(f"    {str(ind):<18} {w*100:>5.2f}%  {'▓'*int(w*100)}")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def factor_turnover_stats(alpha_df: pd.DataFrame, top_n: int = 200) -> dict:
    """计算因子日均换手率和自相关。"""
    from scipy.stats import spearmanr

    rank_corr, top_to = [], []
    for i in range(1, len(alpha_df)):
        prev = alpha_df.iloc[i - 1].dropna()
        curr = alpha_df.iloc[i].dropna()
        common = prev.index.intersection(curr.index)
        if len(common) < 100:
            continue
        rc, _ = spearmanr(prev[common], curr[common])
        rank_corr.append(rc)

        prev_top = set(prev.nlargest(top_n).index)
        curr_top = set(curr.nlargest(top_n).index)
        top_to.append(len(prev_top - curr_top) / top_n)

    return {
        "rank_corr_mean": float(np.mean(rank_corr)),
        "top_turnover_mean": float(np.mean(top_to)),
    }


def main() -> None:
    print(f"\n{'='*58}")
    print(f"  合成 Alpha 生成  IC={IC_MEAN}  IC_Std={IC_STD}")
    print(f"{'='*58}")

    # 加载全期数据
    print("\n[1] 加载全期行情...")
    panel = load_panel(
        date(2024, 1, 1), date(2026, 5, 22),
        columns=[
            "code", "date", "adj_close", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 不同 decay 对比 ──────────────────────────────────────
    print(f"\n[2] decay 参数对比（IC={IC_MEAN}, IC_Std={IC_STD}）\n")
    print(f"  {'decay':>6}  {'IC均值':>8}  {'IC_Std':>8}  {'IC IR':>7}  "
          f"{'自相关':>8}  {'Top200日换手':>12}")
    print(f"  {'-'*62}")

    best_df = None
    for decay in [0.0, 0.70, 0.90, 0.97]:
        adf, ics = build_synthetic_alpha(panel, decay=decay)
        ts = factor_turnover_stats(adf)
        print(f"  {decay:>6.2f}  {ics.mean():>8.4f}  {ics.std():>8.4f}  "
              f"{ics.mean()/ics.std():>7.3f}  "
              f"{ts['rank_corr_mean']:>8.4f}  "
              f"{ts['top_turnover_mean']*100:>11.1f}%")
        if decay == 0.90:
            best_df = (adf, ics)

    # ── 用 decay=0.90 做优化演示 ─────────────────────────────
    alpha_df, ic_series = best_df

    print(f"\n[3] IC 详细统计（decay=0.90）：")
    print_ic_stats(ic_series)

    target_date = alpha_df.index[-1]
    print(f"\n[4] 优化演示日期：{target_date.date()}")
    run_single_opt(target_date.date() if hasattr(target_date, 'date') else target_date,
                   alpha_df, panel)

    # 保存 decay=0.90 的 alpha
    out = "examples/synthetic_alpha.parquet"
    alpha_df.to_parquet(out)
    print(f"\n  Alpha（decay=0.90）已保存：{out}  {alpha_df.shape}")
    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    main()
