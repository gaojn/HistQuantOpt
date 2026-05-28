"""
量化多头批量优化 Demo（AlphaMaxOptimizer）。

Alpha   : 合成因子 IC=0.08, IC_Std=0.10, decay=0.80
回测区间 : 2026-05-01 ~ 2026-05-22
调仓频率 : 每5个交易日
优化目标 : max w'α - γ·‖w‖²
约束     : 单票≤2%、行业≤20%、HS300成分股≥40%、风格暴露≤1σ、换手≤50%

运行：
    cd /Users/guoguo/Desktop/HistQuantOpt
    python examples/demo_batch_alpha_max.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from datetime import date
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer

FACTOR_PATH    = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")
BACKTEST_START = date(2024, 6, 1)
BACKTEST_END   = date(2026, 5, 22)
REBAL_FREQ     = 5
INDEX          = "hs300"
PORTFOLIO_VAL  = 1e8


# ─────────────────────────────────────────────────────────────────
# 合成 Alpha（复用 demo_synthetic_alpha.py 的逻辑）
# ─────────────────────────────────────────────────────────────────

def build_synthetic_alpha(
    panel: pl.DataFrame,
    fwd_days: int  = 5,
    ic_mean: float = 0.10,
    ic_std: float  = 0.07,
    decay: float   = 0.90,
    seed: int      = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    adj = (
        panel.select(["date", "code", "adj_close"])
        .to_pandas()
        .pivot(index="date", columns="code", values="adj_close")
        .sort_index()
    )
    fwd_ret = adj.shift(-fwd_days) / adj - 1
    dates_with_fwd = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    alpha_rows: dict = {}
    f_prev: "pd.Series | None" = None

    for dt in dates_with_fwd:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50:
            continue
        mu, sigma = r.mean(), r.std()
        if sigma < 1e-8:
            continue
        z_r = (r - mu) / sigma
        rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = pd.Series(
            rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps,
            index=r.index,
        )
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        if f_prev is None or decay == 0.0:
            f = new_sig
        else:
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    decay * f_prev[common]
                    + np.sqrt(max(1 - decay**2, 0)) * new_sig[common]
                )
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f
        alpha_rows[dt] = f

    alpha_df = pd.DataFrame(alpha_rows).T
    alpha_df.index.name = "date"
    return alpha_df


def get_alpha_for_date(
    alpha_df: pd.DataFrame,
    target_date: date,
    tickers: list[str],
) -> np.ndarray:
    ts = pd.Timestamp(target_date)
    avail = alpha_df.index[alpha_df.index <= ts]
    if len(avail) == 0:
        return np.zeros(len(tickers))
    row = alpha_df.loc[avail[-1]]
    return row.reindex(tickers).fillna(0.0).values.astype(float)


# ─────────────────────────────────────────────────────────────────
# 批量优化主流程
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    print(f"\n{'='*58}")
    print(f"  量化多头批量优化  {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  调仓频率={REBAL_FREQ}日  指数={INDEX.upper()}  成分股≥40%")
    print(f"{'='*58}")

    # ── 1. 加载全期面板（生成 alpha 需要历史，从 2024-01 起）─────
    print("\n[1] 加载行情数据（2024-01 ~ 2026-05-22）...")
    full_panel = load_panel(
        date(2024, 1, 1), BACKTEST_END,
        columns=[
            "code", "date", "adj_close", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  交易日={full_panel['date'].n_unique()}  股票={full_panel['code'].n_unique()}")

    # ── 2. 生成合成 Alpha ─────────────────────────────────────
    print("\n[2] 生成合成 Alpha（IC=0.08, IC_std=0.10, decay=0.80）...")
    alpha_df = build_synthetic_alpha(
        full_panel, ic_mean=0.08, ic_std=0.10, decay=0.80,
    )
    print(f"  Alpha 矩阵: {alpha_df.shape}")

    # ── 3. 确定回测交易日 & 再平衡日 ─────────────────────────
    trade_dates = (
        full_panel
        .filter(
            (pl.col("date") >= BACKTEST_START) &
            (pl.col("date") <= BACKTEST_END)
        )
        .select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::REBAL_FREQ]
    print(f"\n  回测交易日数={len(trade_dates)}  再平衡日={len(rebal_dates)}")
    print(f"  再平衡日期：{[str(d) for d in rebal_dates]}")

    # ── 4. 优化配置 ───────────────────────────────────────────
    config = AlphaMaxConfig(
        weight_upper=0.02,
        industry_upper=0.20,
        min_constituent_ratio=0.40,
        diversification_penalty=0.05,
        style_bound=1.0,
        max_turnover=0.30,
    )
    optimizer = AlphaMaxOptimizer(config)
    adapter   = RealMarketAdapter()

    # ── 5. 逐期优化 ───────────────────────────────────────────
    print(f"\n[3] 逐期优化...")
    weight_records: dict[date, pd.Series] = {}
    prev_weight_arr: "np.ndarray | None"  = None
    prev_tickers:   "list[str] | None"    = None

    for rebal_date in rebal_dates:
        t0 = time.time()

        # 市场快照
        try:
            snapshot = adapter.build_snapshot_from_panel(
                panel=full_panel,
                target_date=rebal_date,
                index=INDEX,
                portfolio_value=PORTFOLIO_VAL,
            )
        except ValueError as e:
            print(f"  [{rebal_date}] 跳过（快照失败：{e}）")
            continue

        # Barra 因子
        barra = JYBarraFactors(
            snapshot=snapshot,
            target_date=rebal_date,
            factor_path=FACTOR_PATH,
            panel=full_panel,
        )

        # Alpha 对齐
        alpha = get_alpha_for_date(alpha_df, rebal_date, snapshot.tickers)

        # 上期权重对齐到当前 tickers
        if prev_weight_arr is not None and prev_tickers is not None:
            prev_s = pd.Series(prev_weight_arr, index=prev_tickers)
            prev_aligned = prev_s.reindex(snapshot.tickers).fillna(0.0).values
            s = prev_aligned.sum()
            prev_aligned = prev_aligned / s if s > 1e-8 else prev_aligned
        else:
            prev_aligned = None

        # 优化
        result = optimizer.optimize(
            alpha, snapshot,
            style_loading=barra.style_loading,
            prev_weight=prev_aligned,
        )

        elapsed = time.time() - t0

        if result.is_feasible:
            w = pd.Series(result.weights, index=snapshot.tickers)
            weight_records[rebal_date] = w
            prev_weight_arr = result.weights
            prev_tickers    = snapshot.tickers

            const_w  = result.weights[snapshot.constituent_mask].sum()
            turnover = (
                np.abs(result.weights - prev_aligned).sum()
                if prev_aligned is not None else float("nan")
            )
            print(f"  [{rebal_date}]  持仓={result.n_positions:3d}  "
                  f"成分股={const_w*100:.1f}%  "
                  f"换手={turnover*100:.1f}%  "
                  f"耗时={elapsed:.2f}s")
        else:
            print(f"  [{rebal_date}] 求解失败：{result.status}")

    if not weight_records:
        print("所有期均失败，请检查配置")
        return

    # ── 6. 结果汇总 ───────────────────────────────────────────
    weight_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weight_df.index.name = "date"

    print(f"\n{'='*58}")
    print(f"  回测汇总")
    print(f"{'='*58}")
    print(f"  再平衡期数     : {len(weight_df)}")
    print(f"  股票池大小     : {weight_df.shape[1]}")
    print(f"  平均持仓数     : {(weight_df > 1e-6).sum(axis=1).mean():.0f} 只")
    print(f"  总耗时         : {time.time()-t_total:.1f}s")

    # 双边换手
    to_series = weight_df.diff().abs().sum(axis=1).dropna()
    if len(to_series) > 0:
        print(f"  平均双边换手   : {to_series.mean()*100:.1f}%")
        print(f"  换手范围       : {to_series.min()*100:.1f}% ~ {to_series.max()*100:.1f}%")

    # 权重合计验证
    w_sum = weight_df.sum(axis=1)
    print(f"  权重和 min/max : {w_sum.min():.4f} / {w_sum.max():.4f}")

    # 逐期持仓详情
    print(f"\n{'─'*58}")
    print(f"  各期持仓概览")
    print(f"{'─'*58}")
    print(f"  {'日期':<14} {'持仓':>5}  {'前3大持仓'}")
    print(f"  {'-'*56}")
    for d, row in weight_df.iterrows():
        w = row[row > 1e-6].sort_values(ascending=False)
        top3 = "  ".join([f"{t}({v*100:.2f}%)" for t, v in w.head(3).items()])
        print(f"  {str(d):<14} {len(w):>5}  {top3}")

    # 保存
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "batch_weights_alpha_max.parquet"
    weight_df.to_parquet(out)
    print(f"\n  权重矩阵已保存：{out}")
    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    main()
