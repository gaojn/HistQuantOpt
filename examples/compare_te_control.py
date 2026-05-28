"""
TE 控制方案对比：
  基准（当前）: industry=7%, style=0.5, gamma=10
  方案3:       industry=5%, style=0.3, gamma=10
  方案1:       industry=7%, style=0.5, gamma=20

运行：
    python examples/compare_te_control.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace
from datetime import date
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer
from portfolio_optimizer.backtest.engine import Backtester

FACTOR_PATH    = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")
INDEX          = "zz500"
BACKTEST_START = date(2023, 6, 1)
BACKTEST_END   = date(2026, 5, 22)
REBAL_FREQ     = 5
PORTFOLIO_VAL  = 1e8

IC_MEAN = 0.08
IC_STD  = 0.10
DECAY   = 0.80
SEED    = 42

SCENARIOS = {
    "基准(当前)":     dict(industry_active_bound=0.07, style_active_bound=0.50, tracking_penalty=10.0, weight_diff_l2_bound=None),
    "方案1 γ=40":    dict(industry_active_bound=0.07, style_active_bound=0.50, tracking_penalty=40.0, weight_diff_l2_bound=None),
    "方案2 L2硬约束": dict(industry_active_bound=0.07, style_active_bound=0.50, tracking_penalty=10.0, weight_diff_l2_bound=0.088),
    "方案2+1 组合":   dict(industry_active_bound=0.07, style_active_bound=0.50, tracking_penalty=20.0, weight_diff_l2_bound=0.088),
}


# ── Alpha 生成（与 demo_batch_index_enhance 一致）──────────────────

def build_synthetic_alpha(panel, fwd_days=5):
    rng = np.random.default_rng(SEED)
    adj = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    fwd_ret = adj.shift(-fwd_days) / adj - 1
    dates = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    rows: dict = {}
    f_prev = None
    for dt in dates:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50: continue
        mu, sig = r.mean(), r.std()
        if sig < 1e-8: continue
        z_r = (r - mu) / sig
        rho = float(np.clip(rng.normal(IC_MEAN, IC_STD), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = pd.Series(
            rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps,
            index=r.index,
        )
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)
        if f_prev is None or DECAY == 0.0:
            f = new_sig
        else:
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    DECAY * f_prev[common]
                    + np.sqrt(max(1 - DECAY**2, 0)) * new_sig[common]
                )
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f
        rows[dt] = f
    return pd.DataFrame(rows).T


def get_alpha_for_date(alpha_df, target_date, tickers):
    ts = pd.Timestamp(target_date)
    avail = alpha_df.index[alpha_df.index <= ts]
    if len(avail) == 0:
        return np.zeros(len(tickers))
    return alpha_df.loc[avail[-1]].reindex(tickers).fillna(0.0).values.astype(float)


def filter_universe(snapshot, panel, target_date):
    today = (
        panel.filter(pl.col("date") == target_date)
        .select(["code", "is_st"]).to_pandas().set_index("code")
    )
    keep_mask = (~today.index.str.endswith(".BJ")) & (today["is_st"] != 1)
    keep_set  = set(today.index[keep_mask].tolist())
    keep = [t for t in snapshot.tickers if t in keep_set]
    return replace(
        snapshot, tickers=keep,
        industry=snapshot.industry.reindex(keep),
        adv=snapshot.adv.reindex(keep),
        status=snapshot.status.reindex(keep),
        prev_weight=snapshot.prev_weight.reindex(keep).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(keep),
        is_constituent=(
            snapshot.is_constituent.reindex(keep)
            if snapshot.is_constituent is not None else None
        ),
    )


# ── 单场景批量优化 ────────────────────────────────────────────────

def run_scenario(name, cfg_overrides, panel, alpha_df, bm, rebal_dates, adapter):
    weight_upper = {"hs300": 0.05, "zz500": 0.02, "zz1000": 0.01}.get(INDEX, 0.03)
    config = IndexEnhanceConfig(
        weight_upper=weight_upper,
        min_constituent_ratio=0.80,
        max_turnover=0.40,
        industry_active_bound=cfg_overrides.get("industry_active_bound", 0.07),
        style_active_bound=cfg_overrides.get("style_active_bound", 0.50),
        tracking_penalty=cfg_overrides.get("tracking_penalty", 10.0),
        weight_diff_l2_bound=cfg_overrides.get("weight_diff_l2_bound", None),
    )
    optimizer   = IndexEnhanceOptimizer(config)
    weight_records: dict = {}
    prev_w_arr, prev_tickers = None, None
    fail_count  = 0
    t0_total    = time.time()

    for i, rebal_date in enumerate(rebal_dates):
        try:
            snap_full = adapter.build_snapshot_from_panel(
                panel=panel, target_date=rebal_date,
                index=INDEX, portfolio_value=PORTFOLIO_VAL,
            )
        except ValueError:
            continue

        snapshot = filter_universe(snap_full, panel, rebal_date)
        barra    = JYBarraFactors(
            snapshot=snapshot, target_date=rebal_date,
            factor_path=FACTOR_PATH, panel=panel,
        )
        alpha    = get_alpha_for_date(alpha_df, rebal_date, snapshot.tickers)
        bm_series = bm.get_weights(rebal_date, tickers=snapshot.tickers)
        bm_weight = bm_series.values

        if prev_w_arr is not None:
            ps = pd.Series(prev_w_arr, index=prev_tickers) \
                .reindex(snapshot.tickers).fillna(0.0).values
            s  = ps.sum()
            ps = ps / s if s > 1e-8 else ps
        else:
            ps = None

        if ps is None:
            no_to = {**config.__dict__, "max_turnover": None}
            optimizer.config = IndexEnhanceConfig(**no_to)
        else:
            optimizer.config = config

        result = optimizer.optimize(
            alpha=alpha, snapshot=snapshot,
            benchmark_weight=bm_weight,
            style_loading=barra.style_loading,
            prev_weight=ps,
        )

        if result.is_feasible:
            w = pd.Series(result.weights, index=snapshot.tickers)
            weight_records[rebal_date] = w
            prev_w_arr, prev_tickers = result.weights, snapshot.tickers
        else:
            fail_count += 1
            if prev_w_arr is not None:
                w = pd.Series(prev_w_arr, index=prev_tickers) \
                    .reindex(snapshot.tickers).fillna(0.0)
                weight_records[rebal_date] = w

    elapsed = time.time() - t0_total
    print(f"  [{name}] 完成: {len(weight_records)}期成功, {fail_count}期失败, 耗时{elapsed:.0f}s")
    return pd.DataFrame(weight_records).T.fillna(0.0)


# ── 回测 ─────────────────────────────────────────────────────────

def run_backtest(weight_df, panel, bm_calc):
    adj_wide = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    adj_wide.index = pd.to_datetime(adj_wide.index)
    weight_df.index = pd.to_datetime(weight_df.index)

    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)
    daily_ret_all = adj_wide.pct_change(fill_method=None).fillna(0.0)
    w_lag  = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)

    bt = Backtester(cost_one_way=0.0015, risk_free=0.02)
    return bt.run(weight_df, adj_wide, benchmark_ret=bm_ret)


# ── main ─────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print(f"  TE 控制方案对比  {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"{'='*65}")

    data_start = date(BACKTEST_START.year, 1, 1)
    print(f"\n[1] 加载行情...")
    panel = load_panel(
        data_start, BACKTEST_END,
        columns=[
            "code", "date", "adj_close", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )

    print(f"\n[2] 生成 Alpha...")
    alpha_df = build_synthetic_alpha(panel)

    print(f"\n[3] 预计算基准权重...")
    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(BACKTEST_START, BACKTEST_END, panel=panel)

    trade_dates = (
        panel.filter(
            (pl.col("date") >= BACKTEST_START) & (pl.col("date") <= BACKTEST_END)
        ).select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::REBAL_FREQ]
    adapter     = RealMarketAdapter()

    # ── 逐场景运行 ──────────────────────────────────────────────
    results = {}
    print(f"\n[4] 逐场景优化 + 回测...")
    for name, cfg in SCENARIOS.items():
        print(f"\n  >>> {name}")
        weight_df = run_scenario(name, cfg, panel, alpha_df, bm_calc, rebal_dates, adapter)
        bt_result = run_backtest(weight_df, panel, bm_calc)
        results[name] = bt_result

    # ── 对比表 ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  对比结果")
    print(f"{'='*75}")

    metrics = [
        ("年化收益",    lambda r: f"{r.portfolio_metrics.annual_return*100:+.2f}%"),
        ("年化波动",    lambda r: f"{r.portfolio_metrics.annual_vol*100:.2f}%"),
        ("Sharpe",     lambda r: f"{r.portfolio_metrics.sharpe:.3f}"),
        ("最大回撤",    lambda r: f"{r.portfolio_metrics.max_drawdown*100:.2f}%"),
        ("年化超额",    lambda r: f"{r.portfolio_metrics.annual_excess_return*100:+.2f}%"),
        ("跟踪误差TE",  lambda r: f"{r.portfolio_metrics.tracking_error*100:.2f}%"),
        ("信息比率IR",  lambda r: f"{r.portfolio_metrics.info_ratio:.3f}"),
        ("超额最大回撤",lambda r: f"{r.portfolio_metrics.excess_max_drawdown*100:.2f}%"),
        ("超额Calmar",  lambda r: f"{r.portfolio_metrics.excess_calmar:.3f}"),
        ("月度胜率",    lambda r: f"{r.portfolio_metrics.win_rate_monthly*100:.1f}%"),
        ("月均超额",    lambda r: f"{r.portfolio_metrics.avg_monthly_excess*100:+.3f}%"),
        ("平均换手",    lambda r: f"{r.turnover.mean()*100:.1f}%"),
    ]

    names = list(results.keys())
    col_w = 22
    header = f"{'指标':<14}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * (14 + col_w * len(names)))
    for label, fn in metrics:
        row = f"{label:<14}"
        for n in names:
            row += f"{fn(results[n]):>{col_w}}"
        print(row)

    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    main()
