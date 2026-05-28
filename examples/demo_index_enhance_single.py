"""
HS300 指数增强单日优化 Demo（用于验证可行性）。

目标   : max w'α - γ‖w - w_bm‖²
基准   : 沪深300（分级靠档加权）
候选池 : 沪深300 + 中证500 + 中证1000（约1800只）
日期   : 2026-05-21
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
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.index_enhance import (
    IndexEnhanceConfig, IndexEnhanceOptimizer,
)

TARGET_DATE = date(2026, 5, 21)
FACTOR_PATH = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")


def filter_universe(snapshot, panel: pl.DataFrame, target_date: date):
    """限定候选池为 HS300 ∪ ZZ500 ∪ ZZ1000。"""
    today = (
        panel.filter(pl.col("date") == target_date)
        .select(["code", "is_hs300", "is_zz500", "is_zz1000"])
        .to_pandas().set_index("code")
    )
    universe = (
        (today["is_hs300"] == 1) |
        (today["is_zz500"] == 1) |
        (today["is_zz1000"] == 1)
    )
    keep_set = set(today.index[universe].tolist())
    keep_tickers = [t for t in snapshot.tickers if t in keep_set]

    # 重建快照
    from dataclasses import replace
    return replace(
        snapshot,
        tickers=keep_tickers,
        industry=snapshot.industry.reindex(keep_tickers),
        adv=snapshot.adv.reindex(keep_tickers),
        status=snapshot.status.reindex(keep_tickers),
        prev_weight=snapshot.prev_weight.reindex(keep_tickers).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(keep_tickers),
        is_constituent=(
            snapshot.is_constituent.reindex(keep_tickers)
            if snapshot.is_constituent is not None else None
        ),
    )


def main():
    print(f"\n{'='*60}")
    print(f"  HS300 指数增强 - 单日优化  {TARGET_DATE}")
    print(f"{'='*60}")

    # 加载行情
    print("\n[1] 加载行情...")
    panel = load_panel(
        date(2026, 4, 1), TARGET_DATE,
        columns=[
            "code", "date", "close", "adj_close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  全市场股票={panel['code'].n_unique()}")

    # 快照（HS300 基准）
    print("\n[2] 构建市场快照（基准=HS300）...")
    adapter = RealMarketAdapter()
    snap_full = adapter.build_snapshot_from_panel(
        panel=panel, target_date=TARGET_DATE,
        index="hs300", portfolio_value=1e8,
    )
    print(f"  全市场快照: {len(snap_full.tickers)} 只")

    # 过滤候选池
    print("\n[3] 过滤候选池 HS300+ZZ500+ZZ1000...")
    snapshot = filter_universe(snap_full, panel, TARGET_DATE)
    n_hs300 = int(snapshot.constituent_mask.sum())
    print(f"  候选池: {len(snapshot.tickers)} 只  (HS300成分股 {n_hs300} 只)")

    # 基准权重
    print("\n[4] 计算 HS300 基准权重...")
    bm = IndexBenchmarkWeights(index="hs300", panel=panel)
    bm.precompute(date(2026, 4, 1), TARGET_DATE, panel=panel)
    bm_series = bm.get_weights(TARGET_DATE, tickers=snapshot.tickers)
    bm_weight = bm_series.values
    print(f"  基准权重和={bm_weight.sum():.6f}  非零={int((bm_weight>0).sum())}")

    # Barra 因子
    print("\n[5] 加载聚源 Barra 因子...")
    barra = JYBarraFactors(
        snapshot=snapshot, target_date=TARGET_DATE,
        factor_path=FACTOR_PATH, panel=panel,
    )
    style_loading = barra.style_loading

    # 合成 alpha（线性组合 Barra 因子）
    print("\n[6] 合成 Alpha（占位）...")
    weights = {"Momentum": 0.3, "EarningsYield": 0.25, "BookToPrice": 0.2,
               "Growth": 0.15, "Size": -0.1}
    alpha = np.zeros(len(snapshot.tickers))
    for f, w in weights.items():
        if f in style_loading.columns:
            alpha += w * style_loading[f].values

    # 优化
    print("\n[7] 执行 HS300 指数增强优化...")
    cfg = IndexEnhanceConfig(
        weight_upper=0.05,
        min_constituent_ratio=0.80,
        industry_active_bound=0.05,
        style_active_bound=0.30,
        tracking_penalty=10.0,
        max_turnover=None,    # 首期不约束
    )
    optimizer = IndexEnhanceOptimizer(cfg)
    result = optimizer.optimize(
        alpha=alpha, snapshot=snapshot,
        benchmark_weight=bm_weight,
        style_loading=style_loading,
        prev_weight=None,
    )

    # 结果
    print(f"\n{'='*60}\n  优化结果\n{'='*60}")
    print(result.summary())

    # 风格主动暴露
    exp = result.style_active_exposure(style_loading)
    print(f"\n  风格因子主动暴露（相对基准）：")
    for fn, v in exp.items():
        bar = "█" * min(int(abs(v) * 30), 20)
        sign = "+" if v >= 0 else ""
        print(f"    {fn:<22} {sign}{v:+.4f}  {bar}")

    # 行业主动权重
    print(f"\n  行业主动权重（前/后5）：")
    ind_active = result.industry_active_weights()
    for ind, v in ind_active.head(5).items():
        print(f"    [+]  {str(ind):<14} +{v*100:.2f}%")
    print(f"    ...")
    for ind, v in ind_active.tail(5).items():
        print(f"    [-]  {str(ind):<14}  {v*100:+.2f}%")

    # 前10
    print(f"\n  前10大持仓：")
    top = result.top_holdings(10)
    print(f"  {'代码':<13} {'权重':>6} {'基准':>6} {'主动':>7}  {'行业':<14} {'成分股'}")
    print(f"  {'-'*60}")
    for t, row in top.iterrows():
        ic = "✓" if row.get("is_constituent", False) else " "
        ind = str(row.get("industry", ""))[:12]
        print(f"  {t:<13} {row['weight_pct']:>5.2f}% {row['bm_weight_pct']:>5.2f}% "
              f"{row['active_pct']:>+6.2f}%  {ind:<14} {ic}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
