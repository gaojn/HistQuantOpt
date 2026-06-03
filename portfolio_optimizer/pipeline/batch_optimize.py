"""
批量组合优化 pipeline。

支持 index_enhance（指数增强）和 alpha_max（量化多头）两种策略。
通过 YAML 配置文件驱动，不依赖具体 demo 脚本。
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from portfolio_optimizer.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer
from portfolio_optimizer.pipeline.universe import (
    build_cost_vector, build_synthetic_alpha, filter_universe, get_alpha_for_date,
)

FACTOR_PATH = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")

_INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch_optimize(config_path: str | Path) -> pd.DataFrame:
    """
    读取 YAML 配置，执行批量组合优化，保存权重并返回权重矩阵。

    Parameters
    ----------
    config_path : str | Path
        YAML 配置文件路径（参见 configs/ 目录）

    Returns
    -------
    pd.DataFrame  index=date, columns=ticker, values=weight
    """
    cfg = load_config(config_path)
    strategy  = cfg["strategy"]          # "index_enhance" | "alpha_max"
    index     = cfg["index"]
    bt_cfg    = cfg["backtest"]
    uni_cfg   = cfg["universe"]
    opt_cfg   = cfg["optimizer"]
    alpha_cfg = cfg["alpha"]
    out_cfg   = cfg["output"]

    start_date = date.fromisoformat(bt_cfg["start_date"])
    end_date   = date.fromisoformat(bt_cfg["end_date"])
    rebal_freq = int(bt_cfg["rebalance_freq"])
    port_val   = float(bt_cfg["initial_value"])

    index_name = _INDEX_NAMES.get(index, index.upper())
    print(f"\n{'='*65}")
    print(f"  {index_name} {strategy} 批量优化  {start_date} ~ {end_date}")
    print(f"  调仓={rebal_freq}日  候选池: 剔除北交所+ST"
          + (f"  TOP_N={uni_cfg['top_n']}" if uni_cfg.get("top_n") else "  全市场"))
    print(f"{'='*65}")

    # ── 加载行情 ─────────────────────────────────────────────
    data_start = date(start_date.year, 1, 1)
    print(f"\n[1] 加载行情数据（{data_start} ~ {end_date}）...")
    panel = load_panel(
        data_start, end_date,
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

    # ── 合成 Alpha ───────────────────────────────────────────
    print(f"\n[2] 生成合成 Alpha（IC={alpha_cfg['ic_mean']}, decay={alpha_cfg['decay']}）...")
    alpha_df = build_synthetic_alpha(
        panel,
        fwd_days=int(alpha_cfg["fwd_days"]),
        ic_mean=float(alpha_cfg["ic_mean"]),
        ic_std=float(alpha_cfg["ic_std"]),
        decay=float(alpha_cfg["decay"]),
        seed=int(alpha_cfg["seed"]),
    )
    print(f"  Alpha 矩阵: {alpha_df.shape}")

    # ── 再平衡日 ─────────────────────────────────────────────
    trade_dates = (
        panel.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        ).select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::rebal_freq]
    print(f"\n  回测交易日数={len(trade_dates)}  再平衡日数={len(rebal_dates)}")

    # ── 优化器 ───────────────────────────────────────────────
    adapter = RealMarketAdapter()

    if strategy == "index_enhance":
        print(f"\n[3] 预计算 {index.upper()} 基准权重...")
        bm = IndexBenchmarkWeights(index=index, panel=panel)
        bm.precompute(start_date, end_date, panel=panel)

        base_config = IndexEnhanceConfig(
            weight_upper=float(opt_cfg["weight_upper"]),
            weight_lower=float(opt_cfg.get("weight_lower", 0.0)),
            min_constituent_ratio=float(opt_cfg["min_constituent_ratio"]),
            industry_active_bound=float(opt_cfg["industry_active_bound"]),
            style_active_bound=float(opt_cfg["style_active_bound"]),
            tracking_penalty=float(opt_cfg["tracking_penalty"]),
            max_turnover=float(opt_cfg["max_turnover"]) if opt_cfg.get("max_turnover") else None,
            turnover_penalty=float(opt_cfg.get("turnover_penalty", 0.0)),
            weight_diff_l2_bound=float(opt_cfg["weight_diff_l2_bound"]) if opt_cfg.get("weight_diff_l2_bound") else None,
        )
        optimizer = IndexEnhanceOptimizer(base_config)

    else:  # alpha_max
        base_config = AlphaMaxConfig(
            weight_upper=float(opt_cfg["weight_upper"]),
            industry_upper=float(opt_cfg.get("industry_upper", 0.20)),
            min_constituent_ratio=float(opt_cfg.get("min_constituent_ratio", 0.0)),
            diversification_penalty=float(opt_cfg.get("diversification_penalty", 0.05)),
            style_bound=float(opt_cfg["style_bound"]) if opt_cfg.get("style_bound") else None,
            max_turnover=float(opt_cfg["max_turnover"]) if opt_cfg.get("max_turnover") else None,
            turnover_penalty=float(opt_cfg.get("turnover_penalty", 0.0)),
        )
        optimizer = AlphaMaxOptimizer(base_config)

    use_cost_vector = (
        float(opt_cfg.get("turnover_penalty", 0.0)) > 0
        and bool(opt_cfg.get("liquidity_weighted_cost", True))
    )

    # ── 逐期优化 ─────────────────────────────────────────────
    print(f"\n[4] 逐期优化...")
    t_total = time.time()
    weight_records: dict = {}
    prev_w_arr, prev_tickers = None, None
    fail_count = 0
    solve_times = []

    for i, rebal_date in enumerate(rebal_dates):
        t0 = time.time()

        try:
            snap_full = adapter.build_snapshot_from_panel(
                panel=panel, target_date=rebal_date,
                index=index, portfolio_value=port_val,
            )
        except ValueError as e:
            print(f"  [{rebal_date}] 跳过（快照失败：{e}）")
            continue

        snapshot = filter_universe(
            snap_full, panel, rebal_date,
            exclude_bj=bool(uni_cfg.get("exclude_bj", True)),
            exclude_st=bool(uni_cfg.get("exclude_st", True)),
            top_n=int(uni_cfg["top_n"]) if uni_cfg.get("top_n") else None,
        )

        barra = JYBarraFactors(
            snapshot=snapshot, target_date=rebal_date,
            factor_path=FACTOR_PATH, panel=panel,
        )
        alpha = get_alpha_for_date(alpha_df, rebal_date, snapshot.tickers)

        # 上期权重对齐
        if prev_w_arr is not None and prev_tickers is not None:
            ps = pd.Series(prev_w_arr, index=prev_tickers) \
                .reindex(snapshot.tickers).fillna(0.0).values
            s = ps.sum()
            ps = ps / s if s > 1e-8 else ps
        else:
            ps = None

        # 个股冲击成本权重（仅在启用换手软惩罚时计算）
        cost_vec = None
        if use_cost_vector and ps is not None:
            cost_vec = build_cost_vector(
                tickers=snapshot.tickers,
                panel=panel,
                target_date=rebal_date,
            )

        # 优化
        if strategy == "index_enhance":
            bm_series = bm.get_weights(rebal_date, tickers=snapshot.tickers)
            bm_weight = bm_series.values

            cfg_this = base_config if ps is not None else IndexEnhanceConfig(
                **{**base_config.__dict__, "max_turnover": None}
            )
            optimizer.config = cfg_this
            result = optimizer.optimize(
                alpha=alpha, snapshot=snapshot,
                benchmark_weight=bm_weight,
                style_loading=barra.style_loading,
                prev_weight=ps,
                cost_vector=cost_vec,
            )
        else:
            result = optimizer.optimize(
                alpha, snapshot,
                style_loading=barra.style_loading,
                prev_weight=ps,
                cost_vector=cost_vec,
            )

        elapsed = time.time() - t0
        solve_times.append(elapsed)

        if result.is_feasible:
            w = pd.Series(result.weights, index=snapshot.tickers)
            weight_records[rebal_date] = w
            prev_w_arr, prev_tickers = result.weights, snapshot.tickers

            turnover = float(np.abs(result.weights - ps).sum()) if ps is not None else float("nan")
            if i % 10 == 0 or i == len(rebal_dates) - 1:
                extra = ""
                if strategy == "index_enhance":
                    const_w = result.weights[snapshot.constituent_mask].sum()
                    te_l2 = result.tracking_error_l2()
                    extra = f"  {index.upper()}={const_w*100:.1f}%  TE_L2={te_l2:.4f}"
                print(f"  [{i+1:3d}/{len(rebal_dates)}] {rebal_date}  "
                      f"持仓={result.n_positions:3d}  换手={turnover*100:>5.1f}%{extra}  耗时={elapsed:.2f}s")
        else:
            fail_count += 1
            print(f"  [{rebal_date}] ✗ 求解失败：{result.status}")
            if prev_w_arr is not None:
                w = pd.Series(prev_w_arr, index=prev_tickers) \
                    .reindex(snapshot.tickers).fillna(0.0)
                weight_records[rebal_date] = w

    # ── 汇总 & 保存 ──────────────────────────────────────────
    if not weight_records:
        raise RuntimeError("所有期均求解失败，请检查配置")

    weight_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weight_df.index.name = "date"
    turnover_arr = weight_df.diff().abs().sum(axis=1).dropna()

    print(f"\n{'='*65}\n  批量优化汇总\n{'='*65}")
    print(f"  再平衡期数   : {len(weight_df)}")
    print(f"  失败期数     : {fail_count}")
    print(f"  平均持仓数   : {(weight_df > 1e-6).sum(axis=1).mean():.0f} 只")
    print(f"  平均双边换手 : {turnover_arr.mean()*100:.1f}%")
    print(f"  平均耗时     : {np.mean(solve_times):.2f}s  总耗时: {time.time()-t_total:.1f}s")

    out_path = Path(out_cfg["weights"])
    out_path.parent.mkdir(exist_ok=True)
    weight_df.to_parquet(out_path)
    print(f"\n  权重矩阵已保存：{out_path}")
    print(f"\n{'='*65}\n")

    return weight_df
