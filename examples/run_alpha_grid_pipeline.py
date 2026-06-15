"""
VWAP5 合成 Alpha 网格 → 中证1000 指数增强 批量回测。

对 alphas/ 目录下全部 alpha_vwap5_*.parquet（不同 IC/ICIR/decay 组合，
基于未来H=5日 adj_vwap 涨跌幅构造），逐个跑：
  1. 批量优化（指数增强，复用 configs/zz1000_enhance_vwap5_test.yaml 的优化器配置）
  2. 真实执行回测（T+1 VWAP，涨跌停处理，买1‰卖2‰）

⚠️ 因子含未来信息（前视），仅用于模拟标定，不代表真实可交易表现。

输出：output/vwap5_grid/<因子名>/{weights, nav_realistic}.parquet, report_realistic.html
      output/vwap5_grid/summary.csv —— 各因子的绩效汇总

运行：
    python examples/run_alpha_grid_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from datetime import date
from pathlib import Path

import pandas as pd

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester
from portfolio_optimizer.backtest.report import generate_html_report
from portfolio_optimizer.pipeline.batch_optimize import run_batch_optimize, load_config

INDEX       = "zz1000"
INDEX_NAME  = "中证1000"
ALPHAS_DIR  = Path("alphas")
OUT_DIR     = Path("output/vwap5_grid")
BASE_CONFIG = "configs/zz1000_enhance_vwap5_test.yaml"

START_DATE = date(2020, 1, 2)
END_DATE   = date(2026, 5, 22)


def alpha_long_to_wide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    wide = df.pivot(index="date", columns="code", values="alpha").sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(BASE_CONFIG)
    base_cfg["backtest"]["start_date"] = START_DATE.isoformat()
    base_cfg["backtest"]["end_date"]   = END_DATE.isoformat()

    print(f"\n{'='*70}")
    print(f"  VWAP5 合成 Alpha 网格 → {INDEX_NAME} 指数增强批量回测")
    print(f"  区间: {START_DATE} ~ {END_DATE}")
    print(f"{'='*70}")

    data_start = date(START_DATE.year, 1, 1)
    print(f"\n[1] 加载行情数据（{data_start} ~ {END_DATE}）...")
    panel = load_panel(
        data_start, END_DATE,
        columns=[
            "code", "date", "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    print("\n[2] 预计算回测用宽表（adj_close / adj_vwap / limit / status）...")

    def to_wide(col: str) -> pd.DataFrame:
        d = (
            panel.select(["date", "code", col]).to_pandas()
            .pivot(index="date", columns="code", values=col)
            .sort_index()
        )
        d.index = pd.to_datetime(d.index)
        return d

    adj_close_w    = to_wide("adj_close")
    adj_vwap_w     = to_wide("adj_vwap")
    close_raw_w    = to_wide("close")
    limit_up_w     = to_wide("limit_up")
    limit_down_w   = to_wide("limit_down")
    trade_status_w = to_wide("trade_status")

    print(f"\n[3] 构建 {INDEX_NAME} 基准（分级靠档）...")
    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(START_DATE, END_DATE, panel=panel)
    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)

    daily_ret_all = adj_close_w.pct_change(fill_method=None).fillna(0.0)
    w_lag  = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)
    bm_ret.name = INDEX.upper()

    alpha_files = sorted(
        p for p in ALPHAS_DIR.glob("alpha_vwap5_*.parquet")
        if not p.stem.endswith("_wide")
    )
    print(f"\n[4] 因子网格：{len(alpha_files)} 个")
    for p in alpha_files:
        print(f"    {p.name}")

    summary_path = OUT_DIR / "summary.csv"
    summary_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    for i, path in enumerate(alpha_files, 1):
        tag = path.stem
        print(f"\n{'#'*70}")
        print(f"  [{i}/{len(alpha_files)}] {tag}")
        print(f"{'#'*70}")

        factor_out_dir = OUT_DIR / tag
        factor_out_dir.mkdir(parents=True, exist_ok=True)

        if (factor_out_dir / "nav_realistic.parquet").exists():
            print("  已完成，跳过")
            continue

        alpha_wide = alpha_long_to_wide(path)

        cfg = copy.deepcopy(base_cfg)
        weights_path = factor_out_dir / "weights.parquet"
        cfg["output"]["weights"] = str(weights_path)

        if weights_path.exists():
            print(f"  权重已存在，跳过优化：{weights_path}")
            weight_df = pd.read_parquet(weights_path)
        else:
            weight_df = run_batch_optimize(cfg, panel=panel, alpha_df=alpha_wide)

        weight_df.index = pd.to_datetime(weight_df.index)

        bt = RealisticBacktester(cost_buy=0.001, cost_sell=0.002, risk_free=0.02)
        result, exec_stats = bt.run(
            weight_df       = weight_df,
            adj_close       = adj_close_w,
            adj_vwap        = adj_vwap_w,
            close_raw       = close_raw_w,
            limit_up_df     = limit_up_w,
            limit_down_df   = limit_down_w,
            trade_status_df = trade_status_w,
            benchmark_ret   = bm_ret,
            initial_value   = 1e8,
        )

        print(f"\n{result.summary()}")

        nav_out = factor_out_dir / "nav_realistic.parquet"
        pd.DataFrame({
            "nav": result.nav, "bm_nav": result.bm_nav,
            "excess_nav": result.excess_nav,
            "port_ret": result.daily_ret, "bm_ret": result.bm_ret,
        }).to_parquet(nav_out)

        generate_html_report(
            result,
            output_path=factor_out_dir / "report_realistic.html",
            title=f"{INDEX_NAME} 指数增强（{tag}，VWAP5合成Alpha，真实执行）",
        )

        pm, bmm = result.portfolio_metrics, result.benchmark_metrics
        summary_rows.append({
            "tag": tag,
            "ann_ret": pm.annual_return,
            "ann_excess_ret": pm.annual_excess_return,
            "sharpe": pm.sharpe,
            "info_ratio": pm.info_ratio,
            "tracking_error": pm.tracking_error,
            "max_drawdown": pm.max_drawdown,
            "excess_max_drawdown": pm.excess_max_drawdown,
            "win_rate_monthly": pm.win_rate_monthly,
            "bm_ann_ret": bmm.annual_return,
            "turnover_mean": result.turnover.mean(),
            "buy_fail_count": exec_stats["buy_fail_count"],
            "sell_defer_count": exec_stats["sell_defer_count"],
        })

        # 增量保存，便于中途查看进度
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"  完成：{len(alpha_files)} 个因子 -> {OUT_DIR}/")
    print(f"  汇总表 -> {OUT_DIR / 'summary.csv'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
