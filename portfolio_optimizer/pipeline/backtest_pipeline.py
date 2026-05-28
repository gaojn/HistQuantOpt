"""
真实回测 pipeline。

从 YAML 配置或直接参数驱动，执行 T+1 VWAP 真实回测并生成 HTML 报告。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import yaml

from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester
from portfolio_optimizer.backtest.report import generate_html_report
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.io.data_panel import load_panel

_INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest(config_path: str | Path) -> None:
    """
    读取 YAML 配置，执行真实回测并生成 HTML 报告。

    Parameters
    ----------
    config_path : str | Path
        YAML 配置文件路径（参见 configs/ 目录）
    """
    cfg = load_config(config_path)
    index    = cfg["index"]
    bt_cfg   = cfg["backtest"]
    exec_cfg = cfg["execution"]
    out_cfg  = cfg["output"]

    index_name = _INDEX_NAMES.get(index, index.upper())
    strategy_label = {"index_enhance": "指数增强", "alpha_max": "量化多头"}.get(
        cfg["strategy"], cfg["strategy"]
    )

    print(f"\n{'='*60}")
    print(f"  {index_name} {strategy_label}回测（真实执行模式）")
    print(f"  T+1 VWAP 成交 | 涨跌停处理 | 买1‰ 卖2‰")
    print(f"{'='*60}")

    # ── 权重矩阵 ─────────────────────────────────────────────
    print("\n[1] 加载权重矩阵...")
    weight_df = pd.read_parquet(out_cfg["weights"])
    weight_df.index = pd.to_datetime(weight_df.index)
    start_date = weight_df.index[0].date()
    end_date   = weight_df.index[-1].date()
    print(f"  调仓期数={len(weight_df)}  股票池={weight_df.shape[1]}")
    print(f"  区间: {start_date} ~ {end_date}")

    # ── 行情数据 ─────────────────────────────────────────────
    print("\n[2] 加载行情（adj_close / adj_vwap / limit / status）...")
    panel = load_panel(
        start_date, end_date,
        columns=[
            "code", "date",
            "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "trade_status",
            "free_mv", "total_mv",
            "is_hs300", "is_zz500", "is_zz1000",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    def to_wide(col: str) -> pd.DataFrame:
        df = (
            panel.select(["date", "code", col]).to_pandas()
            .pivot(index="date", columns="code", values=col)
            .sort_index()
        )
        df.index = pd.to_datetime(df.index)
        return df

    adj_close_w    = to_wide("adj_close")
    adj_vwap_w     = to_wide("adj_vwap")
    close_raw_w    = to_wide("close")
    limit_up_w     = to_wide("limit_up")
    limit_down_w   = to_wide("limit_down")
    trade_status_w = to_wide("trade_status")

    # ── 基准收益 ─────────────────────────────────────────────
    print(f"\n[3] 构建 {index_name} 基准（分级靠档）...")
    bm_calc = IndexBenchmarkWeights(index=index, panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)

    daily_ret_all = adj_close_w.pct_change(fill_method=None).fillna(0.0)
    w_lag  = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)
    bm_ret.name = index.upper()

    # ── 真实回测 ─────────────────────────────────────────────
    print("\n[4] 执行真实回测（T+1 VWAP，涨跌停处理）...")
    bt = RealisticBacktester(
        cost_buy=float(exec_cfg["cost_buy"]),
        cost_sell=float(exec_cfg["cost_sell"]),
        risk_free=float(exec_cfg["risk_free"]),
    )
    result, exec_stats = bt.run(
        weight_df       = weight_df,
        adj_close       = adj_close_w,
        adj_vwap        = adj_vwap_w,
        close_raw       = close_raw_w,
        limit_up_df     = limit_up_w,
        limit_down_df   = limit_down_w,
        trade_status_df = trade_status_w,
        benchmark_ret   = bm_ret,
        initial_value   = float(bt_cfg["initial_value"]),
    )

    # ── 控制台输出 ────────────────────────────────────────────
    print(f"\n{result.summary()}")
    print(f"\n{'─'*60}")
    print(f"  执行质量统计")
    print(f"{'─'*60}")
    print(f"  涨停/停牌 无法买入次数  : {exec_stats['buy_fail_count']}")
    print(f"  跌停/停牌 延迟卖出次数  : {exec_stats['sell_defer_count']}")
    print(f"  平均现金占比           : {exec_stats['avg_cash_pct']*100:.2f}%")

    print(f"\n{'─'*60}")
    print(f"  年度收益分解（超额=几何）")
    print(f"{'─'*60}")
    print(f"  {'年份':<10} {'组合':>9}  {'基准':>9}  {'超额':>9}  {'最大回撤':>10}")
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        port_y = (1 + grp).prod() - 1
        bm_y   = (1 + result.bm_ret.reindex(grp.index).fillna(0)).prod() - 1
        exc_y  = (1 + port_y) / (1 + bm_y) - 1 if abs(1 + bm_y) > 1e-8 else 0.0
        mdd_y  = float(((1 + grp).cumprod() / (1 + grp).cumprod().cummax() - 1).min())
        print(f"  {year} ({len(grp):>3d}d) "
              f"{port_y*100:>+8.2f}%  {bm_y*100:>+8.2f}%  "
              f"{exc_y*100:>+8.2f}%  {mdd_y*100:>9.2f}%")

    to = result.turnover
    print(f"\n  平均双边换手  : {to.mean()*100:.1f}%  年化: {to.mean()*100*52:.0f}%")

    # ── 保存 & 报告 ───────────────────────────────────────────
    out_dir = Path(out_cfg["nav"]).parent
    out_dir.mkdir(exist_ok=True)

    pd.DataFrame({
        "nav": result.nav, "bm_nav": result.bm_nav,
        "excess_nav": result.excess_nav,
        "port_ret": result.daily_ret, "bm_ret": result.bm_ret,
    }).to_parquet(out_cfg["nav"])

    report_path = generate_html_report(
        result,
        output_path=out_cfg["report"],
        title=out_cfg["report_title"],
    )
    print(f"\n  净值已保存  : {out_cfg['nav']}")
    print(f"  HTML 报告   : {report_path}")
    print(f"\n{'='*60}\n")
