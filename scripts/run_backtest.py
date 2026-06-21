"""
独立回测脚本：传入权重长表 + 回测区间，直接输出净值与报告。

用法（命令行）：
    python scripts/run_backtest.py \
        --weights  output/my_weights.parquet \
        --start    2022-01-01 \
        --end      2025-12-31 \
        --index    zz1000 \
        --out-dir  output/my_backtest

也可作为函数调用：
    from scripts.run_backtest import run_backtest
    result, exec_stats = run_backtest(
        weight_path="output/my_weights.parquet",
        start_date="2022-01-01",
        end_date="2025-12-31",
    )

权重文件格式（二选一）：
  - 长表 parquet：列 [date, code, weight]
  - 宽表 parquet：index=date, columns=ticker
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import polars as pl

from portfolio_optimizer.backtest.engine import RealisticBacktester
from portfolio_optimizer.backtest.report import generate_html_report
from portfolio_optimizer.data.index_close import load_index_returns
from portfolio_optimizer.io.data_panel import load_panel


# ── 辅助函数 ─────────────────────────────────────────────────────


def _load_weights(path: str | Path) -> pd.DataFrame:
    """
    加载权重文件，统一输出宽表（index=DatetimeIndex，columns=ticker）。

    自动识别长表（含 date/code/weight 列）和宽表（index 为日期）。
    """
    df = pd.read_parquet(path)

    # 长表识别
    if {"date", "code", "weight"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="code", values="weight").sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    df.index.name = "date"
    return df


def _to_wide(panel: pl.DataFrame, col: str) -> pd.DataFrame:
    """从 polars 面板 pivot 指定列为宽表，index=DatetimeIndex。"""
    wide = (
        panel.select(["date", "code", col]).to_pandas()
        .pivot(index="date", columns="code", values=col)
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide


def _parse_date(s: str | date | pd.Timestamp) -> date:
    if isinstance(s, date):
        return s
    return pd.Timestamp(s).date()


# ── 核心函数 ─────────────────────────────────────────────────────


def run_backtest(
    weight_path: str | Path,
    start_date: str | date,
    end_date: str | date,
    index: str = "zz1000",
    cost_buy: float = 0.001,
    cost_sell: float = 0.002,
    risk_free: float = 0.02,
    initial_value: float = 1e8,
    out_dir: str | Path | None = None,
    title: str | None = None,
    cache_dir: str | Path | None = None,
    index_close_path: str | Path | None = None,
) -> tuple:
    """
    加载权重文件，执行 T+1 VWAP 真实回测，生成 HTML 报告。

    Parameters
    ----------
    weight_path : 权重 parquet 路径（长表或宽表，见模块文档）
    start_date  : 回测起始日（权重早于此日期的记录会被裁剪）
    end_date    : 回测截止日
    index       : 基准指数 key（hs300 / zz500 / zz1000）
    cost_buy    : 买入费率，默认 0.1‰
    cost_sell   : 卖出费率，默认 0.2‰
    risk_free   : 无风险年化利率（Sharpe 用）
    initial_value : 初始资金（元）
    out_dir     : 报告输出目录，None 则不保存文件
    title       : 报告标题，None 则自动生成
    cache_dir   : 行情 parquet 缓存目录，None 使用 load_panel 默认路径
    index_close_path : 指数收盘价 CSV 路径，None 使用 load_index_returns 默认路径

    Returns
    -------
    (BacktestResult, exec_stats dict)
    """
    t1 = _parse_date(start_date)
    t2 = _parse_date(end_date)

    # ── 1. 权重 ──────────────────────────────────────────────────
    print(f"\n[1] 加载权重：{weight_path}")
    weight_df = _load_weights(weight_path)

    # 裁剪到回测区间
    weight_df = weight_df[
        (weight_df.index >= pd.Timestamp(t1)) &
        (weight_df.index <= pd.Timestamp(t2))
    ]
    if weight_df.empty:
        raise ValueError(f"权重文件在 {t1}~{t2} 无数据，请检查日期区间")

    actual_start = weight_df.index.min().date()
    actual_end   = weight_df.index.max().date()
    n_rebal      = len(weight_df)
    n_stocks     = (weight_df > 1e-6).any(axis=0).sum()
    print(f"  调仓日={n_rebal}  股票池={n_stocks}  区间={actual_start}~{actual_end}")

    # ── 2. 行情面板 ───────────────────────────────────────────────
    # 多取年初数据，确保首期 ADV / VWAP 都有
    data_start = date(t1.year, 1, 1)
    print(f"\n[2] 加载行情面板（{data_start} ~ {t2}）...")
    panel = load_panel(
        data_start, t2,
        columns=[
            "code", "date",
            "adj_close", "adj_vwap",
            "close", "limit_up", "limit_down",
            "trade_status",
        ],
        cache_dir=cache_dir,
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 3. 宽表转换 ───────────────────────────────────────────────
    print("\n[3] 构建回测宽表...")
    adj_close_w    = _to_wide(panel, "adj_close")
    adj_vwap_w     = _to_wide(panel, "adj_vwap")
    close_raw_w    = _to_wide(panel, "close")
    limit_up_w     = _to_wide(panel, "limit_up")
    limit_down_w   = _to_wide(panel, "limit_down")
    trade_status_w = _to_wide(panel, "trade_status")

    # ── 4. 基准收益 ───────────────────────────────────────────────
    print(f"\n[4] 加载基准收益（{index.upper()}）...")
    index_close_kwargs = {} if index_close_path is None else {"path": index_close_path}
    bm_ret = (
        load_index_returns(index, start=str(t1), end=str(t2), **index_close_kwargs)
        .reindex(adj_close_w.index[adj_close_w.index >= pd.Timestamp(t1)])
        .fillna(0.0)
    )

    # ── 5. 回测 ───────────────────────────────────────────────────
    print("\n[5] 执行回测（T+1 VWAP，涨跌停/停牌，非对称成本）...")
    bt = RealisticBacktester(
        cost_buy=cost_buy,
        cost_sell=cost_sell,
        risk_free=risk_free,
    )
    result, exec_stats = bt.run(
        weight_df=weight_df,
        adj_close=adj_close_w,
        adj_vwap=adj_vwap_w,
        close_raw=close_raw_w,
        limit_up_df=limit_up_w,
        limit_down_df=limit_down_w,
        trade_status_df=trade_status_w,
        benchmark_ret=bm_ret,
        initial_value=initial_value,
    )

    print(f"\n{result.summary()}")
    print(
        f"\n  执行统计：涨停买入失败={exec_stats['buy_fail_count']}次  "
        f"跌停延迟卖出={exec_stats['sell_defer_count']}次  "
        f"平均现金占比={exec_stats['avg_cash_pct']*100:.1f}%"
    )

    # ── 6. 输出 ───────────────────────────────────────────────────
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report_title = title or (
            f"回测报告  {index.upper()}基准  {actual_start}~{actual_end}"
        )
        report_path = generate_html_report(
            result,
            output_path=out_path / "report.html",
            title=report_title,
        )
        print(f"\n  HTML 报告：{report_path}")

        # 保存净值 / 超额净值 / 换手率
        nav_df = pd.DataFrame({
            "nav":        result.nav,
            "bm_nav":     result.bm_nav,
            "excess_nav": result.excess_nav,
            "port_ret":   result.daily_ret,
            "bm_ret":     result.bm_ret,
            "excess_ret": result.excess_ret,
        })
        nav_df.to_parquet(out_path / "nav.parquet")
        result.turnover.to_frame("turnover").to_parquet(out_path / "turnover.parquet")
        print(f"  净值数据：{out_path / 'nav.parquet'}")

    return result, exec_stats


# ── CLI 入口 ─────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立回测脚本：权重长表 → 净值报告")
    p.add_argument("--weights",   required=True,       help="权重 parquet 路径（长表/宽表均可）")
    p.add_argument("--start",     required=True,       help="回测起始日，如 2022-01-01")
    p.add_argument("--end",       required=True,       help="回测截止日，如 2025-12-31")
    p.add_argument("--index",     default="zz1000",    help="基准指数 key（hs300/zz500/zz1000）")
    p.add_argument("--out-dir",   default=None,        help="报告输出目录（可选）")
    p.add_argument("--title",     default=None,        help="报告标题（可选）")
    p.add_argument("--cost-buy",  type=float, default=0.001)
    p.add_argument("--cost-sell", type=float, default=0.002)
    p.add_argument("--initial-value", type=float, default=1e8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        weight_path=args.weights,
        start_date=args.start,
        end_date=args.end,
        index=args.index,
        cost_buy=args.cost_buy,
        cost_sell=args.cost_sell,
        initial_value=args.initial_value,
        out_dir=args.out_dir,
        title=args.title,
    )
