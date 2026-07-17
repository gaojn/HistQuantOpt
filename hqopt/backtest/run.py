"""权重 → T+1 VWAP 真实回测 → HTML 报告。

供 CLI（`hqopt backtest` / `hqopt run`）与脚本复用。权重文件支持：
  - 长表 parquet：列 [date, code, weight]
  - 宽表 parquet：index=date, columns=ticker
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqopt.backtest.engine import RealisticBacktester
from hqopt.backtest.report import generate_html_report
from hqopt.constants import SYNTHETIC_ALPHA_WARNING_FILE
from hqopt.data.index_close import load_index_returns
from hqopt.io.data_panel import load_panel

logger = logging.getLogger(__name__)


def _load_warning_banner(weight_path: str | Path) -> str | None:
    warning_path = Path(weight_path).parent / SYNTHETIC_ALPHA_WARNING_FILE
    if not warning_path.exists():
        return None
    return warning_path.read_text(encoding="utf-8").strip() or None


def _load_weights(path: str | Path) -> pd.DataFrame:
    """加载权重，统一输出宽表（index=DatetimeIndex，columns=ticker）。"""
    df = pd.read_parquet(path)
    if {"date", "code", "weight"}.issubset(df.columns):       # 长表
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="code", values="weight").sort_index()
    else:                                                      # 宽表
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
    weight_path : 权重 parquet 路径（长表或宽表）
    start_date / end_date : 回测区间（权重早于 start 的记录会被裁剪）
    index       : 基准指数 key（hs300 / zz500 / zz1000 / csiall ...）
    cost_buy / cost_sell : 买/卖费率（非对称），默认 1‰ / 2‰
    risk_free   : 无风险年化利率（Sharpe 用）
    initial_value : 初始资金（元）
    out_dir     : 报告输出目录，None 则不落地文件
    title       : 报告标题，None 自动生成
    cache_dir   : 行情 parquet 缓存目录，None 用默认
    index_close_path : 指数收盘价 CSV 路径，None 用默认

    Returns
    -------
    (BacktestResult, exec_stats dict)
    """
    t1 = _parse_date(start_date)
    t2 = _parse_date(end_date)

    # ── 1. 权重 ──────────────────────────────────────────────
    logger.info(f"\n[1] 加载权重：{weight_path}")
    weight_df = _load_weights(weight_path)
    weight_df = weight_df[
        (weight_df.index >= pd.Timestamp(t1)) & (weight_df.index <= pd.Timestamp(t2))
    ]
    if weight_df.empty:
        raise ValueError(f"权重文件在 {t1}~{t2} 无数据，请检查日期区间")
    warning_banner = _load_warning_banner(weight_path)

    actual_start = weight_df.index.min().date()
    actual_end = weight_df.index.max().date()
    n_stocks = (weight_df > 1e-6).any(axis=0).sum()
    logger.info(f"  调仓日={len(weight_df)}  股票池={n_stocks}  区间={actual_start}~{actual_end}")

    # ── 2. 行情面板（多取年初数据，确保首期 ADV/VWAP）──────────
    data_start = date(t1.year, 1, 1)
    logger.info(f"\n[2] 加载行情面板（{data_start} ~ {t2}）...")
    panel = load_panel(
        data_start, t2,
        columns=["code", "date", "adj_close", "adj_vwap", "close",
                 "limit_up", "limit_down", "trade_status"],
        cache_dir=cache_dir,
    )
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 3. 宽表 ──────────────────────────────────────────────
    logger.info("\n[3] 构建回测宽表...")
    adj_close_w = _to_wide(panel, "adj_close")
    adj_vwap_w = _to_wide(panel, "adj_vwap")
    close_raw_w = _to_wide(panel, "close")
    limit_up_w = _to_wide(panel, "limit_up")
    limit_down_w = _to_wide(panel, "limit_down")
    trade_status_w = _to_wide(panel, "trade_status")

    # ── 4. 基准收益 ──────────────────────────────────────────
    logger.info(f"\n[4] 加载基准收益（{index.upper()}）...")
    index_close_kwargs = {} if index_close_path is None else {"path": index_close_path}
    bm_ret = (
        load_index_returns(index, start=str(t1), end=str(t2), **index_close_kwargs)
        .reindex(adj_close_w.index[adj_close_w.index >= pd.Timestamp(t1)])
        .fillna(0.0)
    )

    # ── 5. 回测 ──────────────────────────────────────────────
    logger.info("\n[5] 执行回测（T+1 VWAP，涨跌停/停牌，非对称成本）...")
    bt = RealisticBacktester(cost_buy=cost_buy, cost_sell=cost_sell, risk_free=risk_free)
    result, exec_stats = bt.run(
        weight_df=weight_df, adj_close=adj_close_w, adj_vwap=adj_vwap_w,
        close_raw=close_raw_w, limit_up_df=limit_up_w, limit_down_df=limit_down_w,
        trade_status_df=trade_status_w, benchmark_ret=bm_ret, initial_value=initial_value,
    )

    logger.info(f"\n{result.summary()}")
    logger.info(
        f"\n  执行统计：涨停买入失败={exec_stats['buy_fail_count']}次  "
        f"跌停延迟卖出={exec_stats['sell_defer_count']}次  "
        f"平均现金占比={exec_stats['avg_cash_pct']*100:.1f}%"
    )
    if exec_stats.get("delisted_stuck_count", 0) > 0:
        logger.info(
            f"  ⚠️ 退市/长停滞留持仓={exec_stats['delisted_stuck_count']}只  "
            f"陈旧价估值占末日NAV={exec_stats['stale_value_pct']*100:.2f}%"
            f"（无法成交，净值含此部分不可全信）"
        )

    # ── 6. 输出 ──────────────────────────────────────────────
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_title = title or f"回测报告  {index.upper()}基准  {actual_start}~{actual_end}"
        report_path = generate_html_report(
            result, output_path=out_path / "report.html", title=report_title,
            warning_banner=warning_banner,
        )
        logger.info(f"\n  HTML 报告：{report_path}")
        nav_df = pd.DataFrame({
            "nav": result.nav, "bm_nav": result.bm_nav, "excess_nav": result.excess_nav,
            "port_ret": result.daily_ret, "bm_ret": result.bm_ret, "excess_ret": result.excess_ret,
        })
        nav_df.to_parquet(out_path / "nav.parquet")
        result.turnover.to_frame("turnover").to_parquet(out_path / "turnover.parquet")
        if result.actual_weights is not None:
            result.actual_weights.to_parquet(out_path / "actual_weights.parquet")
        logger.info(f"  净值数据：{out_path / 'nav.parquet'}")

    return result, exec_stats
