"""选股列表 → 轮动分仓回测（T+1 开盘买 / T+H 收盘卖）→ HTML 报告。

供 CLI（`hqopt rotate`）与脚本复用。选股文件支持：
  - 长表 parquet / csv：列 [date, code]（多余列忽略），date=T（信号日）
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from hqopt.backtest.report import generate_html_report
from hqopt.backtest.rotate import RotateBacktester
from hqopt.data.benchmark import (
    benchmark_returns_from_rebalance_weights,
    equal_weight_benchmark_weights,
)
from hqopt.data.index_close import load_index_returns
from hqopt.io.data_panel import load_panel

logger = logging.getLogger(__name__)

_EQUAL_WEIGHT_BENCHMARK = "equal_weight"


def _load_picks(path: str | Path) -> pd.DataFrame:
    """加载选股长表（parquet/csv），仅保留 [date, code] 列。"""
    p = Path(path)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    if not {"date", "code"}.issubset(df.columns):
        raise ValueError(f"选股文件须包含 [date, code] 列，当前列为 {list(df.columns)}")
    df = df[["date", "code"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    return df.drop_duplicates().sort_values(["date", "code"]).reset_index(drop=True)


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


def _save_execution_stats(stats: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item(),
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def run_rotate_backtest(
    picks_path: str | Path,
    hold_days: int,
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
    加载选股文件，执行轮动分仓回测（T+1 开盘买 / T+H 收盘卖），生成 HTML 报告。

    Parameters
    ----------
    picks_path  : 选股长表路径（parquet/csv，列 [date, code]，date=信号日 T）
    hold_days   : 持有期 H（交易日）；资金分成 H 份
    start_date / end_date : 回测区间（按信号日裁剪；区间末尾未到期持仓按市价估值）
    index       : 基准 key（equal_weight / hs300 / zz500 / zz1000 / csiall ...）
    cost_buy / cost_sell : 买/卖费率（非对称），默认 1‰ / 2‰
    risk_free   : 无风险年化利率（Sharpe 用）
    initial_value : 初始资金（元）
    out_dir     : 报告输出目录，None 则不落地文件
    title       : 报告标题，None 自动生成
    cache_dir   : 行情 parquet 缓存目录，None 用默认
    index_close_path : 指数收盘价 CSV 路径，None 用默认

    Returns
    -------
    (BacktestResult, exec_stats dict, trades DataFrame)
    """
    t1 = _parse_date(start_date)
    t2 = _parse_date(end_date)

    # ── 1. 选股列表 ──────────────────────────────────────────
    logger.info(f"\n[1] 加载选股文件：{picks_path}")
    all_picks = _load_picks(picks_path)
    picks = all_picks[
        (all_picks["date"] >= pd.Timestamp(t1))
        & (all_picks["date"] <= pd.Timestamp(t2))
    ]
    if picks.empty:
        raise ValueError(f"选股文件在 {t1}~{t2} 无数据，请检查日期区间")
    n_days = picks["date"].nunique()
    n_stocks = picks["code"].nunique()
    logger.info(
        f"  信号日={n_days}  股票池={n_stocks}  "
        f"日均选股={len(picks) / n_days:.1f}  "
        f"区间={picks['date'].min().date()}~{picks['date'].max().date()}"
    )

    # ── 2. 行情面板 ──────────────────────────────────────────
    logger.info(f"\n[2] 加载行情面板（{t1} ~ {t2}）...")
    panel = load_panel(
        t1, t2,
        columns=["code", "date", "open", "adj_open", "adj_close", "close",
                 "limit_up", "limit_down", "trade_status"],
        cache_dir=cache_dir,
    )
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 3. 宽表 ──────────────────────────────────────────────
    logger.info("\n[3] 构建回测宽表...")
    adj_open_w = _to_wide(panel, "adj_open")
    adj_close_w = _to_wide(panel, "adj_close")
    open_raw_w = _to_wide(panel, "open")
    close_raw_w = _to_wide(panel, "close")
    limit_up_w = _to_wide(panel, "limit_up")
    limit_down_w = _to_wide(panel, "limit_down")
    trade_status_w = _to_wide(panel, "trade_status")

    # ── 4. 基准收益 ──────────────────────────────────────────
    first_signal = picks["date"].min()
    if index == _EQUAL_WEIGHT_BENCHMARK:
        logger.info("\n[4] 构建基准收益（全市场等权）...")
        benchmark_weights = equal_weight_benchmark_weights(adj_close_w)
        bm_ret = benchmark_returns_from_rebalance_weights(
            benchmark_weights,
            adj_close_w,
            sorted(picks["date"].unique()),
        )
    else:
        logger.info(f"\n[4] 加载基准收益（{index.upper()}）...")
        index_close_kwargs = {} if index_close_path is None else {"path": index_close_path}
        bm_ret = (
            load_index_returns(index, start=str(t1), end=str(t2), **index_close_kwargs)
            .reindex(adj_close_w.index[adj_close_w.index >= first_signal])
            .fillna(0.0)
        )

    # ── 5. 回测 ──────────────────────────────────────────────
    logger.info(
        f"\n[5] 执行轮动分仓回测（H={hold_days}：T+1 开盘买 / "
        f"T+{hold_days} 收盘卖，资金分 {hold_days} 份）..."
    )
    bt = RotateBacktester(
        hold_days=hold_days, cost_buy=cost_buy,
        cost_sell=cost_sell, risk_free=risk_free,
    )
    result, exec_stats, trades = bt.run(
        picks=picks,
        adj_open=adj_open_w, adj_close=adj_close_w,
        open_raw=open_raw_w, close_raw=close_raw_w,
        limit_up_df=limit_up_w, limit_down_df=limit_down_w,
        trade_status_df=trade_status_w,
        benchmark_ret=bm_ret, initial_value=initial_value,
    )

    logger.info(f"\n{result.summary()}")
    logger.info(
        f"\n  执行统计：买入放弃={exec_stats['buy_fail_count']}次  "
        f"卖出顺延={exec_stats['sell_defer_count']}次  "
        f"退市强制核销={exec_stats['delist_forced_count']}笔  "
        f"末日未平仓={exec_stats['open_position_count']}只  "
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
        benchmark_label = (
            "全市场等权" if index == _EQUAL_WEIGHT_BENCHMARK else index.upper()
        )
        report_title = title or (
            f"轮动分仓回测（H={hold_days}）  {benchmark_label}基准  "
            f"{picks['date'].min().date()}~{picks['date'].max().date()}"
        )
        report_path = generate_html_report(
            result, output_path=out_path / "report.html", title=report_title,
            cache_dir=cache_dir,
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
        trades.to_parquet(out_path / "trades.parquet")
        execution_stats_path = _save_execution_stats(
            exec_stats,
            out_path / "execution_stats.json",
        )
        logger.info(f"  净值数据：{out_path / 'nav.parquet'}")
        logger.info(f"  成交明细：{out_path / 'trades.parquet'}")
        logger.info(f"  执行统计：{execution_stats_path}")

    return result, exec_stats, trades
