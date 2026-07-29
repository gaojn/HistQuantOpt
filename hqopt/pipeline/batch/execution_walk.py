"""成交账本的按日推进游标。

优化阶段与回测阶段必须走完全一致的执行语义，因此推进顺序（信号日先执行旧目标、
收盘后由新目标覆盖、区间末尾补记）固定在 ``_ExecutionWalker`` 一处，不散落到
调用方。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl

from hqopt.backtest.execution import ExecutionLedger

_EXECUTION_COLUMNS = (
    "date", "code", "adj_close", "adj_vwap", "close",
    "limit_up", "limit_down", "trade_status",
)


def _partition_execution_days(
    panel: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[date, pl.DataFrame]:
    """按交易日切分成交所需字段，供逐期优化按日推进实际持仓账本。"""
    missing = [column for column in _EXECUTION_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"行情面板缺少实际成交所需字段：{missing}")
    execution_panel = panel.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    ).select(_EXECUTION_COLUMNS)
    return {
        key[0] if isinstance(key, tuple) else key: day
        for key, day in execution_panel.partition_by("date", as_dict=True).items()
    }


def _execution_day_frame(day: pl.DataFrame) -> pd.DataFrame:
    """把一个 polars 日截面转换为以股票代码为索引的成交数据。"""
    return day.drop("date").to_pandas().set_index("code")


def _advance_execution_day(ledger: ExecutionLedger, day: pl.DataFrame) -> None:
    """把一个 polars 日截面送入共享成交账本。"""
    pdf = _execution_day_frame(day)
    ledger.step(
        adj_close=pdf["adj_close"],
        adj_vwap=pdf["adj_vwap"],
        close_raw=pdf["close"],
        limit_up=pdf["limit_up"],
        limit_down=pdf["limit_down"],
        trade_status=pdf["trade_status"],
    )


def _signal_day_suspended_tickers(day: pl.DataFrame) -> list[str]:
    """返回信号日原始行情中停牌的股票代码。"""
    return (
        day.filter(pl.col("trade_status") == "停牌")
        .get_column("code")
        .cast(pl.Utf8)
        .to_list()
    )


class _ExecutionWalker:
    """按交易日推进共享成交账本，把游标状态收敛在一处。

    优化阶段与回测阶段必须走完全一致的执行语义，因此推进顺序（信号日先执行
    旧目标、收盘后由新目标覆盖、区间末尾补记）都固定在这里。
    """

    def __init__(
        self,
        ledger: ExecutionLedger,
        trade_dates: list[date],
        execution_days: dict[date, pl.DataFrame],
    ) -> None:
        self._ledger = ledger
        self._trade_dates = trade_dates
        self._days = execution_days
        self._cursor = 0

    def _execution_day(self, execution_date: date) -> pl.DataFrame:
        day = self._days.get(execution_date)
        if day is None:
            raise RuntimeError(f"{execution_date} 缺少成交行情截面")
        return day

    def open_signal_day(self, rebal_date: date) -> pl.DataFrame:
        """推进到调仓日并执行旧目标；新目标在当日收盘后才可覆盖。"""
        while (
            self._cursor < len(self._trade_dates)
            and self._trade_dates[self._cursor] < rebal_date
        ):
            _advance_execution_day(
                self._ledger, self._execution_day(self._trade_dates[self._cursor])
            )
            self._cursor += 1

        if (
            self._cursor >= len(self._trade_dates)
            or self._trade_dates[self._cursor] != rebal_date
        ):
            raise RuntimeError(f"{rebal_date} 不在成交交易日序列中")

        signal_day = self._days.get(rebal_date)
        if signal_day is None:
            raise RuntimeError(f"{rebal_date} 缺少信号日行情截面")
        _advance_execution_day(self._ledger, signal_day)
        self._cursor += 1
        return signal_day

    def finish(self) -> None:
        """推进到区间末日，补记最后一批 T+1/T+2/T+3 成交或过期。"""
        while self._cursor < len(self._trade_dates):
            _advance_execution_day(
                self._ledger, self._execution_day(self._trade_dates[self._cursor])
            )
            self._cursor += 1
