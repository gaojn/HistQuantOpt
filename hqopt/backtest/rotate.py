"""轮动分仓回测引擎（T 日选股 → T+1 开盘买入 → T+H 收盘卖出）。

语义
----
- T 日（信号日）收盘后给出选股列表（可为 0~任意只）；
- T+1 日以 **复权开盘价** 买入，桶内逐票等权；
- T+H 日以 **复权收盘价** 卖出（H=1 即 T+1 当日开盘买、收盘卖）；
- 资金分成 H 份（桶）：每日建仓金额 = min(前收盘总资产 / H, 可用现金)。
  连续多日无信号时持仓逐桶卖光、资金全部回到现金，下一次有信号自然
  从「总资产 / H」重新开始建仓，无需显式重置。

执行规则（与 engine.RealisticBacktester 同口径）
------------------------------------------------
- 买入日开盘涨停（open ≥ limit_up × 99.9%）或停牌：放弃该票，资金留现金；
- 卖出日收盘跌停（close ≤ limit_down × 100.1%）或停牌：顺延到下一个
  可交易日收盘卖出；
- 退市（当日行情面板中该票整行消失，价格与交易状态均为 NaN）：不论是否
  到期，当日按**最近有效收盘价**（即前一日价格）强制卖出核销，扣卖出费；
- 估值用 ffill 后的复权收盘价（停牌沿用最近有效价），成交价不 ffill；
- 成本非对称：买入 cost_buy（默认 1‰）、卖出 cost_sell（默认 2‰）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hqopt.backtest.engine import (
    BacktestResult,
    _resolve_benchmark_returns,
    calc_metrics,
)

# 涨跌停判断容差，与 ExecutionLedger 同口径
_LIMIT_UP_TOL = 0.999
_LIMIT_DOWN_TOL = 1.001
_SUSPENDED = "停牌"


@dataclass
class _Bucket:
    """一次信号建仓形成的持仓桶。"""

    signal_day: pd.Timestamp
    due_idx: int                                  # 应卖出的交易日序号（含）
    holdings: dict[str, float] = field(default_factory=dict)  # ticker -> shares


@dataclass(frozen=True)
class _RotateFrames:
    """对齐到回测交易日的行情宽表。"""

    dates: pd.DatetimeIndex
    adj_open: pd.DataFrame
    adj_close_raw: pd.DataFrame
    adj_close_marked: pd.DataFrame    # ffill，仅估值用
    open_raw: pd.DataFrame
    close_raw: pd.DataFrame
    limit_up: pd.DataFrame
    limit_down: pd.DataFrame
    trade_status: pd.DataFrame


def _normalize_picks(
    picks: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> dict[pd.Timestamp, list[str]]:
    """长表 [date, code] → {信号日: 代码列表}，并校验信号日在交易日历内。"""
    if not {"date", "code"}.issubset(picks.columns):
        raise ValueError("选股表必须包含 [date, code] 两列")
    frame = picks[["date", "code"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates()

    missing = pd.DatetimeIndex(frame["date"].unique()).difference(calendar)
    if not missing.empty:
        preview = ", ".join(day.strftime("%Y-%m-%d") for day in missing[:5])
        suffix = f"（共 {len(missing)} 日）" if len(missing) > 5 else ""
        raise ValueError(f"信号日不在行情交易日历中: {preview}{suffix}")

    return {
        day: sorted(group["code"].tolist())
        for day, group in frame.groupby("date")
    }


class RotateBacktester:
    """T 日选股、T+1 开盘买入、T+H 收盘卖出、资金分 H 份的轮动回测。

    Parameters
    ----------
    hold_days : int    持有期 H（交易日）。H=1 即 T+1 开盘买、当日收盘卖。
    cost_buy  : float  买入费率，默认 0.1%（1‰）
    cost_sell : float  卖出费率，默认 0.2%（2‰）
    risk_free : float  年化无风险利率（Sharpe 用）
    """

    def __init__(
        self,
        hold_days: int,
        cost_buy: float = 0.001,
        cost_sell: float = 0.002,
        risk_free: float = 0.02,
    ) -> None:
        if hold_days < 1:
            raise ValueError(f"hold_days 须 ≥ 1，当前为 {hold_days}")
        self.hold_days = int(hold_days)
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.risk_free = risk_free

    # ── 主入口 ─────────────────────────────────────────────

    def run(
        self,
        picks: pd.DataFrame,
        adj_open: pd.DataFrame,
        adj_close: pd.DataFrame,
        open_raw: pd.DataFrame,
        close_raw: pd.DataFrame,
        limit_up_df: pd.DataFrame,
        limit_down_df: pd.DataFrame,
        trade_status_df: pd.DataFrame,
        benchmark_ret: pd.Series | None = None,
        initial_value: float = 1e8,
    ) -> tuple[BacktestResult, dict, pd.DataFrame]:
        """
        执行轮动分仓回测。

        Parameters
        ----------
        picks           : 选股长表，列 [date, code]；date=T（信号日）
        adj_open        : 复权开盘价（T+1 买入成交价）
        adj_close       : 复权收盘价（T+H 卖出成交价 + NAV 估值）
        open_raw        : 原始开盘价（买入日涨停判断）
        close_raw       : 原始收盘价（卖出日跌停判断）
        limit_up_df     : 涨停价（原始）
        limit_down_df   : 跌停价（原始）
        trade_status_df : 交易状态（'交易'/'停牌'/...）
        benchmark_ret   : 基准日收益率（None=等权全股）
        initial_value   : 初始资金（元）

        Returns
        -------
        (BacktestResult, execution_stats dict, trades DataFrame)
        """
        picks_map = _normalize_picks(picks, pd.DatetimeIndex(adj_close.index))
        if not picks_map:
            raise ValueError("选股表为空，无法回测")
        first_signal = min(picks_map)
        frames = self._align_frames(
            adj_open, adj_close, open_raw, close_raw,
            limit_up_df, limit_down_df, trade_status_df, first_signal,
        )
        state = _ReplayState(cash=float(initial_value), equity=float(initial_value))
        records = self._replay(frames, picks_map, state)

        result = self._build_result(
            records, frames, adj_close, benchmark_ret, initial_value, picks_map
        )
        exec_stats = self._build_exec_stats(state, records, frames)
        trades = pd.DataFrame(
            records.trades,
            columns=[
                "date", "signal_date", "code", "side",
                "price", "shares", "notional",
            ],
        )
        return result, exec_stats, trades

    # ── 内部实现 ───────────────────────────────────────────

    @staticmethod
    def _align_frames(
        adj_open, adj_close, open_raw, close_raw,
        limit_up_df, limit_down_df, trade_status_df, first_signal,
    ) -> _RotateFrames:
        dates = adj_close.index[adj_close.index >= first_signal]

        def align(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(index=dates)

        adj_close_raw = align(adj_close)
        return _RotateFrames(
            dates=dates,
            adj_open=align(adj_open),
            adj_close_raw=adj_close_raw,
            adj_close_marked=adj_close_raw.ffill(),
            open_raw=align(open_raw),
            close_raw=align(close_raw),
            limit_up=align(limit_up_df),
            limit_down=align(limit_down_df),
            trade_status=align(trade_status_df),
        )

    def _replay(
        self,
        frames: _RotateFrames,
        picks_map: dict[pd.Timestamp, list[str]],
        state: _ReplayState,
    ) -> _ReplayRecords:
        records = _ReplayRecords(
            port_values=pd.Series(0.0, index=frames.dates),
            turnover_by_signal=dict.fromkeys(sorted(picks_map), 0.0),
        )

        for i, day in enumerate(frames.dates):
            day_traded = 0.0

            # ── 1. 开盘买入：昨日（T）信号今日（T+1）建仓 ──
            if i > 0:
                signal_day = frames.dates[i - 1]
                codes = picks_map.get(signal_day)
                if codes:
                    day_traded += self._buy_bucket(
                        frames, state, records, i, signal_day, codes
                    )

            # ── 2. 退市核销：当日行情整行消失的持仓按前一日价格强制卖出 ──
            day_traded += self._force_sell_delisted(frames, state, records, i)

            # ── 3. 收盘卖出：到期桶（含顺延未卖出的）──
            day_traded += self._sell_due_buckets(frames, state, records, i)

            # ── 4. 收盘估值 ──
            equity = state.cash + self._holdings_value(frames, state, day)
            records.port_values.iloc[i] = equity
            records.actual_weight_rows[day] = self._weights_row(frames, state, day, equity)
            if equity > 1e-8:
                records.cash_ratios.append(state.cash / equity)
                if day_traded > 0:
                    records.turnover[day] = day_traded / equity
            state.equity = equity

        # 信号落在最后一个交易日的无法在 T+1 买入
        last_day = frames.dates[-1]
        state.unexecutable_signals = [
            day for day in picks_map if day >= last_day
        ]
        return records

    def _buy_bucket(
        self,
        frames: _RotateFrames,
        state: _ReplayState,
        records: _ReplayRecords,
        i: int,
        signal_day: pd.Timestamp,
        codes: list[str],
    ) -> float:
        """T+1 开盘按「min(前收盘总资产/H, 现金)」等权建仓，返回成交额。"""
        day = frames.dates[i]
        budget = min(state.equity / self.hold_days, state.cash)
        if budget <= 1e-8:
            # 现金耗尽（如到期桶因跌停/停牌未能回款）→ 本期信号整桶跳过
            state.no_cash_skip_days += 1
            return 0.0
        alloc = budget / len(codes)

        bucket = _Bucket(signal_day=signal_day, due_idx=i + self.hold_days - 1)
        traded = 0.0
        open_adj = frames.adj_open.loc[day]
        open_px = frames.open_raw.loc[day]
        limit_up = frames.limit_up.loc[day]
        status = frames.trade_status.loc[day]

        for code in codes:
            px_adj = open_adj.get(code)
            px_raw = open_px.get(code)
            if pd.isna(px_adj) or pd.isna(px_raw) or px_adj <= 0:
                state.buy_fail_no_quote += 1     # 无行情（未上市/退市/代码错误）
                continue
            if status.get(code) == _SUSPENDED:
                state.buy_fail_suspended += 1
                continue
            up = limit_up.get(code)
            if pd.notna(up) and px_raw >= up * _LIMIT_UP_TOL:
                state.buy_fail_limit_up += 1     # 开盘涨停，放弃
                continue

            notional = alloc / (1.0 + self.cost_buy)   # 含费出资 = alloc
            shares = notional / px_adj
            bucket.holdings[code] = bucket.holdings.get(code, 0.0) + shares
            state.cash -= alloc
            traded += notional
            records.trades.append(
                (day, signal_day, code, "buy", float(px_adj), shares, notional)
            )

        if bucket.holdings:
            state.buckets.append(bucket)
            records.turnover_by_signal[signal_day] += (
                traded / state.equity if state.equity > 1e-8 else 0.0
            )
        return traded

    def _force_sell_delisted(
        self,
        frames: _RotateFrames,
        state: _ReplayState,
        records: _ReplayRecords,
        i: int,
    ) -> float:
        """退市核销：当日行情整行消失的持仓按最近有效价（前一日价格）强制卖出。

        区别于停牌：停牌日面板中仍有该票的行（trade_status='停牌'），走顺延；
        只有价格与交易状态**同时**缺失（该票已从面板消失，即已退市）才触发。
        不论持仓是否到期，当日立即核销、资金回笼，扣正常卖出费。
        """
        day = frames.dates[i]
        close_row = frames.adj_close_raw.loc[day]
        status_row = frames.trade_status.loc[day]
        marked = frames.adj_close_marked.loc[day]
        traded = 0.0

        for bucket in state.buckets:
            for code in list(bucket.holdings):
                if pd.notna(close_row.get(code)) or pd.notna(status_row.get(code)):
                    continue                     # 仍在面板中（含停牌），不属退市
                px = marked.get(code)
                if pd.isna(px) or px <= 0:
                    continue                     # 无任何历史有效价，无法核销
                shares = bucket.holdings.pop(code)
                notional = shares * float(px)
                state.cash += notional * (1.0 - self.cost_sell)
                state.delist_forced_count += 1
                traded += notional
                records.turnover_by_signal[bucket.signal_day] += (
                    notional / state.equity if state.equity > 1e-8 else 0.0
                )
                records.trades.append(
                    (day, bucket.signal_day, code, "sell_delist",
                     float(px), shares, notional)
                )

        state.buckets = [b for b in state.buckets if b.holdings]
        return traded

    def _sell_due_buckets(
        self,
        frames: _RotateFrames,
        state: _ReplayState,
        records: _ReplayRecords,
        i: int,
    ) -> float:
        """卖出所有到期（due_idx ≤ i）的桶；跌停/停牌顺延到下一交易日。"""
        day = frames.dates[i]
        close_adj = frames.adj_close_raw.loc[day]
        close_px = frames.close_raw.loc[day]
        limit_down = frames.limit_down.loc[day]
        status = frames.trade_status.loc[day]
        traded = 0.0

        for bucket in state.buckets:
            if bucket.due_idx > i or not bucket.holdings:
                continue
            deferred = bucket.due_idx < i
            for code in list(bucket.holdings):
                px_adj = close_adj.get(code)
                px_raw = close_px.get(code)
                if pd.isna(px_adj) or px_adj <= 0:
                    state.sell_defer_no_price += 1   # 行在但无真实价（长停等），顺延
                    continue
                if status.get(code) == _SUSPENDED:
                    state.sell_defer_suspended += 1
                    continue
                down = limit_down.get(code)
                if pd.notna(down) and pd.notna(px_raw) and px_raw <= down * _LIMIT_DOWN_TOL:
                    state.sell_defer_limit_down += 1  # 收盘跌停，顺延
                    continue

                shares = bucket.holdings.pop(code)
                notional = shares * float(px_adj)
                state.cash += notional * (1.0 - self.cost_sell)
                traded += notional
                records.turnover_by_signal[bucket.signal_day] += (
                    notional / state.equity if state.equity > 1e-8 else 0.0
                )
                records.trades.append(
                    (
                        day, bucket.signal_day, code,
                        "sell_deferred" if deferred else "sell",
                        float(px_adj), shares, notional,
                    )
                )

        state.buckets = [b for b in state.buckets if b.holdings]
        return traded

    @staticmethod
    def _holdings_value(frames: _RotateFrames, state: _ReplayState, day) -> float:
        marked = frames.adj_close_marked.loc[day]
        value = 0.0
        for bucket in state.buckets:
            for code, shares in bucket.holdings.items():
                px = marked.get(code)
                if pd.notna(px):
                    value += shares * float(px)
        return value

    @staticmethod
    def _weights_row(
        frames: _RotateFrames, state: _ReplayState, day, equity: float
    ) -> pd.Series:
        if equity <= 1e-8:
            return pd.Series(dtype=float)
        marked = frames.adj_close_marked.loc[day]
        agg: dict[str, float] = {}
        for bucket in state.buckets:
            for code, shares in bucket.holdings.items():
                px = marked.get(code)
                if pd.notna(px):
                    agg[code] = agg.get(code, 0.0) + shares * float(px) / equity
        return pd.Series(agg, dtype=float)

    def _build_result(
        self,
        records: _ReplayRecords,
        frames: _RotateFrames,
        adj_close: pd.DataFrame,
        benchmark_ret: pd.Series | None,
        initial_value: float,
        picks_map: dict[pd.Timestamp, list[str]],
    ) -> BacktestResult:
        nav = records.port_values / initial_value
        port_ret = nav.pct_change().fillna(0.0)
        bm_ret = _resolve_benchmark_returns(benchmark_ret, adj_close, frames.dates)
        bm_nav = (1 + bm_ret).cumprod()

        actual_weights = pd.DataFrame(records.actual_weight_rows).T.fillna(0.0)
        actual_weights.index.name = "date"

        # 目标持仓矩阵：每个信号日桶内等权（行内合计 1，代表该桶内部分配，
        # 组合层面的目标权重为其 1/H）。供报告「最后一期目标持仓」表使用。
        target_rows = {
            day: pd.Series(1.0 / len(codes), index=codes)
            for day, codes in picks_map.items() if codes
        }
        target_weights = (
            pd.DataFrame(target_rows).T.sort_index() if target_rows else None
        )
        if target_weights is not None:
            target_weights.index.name = "date"

        return BacktestResult(
            nav=nav,
            bm_nav=bm_nav,
            excess_nav=nav / bm_nav,
            daily_ret=port_ret,
            bm_ret=bm_ret,
            excess_ret=port_ret - bm_ret,
            turnover=pd.Series(records.turnover, name="turnover", dtype=float),
            portfolio_metrics=calc_metrics(port_ret, bm_ret, self.risk_free),
            benchmark_metrics=calc_metrics(
                bm_ret, pd.Series(0.0, index=bm_ret.index), self.risk_free
            ),
            risk_free=self.risk_free,
            actual_weights=actual_weights,
            turnover_by_rebalance=pd.Series(
                records.turnover_by_signal, name="turnover", dtype=float
            ),
            target_weights=target_weights,
        )

    def _build_exec_stats(
        self,
        state: _ReplayState,
        records: _ReplayRecords,
        frames: _RotateFrames,
    ) -> dict:
        final_nav = float(records.port_values.iloc[-1])
        last_day = frames.dates[-1]
        raw_last = frames.adj_close_raw.loc[last_day]
        marked_last = frames.adj_close_marked.loc[last_day]

        # 末日仍持有且无真实价（靠 ffill 陈旧价估值）的滞留持仓
        stuck_count = 0
        stuck_value = 0.0
        open_positions = 0
        for bucket in state.buckets:
            for code, shares in bucket.holdings.items():
                if shares <= 1e-10:
                    continue
                open_positions += 1
                if pd.isna(raw_last.get(code)) and pd.notna(marked_last.get(code)):
                    stuck_count += 1
                    stuck_value += shares * float(marked_last.get(code))

        return {
            "hold_days": self.hold_days,
            "buy_fail_count": state.buy_fail_count,
            "buy_fail_breakdown": {
                "limit_up_open": state.buy_fail_limit_up,    # 开盘涨停
                "suspended": state.buy_fail_suspended,       # 停牌
                "no_quote": state.buy_fail_no_quote,         # 无行情
            },
            "sell_defer_count": state.sell_defer_count,
            "sell_defer_breakdown": {                        # 按被阻断天数计
                "limit_down_close": state.sell_defer_limit_down,  # 收盘跌停
                "suspended": state.sell_defer_suspended,          # 停牌
                "no_price": state.sell_defer_no_price,            # 行在但无价
            },
            "delist_forced_count": state.delist_forced_count,
            "no_cash_skip_days": state.no_cash_skip_days,
            "unexecutable_signal_days": [
                day.strftime("%Y-%m-%d") for day in state.unexecutable_signals
            ],
            "avg_cash_pct": (
                float(np.mean(records.cash_ratios)) if records.cash_ratios else 0.0
            ),
            "final_cash_pct": (
                float(state.cash / final_nav) if final_nav > 1e-8 else 0.0
            ),
            "open_position_count": open_positions,   # 末日未平仓持仓数（区间截断）
            "open_bucket_count": len(state.buckets),
            "delisted_stuck_count": stuck_count,
            "stale_value_pct": (
                float(stuck_value / final_nav) if final_nav > 1e-8 else 0.0
            ),
        }


@dataclass
class _ReplayState:
    """逐日推进中的账户状态。"""

    cash: float
    equity: float                                  # 上一收盘总资产（建仓预算基准）
    buckets: list[_Bucket] = field(default_factory=list)
    buy_fail_suspended: int = 0      # T+1 停牌买不进
    buy_fail_limit_up: int = 0       # T+1 开盘涨停买不进
    buy_fail_no_quote: int = 0       # T+1 无行情（未上市/退市/代码错误）
    sell_defer_limit_down: int = 0   # 收盘跌停卖不出（按被阻断天数计）
    sell_defer_suspended: int = 0    # 停牌卖不出（按被阻断天数计）
    sell_defer_no_price: int = 0     # 行在但无真实价（按被阻断天数计）
    delist_forced_count: int = 0
    no_cash_skip_days: int = 0
    unexecutable_signals: list = field(default_factory=list)

    @property
    def buy_fail_count(self) -> int:
        return self.buy_fail_suspended + self.buy_fail_limit_up + self.buy_fail_no_quote

    @property
    def sell_defer_count(self) -> int:
        return (
            self.sell_defer_limit_down
            + self.sell_defer_suspended
            + self.sell_defer_no_price
        )


@dataclass
class _ReplayRecords:
    """逐日重放产生的时间序列记录。"""

    port_values: pd.Series
    turnover_by_signal: dict
    turnover: dict = field(default_factory=dict)          # key=成交日
    cash_ratios: list[float] = field(default_factory=list)
    actual_weight_rows: dict = field(default_factory=dict)
    trades: list[tuple] = field(default_factory=list)
