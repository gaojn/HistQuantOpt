"""状态化 T+1 成交账本。

同一账本同时供逐期优化和回测使用，确保下一期优化看到的是实际成交持仓，
而不是上一期目标权重。未完成的目标会在后续交易日继续尝试，直到可成交或
被更新的目标替换。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hqopt.constants import LIMIT_TOL


def _valid_price(value: object) -> float:
    """将有效正价格转为 float，无效值返回 0。"""
    if value is None or pd.isna(value):
        return 0.0
    price = float(value)
    return price if price > 0 else 0.0


@dataclass(frozen=True)
class ExecutionDayResult:
    """单日成交结果。"""

    turnover: float = 0.0
    buy_fail_count: int = 0
    sell_defer_count: int = 0
    target_pending: bool = False


class ExecutionLedger:
    """维护股票份额、现金、估值价格和未完成目标组合。"""

    def __init__(
        self,
        initial_value: float,
        cost_buy: float = 0.001,
        cost_sell: float = 0.002,
        min_notional: float = 1.0,
    ) -> None:
        if initial_value <= 0:
            raise ValueError("initial_value 必须为正数")
        self.cash = float(initial_value)
        self.cost_buy = float(cost_buy)
        self.cost_sell = float(cost_sell)
        self.min_notional = float(min_notional)
        self.shares: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.pending_target: pd.Series | None = None
        self.buy_fail_count = 0
        self.sell_defer_count = 0

    def submit_target(self, target_weight: pd.Series) -> None:
        """提交新目标；新目标替换尚未完成的旧目标，并从下一次 ``step`` 开始执行。"""
        target = pd.to_numeric(target_weight, errors="coerce").fillna(0.0).astype(float)
        target = target.clip(lower=0.0)
        if not target.index.is_unique:
            target = target.groupby(level=0).sum()
        total = float(target.sum())
        if total > 1.0 + 1e-6:
            raise ValueError(f"目标权重和不能超过 1，当前为 {total:.8f}")
        self.pending_target = target

    def _update_marks(self, adj_close: pd.Series) -> None:
        for ticker, value in adj_close.items():
            price = _valid_price(value)
            if price > 0:
                self.last_price[str(ticker)] = price

    def position_values(self, prices: pd.Series | None = None) -> pd.Series:
        """计算持仓市值；传入价格时优先使用，否则回退到最近有效收盘价。"""
        values = {}
        for ticker, shares in self.shares.items():
            if shares <= 1e-10:
                continue
            price = _valid_price(prices.get(ticker)) if prices is not None else 0.0
            if price <= 0:
                price = self.last_price.get(ticker, 0.0)
            if price > 0:
                values[ticker] = shares * price
        return pd.Series(values, dtype=float)

    @property
    def nav(self) -> float:
        return float(self.cash + self.position_values().sum())

    def actual_weights(self) -> pd.Series:
        """返回实际股票权重；现金不伪装成股票，因此权重和允许小于 1。"""
        nav = self.nav
        if nav <= 1e-12:
            return pd.Series(dtype=float, name="actual_weight")
        return (self.position_values() / nav).rename("actual_weight")

    @staticmethod
    def _at_limit(
        ticker: str,
        close_raw: pd.Series,
        limit_price: pd.Series,
        side: str,
    ) -> bool:
        close = _valid_price(close_raw.get(ticker))
        limit = _valid_price(limit_price.get(ticker))
        if close <= 0 or limit <= 0:
            return False
        if side == "up":
            return close >= limit * (1 - LIMIT_TOL)
        return close <= limit * (1 + LIMIT_TOL)

    def step(
        self,
        *,
        adj_close: pd.Series,
        adj_vwap: pd.Series,
        close_raw: pd.Series,
        limit_up: pd.Series,
        limit_down: pd.Series,
        trade_status: pd.Series,
    ) -> ExecutionDayResult:
        """推进一个交易日，并尝试执行尚未完成的目标组合。"""
        self._update_marks(adj_close)
        target = self.pending_target
        if target is None:
            return ExecutionDayResult()

        # 订单目标和当前持仓统一按执行日 VWAP 估值；最近收盘价只在 VWAP 缺失时
        # 兜底，并继续用于成交后的 NAV。不能用收盘价差额再除以 VWAP 换算股数。
        current_values = self.position_values(adj_vwap)
        total_val = float(self.cash + current_values.sum())
        if total_val <= 1e-12:
            self.pending_target = None
            return ExecutionDayResult()

        target_values = target * total_val
        all_tickers = current_values.index.union(target_values.index)
        delta = target_values.reindex(all_tickers, fill_value=0.0) - current_values.reindex(
            all_tickers, fill_value=0.0
        )
        sell_orders = (-delta[delta < -self.min_notional]).to_dict()
        buy_orders = delta[delta > self.min_notional].to_dict()

        def suspended(ticker: str) -> bool:
            return trade_status.get(ticker) == "停牌"

        def exec_price(ticker: str) -> float:
            return _valid_price(adj_vwap.get(ticker))

        sell_total = 0.0
        buy_total = 0.0
        deferred_sells = 0
        failed_buys = 0

        # 先卖后买。任何被阻断的订单都会保留整份目标，下一交易日重新计算差额。
        for ticker, sell_value in sell_orders.items():
            price = exec_price(ticker)
            blocked = (
                suspended(ticker)
                or self._at_limit(ticker, close_raw, limit_down, "down")
                or price <= 0
            )
            if blocked:
                deferred_sells += 1
                continue
            held_shares = self.shares.get(ticker, 0.0)
            shares_to_sell = min(sell_value / price, held_shares)
            if shares_to_sell <= 1e-10:
                continue
            self.shares[ticker] = held_shares - shares_to_sell
            proceeds = shares_to_sell * price
            self.cash += proceeds * (1.0 - self.cost_sell)
            sell_total += proceeds

        tradable_buys: dict[str, float] = {}
        for ticker, buy_value in buy_orders.items():
            price = exec_price(ticker)
            blocked = (
                suspended(ticker)
                or self._at_limit(ticker, close_raw, limit_up, "up")
                or price <= 0
            )
            if blocked:
                failed_buys += 1
            else:
                tradable_buys[ticker] = buy_value

        buy_demand = float(sum(tradable_buys.values()))
        scale = (
            min(1.0, self.cash / (buy_demand * (1.0 + self.cost_buy) + 1e-12))
            if buy_demand > 0
            else 0.0
        )
        for ticker, buy_value in tradable_buys.items():
            actual_buy = buy_value * scale
            price = exec_price(ticker)
            if actual_buy < self.min_notional or price <= 0:
                continue
            shares_bought = actual_buy / price
            self.shares[ticker] = self.shares.get(ticker, 0.0) + shares_bought
            # 个别数据源可能有有效 VWAP、但当日复权收盘价缺失。首次成交时至少
            # 用成交价建立估值锚点，避免刚买入的持仓从 NAV 中暂时消失。
            self.last_price.setdefault(ticker, price)
            self.cash -= actual_buy * (1.0 + self.cost_buy)
            buy_total += actual_buy

        if self.cash < -self.min_notional:
            raise RuntimeError(f"成交后现金为负：{self.cash:.2f}")
        self.cash = max(self.cash, 0.0)
        self.shares = {t: sh for t, sh in self.shares.items() if sh > 1e-10}

        retry_needed = deferred_sells > 0 or failed_buys > 0
        if not retry_needed:
            self.pending_target = None

        self.buy_fail_count += failed_buys
        self.sell_defer_count += deferred_sells
        turnover = (sell_total + buy_total) / total_val
        return ExecutionDayResult(
            turnover=float(turnover),
            buy_fail_count=failed_buys,
            sell_defer_count=deferred_sells,
            target_pending=retry_needed,
        )
