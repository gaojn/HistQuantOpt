"""ExecutionLedger 的关键成交语义永久回归测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from hqopt.backtest.execution import ExecutionLedger, OrderState


def _step(
    ledger: ExecutionLedger,
    *,
    prices: pd.Series,
    close_raw: pd.Series | None = None,
    limit_up: pd.Series | None = None,
    limit_down: pd.Series | None = None,
):
    """用全可交易状态推进一天，可按需覆盖原始收盘价和涨跌停价。"""
    tickers = prices.index
    return ledger.step(
        adj_close=prices,
        adj_vwap=prices,
        close_raw=close_raw if close_raw is not None else prices,
        limit_up=(
            limit_up
            if limit_up is not None
            else pd.Series(1_000_000.0, index=tickers)
        ),
        limit_down=(
            limit_down
            if limit_down is not None
            else pd.Series(0.01, index=tickers)
        ),
        trade_status=pd.Series("交易", index=tickers),
    )


def test_limit_up_allows_sell():
    """涨停只阻止买入，不得阻止已有持仓卖出。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.cash = 0.0
    ledger.shares = {"A": 10.0}
    ledger.last_price = {"A": 10.0}
    ledger.submit_target(pd.Series({"A": 0.5}))

    result = _step(
        ledger,
        prices=pd.Series({"A": 10.0}),
        close_raw=pd.Series({"A": 11.0}),
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
    )

    assert result.sell_defer_count == 0
    assert result.target_pending is False
    assert ledger.shares["A"] == pytest.approx(5.0)
    assert ledger.cash == pytest.approx(50.0)
    assert ledger.order_states["A"] == OrderState.FILLED


def test_limit_down_allows_buy():
    """跌停只阻止卖出，不得阻止现金买入。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.submit_target(pd.Series({"A": 1.0}))

    result = _step(
        ledger,
        prices=pd.Series({"A": 10.0}),
        close_raw=pd.Series({"A": 9.0}),
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
    )

    assert result.buy_fail_count == 0
    assert result.target_pending is False
    assert ledger.shares["A"] == pytest.approx(10.0)
    assert ledger.cash == pytest.approx(0.0)
    assert ledger.order_states["A"] == OrderState.FILLED


def test_blocked_sell_and_zero_cash_prevents_all_buys():
    """卖单受阻且现金为零时，买入腿不得发生任何成交。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.cash = 0.0
    ledger.shares = {"SELL": 10.0}
    ledger.last_price = {"SELL": 10.0}
    ledger.submit_target(pd.Series({"SELL": 0.5, "BUY": 0.5}))

    result = _step(
        ledger,
        prices=pd.Series({"SELL": 10.0, "BUY": 10.0}),
        close_raw=pd.Series({"SELL": 9.0, "BUY": 10.0}),
        limit_up=pd.Series({"SELL": 11.0, "BUY": 11.0}),
        limit_down=pd.Series({"SELL": 9.0, "BUY": 9.0}),
    )

    assert result.turnover == pytest.approx(0.0)
    assert result.sell_defer_count == 1
    assert result.target_pending is True
    assert ledger.cash == pytest.approx(0.0)
    assert ledger.shares == pytest.approx({"SELL": 10.0})
    assert ledger.order_states["SELL"] == OrderState.PENDING_SELL
    assert ledger.order_states["BUY"] == OrderState.PENDING_BUY


def test_exact_100_orders_fill_90_then_only_retry_remaining_10():
    """T+1 成交 90 只后锁定；T+2 仅继续处理其余 10 只。"""
    tickers = [f"S{i:03d}" for i in range(100)]
    filled_tickers = tickers[:90]
    pending_tickers = tickers[90:]
    target = pd.Series(0.01, index=tickers)
    prices = pd.Series(10.0, index=tickers)

    ledger = ExecutionLedger(
        initial_value=10_000.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.submit_target(target)

    first_close = prices.copy()
    first_limit_up = pd.Series(11.0, index=tickers)
    first_close.loc[pending_tickers] = 11.0
    first = _step(
        ledger,
        prices=prices,
        close_raw=first_close,
        limit_up=first_limit_up,
        limit_down=pd.Series(9.0, index=tickers),
    )
    filled_shares_after_t1 = {
        ticker: ledger.shares[ticker] for ticker in filled_tickers
    }

    assert first.attempt_number == 1
    assert first.filled_count == 90
    assert first.pending_count == 10
    assert first.target_pending is True
    assert set(ledger.shares) == set(filled_tickers)
    assert {
        ticker
        for ticker, state in ledger.order_states.items()
        if state == OrderState.FILLED
    } == set(filled_tickers)
    assert {
        ticker
        for ticker, state in ledger.order_states.items()
        if state == OrderState.PENDING_BUY
    } == set(pending_tickers)

    second = _step(
        ledger,
        prices=prices,
        limit_up=pd.Series(11.0, index=tickers),
        limit_down=pd.Series(9.0, index=tickers),
    )

    assert second.attempt_number == 2
    assert second.filled_count == 10
    assert second.pending_count == 0
    assert second.target_pending is False
    assert {
        ticker: ledger.shares[ticker] for ticker in filled_tickers
    } == pytest.approx(filled_shares_after_t1)
    assert set(ledger.shares) == set(tickers)
    assert all(state == OrderState.FILLED for state in ledger.order_states.values())


def test_cancel_pending_target_resets_without_counting_expiry():
    """新调仓撤销旧目标时，不成交、不计过期，新目标从第 1 次尝试开始。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.submit_target(pd.Series({"OLD": 1.0}))
    blocked = _step(
        ledger,
        prices=pd.Series({"OLD": 10.0}),
        close_raw=pd.Series({"OLD": 11.0}),
        limit_up=pd.Series({"OLD": 11.0}),
        limit_down=pd.Series({"OLD": 9.0}),
    )
    assert blocked.attempt_number == 1

    ledger.cancel_pending_target()

    assert ledger.pending_target is None
    assert ledger.pending_tickers == set()
    assert ledger.order_states == {}
    assert ledger.target_attempts == 0
    assert ledger.expired_order_count == 0
    assert ledger.expired_notional == pytest.approx(0.0)

    ledger.submit_target(pd.Series({"NEW": 1.0}))
    result = _step(ledger, prices=pd.Series({"NEW": 10.0}))

    assert result.attempt_number == 1
    assert ledger.shares == pytest.approx({"NEW": 10.0})
    assert ledger.expired_order_count == 0
