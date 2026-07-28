"""回测引擎执行规则单测（T+1 / 涨跌停 / 停牌 / 退市估值）。

用最小宽表（少量股票、少量交易日）构造可手算的场景，
覆盖 RealisticBacktester 最关键、也最缺测试的执行路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hqopt.backtest.engine as engine_module
from hqopt.backtest.engine import RealisticBacktester, calc_metrics
from hqopt.backtest.execution import ExecutionLedger, OrderState

INIT = 1e8


def _frames(dates, tickers, close=10.0):
    """构造一组「全可交易、无涨跌停」的回测宽表，便于按需覆写。

    Returns dict: adj_close / adj_vwap / close_raw / limit_up / limit_down / trade_status
    """
    def const(v, dtype=float):
        return pd.DataFrame(v, index=dates, columns=tickers, dtype=dtype)

    return {
        "adj_close":    const(close),
        "adj_vwap":     const(close),
        "close_raw":    const(close),
        "limit_up":     const(1e6),     # 极大 → 永不涨停
        "limit_down":   const(0.01),    # 极小 → 永不跌停
        "trade_status": pd.DataFrame("交易", index=dates, columns=tickers),
    }


def _run(weight_df, frames, sell_only_df=None, benchmark_ret=None):
    bt = RealisticBacktester()
    return bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        benchmark_ret=benchmark_ret,
        initial_value=INIT,
        sell_only_df=sell_only_df,
    )


def _run_with_benchmark(weight_df, frames, benchmark_ret):
    return _run(weight_df, frames, benchmark_ret=benchmark_ret)


# ── T+1 成交时序 ─────────────────────────────────────────────────


def test_signal_executes_next_day_not_same_day():
    """T 日出信号当天不成交（NAV 仍为初始）；T+1 才建仓。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    weight_df = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates[:1])

    result, stats = _run(weight_df, frames)

    # 信号日（首日）尚无持仓 → NAV 恰为 1.0
    assert result.nav.iloc[0] == pytest.approx(1.0, abs=1e-9)
    # T+1 建仓后扣买入成本 0.1% → NAV≈0.999，且此后价格不变保持平稳
    assert result.nav.iloc[1] == pytest.approx(0.999, abs=2e-3)
    assert result.nav.iloc[-1] == pytest.approx(result.nav.iloc[1], abs=1e-6)
    assert stats["target_pending"] is False
    assert stats["expired_order_count"] == 0


def test_non_trading_rebalance_date_rejected():
    """调仓日不在行情交易日历中时应明确报错，不能静默丢弃目标。"""
    dates = pd.bdate_range("2024-01-08", periods=3)
    frames = _frames(dates, ["A"])
    weight_df = pd.DataFrame(
        {"A": [1.0]},
        index=[pd.Timestamp("2024-01-06")],  # 周六
    )

    with pytest.raises(ValueError, match="调仓日不在行情交易日历"):
        _run(weight_df, frames)


def test_new_rebalance_cancels_old_target_before_step(monkeypatch):
    """新调仓日直接撤销旧单；新目标仅在下一交易日从第 1 次尝试开始。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A", "B"])

    # 旧目标在 T+1 涨停而未成交；新调仓日 A 已恢复可交易。
    frames["close_raw"].loc[dates[1], "A"] = 11.0
    frames["limit_up"].loc[dates[1], "A"] = 11.0
    weights = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=[dates[0], dates[2]],
    )

    attempts = []

    class RecordingLedger(ExecutionLedger):
        def step(self, **kwargs):
            result = super().step(**kwargs)
            attempts.append(result.attempt_number)
            return result

    monkeypatch.setattr(engine_module, "ExecutionLedger", RecordingLedger)
    result, stats = _run(weights, frames)

    assert result.actual_weights.loc[dates[2]].sum() == pytest.approx(0.0)
    assert result.actual_weights.loc[dates[3]].get("A", 0.0) == pytest.approx(0.0)
    assert result.actual_weights.loc[dates[3], "B"] > 0.99
    assert attempts == [0, 1, 0, 1]
    assert stats["order_states"]["B"] == "filled"


# ── 涨停拦截买入 ─────────────────────────────────────────────────


def test_limit_up_blocks_buy():
    """执行日涨停的票无法买入，计入 buy_fail_count，资金留现金。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    exec_day = dates[1]
    # B 在 T+1 执行日涨停：close == limit_up
    frames["limit_up"].loc[exec_day, "B"] = 11.0
    frames["close_raw"].loc[exec_day, "B"] = 11.0
    frames["adj_vwap"].loc[exec_day, "B"] = 11.0

    weight_df = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=dates[:1])
    _, stats = _run(weight_df, frames)

    assert stats["buy_fail_count"] >= 1


# ── 跌停延期卖出 ─────────────────────────────────────────────────


def test_limit_down_defers_sell():
    """目标清仓的票若执行日跌停，进延期队列，计入 sell_defer_count。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    # 先持有 A、B（day0 信号，day1 建仓）；再 day2 信号清掉 B，day3 执行
    sell_day = dates[3]
    frames["limit_down"].loc[sell_day, "B"] = 9.0
    frames["close_raw"].loc[sell_day, "B"] = 9.0
    frames["adj_vwap"].loc[sell_day, "B"] = 9.0

    weight_df = pd.DataFrame(
        {"A": [0.5, 1.0], "B": [0.5, 0.0]},
        index=[dates[0], dates[2]],
    )
    _, stats = _run(weight_df, frames)

    assert stats["sell_defer_count"] >= 1


# ── 停牌不可成交 ─────────────────────────────────────────────────


def test_suspension_blocks_then_retries_buy():
    """执行日停牌时买入失败，复牌后应继续追踪原目标并完成买入。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A"]
    frames = _frames(dates, tickers)
    frames["trade_status"].loc[dates[1], "A"] = "停牌"

    weight_df = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result, stats = _run(weight_df, frames)

    assert stats["buy_fail_count"] >= 1
    assert stats["target_pending"] is False
    assert stats["final_actual_weights"]["A"] > 0.99
    assert result.actual_weights.loc[dates[1]].sum() == 0.0
    assert result.actual_weights.loc[dates[2], "A"] > 0.99
    # 复牌后完成建仓并扣除买入成本
    assert result.nav.iloc[-1] == pytest.approx(0.999, abs=2e-3)


# ── 退市估值回归（#2 修复）─────────────────────────────────────────


def test_delisting_no_phantom_nav_collapse():
    """持仓股退市（后续行情缺失）时，NAV 用最后有效价估值，不应凭空崩塌。"""
    dates = pd.bdate_range("2024-01-02", periods=8)
    tickers = ["A"]
    frames = _frames(dates, tickers)
    # A 自 day5 起退市：所有价格列置 NaN（trade_status 也无意义，但保持"交易"不影响）
    for key in ("adj_close", "adj_vwap", "close_raw"):
        frames[key].loc[dates[5:], "A"] = np.nan

    weight_df = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result, stats = _run(weight_df, frames)

    # 建仓后 NAV≈0.999；退市后应保持该水平而非掉到现金（~0）
    held_nav = result.nav.iloc[4]
    assert held_nav == pytest.approx(0.999, abs=2e-3)
    assert result.nav.iloc[-1] == pytest.approx(held_nav, abs=1e-6)
    # 告警字段：末日有 1 只滞留、且占 NAV 绝大部分
    assert stats["delisted_stuck_count"] == 1
    assert stats["stale_value_pct"] > 0.9


# ── 换手记录 & 指标 ──────────────────────────────────────────────


def test_turnover_recorded_on_execution():
    """建仓日应记录非零换手率。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=dates[:1])
    result, _ = _run(weight_df, frames)

    assert len(result.turnover) >= 1
    assert result.turnover.iloc[0] > 0.0


# ── 首日基准口径 ─────────────────────────────────────────────────


def test_first_rebalance_day_benchmark_return_is_zeroed():
    """首个调仓日组合空仓，基准不得计息，否则超额被一次性侵蚀。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=dates[:1])
    # 首日基准 +5%：修复前会全额记为 -5% 超额
    bm = pd.Series([0.05, 0.01, 0.0, 0.0, 0.0], index=dates)

    result, _ = _run_with_benchmark(weight_df, frames, bm)

    assert result.daily_ret.iloc[0] == pytest.approx(0.0)
    assert result.bm_ret.iloc[0] == pytest.approx(0.0)
    assert result.excess_ret.iloc[0] == pytest.approx(0.0)
    assert result.bm_nav.iloc[0] == pytest.approx(1.0)
    assert result.excess_nav.iloc[0] == pytest.approx(1.0)
    # 首日之后的基准收益不受影响
    assert result.bm_ret.iloc[1] == pytest.approx(0.01)


def test_first_day_zeroing_does_not_mutate_caller_benchmark():
    """置零只作用于回测内部副本，调用方传入的 Series 不被改写。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    tickers = ["A"]
    frames = _frames(dates, tickers)
    weight_df = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    bm = pd.Series([0.05, 0.0, 0.0], index=dates)

    _run_with_benchmark(weight_df, frames, bm)

    assert bm.iloc[0] == pytest.approx(0.05)


def test_default_equal_weight_benchmark_also_zeroed_on_first_day():
    """benchmark_ret=None 走等权全股分支时，首日同样置零。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    # 价格逐日上涨 → 等权基准每日均为正收益
    rising = pd.DataFrame(
        [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]],
        index=dates, columns=tickers,
    )
    frames["adj_close"] = rising
    frames["adj_vwap"] = rising
    frames["close_raw"] = rising
    weight_df = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=dates[:1])

    result, _ = _run(weight_df, frames)

    assert result.bm_ret.iloc[0] == pytest.approx(0.0)
    assert result.bm_ret.iloc[1] > 0.0


# ── XD/XR/N 不被引擎误拦 ─────────────────────────────────────────


def test_xd_xr_n_tradable_not_blocked():
    """XD/XR/N 日股票可正常成交，引擎不应拦截买入。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    # A 在 T+1 执行日为 XD 状态（除息，但可正常买入）
    frames["trade_status"].loc[dates[1], "A"] = "XD"
    # 价格正常，不触涨跌停
    weight_df = pd.DataFrame({"A": [0.8], "B": [0.2]}, index=dates[:1])
    result, stats = _run(weight_df, frames)
    # XD 日不应被拦截，buy_fail_count 应为 0
    assert stats["buy_fail_count"] == 0
    # 建仓完成后 NAV 约为 0.999（扣买入费率）
    assert result.nav.iloc[1] == pytest.approx(0.999, abs=2e-3)


# ── 停牌阻断卖出 ─────────────────────────────────────────────────


def test_suspension_blocks_sell():
    """持仓股停牌时无法卖出，应进延期队列。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    # day0 先建仓 A:0.5 B:0.5；day2 信号清仓 B；day3 执行但 B 停牌
    sell_day = dates[3]
    frames["trade_status"].loc[sell_day, "B"] = "停牌"
    weight_df = pd.DataFrame(
        {"A": [0.5, 1.0], "B": [0.5, 0.0]},
        index=[dates[0], dates[2]],
    )
    _, stats = _run(weight_df, frames)
    # B 停牌无法卖出 → 进延期队列
    assert stats["sell_defer_count"] >= 1


def test_deferred_sell_also_completes_buy_leg():
    """卖出恢复后必须继续买入目标股票，不能留下长期现金。"""
    dates = pd.bdate_range("2024-01-02", periods=7)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    frames["trade_status"].loc[dates[3], "A"] = "停牌"
    weight_df = pd.DataFrame(
        {"A": [1.0, 0.5], "B": [0.0, 0.5]},
        index=[dates[0], dates[2]],
    )

    bt = RealisticBacktester(cost_buy=0.0, cost_sell=0.0, risk_free=0.0)
    result, stats = bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=INIT,
    )

    assert stats["sell_defer_count"] >= 1
    assert result.actual_weights.loc[dates[3], "B"] == pytest.approx(0.0)
    assert stats["target_pending"] is False
    assert stats["final_cash_pct"] < 1e-8
    assert stats["final_actual_weights"]["A"] == pytest.approx(0.5, abs=1e-6)
    assert stats["final_actual_weights"]["B"] == pytest.approx(0.5, abs=1e-6)
    assert result.nav.iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_rebalance_order_sizing_uses_vwap_not_close():
    """收盘价与 VWAP 不同时，订单差额和成交股数必须使用同一 VWAP 口径。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    frames = _frames(dates, ["A", "B"])
    # day2 收盘提交 50/50 目标；day3 执行时 A 的 VWAP=10、收盘=20。
    frames["adj_close"].loc[dates[3]:, "A"] = 20.0
    weight_df = pd.DataFrame(
        {"A": [1.0, 0.5], "B": [0.0, 0.5]},
        index=[dates[0], dates[2]],
    )

    bt = RealisticBacktester(cost_buy=0.0, cost_sell=0.0, risk_free=0.0)
    result, stats = bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=INIT,
    )

    # VWAP 时点成交后 A/B 各 5 百万股；A 随后按 20 元收盘，故收盘权重为 2/3、1/3。
    assert stats["final_actual_weights"]["A"] == pytest.approx(2 / 3, abs=1e-6)
    assert stats["final_actual_weights"]["B"] == pytest.approx(1 / 3, abs=1e-6)
    assert result.nav.iloc[-1] == pytest.approx(1.5, abs=1e-9)


def test_signal_day_suspension_freezes_shares_for_whole_target():
    """T 日停牌票即使 T+1 复牌，也应保持股数不变。"""
    dates = pd.bdate_range("2024-01-02", periods=7)
    frames = _frames(dates, ["A", "B", "C"])
    frames["trade_status"].loc[dates[2], "A"] = "停牌"
    frames["adj_close"].loc[dates[3]:, "A"] = 20.0
    frames["adj_vwap"].loc[dates[3]:, "A"] = 20.0
    frames["close_raw"].loc[dates[3]:, "A"] = 20.0

    weight_df = pd.DataFrame(
        {
            "A": [0.5, 0.5],
            "B": [0.5, 0.0],
            "C": [0.0, 0.5],
        },
        index=[dates[0], dates[2]],
    )
    bt = RealisticBacktester(cost_buy=0.0, cost_sell=0.0, risk_free=0.0)
    result, stats = bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=INIT,
    )

    assert stats["final_shares"]["A"] == pytest.approx(5_000_000.0)
    assert result.actual_weights.iloc[-1]["A"] == pytest.approx(2 / 3)
    assert result.actual_weights.iloc[-1]["C"] == pytest.approx(1 / 3)
    assert stats["order_states"]["A"] == "frozen"


def test_filled_symbols_locked_and_partial_buys_remain_pending_until_expiry():
    """已成交股票不回调；不等额买单按同一比例部分成交，T+3 后过期。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    frames = _frames(dates, ["A", "B", "C"])
    for ticker in ("B", "C"):
        frames["close_raw"].loc[dates[1], ticker] = 11.0
        frames["limit_up"].loc[dates[1], ticker] = 11.0
    frames["adj_close"].loc[dates[2]:, "A"] = 20.0
    frames["adj_vwap"].loc[dates[2]:, "A"] = 20.0
    frames["close_raw"].loc[dates[2]:, "A"] = 20.0

    weight_df = pd.DataFrame(
        {"A": [0.4], "B": [0.4], "C": [0.2]},
        index=[dates[0]],
    )
    bt = RealisticBacktester(cost_buy=0.0, cost_sell=0.0, risk_free=0.0)
    _, stats = bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=100.0,
    )

    # T+2 时 NAV=140、现金=60；B/C 需求分别为 56/28，统一乘 5/7 后买 40/20。
    assert stats["final_shares"] == pytest.approx({"A": 4.0, "B": 4.0, "C": 2.0})
    assert stats["order_states"]["A"] == "filled"
    assert stats["order_states"]["B"] == "expired"
    assert stats["order_states"]["C"] == "expired"
    assert stats["expired_order_count"] == 2
    assert stats["expired_notional"] == pytest.approx(24.0)


def test_orders_expire_after_t3_and_new_target_reactivates():
    """T+3 后旧单失效，T+4 不成交；下一次新目标可重新激活交易。"""
    dates = pd.bdate_range("2024-01-02", periods=7)
    frames = _frames(dates, ["A"])
    frames["close_raw"].loc[dates[1], "A"] = 11.0
    frames["limit_up"].loc[dates[1], "A"] = 11.0
    frames["trade_status"].loc[dates[2], "A"] = "停牌"
    frames["adj_vwap"].loc[dates[3], "A"] = np.nan

    weight_df = pd.DataFrame({"A": [1.0, 1.0]}, index=[dates[0], dates[5]])
    bt = RealisticBacktester(cost_buy=0.0, cost_sell=0.0, risk_free=0.0)
    result, stats = bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=INIT,
    )

    assert result.actual_weights.loc[dates[4]].sum() == 0.0
    assert result.actual_weights.loc[dates[6], "A"] > 0.99
    assert stats["expired_order_count"] == 1
    assert stats["order_states"]["A"] == "filled"


def test_t3_recovery_executes_before_expiry_and_attempts_are_exact():
    """T+3 是最后一次有效尝试；当日恢复交易应成交，而非先过期。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.submit_target(pd.Series({"A": 1.0}))
    price = pd.Series({"A": 10.0})

    first = ledger.step(
        adj_close=price,
        adj_vwap=price,
        close_raw=pd.Series({"A": 11.0}),
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
        trade_status=pd.Series({"A": "交易"}),
    )
    second = ledger.step(
        adj_close=price,
        adj_vwap=price,
        close_raw=price,
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
        trade_status=pd.Series({"A": "停牌"}),
    )
    third = ledger.step(
        adj_close=price,
        adj_vwap=price,
        close_raw=price,
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
        trade_status=pd.Series({"A": "交易"}),
    )

    assert [first.attempt_number, second.attempt_number, third.attempt_number] == [1, 2, 3]
    assert third.expired_count == 0
    assert ledger.expired_order_count == 0
    assert ledger.shares["A"] == pytest.approx(10.0)
    assert ledger.order_states["A"] == OrderState.FILLED


def test_pending_order_recalculates_direction_but_sell_only_never_buys():
    """普通 pending 按最新目标变向；制度性 sell_only 仍禁止反向买入。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.cash = 0.0
    ledger.shares = {"A": 5.0, "D": 5.0}
    ledger.last_price = {"A": 10.0, "D": 10.0}
    ledger.submit_target(pd.Series({"A": 0.4, "D": 0.6}))

    day1_price = pd.Series({"A": 10.0, "D": 10.0})
    ledger.step(
        adj_close=day1_price,
        adj_vwap=day1_price,
        close_raw=pd.Series({"A": 9.0, "D": 10.0}),
        limit_up=pd.Series({"A": 11.0, "D": 11.0}),
        limit_down=pd.Series({"A": 9.0, "D": 9.0}),
        trade_status=pd.Series({"A": "交易", "D": "交易"}),
    )
    assert ledger.order_states["A"] == OrderState.PENDING_SELL
    assert ledger.order_states["D"] == OrderState.PENDING_BUY

    # D 翻倍后，A 的原卖单差额转正、D 的原买单差额转负；普通订单按最新差额变向。
    day2_price = pd.Series({"A": 10.0, "D": 20.0})
    ledger.step(
        adj_close=day2_price,
        adj_vwap=day2_price,
        close_raw=day2_price,
        limit_up=pd.Series({"A": 11.0, "D": 22.0}),
        limit_down=pd.Series({"A": 9.0, "D": 18.0}),
        trade_status=pd.Series({"A": "交易", "D": "交易"}),
    )

    assert ledger.shares == pytest.approx({"A": 6.0, "D": 4.5})

    sell_only_ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    sell_only_ledger.cash = 0.0
    sell_only_ledger.shares = {"A": 5.0, "D": 5.0}
    sell_only_ledger.last_price = {"A": 10.0, "D": 10.0}
    sell_only_ledger.submit_target(
        pd.Series({"A": 0.4, "D": 0.6}),
        sell_only_tickers=["A"],
    )
    sell_only_ledger.step(
        adj_close=day1_price,
        adj_vwap=day1_price,
        close_raw=pd.Series({"A": 9.0, "D": 10.0}),
        limit_up=pd.Series({"A": 11.0, "D": 11.0}),
        limit_down=pd.Series({"A": 9.0, "D": 9.0}),
        trade_status=pd.Series({"A": "交易", "D": "交易"}),
    )
    sell_only_ledger.step(
        adj_close=day2_price,
        adj_vwap=day2_price,
        close_raw=day2_price,
        limit_up=pd.Series({"A": 11.0, "D": 22.0}),
        limit_down=pd.Series({"A": 9.0, "D": 18.0}),
        trade_status=pd.Series({"A": "交易", "D": "交易"}),
    )

    assert sell_only_ledger.shares["A"] == pytest.approx(5.0)
    assert sell_only_ledger.shares["D"] == pytest.approx(4.5)


def test_completed_sell_is_locked_while_buy_remains_pending():
    """卖单完成后立即锁定；其余买单 pending 时价格漂移也不得再次卖该股票。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.0,
    )
    ledger.cash = 0.0
    ledger.shares = {"A": 10.0}
    ledger.last_price = {"A": 10.0}
    ledger.submit_target(pd.Series({"A": 0.5, "B": 0.5}))

    day1_price = pd.Series({"A": 10.0, "B": 10.0})
    ledger.step(
        adj_close=day1_price,
        adj_vwap=day1_price,
        close_raw=pd.Series({"A": 10.0, "B": 11.0}),
        limit_up=pd.Series({"A": 11.0, "B": 11.0}),
        limit_down=pd.Series({"A": 9.0, "B": 9.0}),
        trade_status=pd.Series({"A": "交易", "B": "交易"}),
    )
    assert ledger.shares["A"] == pytest.approx(5.0)
    assert ledger.order_states["A"] == OrderState.FILLED
    assert ledger.order_states["B"] == OrderState.PENDING_BUY

    # A 翻倍使其高于原目标，但 A 已成交锁定；T+2 只能继续处理 B。
    day2_price = pd.Series({"A": 20.0, "B": 10.0})
    ledger.step(
        adj_close=day2_price,
        adj_vwap=day2_price,
        close_raw=day2_price,
        limit_up=pd.Series({"A": 22.0, "B": 11.0}),
        limit_down=pd.Series({"A": 18.0, "B": 9.0}),
        trade_status=pd.Series({"A": "交易", "B": "交易"}),
    )

    assert ledger.shares["A"] == pytest.approx(5.0)
    assert ledger.shares["B"] == pytest.approx(5.0)


def test_buy_fee_crossing_target_completes_instead_of_expiring():
    """买费令成交后轻微越过目标时应视为完成，不能继续 pending 到过期。"""
    ledger = ExecutionLedger(
        initial_value=1e8,
        cost_buy=0.001,
        cost_sell=0.0,
    )
    ledger.submit_target(pd.Series({"A": 1.0 - 3e-7}))
    price = pd.Series({"A": 10.0})

    result = ledger.step(
        adj_close=price,
        adj_vwap=price,
        close_raw=price,
        limit_up=pd.Series({"A": 11.0}),
        limit_down=pd.Series({"A": 9.0}),
        trade_status=pd.Series({"A": "交易"}),
    )

    assert result.target_pending is False
    assert ledger.order_states["A"] == OrderState.FILLED
    assert ledger.expired_order_count == 0


def test_t3_recomputes_and_executes_new_sell_after_sell_fee():
    """T+3 首笔卖出降 NAV 后产生的新卖单，必须在同日继续执行而非过期。"""
    ledger = ExecutionLedger(
        initial_value=1e8,
        cost_buy=0.001,
        cost_sell=0.002,
    )
    ledger.cash = 0.0
    ledger.shares = {"A": 49_990_000.0, "B": 50_010_000.0}
    ledger.last_price = {"A": 1.0, "B": 1.0}
    ledger.submit_target(pd.Series({"A": 0.5, "B": 0.0}))

    price = pd.Series({"A": 1.0, "B": 1.0})
    for _ in range(2):
        ledger.step(
            adj_close=price,
            adj_vwap=price,
            close_raw=pd.Series({"A": 1.0, "B": 0.9}),
            limit_up=pd.Series({"A": 1.1, "B": 1.1}),
            limit_down=pd.Series({"A": 0.9, "B": 0.9}),
            trade_status=pd.Series({"A": "交易", "B": "交易"}),
        )

    third = ledger.step(
        adj_close=price,
        adj_vwap=price,
        close_raw=price,
        limit_up=pd.Series({"A": 1.1, "B": 1.1}),
        limit_down=pd.Series({"A": 0.9, "B": 0.9}),
        trade_status=pd.Series({"A": "交易", "B": "交易"}),
    )

    assert third.attempt_number == 3
    assert third.expired_count == 0
    assert ledger.expired_order_count == 0
    assert ledger.order_states == {
        "A": OrderState.FILLED,
        "B": OrderState.FILLED,
    }
    assert ledger.shares["A"] == pytest.approx(49_949_990.0)
    assert "B" not in ledger.shares


def test_sell_phase_handles_three_level_fee_cascade_in_one_day():
    """卖费连续压低 NAV 时，C→A→B 三层新卖单都应在进入买入前完成。"""
    ledger = ExecutionLedger(
        initial_value=100.0,
        cost_buy=0.0,
        cost_sell=0.1,
        min_notional=0.001,
        max_attempts=1,
    )
    ledger.cash = 0.0
    ledger.shares = {"A": 49.99, "B": 39.57, "C": 10.44}
    ledger.last_price = {"A": 1.0, "B": 1.0, "C": 1.0}
    ledger.submit_target(pd.Series({"A": 0.5, "B": 0.4, "C": 0.0}))
    prices = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})

    result = ledger.step(
        adj_close=prices,
        adj_vwap=prices,
        close_raw=prices,
        limit_up=pd.Series({"A": 1.1, "B": 1.1, "C": 1.1}),
        limit_down=pd.Series({"A": 0.9, "B": 0.9, "C": 0.9}),
        trade_status=pd.Series({"A": "交易", "B": "交易", "C": "交易"}),
    )

    assert result.filled_count == 3
    assert result.expired_count == 0
    assert ledger.expired_order_count == 0
    assert ledger.order_states == {
        "A": OrderState.FILLED,
        "B": OrderState.FILLED,
        "C": OrderState.FILLED,
    }


@pytest.mark.parametrize("max_attempts", [0, -1, 0.5, True])
def test_execution_attempt_limit_requires_positive_integer(max_attempts):
    with pytest.raises(ValueError, match="正整数"):
        ExecutionLedger(initial_value=100.0, max_attempts=max_attempts)


def test_target_weight_sum_above_one_is_rejected():
    ledger = ExecutionLedger(initial_value=100.0)
    with pytest.raises(ValueError, match="不能超过 1"):
        ledger.submit_target(pd.Series({"A": 1.0 + 5e-7}))


def test_engine_direct_call_rejects_invalid_sell_only_values_and_shape():
    """公开引擎入口也必须 fail-closed，不能把字符串 False 当成布尔 True。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A", "B"])
    weights = pd.DataFrame(
        {"A": [0.6], "B": [0.4]},
        index=dates[:1],
    )

    invalid_values = pd.DataFrame(
        {"A": ["False"], "B": [False]},
        index=weights.index,
    )
    with pytest.raises(ValueError, match="只能包含"):
        _run(weights, frames, invalid_values)

    missing_column = pd.DataFrame(
        {"A": [False]},
        index=weights.index,
    )
    with pytest.raises(ValueError, match="股票列与权重文件不一致"):
        _run(weights, frames, missing_column)


def test_calc_metrics_constant_returns():
    """恒定正收益：年化为正、Sharpe 为正、回撤≈0。"""
    idx = pd.bdate_range("2024-01-02", periods=252)
    ret = pd.Series(0.001, index=idx)
    bm = pd.Series(0.0, index=idx)

    m = calc_metrics(ret, bm)

    assert m.annual_return == pytest.approx((1.001) ** 252 - 1, rel=1e-6)
    assert m.sharpe > 0
    assert m.max_drawdown <= 1e-9
    # 几何超额（bm=0 时等于组合年化收益）
    assert m.annual_excess_return == pytest.approx((1.001) ** 252 - 1, rel=1e-4)


# ── 拆分后的内部构件 ─────────────────────────────────────────────


def _min_frames(dates, tickers, close=10.0):
    """用真实对齐路径构造 _MarketFrames，避免测试自造与生产不一致的结构。"""
    f = _frames(dates, tickers, close)
    return engine_module._align_market_frames(
        f["adj_close"], f["adj_vwap"], f["close_raw"],
        f["limit_up"], f["limit_down"], f["trade_status"],
        first_rebal=dates[0],
    )


def test_normalize_weight_inputs_casts_non_datetime_index():
    """权重索引是 date/字符串时必须转成 Timestamp，否则调仓日永远匹配不上。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    adj_close = pd.DataFrame(10.0, index=dates, columns=["A"])
    weights = pd.DataFrame({"A": [1.0]}, index=[dates[0].date()])

    normalized, sell_only, rebal_dates = engine_module._normalize_weight_inputs(
        weights, None, adj_close
    )

    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert rebal_dates == [dates[0]]
    assert sell_only is None
    assert not isinstance(weights.index, pd.DatetimeIndex)   # 未就地改写调用方


def test_normalize_weight_inputs_rejects_dates_outside_calendar():
    dates = pd.bdate_range("2024-01-02", periods=3)
    adj_close = pd.DataFrame(10.0, index=dates, columns=["A"])
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2023-06-01"]))

    with pytest.raises(ValueError, match="调仓日不在行情交易日历中: 2023-06-01"):
        engine_module._normalize_weight_inputs(weights, None, adj_close)


def test_normalize_weight_inputs_truncates_long_missing_date_list():
    """缺失日超过 5 个时只列前 5 个并附总数，避免刷屏。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    adj_close = pd.DataFrame(10.0, index=dates, columns=["A"])
    missing = pd.bdate_range("2023-06-01", periods=7)
    weights = pd.DataFrame({"A": [1.0] * 7}, index=missing)

    with pytest.raises(ValueError, match=r"（共 7 日）"):
        engine_module._normalize_weight_inputs(weights, None, adj_close)


def test_stale_holding_value_counts_only_price_missing_positions():
    """只统计「末日无真实价、靠 ffill 陈旧价估值」的持仓。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    tickers = ["ALIVE", "DEAD", "DUST"]
    f = _frames(dates, tickers)
    f["adj_close"].loc[dates[-1], "DEAD"] = np.nan       # 退市：末日无价
    f["adj_close"].loc[dates[-1], "DUST"] = np.nan
    frames = engine_module._align_market_frames(
        f["adj_close"], f["adj_vwap"], f["close_raw"],
        f["limit_up"], f["limit_down"], f["trade_status"],
        first_rebal=dates[0],
    )
    ledger = ExecutionLedger(initial_value=INIT)
    ledger.shares = {"ALIVE": 100.0, "DEAD": 50.0, "DUST": 1e-12}

    count, value = engine_module._stale_holding_value(ledger, frames)

    assert count == 1                       # ALIVE 有真实价、DUST 是粉尘仓位
    assert value == pytest.approx(50.0 * 10.0)


def test_stale_holding_value_zero_when_all_prices_present():
    dates = pd.bdate_range("2024-01-02", periods=3)
    frames = _min_frames(dates, ["A"])
    ledger = ExecutionLedger(initial_value=INIT)
    ledger.shares = {"A": 100.0}

    assert engine_module._stale_holding_value(ledger, frames) == (0, 0.0)


def test_resolve_benchmark_returns_zeroes_first_day_for_both_sources():
    dates = pd.bdate_range("2024-01-02", periods=4)
    rising = pd.DataFrame(
        [[10.0], [11.0], [12.0], [13.0]], index=dates, columns=["A"]
    )
    explicit = pd.Series([0.05, 0.01, 0.02, 0.03], index=dates)

    from_explicit = engine_module._resolve_benchmark_returns(explicit, rising, dates)
    from_equal_weight = engine_module._resolve_benchmark_returns(None, rising, dates)

    assert from_explicit.iloc[0] == pytest.approx(0.0)
    assert from_explicit.iloc[1] == pytest.approx(0.01)     # 其余日不受影响
    assert from_equal_weight.iloc[0] == pytest.approx(0.0)
    assert from_equal_weight.iloc[1] > 0.0
    assert explicit.iloc[0] == pytest.approx(0.05)          # 未改写调用方


def test_resolve_benchmark_returns_fills_missing_dates_with_zero():
    dates = pd.bdate_range("2024-01-02", periods=4)
    adj_close = pd.DataFrame(10.0, index=dates, columns=["A"])
    partial = pd.Series([0.02], index=dates[2:3])

    resolved = engine_module._resolve_benchmark_returns(partial, adj_close, dates)

    assert list(resolved.index) == list(dates)
    assert resolved.iloc[1] == pytest.approx(0.0)
    assert resolved.iloc[2] == pytest.approx(0.02)


def test_signal_day_orders_extracts_frozen_and_sell_only():
    dates = pd.bdate_range("2024-01-02", periods=2)
    tickers = ["A", "B", "C"]
    f = _frames(dates, tickers)
    f["trade_status"].loc[dates[0], "B"] = "停牌"
    frames = engine_module._align_market_frames(
        f["adj_close"], f["adj_vwap"], f["close_raw"],
        f["limit_up"], f["limit_down"], f["trade_status"],
        first_rebal=dates[0],
    )
    weights = pd.DataFrame([[0.3, 0.3, 0.4]], index=dates[:1], columns=tickers)
    sell_only = pd.DataFrame(
        [[False, False, True]], index=dates[:1], columns=tickers
    )

    frozen, sell_only_tickers = engine_module._signal_day_orders(
        frames, weights, sell_only, dates[0]
    )

    assert frozen == ["B"]
    assert sell_only_tickers == ["C"]


def test_signal_day_orders_without_sell_only_matrix():
    dates = pd.bdate_range("2024-01-02", periods=2)
    frames = _min_frames(dates, ["A"])
    weights = pd.DataFrame({"A": [1.0]}, index=dates[:1])

    frozen, sell_only_tickers = engine_module._signal_day_orders(
        frames, weights, None, dates[0]
    )

    assert frozen == []
    assert sell_only_tickers == []
