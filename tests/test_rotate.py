"""轮动分仓回测引擎单测（T+1 开盘买 / T+H 收盘卖 / 资金分 H 份）。

用最小宽表（少量股票、少量交易日）构造可手算的场景，
覆盖 RotateBacktester 的建仓预算、等权分配、涨跌停/停牌、顺延卖出等路径。
"""
from __future__ import annotations

import pandas as pd
import pytest

from hqopt.backtest.rotate import RotateBacktester

INIT = 1e8
COST_BUY = 0.001
COST_SELL = 0.002


def _frames(dates, tickers, price=10.0):
    """构造一组「全可交易、无涨跌停、开=收=price」的宽表，便于按需覆写。"""
    def const(v, dtype=float):
        return pd.DataFrame(v, index=dates, columns=tickers, dtype=dtype)

    return {
        "adj_open":     const(price),
        "adj_close":    const(price),
        "open_raw":     const(price),
        "close_raw":    const(price),
        "limit_up":     const(1e6),     # 极大 → 永不涨停
        "limit_down":   const(0.01),    # 极小 → 永不跌停
        "trade_status": pd.DataFrame("交易", index=dates, columns=tickers),
    }


def _picks(rows):
    """rows: [(date, code), ...] → 选股长表。"""
    return pd.DataFrame(rows, columns=["date", "code"])


def _run(picks, frames, hold_days, initial_value=INIT):
    bt = RotateBacktester(
        hold_days=hold_days, cost_buy=COST_BUY, cost_sell=COST_SELL
    )
    return bt.run(
        picks=picks,
        adj_open=frames["adj_open"],
        adj_close=frames["adj_close"],
        open_raw=frames["open_raw"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        initial_value=initial_value,
    )


# ── 基本时序与资金分份 ────────────────────────────────────────────


def test_h2_buy_next_open_sell_second_close():
    """H=2：T 日信号当天不动，T+1 开盘用 1/2 资金买入，T+2 收盘卖出。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    frames = _frames(dates, ["A"])
    picks = _picks([(dates[0], "A")])

    result, stats, trades = _run(picks, frames, hold_days=2)

    # 信号日（首日）全现金 → NAV = 1.0
    assert result.nav.iloc[0] == pytest.approx(1.0, abs=1e-12)
    # T+1：出资 0.5，买入净额 0.5/1.001（价格不变）
    nav1_expected = 0.5 + 0.5 / (1 + COST_BUY)
    assert result.nav.iloc[1] == pytest.approx(nav1_expected, abs=1e-9)
    # T+2：收盘按原价卖出，扣 2‰ 卖出费
    nav2_expected = 0.5 + 0.5 * (1 - COST_SELL) / (1 + COST_BUY)
    assert result.nav.iloc[2] == pytest.approx(nav2_expected, abs=1e-9)
    # 之后无持仓，NAV 平稳
    assert result.nav.iloc[-1] == pytest.approx(nav2_expected, abs=1e-9)
    assert stats["open_position_count"] == 0

    # 成交明细：1 买 1 卖，均归属同一信号日
    assert list(trades["side"]) == ["buy", "sell"]
    assert (trades["signal_date"] == dates[0]).all()


def test_h1_intraday_buy_open_sell_close():
    """H=1：T+1 开盘全仓买入、当日收盘卖出，赚足日内价差。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    frames = _frames(dates, ["A"])
    # T+1 开盘 10 → 收盘 11
    frames["adj_close"].loc[dates[1], "A"] = 11.0
    frames["close_raw"].loc[dates[1], "A"] = 11.0
    picks = _picks([(dates[0], "A")])

    result, stats, trades = _run(picks, frames, hold_days=1)

    gross = 1.0 / (1 + COST_BUY)                 # 买入净额（价格 10 的市值占比）
    nav_expected = gross * 1.1 * (1 - COST_SELL)  # 10 → 11 涨 10% 后扣卖出费
    assert result.nav.iloc[1] == pytest.approx(nav_expected, abs=1e-9)
    assert list(trades["side"]) == ["buy", "sell"]
    assert stats["open_position_count"] == 0


def test_equal_weight_within_bucket():
    """桶内多只股票等权分配。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A", "B"])
    picks = _picks([(dates[0], "A"), (dates[0], "B")])

    _, _, trades = _run(picks, frames, hold_days=2)

    buys = trades[trades["side"] == "buy"]
    assert len(buys) == 2
    # 每票出资 = (INIT/2)/2，成交净额 = 出资 / (1+费率)
    expected_notional = INIT / 4 / (1 + COST_BUY)
    assert buys["notional"].tolist() == pytest.approx(
        [expected_notional] * 2, rel=1e-12
    )


def test_daily_budget_is_total_equity_over_h():
    """H=2 连续两天出信号：第二桶预算按「前收盘总资产/2」计。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    frames = _frames(dates, ["A", "B"])
    picks = _picks([(dates[0], "A"), (dates[1], "B")])

    _, _, trades = _run(picks, frames, hold_days=2)

    buys = trades[trades["side"] == "buy"].set_index("code")
    # 第一桶出资 INIT/2；第一天买入费导致总资产微降，第二桶预算略小
    equity_day1 = INIT / 2 + INIT / 2 / (1 + COST_BUY)
    assert buys.loc["A", "notional"] == pytest.approx(
        INIT / 2 / (1 + COST_BUY), rel=1e-12
    )
    assert buys.loc["B", "notional"] == pytest.approx(
        equity_day1 / 2 / (1 + COST_BUY), rel=1e-12
    )


def test_budget_capped_by_available_cash():
    """持仓浮盈推高总资产时，建仓预算受可用现金约束（不透支）。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    frames = _frames(dates, ["A", "B"])
    # A 在 T+1 收盘翻倍 → 总资产/2 > 剩余现金
    frames["adj_close"].loc[dates[1]:, "A"] = 20.0
    frames["close_raw"].loc[dates[1]:, "A"] = 20.0
    picks = _picks([(dates[0], "A"), (dates[1], "B")])

    _, stats, trades = _run(picks, frames, hold_days=2)

    buys = trades[trades["side"] == "buy"].set_index("code")
    # 第二桶预算 = min(总资产/2, 现金 5e7) = 5e7
    assert buys.loc["B", "notional"] == pytest.approx(
        INIT / 2 / (1 + COST_BUY), rel=1e-12
    )


def test_no_signal_days_leave_cash_idle_then_natural_reset():
    """连续无信号 → 持仓卖光后全现金；再次出信号时按总资产/H 重新建仓。"""
    dates = pd.bdate_range("2024-01-02", periods=10)
    frames = _frames(dates, ["A"])
    # day0 信号 → day1 买、day2 卖；day3~5 无信号；day6 再出信号
    picks = _picks([(dates[0], "A"), (dates[6], "A")])

    result, _, trades = _run(picks, frames, hold_days=2)

    # day3~day6 无持仓，NAV 平稳
    for i in range(3, 7):
        assert result.nav.iloc[i] == pytest.approx(result.nav.iloc[2], abs=1e-12)
    # day7 重新建仓：出资 = 当时总资产 / 2
    equity_before = result.nav.iloc[6] * INIT
    buys = trades[trades["side"] == "buy"]
    assert buys.iloc[1]["notional"] == pytest.approx(
        equity_before / 2 / (1 + COST_BUY), rel=1e-12
    )


# ── 涨跌停 / 停牌 / 缺行情 ────────────────────────────────────────


def test_buy_blocked_by_limit_up_open_leaves_cash():
    """T+1 开盘一字涨停：放弃买入，资金留现金。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A", "B"])
    frames["open_raw"].loc[dates[1], "A"] = 11.0
    frames["limit_up"].loc[dates[1], "A"] = 11.0
    picks = _picks([(dates[0], "A"), (dates[0], "B")])

    result, stats, trades = _run(picks, frames, hold_days=2)

    assert stats["buy_fail_count"] == 1
    buys = trades[trades["side"] == "buy"]
    assert buys["code"].tolist() == ["B"]
    # A 的份额留现金：只买入了 B 的一半仓位
    nav1_expected = 0.75 + 0.25 / (1 + COST_BUY)
    assert result.nav.iloc[1] == pytest.approx(nav1_expected, abs=1e-9)


def test_buy_blocked_by_suspension():
    """T+1 停牌：放弃买入。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A"])
    frames["trade_status"].loc[dates[1], "A"] = "停牌"
    picks = _picks([(dates[0], "A")])

    result, stats, _ = _run(picks, frames, hold_days=2)

    assert stats["buy_fail_count"] == 1
    assert result.nav.iloc[-1] == pytest.approx(1.0, abs=1e-12)


def test_unknown_code_counted_as_buy_fail():
    """行情里不存在的代码：计入买入放弃，不影响其余票。"""
    dates = pd.bdate_range("2024-01-02", periods=4)
    frames = _frames(dates, ["A"])
    picks = _picks([(dates[0], "A"), (dates[0], "ZZZ")])

    _, stats, trades = _run(picks, frames, hold_days=2)

    assert stats["buy_fail_count"] == 1
    assert trades[trades["side"] == "buy"]["code"].tolist() == ["A"]


def test_sell_blocked_by_limit_down_defers_to_next_day():
    """T+H 收盘跌停：顺延到下一个可交易日收盘卖出。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    frames = _frames(dates, ["A"])
    # 应卖日（day2）收盘跌停
    frames["close_raw"].loc[dates[2], "A"] = 9.0
    frames["limit_down"].loc[dates[2], "A"] = 9.0
    picks = _picks([(dates[0], "A")])

    _, stats, trades = _run(picks, frames, hold_days=2)

    assert stats["sell_defer_count"] == 1
    sells = trades[trades["side"].str.startswith("sell")]
    assert len(sells) == 1
    assert sells.iloc[0]["side"] == "sell_deferred"
    assert sells.iloc[0]["date"] == dates[3]


def test_sell_blocked_by_suspension_defers():
    """T+H 停牌：顺延卖出；期间按最近有效价估值。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    frames = _frames(dates, ["A"])
    frames["trade_status"].loc[dates[2], "A"] = "停牌"
    frames["adj_close"].loc[dates[2], "A"] = float("nan")
    frames["close_raw"].loc[dates[2], "A"] = float("nan")
    picks = _picks([(dates[0], "A")])

    result, stats, trades = _run(picks, frames, hold_days=2)

    assert stats["sell_defer_count"] >= 1
    sells = trades[trades["side"].str.startswith("sell")]
    assert sells.iloc[0]["date"] == dates[3]
    # 停牌日估值沿用前一日价格 → NAV 与 day1 相同
    assert result.nav.iloc[2] == pytest.approx(result.nav.iloc[1], abs=1e-12)


# ── 输入校验 ─────────────────────────────────────────────────────


def test_non_trading_signal_date_rejected():
    """信号日不在交易日历中时应明确报错。"""
    dates = pd.bdate_range("2024-01-08", periods=3)
    frames = _frames(dates, ["A"])
    picks = _picks([(pd.Timestamp("2024-01-06"), "A")])  # 周六

    with pytest.raises(ValueError, match="信号日不在行情交易日历"):
        _run(picks, frames, hold_days=2)


def test_hold_days_must_be_positive():
    with pytest.raises(ValueError, match="hold_days"):
        RotateBacktester(hold_days=0)


def test_empty_picks_rejected():
    dates = pd.bdate_range("2024-01-02", periods=3)
    frames = _frames(dates, ["A"])
    with pytest.raises(ValueError, match="选股表为空"):
        _run(_picks([]), frames, hold_days=2)


def test_last_day_signal_reported_unexecutable():
    """信号落在最后一个交易日：无法 T+1 买入，计入统计。"""
    dates = pd.bdate_range("2024-01-02", periods=3)
    frames = _frames(dates, ["A"])
    picks = _picks([(dates[-1], "A")])

    result, stats, trades = _run(picks, frames, hold_days=2)

    assert trades.empty
    assert stats["unexecutable_signal_days"] == [dates[-1].strftime("%Y-%m-%d")]


# ── 结果结构 ─────────────────────────────────────────────────────


def test_actual_weights_and_turnover_recorded():
    """实际权重、双边换手按日记录；换手按信号日聚合。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    frames = _frames(dates, ["A"])
    picks = _picks([(dates[0], "A")])

    result, _, _ = _run(picks, frames, hold_days=2)

    # day1 持仓权重 ≈ 0.4998（半仓）
    w1 = result.actual_weights.loc[dates[1], "A"]
    assert w1 == pytest.approx(0.5 / (1 + COST_BUY) / (0.5 + 0.5 / (1 + COST_BUY)), rel=1e-9)
    # 信号日聚合换手 = 买入 + 卖出 双边
    to = result.turnover_by_rebalance
    assert to.index.tolist() == [dates[0]]
    assert to.iloc[0] > 0
