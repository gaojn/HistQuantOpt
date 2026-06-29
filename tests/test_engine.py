"""回测引擎执行规则单测（T+1 / 涨跌停 / 停牌 / 退市估值）。

用最小宽表（少量股票、少量交易日）构造可手算的场景，
覆盖 RealisticBacktester 最关键、也最缺测试的执行路径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.backtest.engine import RealisticBacktester, calc_metrics

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


def _run(weight_df, frames):
    bt = RealisticBacktester()
    return bt.run(
        weight_df=weight_df,
        adj_close=frames["adj_close"],
        adj_vwap=frames["adj_vwap"],
        close_raw=frames["close_raw"],
        limit_up_df=frames["limit_up"],
        limit_down_df=frames["limit_down"],
        trade_status_df=frames["trade_status"],
        benchmark_ret=None,
        initial_value=INIT,
    )


# ── T+1 成交时序 ─────────────────────────────────────────────────


def test_signal_executes_next_day_not_same_day():
    """T 日出信号当天不成交（NAV 仍为初始）；T+1 才建仓。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A", "B"]
    frames = _frames(dates, tickers)
    weight_df = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates[:1])

    result, _ = _run(weight_df, frames)

    # 信号日（首日）尚无持仓 → NAV 恰为 1.0
    assert result.nav.iloc[0] == pytest.approx(1.0, abs=1e-9)
    # T+1 建仓后扣买入成本 0.1% → NAV≈0.999，且此后价格不变保持平稳
    assert result.nav.iloc[1] == pytest.approx(0.999, abs=2e-3)
    assert result.nav.iloc[-1] == pytest.approx(result.nav.iloc[1], abs=1e-6)


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


def test_suspension_blocks_buy():
    """执行日停牌的票无法买入，资金滞留现金。"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    tickers = ["A"]
    frames = _frames(dates, tickers)
    frames["trade_status"].loc[dates[1], "A"] = "停牌"

    weight_df = pd.DataFrame({"A": [1.0]}, index=dates[:1])
    result, stats = _run(weight_df, frames)

    assert stats["buy_fail_count"] >= 1
    # 买入失败 → 全程几乎满仓现金，NAV 维持 1.0
    assert result.nav.iloc[-1] == pytest.approx(1.0, abs=1e-6)


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


def test_calc_metrics_constant_returns():
    """恒定正收益：年化为正、Sharpe 为正、回撤≈0。"""
    idx = pd.bdate_range("2024-01-02", periods=252)
    ret = pd.Series(0.001, index=idx)
    bm = pd.Series(0.0, index=idx)

    m = calc_metrics(ret, bm)

    assert m.annual_return == pytest.approx((1.001) ** 252 - 1, rel=1e-6)
    assert m.sharpe > 0
    assert m.max_drawdown <= 1e-9
    assert m.annual_excess_return == pytest.approx(0.001 * 252, rel=1e-6)
