"""hypothesis 性质测试：随机场景下的核心不变量。

三块组合爆炸、逐例测试追不上的代码，各自有清晰的不变量：

1. ``TopNEqualOptimizer.optimize``——贪心构造 + 残差分摊的任何路径都不得
   违反 long-only / 预算上限 / frozen 精确 / zero 禁持 / sell-only 不增持 /
   换手预算。
2. ``finalize_weights``——削减循环只能向下调整，绝不放大任何硬约束。
3. ``ExecutionLedger.step``——现金/股数非负；零成本 + 价格不变时 NAV 严格
   守恒；冻结持仓在目标生命周期内股数不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hqopt.backtest.execution import ExecutionLedger, OrderState
from hqopt.data.generator import MarketSnapshot, TradingStatus
from hqopt.optimizer._common import build_trading_masks, finalize_weights
from hqopt.optimizer.topn_equal import TopNEqualConfig, TopNEqualOptimizer

_N = 24  # 随机场景的股票数：足够触发换股/补仓/削减分支，又不拖慢用例


def _make_snapshot(
    statuses: list[TradingStatus],
    sell_only_flags: np.ndarray,
    prev: np.ndarray,
) -> MarketSnapshot:
    tickers = [f"S{i:03d}" for i in range(_N)]
    idx = pd.Index(tickers)
    return MarketSnapshot(
        tickers=tickers,
        industry=pd.Series("行业A", index=idx),
        adv=pd.Series(1e8, index=idx),
        status=pd.Series(statuses, index=idx),
        prev_weight=pd.Series(prev, index=idx),
        market_cap=pd.Series(1e10, index=idx),
        portfolio_value=1e8,
        sell_only=pd.Series(sell_only_flags, index=idx),
    )


@st.composite
def _scenario(draw):
    """随机市场截面：alpha（含 NaN）、交易状态、sell-only、上期持仓。"""
    alpha = draw(
        hnp.arrays(
            np.float64,
            _N,
            elements=st.floats(-3.0, 3.0, allow_nan=False),
        )
    )
    nan_mask = draw(hnp.arrays(np.bool_, _N))
    alpha = np.where(nan_mask, np.nan, alpha)

    statuses = draw(
        st.lists(
            st.sampled_from(list(TradingStatus)), min_size=_N, max_size=_N
        )
    )
    sell_only_flags = draw(hnp.arrays(np.bool_, _N))

    has_prev = draw(st.booleans())
    prev = np.zeros(_N)
    if has_prev:
        raw = draw(
            hnp.arrays(np.float64, _N, elements=st.floats(0.0, 1.0))
        )
        raw = np.where(raw > 0.5, raw, 0.0)  # 稀疏化：模拟持有部分股票
        total = float(raw.sum())
        if total > 0:
            target_sum = draw(st.floats(0.2, 0.99))
            prev = raw / total * target_sum
    return alpha, statuses, sell_only_flags, (prev if has_prev else None)


@st.composite
def _config(draw) -> TopNEqualConfig:
    return TopNEqualConfig(
        top_n=draw(st.integers(1, 12)),
        max_turnover=draw(
            st.one_of(st.none(), st.floats(0.05, 0.8))
        ),
        no_trade_band=draw(st.floats(0.0, 0.02)),
    )


# ──────────────────────────────────────────────────────────────────
# 1. TopNEqual：任意场景不得违约
# ──────────────────────────────────────────────────────────────────

@settings(max_examples=200, deadline=None)
@given(scenario=_scenario(), config=_config())
def test_topn_equal_never_violates_constraints(scenario, config):
    alpha, statuses, sell_only_flags, prev = scenario
    snap = _make_snapshot(statuses, sell_only_flags, prev if prev is not None else np.zeros(_N))
    optimizer = TopNEqualOptimizer(config)

    result = optimizer.optimize(alpha=alpha, snapshot=snap, prev_weight=prev)

    # 规则式构造理应永不 infeasible——出现即为贪心/分摊实现的回归
    assert result.status == "optimal", (
        f"infeasible: {result.failure_reason} violations={result.constraint_violations}"
    )

    # 用独立重算的掩码验证（不复用 optimizer 自己的 _violations）
    masks = build_trading_masks(snap, prev)
    w = result.weights
    assert (w >= 0.0).all()
    assert float(w.sum()) <= 1.0 + 1e-9
    assert np.array_equal(w[masks.frozen], masks.prev_weight[masks.frozen])
    assert (w[masks.zero] == 0.0).all()
    assert (
        w[masks.sell_only] <= masks.prev_weight[masks.sell_only] + 1e-9
    ).all()
    if prev is not None and config.max_turnover is not None:
        cash_gap = max(0.0, 1.0 - float(masks.prev_weight.sum()))
        gross = float(np.abs(w - masks.prev_weight).sum())
        assert gross <= config.max_turnover + cash_gap + 1e-9


# ──────────────────────────────────────────────────────────────────
# 2. finalize_weights：削减只向下，硬约束绝不放大
# ──────────────────────────────────────────────────────────────────

@settings(max_examples=200, deadline=None)
@given(
    scenario=_scenario(),
    raw=hnp.arrays(
        np.float64,
        _N,
        elements=st.floats(-0.5, 0.5, allow_nan=False),
    ),
)
def test_finalize_weights_never_amplifies(scenario, raw):
    _, statuses, sell_only_flags, prev = scenario
    snap = _make_snapshot(
        statuses, sell_only_flags, prev if prev is not None else np.zeros(_N)
    )
    masks = build_trading_masks(snap, prev)

    out = finalize_weights(raw, masks)

    assert (out >= 0.0).all()
    assert (out[masks.zero] == 0.0).all()
    # 冻结值精确保留（逐比特，不是近似）
    assert np.array_equal(out[masks.frozen], masks.prev_weight[masks.frozen])
    # 非冻结票只可能被削减，绝不高于 clip 后的求解器原始值
    non_frozen = ~masks.frozen
    assert (
        out[non_frozen] <= np.clip(raw, 0.0, None)[non_frozen] + 1e-12
    ).all()
    # 预算：非冻结部分被削到不超预算；冻结权重本身来自已归一化的上期持仓
    frozen_sum = float(masks.prev_weight[masks.frozen].sum())
    assert float(out.sum()) <= max(1.0, frozen_sum) + 1e-9


# ──────────────────────────────────────────────────────────────────
# 3. ExecutionLedger.step：守恒与非负
# ──────────────────────────────────────────────────────────────────

@st.composite
def _ledger_case(draw):
    n = 8
    tickers = [f"T{i}" for i in range(n)]
    prices = draw(
        hnp.arrays(np.float64, n, elements=st.floats(0.5, 100.0))
    )
    targets = []
    for _ in range(2):
        raw = draw(hnp.arrays(np.float64, n, elements=st.floats(0.0, 1.0)))
        total = float(raw.sum())
        scale = draw(st.floats(0.2, 0.99))
        targets.append(raw / total * scale if total > 0 else raw)
    suspended_days = draw(
        st.lists(hnp.arrays(np.bool_, n), min_size=6, max_size=6)
    )
    return tickers, prices, targets, suspended_days


def _step(ledger, tickers, prices, suspended):
    px = pd.Series(prices, index=tickers)
    status = pd.Series(
        ["停牌" if flag else "交易" for flag in suspended], index=tickers
    )
    return ledger.step(
        adj_close=px,
        adj_vwap=px,
        close_raw=px,
        limit_up=px * 1.10,
        limit_down=px * 0.90,
        trade_status=status,
    )


@settings(max_examples=150, deadline=None)
@given(case=_ledger_case())
def test_ledger_zero_cost_nav_conservation(case):
    """零成本 + 价格不变：任意停牌/换目标序列下 NAV 严格守恒。"""
    tickers, prices, targets, suspended_days = case
    ledger = ExecutionLedger(initial_value=1e6, cost_buy=0.0, cost_sell=0.0)

    for day, suspended in enumerate(suspended_days):
        if day % 3 == 0:
            target = targets[(day // 3) % len(targets)]
            ledger.submit_target(pd.Series(target, index=tickers))
        _step(ledger, tickers, prices, suspended)

        assert ledger.cash >= -1e-9
        assert all(shares > 0 for shares in ledger.shares.values())
        assert ledger.nav == pytest.approx(1e6, rel=1e-9)
        # 订单簿一致性：pending 必须有 PENDING_* 状态
        for ticker in ledger.pending_tickers:
            assert ledger.order_states[ticker] in (
                OrderState.PENDING_BUY,
                OrderState.PENDING_SELL,
            )


@settings(max_examples=150, deadline=None)
@given(case=_ledger_case())
def test_ledger_with_costs_nav_never_increases(case):
    """价格不变时交易成本只会消耗 NAV，绝不凭空创造价值。"""
    tickers, prices, targets, suspended_days = case
    ledger = ExecutionLedger(initial_value=1e6, cost_buy=0.001, cost_sell=0.002)

    nav_prev = ledger.nav
    for day, suspended in enumerate(suspended_days):
        if day % 3 == 0:
            ledger.submit_target(
                pd.Series(targets[(day // 3) % len(targets)], index=tickers)
            )
        _step(ledger, tickers, prices, suspended)
        assert ledger.cash >= -1e-9
        assert ledger.nav <= nav_prev + 1e-6
        nav_prev = ledger.nav


@settings(max_examples=100, deadline=None)
@given(case=_ledger_case())
def test_ledger_frozen_shares_immutable(case):
    """冻结票在目标生命周期内股数逐比特不变。"""
    tickers, prices, targets, suspended_days = case
    ledger = ExecutionLedger(initial_value=1e6, cost_buy=0.001, cost_sell=0.002)

    # 先建仓（无冻结）
    ledger.submit_target(pd.Series(targets[0], index=tickers))
    for suspended in suspended_days[:3]:
        _step(ledger, tickers, prices, suspended)
    if not ledger.shares:
        return  # 全部被停牌阻断，无持仓可冻结

    frozen = [next(iter(ledger.shares))]
    frozen_shares = {t: ledger.shares[t] for t in frozen}
    ledger.submit_target(
        pd.Series(targets[1], index=tickers), frozen_tickers=frozen
    )
    for suspended in suspended_days[3:]:
        _step(ledger, tickers, prices, suspended)
        for ticker, shares in frozen_shares.items():
            assert ledger.shares.get(ticker, 0.0) == shares
        if ledger.pending_target is None:
            break
