"""Top-N 等权持有策略（topn_equal）的规则正确性测试。

覆盖：首期建仓、免交易带、换手预算、换股优先级、三类交易状态掩码、
现金重投口径，以及 pipeline 接线（_build_optimizer / 默认配置）。
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hqopt.data.generator import MarketDataGenerator, TradingStatus
from hqopt.optimizer.topn_equal import (
    TopNEqualConfig,
    TopNEqualOptimizer,
    TopNEqualResult,
)

_N = 60          # 合成快照股票数
_TOP = 10        # 目标持仓只数


@pytest.fixture
def snap():
    """无停牌/次新/ST 的干净快照：交易状态由各测试自行注入。"""
    return MarketDataGenerator(
        n_stocks=_N, seed=3,
        suspended_ratio=0.0, new_listing_ratio=0.0,
    ).generate()


@pytest.fixture
def alpha():
    """严格递减的 alpha：股票 0 最好，股票 N-1 最差，排名无歧义。"""
    return np.linspace(1.0, -1.0, _N)


def _opt(**kwargs) -> TopNEqualOptimizer:
    cfg = TopNEqualConfig(**{"top_n": _TOP, **kwargs})
    return TopNEqualOptimizer(cfg)


def _equal_weight_prev(snap, idx: list[int]) -> np.ndarray:
    prev = np.zeros(len(snap.tickers))
    prev[idx] = 1.0 / len(idx)
    return prev


def _gross(res: TopNEqualResult, prev: np.ndarray) -> float:
    return float(np.abs(res.weights - prev).sum())


# ──────────────────────────────────────────────────────────────────
# 配置校验
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0}, "top_n 必须"),
        ({"top_n": 10, "max_turnover": 0.0}, "max_turnover 必须"),
        ({"top_n": 10, "max_turnover": 2.5}, "max_turnover 必须"),
        ({"top_n": 10, "no_trade_band": -0.01}, "no_trade_band 必须"),
    ],
)
def test_config_rejects_invalid_params(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TopNEqualConfig(**kwargs)


# ──────────────────────────────────────────────────────────────────
# 首期建仓
# ──────────────────────────────────────────────────────────────────

def test_initial_build_equal_weights_top_alpha(snap, alpha):
    res = _opt().optimize(alpha, snap)
    assert res.is_feasible
    assert res.n_positions == _TOP
    held = np.where(res.weights > 0)[0]
    # 持有的正是 alpha 前 N 只，逐票等权、满仓
    assert set(held.tolist()) == set(range(_TOP))
    np.testing.assert_allclose(res.weights[held], 1.0 / _TOP)
    assert abs(res.weights.sum() - 1.0) < 1e-12


def test_initial_build_skips_untradable(snap, alpha):
    """首期禁开仓的票（停牌/次新/ST）即使 alpha 最高也不入选。"""
    snap.status.iloc[0] = TradingStatus.SUSPENDED
    snap.status.iloc[1] = TradingStatus.ST
    res = _opt().optimize(alpha, snap)
    assert res.weights[0] == 0.0
    assert res.weights[1] == 0.0
    held = set(np.where(res.weights > 0)[0].tolist())
    assert held == set(range(2, _TOP + 2))


# ──────────────────────────────────────────────────────────────────
# 免交易带：尽量少交易
# ──────────────────────────────────────────────────────────────────

def test_steady_state_no_trades(snap, alpha):
    """持仓已是理想等权组合 → 一笔不交易。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    res = _opt().optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    assert res.n_trades == 0
    np.testing.assert_array_equal(res.weights, prev)


def test_drift_within_band_not_traded(snap, alpha):
    """漂移在带内（|w−w_eq|<0.005）→ 保持原权重，不交易。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    drift = np.array([0.004, -0.004] * (_TOP // 2))
    prev[: _TOP] += drift          # 总和不变，逐票带内漂移
    res = _opt().optimize(alpha, snap, prev_weight=prev)
    assert res.n_trades == 0
    np.testing.assert_array_equal(res.weights, prev)


def test_drift_beyond_band_rebalanced_only_that_stock(snap, alpha):
    """一只票漂出带外 → 只有它被调回等权，其余带内票不动。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    prev[0] += 0.03                # 0.13，带外
    prev[1] -= 0.004               # 带内
    prev[2] -= 0.004               # 带内
    prev[3] -= 0.004               # 带内
    # 归一：留 0.018 现金缺口（模拟成交留残）
    res = _opt().optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    w_eq = 1.0 / _TOP
    assert abs(res.weights[0] - w_eq) < 1e-12      # 超重票调回等权
    np.testing.assert_array_equal(res.weights[1:4], prev[1:4])  # 带内票原样
    # 卖出释放的资金无买方吸纳时留作现金，不得回补给刚卖出的票
    assert res.weights.sum() <= 1.0 + 1e-12


def test_no_trade_band_zero_restores_exact_equal_weight(snap, alpha):
    """band=0 时每期精确调回等权。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    prev[: _TOP] += np.array([0.002, -0.002] * (_TOP // 2))
    res = _opt(no_trade_band=0.0, max_turnover=None).optimize(
        alpha, snap, prev_weight=prev
    )
    np.testing.assert_allclose(res.weights[: _TOP], 1.0 / _TOP, atol=1e-12)


# ──────────────────────────────────────────────────────────────────
# 换手预算与换股优先级
# ──────────────────────────────────────────────────────────────────

def test_turnover_capped_on_full_alpha_reversal(snap):
    """alpha 完全反转（理想组合 0 重叠）时，换手不得超预算。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    reversed_alpha = np.linspace(-1.0, 1.0, _N)   # 现在股票 N-1 最好
    res = _opt(max_turnover=0.40).optimize(
        reversed_alpha, snap, prev_weight=prev
    )
    assert res.is_feasible
    assert _gross(res, prev) <= 0.40 + 1e-9
    # 双边 40%、每组换股 ≈ 2/N=0.2 → 约 2 组；必须发生了换股
    new_entries = np.where((res.weights > 0) & (prev == 0))[0]
    assert len(new_entries) >= 1
    # 新入选的是最优候选（数组尾部），退出的是最差持仓（此 alpha 下为头部）
    assert set(new_entries.tolist()) <= set(range(_N - _TOP, _N))
    exited = np.where((res.weights == 0) & (prev > 0))[0]
    assert set(exited.tolist()) <= set(range(_TOP))
    assert len(exited) == len(new_entries)
    # 卖最差：退出票的 alpha 排名必须差于所有留下的持仓
    kept = np.where((res.weights > 0) & (prev > 0))[0]
    assert reversed_alpha[exited].max() < reversed_alpha[kept].min()


def test_unlimited_turnover_reaches_ideal_portfolio(snap):
    """不限换手时一步到位：完整换成新的 top-N 等权。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    reversed_alpha = np.linspace(-1.0, 1.0, _N)
    res = _opt(max_turnover=None).optimize(reversed_alpha, snap, prev_weight=prev)
    held = np.where(res.weights > 0)[0]
    assert set(held.tolist()) == set(range(_N - _TOP, _N))
    np.testing.assert_allclose(res.weights[held], 1.0 / _TOP, atol=1e-12)


def test_cash_gap_not_charged_to_turnover_budget(snap, alpha):
    """现金重投不挤占换手预算：gross ≤ max_turnover + cash_gap。"""
    prev = _equal_weight_prev(snap, list(range(_TOP))) * 0.9   # 10% 现金
    res = _opt(max_turnover=0.10).optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    assert _gross(res, prev) <= 0.10 + 0.10 + 1e-9
    # 现金应被重新投出去（买方向交易存在时满仓）
    assert res.weights.sum() > 0.99


def test_position_deficit_replenished_to_top_n(snap, alpha):
    """执行损耗后持仓少于 N → 净买入补足，不需要卖出配对。"""
    prev = np.zeros(_N)
    prev[:6] = 1.0 / _TOP          # 只剩 6 只在手，40% 现金
    res = _opt(max_turnover=None).optimize(alpha, snap, prev_weight=prev)
    assert res.n_positions == _TOP
    held = np.where(res.weights > 0)[0]
    assert set(held.tolist()) == set(range(_TOP))
    assert abs(res.weights.sum() - 1.0) < 1e-9


def test_position_deficit_replenish_respects_budget(snap, alpha):
    """补足数量同样受换手预算限制（含 cash_gap 口径）。"""
    prev = np.zeros(_N)
    prev[:6] = 1.0 / _TOP          # cash_gap = 0.4
    res = _opt(max_turnover=0.05).optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    assert _gross(res, prev) <= 0.05 + 0.40 + 1e-9
    # 预算 0.45、每只补仓约 0.1 → 至少补 4 只
    assert res.n_positions > 6


def test_fully_invested_deficit_funded_by_selling_overweights(snap, alpha):
    """满仓但持仓少于 N（如调大 N）→ 卖出超重票融资补仓，最终 N 只等权。"""
    prev = np.zeros(_N)
    prev[:5] = 0.2                 # 满仓 5 只，需要补到 10 只
    res = _opt(max_turnover=None, no_trade_band=0.0).optimize(
        alpha, snap, prev_weight=prev
    )
    assert res.is_feasible
    assert res.n_positions == _TOP
    np.testing.assert_allclose(res.weights[: _TOP], 1.0 / _TOP, atol=1e-9)
    assert abs(res.weights.sum() - 1.0) < 1e-9


def test_fully_invested_deficit_no_phantom_fills_under_tight_budget(snap, alpha):
    """满仓补仓预算不足时不得出现幽灵补仓：要么真买进，要么保持原持仓数。

    回归：曾经补仓贪心只看换手（新买票被残差打回零 → 零换手全通过），
    把 |net_gap| 个虚位全部推进，结果既没加到仓、又虚占了换股名额。
    """
    prev = np.zeros(_N)
    prev[:5] = 0.2                 # 满仓 5 只，每笔补仓需买 w_eq + 卖出融资
    res = _opt(max_turnover=0.05).optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    assert _gross(res, prev) <= 0.05 + 1e-9
    # 不存在"权重为 0 的新持仓"：所有正持仓都是真实买入
    tiny = (res.weights > 0) & (res.weights < 1e-8)
    assert not tiny.any()


def test_position_surplus_trimmed_toward_top_n(snap, alpha):
    """持仓多于 N（如调小 N）→ 净卖出最差持仓，逼近 N 只。"""
    prev = np.zeros(_N)
    prev[:20] = 1.0 / 20
    res = _opt(max_turnover=None).optimize(alpha, snap, prev_weight=prev)
    assert res.n_positions == _TOP
    held = np.where(res.weights > 0)[0]
    assert set(held.tolist()) == set(range(_TOP))   # 留下 alpha 最好的 N 只


# ──────────────────────────────────────────────────────────────────
# 交易状态掩码
# ──────────────────────────────────────────────────────────────────

def test_frozen_holding_keeps_exact_weight(snap, alpha):
    """停牌持仓票权重冻结，且占用 N 名额。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    prev[0] = 0.13
    snap.status.iloc[0] = TradingStatus.SUSPENDED
    res = _opt().optimize(alpha, snap, prev_weight=prev)
    assert res.is_feasible
    assert res.weights[0] == 0.13


def test_suspended_unheld_never_bought(snap):
    """停牌且无持仓的票即使 alpha 冲进 top-N 也不得开仓。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    alpha2 = np.linspace(1.0, -1.0, _N)
    alpha2[20] = 99.0                    # 未持有的高 alpha 票
    snap.status.iloc[20] = TradingStatus.SUSPENDED
    res = _opt(max_turnover=None).optimize(alpha2, snap, prev_weight=prev)
    assert res.weights[20] == 0.0


def test_sell_only_holding_not_increased(snap, alpha):
    """只卖不买的持仓票：可保留、可卖出，但绝不加仓。"""
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    prev[0] = 0.05                       # 低于等权 0.1
    sell_only = pd.Series(False, index=snap.tickers)
    sell_only.iloc[0] = True
    snap2 = replace(snap, sell_only=sell_only)
    res = _opt(max_turnover=None, no_trade_band=0.0).optimize(
        alpha, snap2, prev_weight=prev
    )
    assert res.is_feasible
    assert res.weights[0] <= 0.05 + 1e-12


# ──────────────────────────────────────────────────────────────────
# 结果契约与 pipeline 接线
# ──────────────────────────────────────────────────────────────────

def test_result_reports_trade_count_and_violations(snap, alpha):
    prev = _equal_weight_prev(snap, list(range(_TOP)))
    reversed_alpha = np.linspace(-1.0, 1.0, _N)
    res = _opt().optimize(reversed_alpha, snap, prev_weight=prev)
    assert res.status == "optimal"
    assert res.n_trades == int((np.abs(res.weights - prev) > 1e-12).sum())
    assert "turnover" in res.constraint_violations
    assert max(res.constraint_violations.values()) < 1e-9
    assert "交易只数" in res.summary()


def test_nan_alpha_treated_as_worst(snap, alpha):
    alpha2 = alpha.copy()
    alpha2[0] = np.nan
    res = _opt().optimize(alpha2, snap)
    assert res.weights[0] == 0.0
    held = set(np.where(res.weights > 0)[0].tolist())
    assert held == set(range(1, _TOP + 1))


def test_build_optimizer_wires_topn_equal():
    from hqopt.pipeline.batch_optimize import _build_optimizer

    optimizer, cfg = _build_optimizer(
        "topn_equal",
        {"top_n": 50, "max_turnover": 0.3, "no_trade_band": 0.01},
        risk_aversion=None,
    )
    assert isinstance(optimizer, TopNEqualOptimizer)
    assert cfg.top_n == 50
    assert cfg.max_turnover == 0.3
    assert cfg.no_trade_band == 0.01


def test_default_topn_config_loads():
    from hqopt.pipeline.batch_optimize import load_config

    cfg = load_config("configs/topn_equal_default.yaml")
    assert cfg["strategy"] == "topn_equal"
    assert cfg["optimizer"]["top_n"] == 100
    assert cfg["optimizer"]["max_turnover"] == 0.40
    assert cfg["optimizer"]["no_trade_band"] == 0.005
    assert cfg["backtest"]["benchmark"] == "equal_weight"
