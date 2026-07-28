"""优化器约束不变量测试（合成快照）。"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hqopt.data.generator import MarketDataGenerator, TradingStatus
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.optimizer.index_enhance import (
    IndexEnhanceConfig,
    IndexEnhanceOptimizer,
)

CNE6 = ["Size", "Beta", "Momentum", "Liquidity", "Value"]


@pytest.fixture
def snap():
    return MarketDataGenerator(n_stocks=120, seed=0).generate()


def _style(snap):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.standard_normal((len(snap.tickers), len(CNE6))),
        index=snap.tickers, columns=CNE6,
    )


def test_alpha_max_invariants(snap):
    alpha = np.random.default_rng(1).standard_normal(len(snap.tickers))
    cfg = AlphaMaxConfig(
        weight_upper=0.05,
        industry_upper=0.5,
        style_bound={"default": 1.0, "Size": 0.4},
    )
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, style_loading=_style(snap))
    assert res.is_feasible
    w = res.weights
    assert abs(w.sum() - 1.0) < 1e-5
    assert w.min() >= -1e-8
    assert w.max() <= 0.05 + 1e-6
    # 停牌 / 次新权重必须为 0
    assert float(w[snap.suspended_mask].max(initial=0.0)) < 1e-6
    assert float(w[snap.new_listing_mask].max(initial=0.0)) < 1e-6


def test_st_stocks_banned(snap):
    """ST 票（独立状态）应被禁持仓，权重为 0，且 st_mask 正确反映。"""
    snap.status.iloc[:3] = TradingStatus.ST
    assert snap.st_mask[:3].all()
    assert not snap.tradable_mask[:3].any()

    alpha = np.random.default_rng(5).standard_normal(len(snap.tickers))
    cfg = AlphaMaxConfig(weight_upper=0.05, industry_upper=0.5)
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, style_loading=_style(snap))
    assert res.is_feasible
    assert float(res.weights[snap.st_mask].max(initial=0.0)) < 1e-6


def test_alpha_max_per_factor_style_bound(snap):
    """收紧某因子上限后，该因子的绝对暴露应被约束住。"""
    alpha = np.random.default_rng(11).standard_normal(len(snap.tickers))
    style = _style(snap)
    cfg = AlphaMaxConfig(
        weight_upper=0.1,
        industry_upper=0.6,
        style_bound={"default": 5.0, "Size": 0.05},
    )
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, style_loading=style)
    assert res.is_feasible
    size_exp = float(style["Size"].values @ res.weights)
    assert abs(size_exp) <= 0.05 + 1e-4


def test_index_enhance_invariants(snap):
    alpha = np.random.default_rng(2).standard_normal(len(snap.tickers))
    bm = snap.constituent_mask.astype(float)
    bm = bm / bm.sum()
    cfg = IndexEnhanceConfig(
        weight_upper=0.05, min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound={"default": 1.0, "Size": 0.3},
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm, style_loading=_style(snap)
    )
    assert res.is_feasible
    w = res.weights
    assert abs(w.sum() - 1.0) < 1e-5
    assert w.max() <= 0.05 + 1e-6
    assert float(w[snap.suspended_mask].max(initial=0.0)) < 1e-6


def test_index_enhance_per_factor_style_bound(snap):
    """收紧某因子上限后，该因子的主动暴露应被约束住。"""
    alpha = np.random.default_rng(3).standard_normal(len(snap.tickers))
    bm = snap.constituent_mask.astype(float)
    bm = bm / bm.sum()
    style = _style(snap)
    cfg = IndexEnhanceConfig(
        weight_upper=0.1, min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound={"default": 5.0, "Size": 0.05},  # 仅收紧 Size
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm, style_loading=style
    )
    assert res.is_feasible
    active = res.weights - bm
    size_exp = float(style["Size"].values @ active)
    assert abs(size_exp) <= 0.05 + 1e-4


# ═══════════════════════════════════════════════════════════════
#  批次2 新口径测试：停牌冻结 / 掉池只卖不买
# ═══════════════════════════════════════════════════════════════


def _make_snap_with_suspended_holding(n: int = 50, seed: int = 99) -> tuple:
    """
    构造一个快照：
    - tickers[0] 停牌且有持仓（frozen 预期）
    - tickers[1] 停牌无持仓（zero 预期）
    - tickers[2] 正常有持仓，alpha>0（sell_only 测试用）
    - 其余正常

    返回 (snap, prev_w, bm_w)
    """
    gen = MarketDataGenerator(n_stocks=n, seed=seed)
    snap = gen.generate()

    # 强制 status
    snap.status.iloc[0] = TradingStatus.SUSPENDED   # 停牌有持仓
    snap.status.iloc[1] = TradingStatus.SUSPENDED   # 停牌无持仓
    # tickers[2] 保持正常（用于 sell_only 测试时手动设置 sell_only）

    # 上期权重（仅 tickers[0] 和 tickers[2] 有持仓）
    prev_w = np.zeros(n)
    prev_w[0] = 0.05   # 停牌持仓
    prev_w[2] = 0.08   # 正常持仓（将被 sell_only）

    # 基准权重（均匀分配给全部成分股）
    const = snap.constituent_mask
    bm_w = np.zeros(n)
    bm_w[const] = 1.0 / const.sum()

    return snap, prev_w, bm_w


# ── index_enhance ────────────────────────────────────────────────

def test_ie_suspended_with_holding_frozen():
    """停牌持仓票：结果 w == w_prev（冻结），换手不计。"""
    snap, prev_w, bm_w = _make_snap_with_suspended_holding()
    rng = np.random.default_rng(10)
    alpha = rng.standard_normal(snap.n_stocks)
    alpha[0] = 10.0  # 超高 alpha，确保不是因为"alpha 低"而权重=w_prev

    # max_turnover 非常宽松，避免因换手不够而误判
    cfg = IndexEnhanceConfig(
        weight_upper=0.10,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        max_turnover=1.0,
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w, prev_weight=prev_w
    )
    assert res.is_feasible
    # 求解后处理也必须精确保留冻结值，不能被全组合归一化再次改变。
    assert res.weights[0] == prev_w[0]
    assert res.weights[1] == 0.0
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert res.weights.sum() <= 1.0 + 1e-12


def test_ie_suspended_holding_no_turnover_budget():
    """停牌持仓不消耗换手预算。

    场景：全量已投资均匀组合，stock[0] 停牌有持仓 w_prev[0]=W0。
    若旧口径强制归零需 2*W0 换手；新口径冻结后占用 0，允许较小的 max_turnover。
    """
    snap, _, bm_w = _make_snap_with_suspended_holding(n=50)
    n = snap.n_stocks

    # 全量已投资均匀权重，stock[0] 略高（模拟真实持仓状态）
    prev_w = np.full(n, 1.0 / n)
    prev_w[0] = 0.06   # 停牌有持仓
    prev_w = prev_w / prev_w.sum()   # 归一
    w0 = prev_w[0]

    # alpha=0 → 优化器只想满足约束，不做方向性买入，换手极低
    alpha = np.zeros(n)

    # max_turnover = 0.04 < 2*w0 ≈ 0.12
    # 旧口径：停牌强制归零需 ≈ 0.12 换手 → infeasible
    # 新口径：冻结后换手 ≈ 0 → feasible
    cfg = IndexEnhanceConfig(
        weight_upper=0.10,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        max_turnover=0.04,
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w, prev_weight=prev_w
    )
    assert res.is_feasible, f"冻结票应不消耗换手预算: {res.status}"
    assert abs(res.weights[0] - w0) < 1e-5


def test_ie_frozen_exceeds_weight_upper_feasible():
    """冻结票 w_prev > weight_upper 时：问题 feasible，w == w_prev。"""
    snap, prev_w, bm_w = _make_snap_with_suspended_holding()
    prev_w[0] = 0.12  # 超过 weight_upper=0.10

    alpha = np.random.default_rng(12).standard_normal(snap.n_stocks)
    cfg = IndexEnhanceConfig(
        weight_upper=0.10,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        max_turnover=1.0,
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w, prev_weight=prev_w
    )
    assert res.is_feasible, f"冻结票超上界时应仍 feasible: {res.status}"
    assert abs(res.weights[0] - 0.12) < 1e-5


def test_ie_sell_only_no_new_buy():
    """sell_only 票：w <= w_prev，alpha 为正也不加仓。"""
    snap, prev_w, bm_w = _make_snap_with_suspended_holding()
    alpha = np.random.default_rng(13).standard_normal(snap.n_stocks)
    alpha[2] = 50.0  # 极高 alpha，确保不是因 alpha 低而自然不加仓

    # 手动给快照加 sell_only 标记（模拟掉池持仓票）
    sell_only_s = pd.Series(False, index=snap.tickers, dtype=bool)
    sell_only_s.iloc[2] = True
    snap = replace(snap, sell_only=sell_only_s)

    cfg = IndexEnhanceConfig(
        weight_upper=0.15,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        max_turnover=1.0,
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w, prev_weight=prev_w
    )
    assert res.is_feasible
    # sell_only 票权重不超过 w_prev（+小容差）
    assert res.weights[2] <= prev_w[2] + 1e-5


# ── alpha_max ────────────────────────────────────────────────────

def test_am_suspended_with_holding_frozen():
    """AlphaMax：停牌持仓票冻结 w == w_prev。"""
    snap, prev_w, _ = _make_snap_with_suspended_holding()
    alpha = np.random.default_rng(20).standard_normal(snap.n_stocks)
    alpha[0] = 10.0

    cfg = AlphaMaxConfig(weight_upper=0.10, industry_upper=0.5, max_turnover=1.0)
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, prev_weight=prev_w)
    assert res.is_feasible
    assert res.weights[0] == prev_w[0]
    assert res.weights[1] == 0.0
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert res.weights.sum() <= 1.0 + 1e-12


def test_am_sell_only_no_new_buy():
    """AlphaMax：sell_only 票 w <= w_prev，极高 alpha 也不加仓。"""
    snap, prev_w, _ = _make_snap_with_suspended_holding()
    alpha = np.random.default_rng(21).standard_normal(snap.n_stocks)
    alpha[2] = 50.0

    sell_only_s = pd.Series(False, index=snap.tickers, dtype=bool)
    sell_only_s.iloc[2] = True
    snap = replace(snap, sell_only=sell_only_s)

    cfg = AlphaMaxConfig(weight_upper=0.15, industry_upper=0.5, max_turnover=1.0)
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, prev_weight=prev_w)
    assert res.is_feasible
    assert res.weights[2] <= prev_w[2] + 1e-5


@pytest.mark.parametrize("strategy", ["alpha_max", "index_enhance"])
def test_signal_day_limits_do_not_constrain_next_day_target(strategy):
    """T 日涨跌停不代表 T+1 不可交易，不得限制目标权重方向。"""
    snap = MarketDataGenerator(n_stocks=20, seed=123).generate()
    snap.status[:] = TradingStatus.NORMAL
    snap.status.iloc[0] = TradingStatus.LIMIT_UP
    snap.status.iloc[1] = TradingStatus.LIMIT_DOWN

    prev_w = np.full(snap.n_stocks, 0.8 / (snap.n_stocks - 2))
    prev_w[0] = 0.0
    prev_w[1] = 0.2
    alpha = np.zeros(snap.n_stocks)
    alpha[0] = 100.0
    alpha[1] = -100.0

    if strategy == "alpha_max":
        cfg = AlphaMaxConfig(
            weight_upper=0.5,
            industry_upper=1.0,
            diversification_penalty=0.0,
        )
        res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, prev_weight=prev_w)
    else:
        bm_w = np.full(snap.n_stocks, 1.0 / snap.n_stocks)
        cfg = IndexEnhanceConfig(
            weight_upper=0.5,
            min_constituent_ratio=0.0,
            industry_active_bound=1.0,
            style_active_bound=10.0,
            tracking_penalty=0.0,
        )
        res = IndexEnhanceOptimizer(cfg).optimize(
            alpha,
            snap,
            benchmark_weight=bm_w,
            prev_weight=prev_w,
        )

    assert res.is_feasible
    assert res.weights[0] > 0.1, "T 日涨停票应允许生成 T+1 买入目标"
    assert res.weights[1] < prev_w[1] - 0.1, "T 日跌停票应允许生成 T+1 卖出目标"


# ═══════════════════════════════════════════════════════════════
#  T3: active_weight_upper 约束 + 冲突豁免
# ═══════════════════════════════════════════════════════════════


def test_ie_active_weight_upper_binds():
    """active_weight_upper=0.01 时，非冻结票 |w - w_bm| <= 0.011（含容差）。"""
    snap, _, bm_w = _make_snap_with_suspended_holding(n=60, seed=77)
    alpha = np.random.default_rng(30).standard_normal(snap.n_stocks)
    cfg = IndexEnhanceConfig(
        weight_upper=0.10,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        max_turnover=1.0,
        active_weight_upper=0.01,
    )
    # 无上期持仓：所有非停牌票均非冻结
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w
    )
    assert res.is_feasible
    w = res.weights
    non_susp = ~snap.suspended_mask
    max_active = float(np.abs(w[non_susp] - bm_w[non_susp]).max(initial=0.0))
    assert max_active <= 0.011, f"最大主动偏离 {max_active:.4f} 超出上限"


def test_ie_active_weight_upper_conflict_feasible():
    """冲突场景：成分股 w_bm=0.03 > delta=0.01 且被停牌无持仓（zero）→ 仍 feasible 且 w=0。

    若下界约束未豁免 zero 票，w_bm - w <= delta 要求 w >= 0.02，而 w==0 约束矛盾 → infeasible。
    修复后 zero 票豁免下界，问题应 feasible 且该票 w=0。
    """
    # 构造成分股 w_bm > delta、该票被停牌且无持仓
    n = 40
    gen = MarketDataGenerator(n_stocks=n, seed=50)
    snap = gen.generate()
    snap.status.iloc[0] = TradingStatus.SUSPENDED  # 停牌
    # tickers[0] 无上期持仓 → zero_mask

    # 基准权重：tickers[0] 占 3%（大于 delta=0.01）
    bm_w = np.full(n, 1.0 / n)
    bm_w[0] = 0.03
    bm_w[1:] = (1 - 0.03) / (n - 1)

    alpha = np.zeros(n)
    cfg = IndexEnhanceConfig(
        weight_upper=0.10,
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
        active_weight_upper=0.01,  # delta < w_bm[0]
    )
    res = IndexEnhanceOptimizer(cfg).optimize(
        alpha, snap, benchmark_weight=bm_w
    )
    assert res.is_feasible, (
        f"zero 票 w_bm > delta 时下界应豁免，问题应 feasible，实为 {res.status}"
    )
    # 停牌无持仓票 w==0
    assert res.weights[0] < 1e-6, f"停牌无持仓票权重应为 0，实为 {res.weights[0]:.6f}"
