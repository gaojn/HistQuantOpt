"""候选池瘦身（``universe.alpha_top_m``）契约测试。

瘦身的正确性依赖一个数学事实：纯多头 + 线性 alpha + 凸惩罚下，深度负 alpha
的股票不进最优解支撑集。但约束相关股票必须显式保留，否则会静默改变问题语义：

1. ``candidate_pool_mask``：持仓 / 基准股 / 成分股容量必须在池内；
2. ``subset_snapshot``：切片后所有字段按 ticker 重对齐；
3. **等价性**：瘦身池的解与全池解逐票一致（这是允许该优化上生产的前提）；
4. ``_reduce_candidate_pool`` 接线：未配置时零改动；配置后 alpha 只切片、
   不重新标准化（重新标准化会改变与惩罚项的量纲耦合，破坏等价性）。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hqopt.data.generator import MarketDataGenerator
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.pipeline.batch.periods import _reduce_candidate_pool
from hqopt.pipeline.batch.types import _PeriodContext
from hqopt.pipeline.universe import candidate_pool_mask, subset_snapshot

# N 取 600：风格对冲规则每因子两端各保留 60 只，规模太小时池不会真的收缩，
# 等价性测试会退化成"全池 vs 全池"的空转
_N = 600
_TOP_M = 150


@pytest.fixture
def snap():
    return MarketDataGenerator(
        n_stocks=_N, seed=7, n_constituents=200
    ).generate()


@pytest.fixture
def alpha(snap):
    values = np.random.default_rng(11).standard_normal(snap.n_stocks)
    return (values - values.mean()) / values.std()


@pytest.fixture
def style(snap):
    """带截面偏移的风格载荷——使绝对风格约束成为紧约束，复现对冲票场景。"""
    rng = np.random.default_rng(23)
    data = {
        "Size": rng.standard_normal(_N) - 0.8,   # 均值偏负：组合需买高 Size 票对冲
        "Beta": rng.standard_normal(_N) + 0.5,
        "Liquidity": rng.standard_normal(_N),
    }
    return pd.DataFrame(data, index=snap.tickers)


# ── candidate_pool_mask ───────────────────────────────────────────


def test_mask_keeps_all_constraint_relevant_stocks(snap, alpha):
    """持仓、基准股、alpha top-M、成分股容量四类都必须保留。"""
    prev = np.zeros(_N)
    prev[[3, 50, 199]] = [0.1, 0.2, 0.05]          # 含深度负 alpha 也要保留
    bm = np.zeros(_N)
    bm[[7, 120]] = [0.5, 0.5]

    keep = candidate_pool_mask(alpha, snap, prev, bm, _TOP_M)

    assert keep[[3, 50, 199]].all(), "持仓票必须在池内（卖出/冻结约束挂在它身上）"
    assert keep[[7, 120]].all(), "基准权重非零的票必须在池内"
    top_idx = np.argsort(alpha)[::-1][:_TOP_M]
    assert keep[top_idx].all(), "alpha top-M 必须在池内"
    cmask = snap.constituent_mask
    n_const_kept = int((keep & cmask).sum())
    assert n_const_kept >= int(_TOP_M * 0.6), "成分股容量兜底不足，成分下限约束可能不可行"


def test_mask_without_prev_and_benchmark(snap, alpha):
    """首期（无持仓、无基准）也能工作，池 = top-M ∪ 成分股容量。"""
    keep = candidate_pool_mask(alpha, snap, None, None, _TOP_M)
    assert keep.sum() >= _TOP_M
    assert keep.sum() < _N, "瘦身后必须真的比全池小，否则配置空转"


def test_mask_keeps_style_extreme_hedge_stocks(snap, alpha, style):
    """风格载荷两端极值股票必须保留——紧风格约束下它们是对冲候选。

    实测教训：漏掉对冲票时全池解有 ~4% 权重落在池外，L1 漂移 0.16。
    """
    keep = candidate_pool_mask(alpha, snap, None, None, _TOP_M, style_loading=style)
    for col in style.columns:
        order = np.argsort(style[col].values)
        # 两端各 min(60, N) 只都在池内（测试规模 N=200 < 2×60，故取实际尾部）
        tail = min(60, _N)
        assert keep[order[:tail]].all(), f"{col} 负端极值票被漏掉"
        assert keep[order[-tail:]].all(), f"{col} 正端极值票被漏掉"


def test_mask_rejects_nonpositive_top_m(snap, alpha):
    with pytest.raises(ValueError, match="top_m"):
        candidate_pool_mask(alpha, snap, None, None, 0)


# ── subset_snapshot ───────────────────────────────────────────────


def test_subset_snapshot_realigns_all_fields(snap):
    keep_idx = np.array([0, 5, 17, 42, 199])
    sub = subset_snapshot(snap, keep_idx)

    expected = [snap.tickers[i] for i in keep_idx]
    assert sub.tickers == expected
    assert list(sub.industry.index) == expected
    assert list(sub.market_cap.index) == expected
    assert sub.n_stocks == len(keep_idx)
    for i, full_i in enumerate(keep_idx):
        assert sub.status.iloc[i] == snap.status.iloc[full_i]
        assert sub.constituent_mask[i] == snap.constituent_mask[full_i]


# ── 等价性：瘦身池解 == 全池解 ────────────────────────────────────


def _solve(snap, alpha, prev=None, style_loading=None, style_bound=None):
    cfg = AlphaMaxConfig(
        weight_upper=0.05,
        industry_upper=0.5,
        diversification_penalty=0.05,
        style_bound=style_bound,
        max_turnover=0.5 if prev is not None else None,
    )
    return AlphaMaxOptimizer(cfg).optimize(
        alpha, snap, prev_weight=prev, style_loading=style_loading
    )


def _assert_equivalent(full, reduced, snap):
    """瘦身池解与全池解逐票一致（池外权重为零 + 池内逐票相等）。"""
    full_w = pd.Series(full.weights, index=snap.tickers)
    reduced_w = pd.Series(reduced.weights, index=reduced.tickers)
    dropped = full_w.drop(index=reduced.tickers)
    assert float(dropped.abs().max()) < 1e-6, (
        "全池解在被剔除股票上有非零权重——瘦身规则漏保留了会进支撑集的票"
    )
    diff = (full_w.reindex(reduced.tickers) - reduced_w).abs().max()
    assert float(diff) < 1e-5, f"瘦身池解与全池解偏差 {diff:.2e}，超出求解器噪声"


def test_reduced_solve_matches_full_solve(snap, alpha):
    """瘦身池解与全池解逐票一致——允许该优化上生产的核心前提。"""
    full = _solve(snap, alpha)
    assert full.is_feasible

    keep = candidate_pool_mask(alpha, snap, None, None, _TOP_M)
    assert keep.sum() < _N, "池没有真的收缩，等价性测试空转"
    keep_idx = np.where(keep)[0]
    reduced = _solve(subset_snapshot(snap, keep_idx), alpha[keep_idx])
    assert reduced.is_feasible
    _assert_equivalent(full, reduced, snap)


def test_reduced_solve_matches_full_solve_with_turnover(snap, alpha):
    """带上期持仓与换手约束时同样等价（现金缺口/换手预算语义不变）。"""
    prev = _solve(snap, alpha).weights
    full = _solve(snap, alpha, prev=prev)
    assert full.is_feasible

    keep = candidate_pool_mask(alpha, snap, prev, None, _TOP_M)
    keep_idx = np.where(keep)[0]
    reduced = _solve(
        subset_snapshot(snap, keep_idx), alpha[keep_idx], prev=prev[keep_idx]
    )
    assert reduced.is_feasible
    _assert_equivalent(full, reduced, snap)


def test_reduced_solve_matches_full_solve_with_binding_style(snap, alpha, style):
    """紧的绝对风格约束下同样等价——最能暴露「漏掉对冲票」的场景。

    风格载荷截面均值偏离 0（如 Size 均值 −0.8）时约束必然 binding，
    最优解含 alpha 平庸的对冲票；无风格对冲规则的池会解出不同组合。
    """
    full = _solve(snap, alpha, style_loading=style, style_bound=0.15)
    assert full.is_feasible

    keep = candidate_pool_mask(
        alpha, snap, None, None, _TOP_M, style_loading=style
    )
    assert keep.sum() < _N, "池没有真的收缩，等价性测试空转"
    keep_idx = np.where(keep)[0]
    reduced = _solve(
        subset_snapshot(snap, keep_idx), alpha[keep_idx],
        style_loading=style, style_bound=0.15,
    )
    assert reduced.is_feasible
    _assert_equivalent(full, reduced, snap)


# ── _reduce_candidate_pool 接线 ───────────────────────────────────


class _FakeRiskModel:
    """按 tickers 返回带 style_loading 的假风险切片，记录调用。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def at(self, target_date, tickers):
        self.calls.append(list(tickers))
        return SimpleNamespace(
            style_loading=lambda: pd.DataFrame(
                0.0, index=list(tickers), columns=["Size"]
            )
        )


def _fake_inputs(alpha_top_m, risk_model=None):
    return SimpleNamespace(
        universe_cfg={"alpha_top_m": alpha_top_m},
        benchmark=None,
        risk_model=risk_model or _FakeRiskModel(),
    )


def _ctx_for(snap, prev=None):
    return _PeriodContext(
        snapshot=snap,
        risk_snapshot=SimpleNamespace(),
        style_loading=pd.DataFrame(0.0, index=snap.tickers, columns=["Size"]),
        prev_weight=prev,
    )


def test_reduce_disabled_returns_inputs_unchanged(snap, alpha):
    """alpha_top_m 未配置（None/缺省）时必须零改动直通。"""
    ctx = _ctx_for(snap)
    cost = np.ones(_N)
    out_ctx, out_alpha, out_cost = _reduce_candidate_pool(
        _fake_inputs(None), ctx, alpha, cost, date(2025, 1, 2)
    )
    assert out_ctx is ctx
    assert out_alpha is alpha
    assert out_cost is cost


def test_reduce_slices_alpha_and_cost_without_recompute(snap, alpha):
    """瘦身后 alpha 与成本向量都是全池版本的切片，绝不重算。

    alpha 重新标准化、或成本向量按瘦身池重新做中位数归一化，都会悄悄
    改变目标函数——这正是等价性被破坏的两条最隐蔽路径。
    """
    prev = np.zeros(_N)
    prev[190] = 0.3                      # 深度排名外的持仓票
    ctx = _ctx_for(snap, prev=prev)
    risk_model = _FakeRiskModel()
    cost = np.random.default_rng(3).uniform(0.1, 10.0, _N)

    out_ctx, out_alpha, out_cost = _reduce_candidate_pool(
        _fake_inputs(_TOP_M, risk_model), ctx, alpha, cost, date(2025, 1, 2)
    )

    n_reduced = len(out_ctx.snapshot.tickers)
    assert n_reduced < _N
    assert out_ctx.prev_weight is not None
    assert float(out_ctx.prev_weight.sum()) == pytest.approx(0.3), (
        "持仓票必须完整保留，prev_weight 总和不得因瘦身丢失"
    )
    # alpha / cost 逐票等于全池版本的切片
    ticker_pos = {t: i for i, t in enumerate(snap.tickers)}
    keep_pos = [ticker_pos[t] for t in out_ctx.snapshot.tickers]
    np.testing.assert_array_equal(out_alpha, alpha[keep_pos])
    assert out_cost is not None
    np.testing.assert_array_equal(out_cost, cost[keep_pos])
    # 风险模型按瘦身后的 tickers 重新取切片
    assert risk_model.calls[-1] == out_ctx.snapshot.tickers


def test_reduce_skipped_when_pool_already_small(snap, alpha):
    """池子本来就不大于 top_m 时不动它（避免无谓的重建开销）。"""
    ctx = _ctx_for(snap)
    out_ctx, out_alpha, out_cost = _reduce_candidate_pool(
        _fake_inputs(_N + 100), ctx, alpha, None, date(2025, 1, 2)
    )
    assert out_ctx is ctx
    assert out_alpha is alpha
    assert out_cost is None
