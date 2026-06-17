"""优化器约束不变量测试（合成快照）。"""
import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.data.generator import MarketDataGenerator
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from portfolio_optimizer.optimizer.index_enhance import (
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
    cfg = AlphaMaxConfig(weight_upper=0.05, industry_upper=0.5, style_bound=1.0)
    res = AlphaMaxOptimizer(cfg).optimize(alpha, snap, style_loading=_style(snap))
    assert res.is_feasible
    w = res.weights
    assert abs(w.sum() - 1.0) < 1e-5
    assert w.min() >= -1e-8
    assert w.max() <= 0.05 + 1e-6
    # 停牌 / 次新权重必须为 0
    assert float(w[snap.suspended_mask].max(initial=0.0)) < 1e-6
    assert float(w[snap.new_listing_mask].max(initial=0.0)) < 1e-6


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
