"""风格约束上限解析测试（float / dict / 缺 default）。"""
import numpy as np

from portfolio_optimizer.optimizer.index_enhance import _resolve_style_bounds, _UNBOUNDED
from portfolio_optimizer.pipeline.batch_optimize import _parse_style_bound

FACTORS = ["Size", "Momentum", "Beta"]


def test_scalar_applies_to_all():
    vec = _resolve_style_bounds(0.3, FACTORS)
    assert np.allclose(vec, [0.3, 0.3, 0.3])


def test_dict_with_default():
    vec = _resolve_style_bounds({"default": 0.5, "Momentum": 0.2}, FACTORS)
    assert np.allclose(vec, [0.5, 0.2, 0.5])


def test_dict_without_default_is_unbounded():
    vec = _resolve_style_bounds({"Size": 0.3}, FACTORS)
    assert vec[0] == 0.3
    assert vec[1] >= _UNBOUNDED and vec[2] >= _UNBOUNDED   # 未列出且无 default → 不约束


def test_order_follows_factor_names():
    vec = _resolve_style_bounds({"default": 1.0, "Beta": 0.1}, FACTORS)
    assert vec[2] == 0.1   # Beta 是第三个


def test_parse_style_bound_passthrough():
    assert _parse_style_bound(0.4) == 0.4
    d = _parse_style_bound({"default": 0.5, "Size": 0.3})
    assert d["Size"] == 0.3 and d["default"] == 0.5
