"""绩效指标的跨模块口径一致性。

`engine.calc_metrics`（全期/摘要卡片）与 `report._annual_metrics`（年度分解表）
是两套独立实现。历史上二者的 IR 分子一度一个用几何年化超额、一个用算术
`日均超额×252`，导致同一张 HTML 报告里年度行与全期行的 IR 不可比（实测差
3%~59%），而两边各自的单测都是绿的。

本文件专门锁住「两套实现在同一输入下必须给出相同口径」这个不变量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hqopt.backtest.report as report_module
from hqopt.backtest.engine import calc_metrics

RISK_FREE = 0.02


def _series(seed: int, n: int = 252, excess: float = 0.0008):
    """构造一对（组合、基准）日收益序列。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    bm = pd.Series(rng.normal(0.0002, 0.011, n), index=idx)
    ret = bm + excess + pd.Series(rng.normal(0, 0.004, n), index=idx)
    return ret, bm


# 覆盖正负超额、长短样本；短样本正是年化外推最容易出问题的地方
CASES = [
    pytest.param(0, 252, 0.0008, id="满一年-正超额"),
    pytest.param(1, 252, -0.0006, id="满一年-负超额"),
    pytest.param(2, 252, 0.0016, id="满一年-高超额"),
    pytest.param(3, 60, 0.0008, id="仅3个月"),
    pytest.param(4, 20, 0.0010, id="仅1个月"),
]


@pytest.mark.parametrize(("seed", "n", "excess"), [
    (c.values[0], c.values[1], c.values[2]) for c in CASES
], ids=[c.id for c in CASES])
def test_info_ratio_same_across_modules(seed, n, excess):
    """IR 必须完全一致：分子同为几何年化超额，分母同为年化 TE。"""
    ret, bm = _series(seed, n, excess)

    engine_ir = calc_metrics(ret, bm, RISK_FREE).info_ratio
    report_ir = report_module._annual_metrics(ret, bm, risk_free=RISK_FREE)["info_ratio"]

    assert report_ir == pytest.approx(engine_ir, rel=1e-12)


@pytest.mark.parametrize(("seed", "n", "excess"), [
    (c.values[0], c.values[1], c.values[2]) for c in CASES
], ids=[c.id for c in CASES])
def test_shared_metrics_same_across_modules(seed, n, excess):
    """两套实现的共同指标必须完全一致。"""
    ret, bm = _series(seed, n, excess)

    e = calc_metrics(ret, bm, RISK_FREE)
    r = report_module._annual_metrics(ret, bm, risk_free=RISK_FREE)

    assert r["annual_return"] == pytest.approx(e.annual_return, rel=1e-12)
    assert r["annual_vol"] == pytest.approx(e.annual_vol, rel=1e-12)
    assert r["sharpe"] == pytest.approx(e.sharpe, rel=1e-12)
    assert r["max_dd"] == pytest.approx(e.max_drawdown, rel=1e-12)
    assert r["calmar"] == pytest.approx(e.calmar, rel=1e-12)
    assert r["excess_vol"] == pytest.approx(e.tracking_error, rel=1e-12)
    assert r["excess_max_dd"] == pytest.approx(e.excess_max_drawdown, rel=1e-12)
    assert r["annual_excess_return"] == pytest.approx(e.annual_excess_return, rel=1e-12)


def test_annual_ir_numerator_is_geometric_not_arithmetic():
    """回归锁：IR 分子若退回算术口径（日均超额×252），本测试必须失败。"""
    ret, bm = _series(seed=7, n=252, excess=0.0016)
    r = report_module._annual_metrics(ret, bm, risk_free=RISK_FREE)

    exc = ret - bm
    arithmetic_ir = (exc.mean() * 252) / (exc.std() * np.sqrt(252) + 1e-12)

    assert r["info_ratio"] != pytest.approx(arithmetic_ir, rel=1e-6)
    assert r["info_ratio"] == pytest.approx(
        calc_metrics(ret, bm, RISK_FREE).info_ratio, rel=1e-12
    )


# ── Calmar：年度与全期统一使用年化收益分子（行业标准 CAGR/MDD） ─────


def test_annual_calmar_uses_annualized_return():
    """年度 Calmar 分子是年化收益(CAGR)，与 engine.calc_metrics 同口径。"""
    ret, bm = _series(seed=8, n=60, excess=0.0012)
    r = report_module._annual_metrics(ret, bm, risk_free=RISK_FREE)

    nav = (1 + ret).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())

    assert r["calmar"] == pytest.approx(r["annual_return"] / (abs(mdd) + 1e-12), rel=1e-12)


def test_full_period_calmar_uses_annualized_return_in_engine():
    """全期 Calmar 必须是年化收益(CAGR)/同期最大回撤，不得用累计收益。"""
    ret, bm = _series(seed=10, n=60, excess=0.0008)

    e = calc_metrics(ret, bm, RISK_FREE)
    total = (1 + ret).prod() - 1
    nav = (1 + ret).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())

    assert e.calmar == pytest.approx(e.annual_return / (abs(mdd) + 1e-12), rel=1e-12)
    assert e.calmar != pytest.approx(
        total / (abs(mdd) + 1e-12), rel=1e-6
    )


def test_full_period_excess_calmar_uses_annualized_excess():
    """超额 Calmar 同样使用年化几何超额，不得混入累计分子。"""
    ret, bm = _series(seed=11, n=60, excess=0.0010)
    e = calc_metrics(ret, bm, RISK_FREE)

    total = (1 + ret).prod() - 1
    bm_total = (1 + bm).prod() - 1
    total_exc = (1 + total) / (1 + bm_total) - 1
    exc_nav = (1 + ret).cumprod() / (1 + bm).cumprod()
    exc_mdd = float((exc_nav / exc_nav.cummax() - 1).min())

    assert e.excess_calmar == pytest.approx(
        e.annual_excess_return / (abs(exc_mdd) + 1e-12), rel=1e-12
    )
    assert e.excess_calmar != pytest.approx(
        total_exc / (abs(exc_mdd) + 1e-12), rel=1e-6
    )
