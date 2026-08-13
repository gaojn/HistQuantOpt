"""求解器降级路径测试：PIQP → CLARABEL → SCS → 全部失败。

这条链路平时不触发，但一旦悄悄坏掉，表现形式往往是"业绩变好"而不是报错——
求解失败若被当成正常结果返回，会得到一个全零或半成品权重向量，回测照跑不误。
因此单独守护三条契约：

1. ``solve_with_fallback``：PIQP 首选（纯 QP 求解器，全市场规模实测快 3~4 倍）；
   遇到非 QP 结构（如 te_upper/vol_upper 的二次硬约束）或异常时必须降级
   CLARABEL，再降级 SCS；全部抛异常时如实返回失败原因，绝不吞掉。
2. 两个优化器：拿到失败原因或非最优状态时必须返回 **不可行结果**
   （``is_feasible=False``、权重全零、目标值 NaN），供
   ``batch_optimize._run_periods`` 跳过该期并计入失败期数。

契约对 ``alpha_max`` 与 ``index_enhance`` 完全一致（同出 ``_common``），故按
优化器参数化——避免"改一处漏一处"。
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from hqopt.data.generator import MarketDataGenerator
from hqopt.optimizer import _common
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer

# 单票上限 × 股票数 < 1，预算约束 sum(w)==1 必然无解
_INFEASIBLE_WEIGHT_UPPER = 0.001


@pytest.fixture
def snap():
    return MarketDataGenerator(n_stocks=40, seed=0).generate()


def _trivial_qp(n: int = 10) -> cp.Problem:
    """一个有解的小 QP（二次目标 + 线性约束），PIQP 可直接求解。"""
    w = cp.Variable(n, nonneg=True)
    return cp.Problem(
        cp.Maximize(np.arange(n) @ w - cp.sum_squares(w)), [cp.sum(w) == 1.0]
    )


def _non_qp_problem(n: int = 10) -> cp.Problem:
    """一个有解但无法化为 QP 的凸问题（sqrt 目标），PIQP 必然拒绝。"""
    w = cp.Variable(n, nonneg=True)
    return cp.Problem(cp.Maximize(cp.sum(cp.sqrt(w))), [cp.sum(w) == 1.0])


def _quad_constraint_problem(n: int = 10) -> cp.Problem:
    """带二次硬约束的问题——te_upper / vol_upper 模式的最小复刻。"""
    w = cp.Variable(n, nonneg=True)
    return cp.Problem(
        cp.Maximize(cp.sum(w)),
        [cp.sum(w) == 1.0, cp.sum_squares(w) <= 0.5],
    )


def _infeasible_problem(n: int = 10) -> cp.Problem:
    w = cp.Variable(n, nonneg=True)
    return cp.Problem(
        cp.Maximize(cp.sum(w)),
        [cp.sum(w) == 1.0, w <= _INFEASIBLE_WEIGHT_UPPER],
    )


def _spy_solver(monkeypatch, *, raise_on: tuple = ()) -> list:
    """记录每次 solve 用了哪个求解器；``raise_on`` 中的求解器改为抛异常。

    返回被填充的调用列表（按调用顺序记录 solver 对象）。
    """
    original = cp.Problem.solve
    calls: list = []

    def fake_solve(self, *args, solver=None, **kwargs):
        calls.append(solver)
        if solver in raise_on:
            raise RuntimeError(f"{solver} boom")
        return original(self, *args, solver=solver, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    return calls


# ── solve_with_fallback 本身 ──────────────────────────────────────


def test_piqp_solves_qp_without_fallback(monkeypatch):
    """QP 问题（默认软惩罚配置的结构）应由 PIQP 一次解决，不触发降级。"""
    calls = _spy_solver(monkeypatch)
    problem = _trivial_qp()

    failure = _common.solve_with_fallback(problem)

    assert calls == [cp.PIQP], "QP 问题不应降级——降级说明 PIQP 首选路径坏了"
    assert failure is None
    assert problem.status in ("optimal", "optimal_inaccurate")


def test_non_qp_structure_falls_back_to_clarabel(monkeypatch):
    """无法化为 QP 的问题（如 te_upper/vol_upper 的二次硬约束）→ 降级 CLARABEL。

    cvxpy 对 PIQP 抛 SolverError，链条必须吞掉并继续，语义不变只是变慢。
    """
    calls = _spy_solver(monkeypatch)
    problem = _quad_constraint_problem()

    failure = _common.solve_with_fallback(problem)

    assert calls == [cp.PIQP, cp.CLARABEL]
    assert failure is None
    assert problem.status in ("optimal", "optimal_inaccurate")


def test_clarabel_exception_falls_back_to_scs(monkeypatch):
    """PIQP 拒绝、CLARABEL 抛异常 → 仍要真的再试 SCS，取到最优解。"""
    calls = _spy_solver(monkeypatch, raise_on=(cp.CLARABEL,))
    problem = _non_qp_problem()

    failure = _common.solve_with_fallback(problem)

    assert calls == [cp.PIQP, cp.CLARABEL, cp.SCS], "CLARABEL 失败后必须真的再试 SCS"
    assert failure is None, "SCS 成功时不应上报失败原因"
    assert problem.status in ("optimal", "optimal_inaccurate")


def test_non_optimal_status_falls_through_all_solvers(monkeypatch):
    """求解器不抛异常但返回非最优状态，必须逐级降级到底。

    只判异常不判 status 的写法会在这里静默停在首个求解器的劣质解上。
    """
    calls = _spy_solver(monkeypatch)
    problem = _infeasible_problem()

    failure = _common.solve_with_fallback(problem)

    assert calls == [cp.PIQP, cp.CLARABEL, cp.SCS]
    # SCS 未抛异常 → 不算"求解器失败"，由调用方按 status 判定不可行
    assert failure is None
    assert problem.status not in ("optimal", "optimal_inaccurate")


def test_all_solvers_fail_returns_reason(monkeypatch):
    """全部求解器都抛异常时必须如实返回原因，不得吞掉。"""
    calls = _spy_solver(monkeypatch, raise_on=(cp.PIQP, cp.CLARABEL, cp.SCS))
    problem = _trivial_qp()

    failure = _common.solve_with_fallback(problem)

    assert calls == [cp.PIQP, cp.CLARABEL, cp.SCS]
    assert failure is not None
    assert failure.startswith("all solvers failed:")
    assert "boom" in failure, "失败原因应保留底层异常信息，便于定位"


# ── 优化器对降级结果的处理 ────────────────────────────────────────


def _optimize(kind: str, snap, cfg_overrides: dict | None = None):
    """按策略跑一次优化，返回结果；两策略共用同一批宽松约束。"""
    alpha = np.random.default_rng(4).standard_normal(snap.n_stocks)
    overrides = cfg_overrides or {}
    if kind == "alpha_max":
        cfg = AlphaMaxConfig(
            weight_upper=overrides.get("weight_upper", 0.10),
            industry_upper=0.5,
        )
        return AlphaMaxOptimizer(cfg).optimize(alpha, snap)
    bm_w = snap.constituent_mask.astype(float)
    bm_w = bm_w / bm_w.sum()
    cfg = IndexEnhanceConfig(
        weight_upper=overrides.get("weight_upper", 0.10),
        min_constituent_ratio=0.0,
        industry_active_bound=0.5,
        style_active_bound=5.0,
    )
    return IndexEnhanceOptimizer(cfg).optimize(alpha, snap, benchmark_weight=bm_w)


@pytest.mark.parametrize("kind", ["index_enhance", "alpha_max"])
def test_solver_failure_yields_infeasible_result(monkeypatch, snap, kind):
    """两求解器都失败 → 不可行结果（零权重 / NaN 目标值），且保留失败原因。"""
    module = f"hqopt.optimizer.{kind}"
    monkeypatch.setattr(
        f"{module}.solve_with_fallback",
        lambda problem: "all solvers failed: boom",
    )

    res = _optimize(kind, snap)

    assert not res.is_feasible
    assert "boom" in res.status, "失败原因必须透传到结果状态，供日志定位"
    assert np.array_equal(res.weights, np.zeros(snap.n_stocks)), (
        "求解失败必须给零权重，绝不能返回半成品向量让回测继续跑"
    )
    assert np.isnan(res.objective_value)
    # 快照被清空 → 下游拿不到"看似正常"的组合信息
    assert res.snapshot is None


@pytest.mark.parametrize("kind", ["index_enhance", "alpha_max"])
def test_non_optimal_status_yields_infeasible_result(snap, kind):
    """真实无解问题（单票上限 × 股票数 < 1）→ 不可行结果，不抛异常。

    单期无解不应中断整段回测，而是由 ``_run_periods`` 跳过该期。
    """
    res = _optimize(kind, snap, {"weight_upper": _INFEASIBLE_WEIGHT_UPPER})

    assert not res.is_feasible, f"上限 {_INFEASIBLE_WEIGHT_UPPER} 时问题应无解"
    assert res.status.startswith("infeasible: ")
    assert float(res.weights.sum()) == 0.0


def test_ie_infeasible_result_has_no_benchmark(monkeypatch, snap):
    """指增不可行结果不携带基准权重 → active_weight 为 None 而非虚假零偏离。"""
    monkeypatch.setattr(
        "hqopt.optimizer.index_enhance.solve_with_fallback",
        lambda problem: "all solvers failed: boom",
    )

    res = _optimize("index_enhance", snap)

    assert res.benchmark_weight is None
    assert res.active_weight is None
    assert np.isnan(res.tracking_error_l2())
    assert res.industry_active_weights().empty
    style = pd.DataFrame(0.0, index=snap.tickers, columns=["Size"])
    assert res.style_active_exposure(style).empty

# ── 求解后硬约束门禁 ──────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["index_enhance", "alpha_max"])
def test_post_solve_rejects_cleanup_that_breaks_budget(monkeypatch, snap, kind):
    """即使原始求解成功，清理后的权重破坏预算也必须拒绝发布。"""
    module = f"hqopt.optimizer.{kind}"
    monkeypatch.setattr(
        f"{module}.finalize_weights",
        lambda raw, masks: np.zeros_like(raw),
    )

    result = _optimize(kind, snap)

    assert not result.is_feasible
    assert "post-solve constraint violation" in result.status
    assert result.constraint_violations["budget"] == pytest.approx(1.0)
    assert np.array_equal(result.weights, np.zeros(snap.n_stocks))


@pytest.mark.parametrize("kind", ["index_enhance", "alpha_max"])
def test_optimal_inaccurate_is_not_trusted_without_postcheck(
    monkeypatch, snap, kind
):
    """optimal_inaccurate 只表示求解器状态；违规半成品仍必须被门禁拦截。"""
    module = f"hqopt.optimizer.{kind}"

    def fake_inaccurate(problem):
        variable = problem.variables()[0]
        variable.value = np.zeros(variable.shape)
        problem._status = "optimal_inaccurate"
        return None

    monkeypatch.setattr(f"{module}.solve_with_fallback", fake_inaccurate)

    result = _optimize(kind, snap)

    assert not result.is_feasible
    assert "post-solve constraint violation" in result.status
    assert result.constraint_violations["budget"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["index_enhance", "alpha_max"])
def test_scs_fallback_must_also_pass_postcheck(monkeypatch, snap, kind):
    """PIQP/CLARABEL 均异常后采用 SCS；仅在全部硬约束残差过门时才可发布。"""
    calls = _spy_solver(monkeypatch, raise_on=(cp.PIQP, cp.CLARABEL))

    result = _optimize(kind, snap)

    assert calls == [cp.PIQP, cp.CLARABEL, cp.SCS]
    assert result.is_feasible
    assert max(result.constraint_violations.values()) <= _common.POST_SOLVE_ABS_TOL
