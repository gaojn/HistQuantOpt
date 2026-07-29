"""阶段二：逐期优化。

每个调仓日的流程是「推进成交账本 → 构建优化域 → 取 Alpha → 求解 → 落账」。
任何一步失败都不中断整段回测：信号日先执行旧目标；若未发布新权重行，旧目标继续
保留并在后续交易日尝试。

本模块的依赖全部经 ``_BatchInputs`` 实例传入（优化器、风险模型、基准、账本均为
已构造好的实例），因此不持有任何可被测试 monkeypatch 的模块级全局。
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from hqopt.optimizer._common import POST_SOLVE_ABS_TOL
from hqopt.optimizer.index_enhance import IndexEnhanceConfig
from hqopt.pipeline.batch.execution_walk import (
    _ExecutionWalker,
    _signal_day_suspended_tickers,
)
from hqopt.pipeline.batch.types import (
    _BatchInputs,
    _PeriodContext,
    _PeriodOutcome,
    _RunStats,
)
from hqopt.pipeline.universe import (
    AlphaZeroVarianceError,
    build_cost_vector,
    filter_universe,
    get_alpha_for_date,
)

logger = logging.getLogger(__name__)

# 求解器数值粉尘阈值：仅当所有单票粉尘的总和也不超过该阈值时才清零，
# 避免微量持仓，同时防止累计清零破坏优化器硬约束。
_DUST_WEIGHT_TOL = 1e-6
# 滞留持仓"显著"阈值：仅用于告警分级展示，不影响处理逻辑。
_STUCK_WEIGHT_TOL = 1e-4
# Alpha 对优化域的最低覆盖率；低于此值仅告警（不跳过），提示选股信息量不足。
_MIN_ALPHA_COVERAGE = 0.5

def _clean_target_weights(
    weights: np.ndarray,
    snapshot,
    prev_weight: np.ndarray | None,
) -> pd.Series:
    """清除数值粉尘，同时严格保留 T 日停牌持仓的实际权重。

    停牌且有持仓的股票由优化器固定为 ``prev_weight``；停牌且无持仓的股票
    固定为零。清除其他股票的数值粉尘后不再二次归一化，避免改变停牌股目标
    或放大 sell_only、个股上限等硬约束；微小预算缺口保留为现金。
    """
    target = pd.Series(
        np.asarray(weights, dtype=float),
        index=snapshot.tickers,
    ).clip(lower=0.0)
    suspended = pd.Series(snapshot.suspended_mask, index=snapshot.tickers, dtype=bool)
    previous = (
        pd.Series(0.0, index=snapshot.tickers, dtype=float)
        if prev_weight is None
        else pd.Series(np.asarray(prev_weight, dtype=float), index=snapshot.tickers)
    )
    frozen = suspended & previous.gt(1e-12)
    suspended_without_holding = suspended & ~frozen

    target.loc[frozen] = previous.loc[frozen]
    target.loc[suspended_without_holding] = 0.0

    adjustable = ~suspended
    dust = adjustable & target.gt(0.0) & target.lt(_DUST_WEIGHT_TOL)
    # 只有总粉尘也处于数值容差内时才清零；否则逐票清零可能累计破坏成分下限、
    # 行业/风格下界等硬约束。
    if float(target.loc[dust].sum()) <= _DUST_WEIGHT_TOL:
        target.loc[dust] = 0.0
    if float(target.sum()) > 1.0 + 1e-10:
        raise ValueError(f"清理后目标权重和超过 1：{float(target.sum()):.12f}")
    return target


def _build_period_snapshot(
    inputs: _BatchInputs,
    rebal_date: date,
    actual_holdings: pd.Series,
    has_prior_target: bool,
):
    """构建当期优化域快照；当日无行情则返回 None（由调用方跳过该期）。"""
    try:
        snap_full = inputs.adapter.build_snapshot_from_panel(
            panel=inputs.panel, target_date=rebal_date,
            index=inputs.index, portfolio_value=inputs.ledger.nav,
        )
    except ValueError as e:
        logger.info(f"  [{rebal_date}] 跳过（快照失败：{e}）")
        return None

    # 有实际持仓却无当日行情（退市/长停滞留）：这些票无法交易也无法估计风险，
    # 不进优化域，按场外滞留资产处理——继续优化其余部分。账本只按真实现金成交，
    # 不会把滞留市值误当现金再分配（超买部分会被现金约束等比例缩减）。
    missing_actual = actual_holdings[
        ~actual_holdings.index.isin(snap_full.tickers) & (actual_holdings > 1e-8)
    ]
    if not missing_actual.empty:
        significant = missing_actual[missing_actual > _STUCK_WEIGHT_TOL]
        detail = (
            f"，其中显著持仓 {significant.round(6).to_dict()}"
            if not significant.empty else "（均为粉尘仓位）"
        )
        logger.warning(
            f"  [{rebal_date}] 实际持仓中 {len(missing_actual)} 只缺当日行情"
            f"（合计权重 {float(missing_actual.sum()):.4%}），"
            f"作为滞留资产排除在优化域外{detail}"
        )

    uni_cfg = inputs.universe_cfg
    return filter_universe(
        snap_full, inputs.panel, rebal_date,
        exclude_bj=bool(uni_cfg.get("exclude_bj", True)),
        exclude_st=bool(uni_cfg.get("exclude_st", True)),
        top_n=int(uni_cfg["top_n"]) if uni_cfg.get("top_n") else None,
        prev_holdings=actual_holdings if has_prior_target else None,
    )


def _prepare_period(
    inputs: _BatchInputs,
    rebal_date: date,
    actual_holdings: pd.Series,
    has_prior_target: bool,
) -> _PeriodContext | None:
    """快照 + 风险模型对齐；任一前置条件不满足则返回 None。"""
    snapshot = _build_period_snapshot(
        inputs, rebal_date, actual_holdings, has_prior_target
    )
    if snapshot is None:
        return None

    # 风格载荷 + 风险模型：均来自 CNE6 面板（S 20 风格或 L 16 风格 + 行业）。
    # 暴露用于 style_active_bound 约束；risk_aversion 设置时协方差进目标。
    # 无 CNE6 覆盖的调仓日跳过。
    risk_snap = inputs.risk_model.at(rebal_date, snapshot.tickers)
    if risk_snap is None:
        logger.info(f"  [{rebal_date}] 跳过（CNE6 风险面板无覆盖）")
        return None

    uncovered = pd.Series(~risk_snap.covered_mask, index=snapshot.tickers, dtype=bool)
    if uncovered.any():
        existing_sell_only = (
            snapshot.sell_only.reindex(snapshot.tickers).fillna(False).astype(bool)
            if snapshot.sell_only is not None
            else pd.Series(False, index=snapshot.tickers, dtype=bool)
        )
        snapshot = replace(snapshot, sell_only=(existing_sell_only | uncovered))
        logger.warning(
            f"  [{rebal_date}] CNE6 未覆盖 {int(uncovered.sum())} 只："
            "禁止新开仓，已有持仓只卖不买"
        )

    # 上期权重来自实际成交账本，现金保留为权重缺口，不做重新归一化。
    prev_weight = (
        actual_holdings.reindex(snapshot.tickers).fillna(0.0).values
        if has_prior_target else None
    )
    return _PeriodContext(
        snapshot=snapshot,
        risk_snapshot=risk_snap,
        style_loading=risk_snap.style_loading(),
        prev_weight=prev_weight,
    )


def _resolve_period_alpha(
    inputs: _BatchInputs,
    ctx: _PeriodContext,
    rebal_date: date,
    stats: _RunStats,
) -> np.ndarray | None:
    """取当期 Alpha 并做陈旧度/覆盖率告警；不可用时返回 None。"""
    policy = inputs.alpha_policy
    try:
        alpha_slice = get_alpha_for_date(
            inputs.alpha_df,
            rebal_date,
            ctx.snapshot.tickers,
            max_staleness_days=policy.max_staleness_days,
            standardize=policy.standardize,
        )
    except AlphaZeroVarianceError as exc:
        stats.alpha_zero_variance_periods += 1
        stats.alpha_skipped_periods += 1
        stats.fail_count += 1
        logger.warning(f"  [{rebal_date}] 跳过优化：{exc}")
        return None
    if alpha_slice is None:
        # 信号过期或对优化域零覆盖：宁可不调仓，也不用陈旧/空信号驱动优化。
        stats.alpha_skipped_periods += 1
        stats.fail_count += 1
        logger.warning(
            f"  [{rebal_date}] 跳过优化：Alpha 不可用"
            f"（超过 max_staleness_days={policy.max_staleness_days} 或优化域零覆盖）"
        )
        return None

    stats.alpha_max_staleness = max(
        stats.alpha_max_staleness, alpha_slice.staleness_days
    )
    period_key = rebal_date.isoformat()
    stats.alpha_as_of_by_period[period_key] = alpha_slice.as_of.isoformat()
    stats.alpha_staleness_days_by_period[period_key] = alpha_slice.staleness_days
    if alpha_slice.staleness_days > policy.stale_warn_days:
        stats.alpha_stale_periods += 1
        logger.warning(
            f"  [{rebal_date}] Alpha 陈旧 {alpha_slice.staleness_days} 日"
            f"（取自 {alpha_slice.as_of}，有效 {alpha_slice.n_valid} 只）"
            "——信号可能已过期，业绩不可直接采信"
        )
    n_tickers = len(ctx.snapshot.tickers)
    alpha_coverage = alpha_slice.n_valid / max(n_tickers, 1)
    if alpha_coverage < _MIN_ALPHA_COVERAGE:
        logger.warning(
            f"  [{rebal_date}] Alpha 仅覆盖优化域 {alpha_coverage:.1%}"
            f"（{alpha_slice.n_valid}/{n_tickers} 只）："
            "未覆盖股票按截面中性(0)处理，选股信息量有限"
        )
    return alpha_slice.values


def _solve_period(
    inputs: _BatchInputs,
    ctx: _PeriodContext,
    alpha: np.ndarray,
    rebal_date: date,
    stats: _RunStats,
):
    """执行当期优化；未进入求解（基准风险覆盖不足）时返回 None。"""
    prev_weight = ctx.prev_weight

    # 个股冲击成本权重（仅在启用换手软惩罚时计算）
    cost_vec = None
    if inputs.use_cost_vector and prev_weight is not None:
        cost_vec = build_cost_vector(
            tickers=ctx.snapshot.tickers,
            panel=inputs.panel,
            target_date=rebal_date,
        )

    if inputs.strategy != "index_enhance":
        return inputs.optimizer.optimize(
            alpha, ctx.snapshot,
            style_loading=ctx.style_loading,
            prev_weight=prev_weight,
            cost_vector=cost_vec,
            risk_snapshot=ctx.risk_snapshot,
        )

    bm_series = inputs.benchmark.get_weights(rebal_date, tickers=ctx.snapshot.tickers)
    bm_total = float(bm_series.sum())
    covered_weight = (
        float(bm_series.values[ctx.risk_snapshot.covered_mask].sum()) / bm_total
        if bm_total > 1e-12 else 0.0
    )
    if covered_weight < inputs.min_risk_coverage:
        # 覆盖率异常多为面板数据阶段性缺口（如 2020 年初），跳过该期
        # 并保留账本原目标继续执行；不中断整段回测。
        stats.fail_count += 1
        logger.warning(
            f"  [{rebal_date}] 跳过优化：基准风险覆盖率 {covered_weight:.2%} "
            f"低于阈值 {inputs.min_risk_coverage:.2%}（零暴露填充不可信）"
        )
        return None

    # 首期无实际持仓时不施加换手硬上限（无「上期」可比）
    inputs.optimizer.config = (
        inputs.base_config if prev_weight is not None
        else IndexEnhanceConfig(**{**inputs.base_config.__dict__, "max_turnover": None})
    )
    return inputs.optimizer.optimize(
        alpha=alpha, snapshot=ctx.snapshot,
        benchmark_weight=bm_series.values,
        style_loading=ctx.style_loading,
        prev_weight=prev_weight,
        cost_vector=cost_vec,
        risk_snapshot=ctx.risk_snapshot,
    )


def _record_period_success(
    inputs: _BatchInputs,
    ctx: _PeriodContext,
    result: Any,
    rebal_date: date,
    signal_day: pl.DataFrame,
    period_index: int,
    elapsed: float,
    outcome: _PeriodOutcome,
) -> None:
    """落账当期目标：清理粉尘、记录只卖矩阵、提交到成交账本。"""
    snapshot = ctx.snapshot
    prev_weight = ctx.prev_weight

    # 清除求解器数值粉尘，但绝不能通过全组合重归一改变停牌股权重。
    w = _clean_target_weights(result.weights, snapshot, prev_weight)
    outcome.weight_records[rebal_date] = w

    held = (
        np.zeros(snapshot.n_stocks, dtype=bool) if prev_weight is None
        else prev_weight > 1e-12
    )
    effective_sell_only = (
        snapshot.sell_only_mask
        | ((snapshot.new_listing_mask | snapshot.st_mask) & held)
    ) & ~snapshot.suspended_mask
    sell_only = pd.Series(effective_sell_only, index=snapshot.tickers, dtype=bool)
    outcome.sell_only_records[rebal_date] = sell_only

    inputs.ledger.submit_target(
        w,
        frozen_tickers=_signal_day_suspended_tickers(signal_day),
        sell_only_tickers=sell_only.index[sell_only].tolist(),
    )

    turnover = (
        float(np.abs(w.values - prev_weight).sum()) if prev_weight is not None
        else float("nan")
    )
    if prev_weight is not None:
        outcome.stats.target_turnovers.append(turnover)

    n_periods = len(inputs.rebal_dates)
    if period_index % 10 == 0 or period_index == n_periods - 1:
        extra = ""
        if inputs.strategy == "index_enhance":
            const_w = result.weights[snapshot.constituent_mask].sum()
            extra = (
                f"  {inputs.index.upper()}={const_w*100:.1f}%  "
                f"TE_L2={result.tracking_error_l2():.4f}"
            )
        logger.info(
            f"  [{period_index+1:3d}/{n_periods}] {rebal_date}  "
            f"持仓={result.n_positions:3d}  换手={turnover*100:>5.1f}%{extra}  "
            f"耗时={elapsed:.2f}s"
        )


def _run_periods(inputs: _BatchInputs) -> _PeriodOutcome:
    """逐期优化：推进成交账本 → 构建优化域 → 取 Alpha → 求解 → 落账。

    任何一期在求解前后失败都不中断整段回测：信号日已正常执行旧目标；不发布
    新权重行时，旧目标继续保留并在后续交易日尝试。
    """
    logger.info("\n[4] 逐期优化...")
    t_total = time.time()
    outcome = _PeriodOutcome(
        weight_records={}, sell_only_records={}, stats=_RunStats(), elapsed=0.0
    )
    walker = _ExecutionWalker(inputs.ledger, inputs.trade_dates, inputs.execution_days)
    has_prior_target = False

    for period_index, rebal_date in enumerate(inputs.rebal_dates):
        t0 = time.time()
        signal_day = walker.open_signal_day(rebal_date)
        actual_holdings = inputs.ledger.actual_weights()

        ctx = _prepare_period(inputs, rebal_date, actual_holdings, has_prior_target)
        if ctx is None:
            continue

        alpha = _resolve_period_alpha(inputs, ctx, rebal_date, outcome.stats)
        if alpha is None:
            continue

        result = _solve_period(inputs, ctx, alpha, rebal_date, outcome.stats)
        if result is None:
            continue

        elapsed = time.time() - t0
        outcome.stats.solve_times.append(elapsed)
        violations = getattr(result, "constraint_violations", {})
        if violations:
            max_violation = max(float(value) for value in violations.values())
            period_key = rebal_date.isoformat()
            outcome.stats.postcheck_max_violation = max(
                outcome.stats.postcheck_max_violation, max_violation
            )
            outcome.stats.postcheck_max_violation_by_period[period_key] = (
                max_violation
            )
        if not result.is_feasible:
            outcome.stats.fail_count += 1
            if "post-solve constraint violation" in result.status:
                outcome.stats.postcheck_failure_count += 1
                outcome.stats.postcheck_failures_by_period[
                    rebal_date.isoformat()
                ] = {
                    name: float(value)
                    for name, value in violations.items()
                    if not np.isfinite(value) or value > POST_SOLVE_ABS_TOL
                }
            logger.info(f"  [{rebal_date}] ✗ 求解失败：{result.status}")
            continue

        _record_period_success(
            inputs, ctx, result, rebal_date, signal_day,
            period_index, elapsed, outcome,
        )
        has_prior_target = True

    # 最后一个调仓目标也要推进到配置结束日，否则成交统计会停在信号日收盘，
    # 漏记最后一批 T+1/T+2/T+3 成交或过期。
    walker.finish()
    # 失败期数以“候选期数 - 成功发布目标期数”为唯一口径，覆盖快照、风险、
    # Alpha、求解等全部失败路径，避免各分支手工累计发生漏记或重复。
    outcome.stats.fail_count = len(inputs.rebal_dates) - len(outcome.weight_records)
    outcome.elapsed = time.time() - t_total
    return outcome
