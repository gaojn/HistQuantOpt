"""阶段三：汇总与原子发布。

把逐期权重汇成矩阵，连同只卖矩阵、成交统计与内容绑定清单一次性原子发布，
避免读取端看到写了一半的 bundle。合成 Alpha 的警告文件也在此处落盘/清除。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hqopt.backtest.execution import (
    publish_batch_bundle,
    synthetic_alpha_warning_path_for_weights,
)
from hqopt.constants import SYNTHETIC_ALPHA_WARNING_TEXT
from hqopt.optimizer._common import POST_SOLVE_ABS_TOL
from hqopt.pipeline.batch.types import _BatchInputs, _PeriodOutcome

logger = logging.getLogger(__name__)


def _target_turnover_summary(stats) -> dict[str, float]:
    """目标换手的两口径均值。首期无「上期」故不计入。"""
    records = list(stats.target_turnover_by_period.values())
    if not records:
        nan = float("nan")
        return {"gross_mean": nan, "net_mean": nan, "cash_gap_mean": nan}
    return {
        "gross_mean": float(np.mean([r["gross"] for r in records])),
        "net_mean": float(np.mean([r["net"] for r in records])),
        "cash_gap_mean": float(np.mean([r["cash_gap"] for r in records])),
    }


def _log_run_summary(weight_df: pd.DataFrame, outcome: _PeriodOutcome) -> None:
    stats = outcome.stats
    logger.info(f"\n{'='*65}\n  批量优化汇总\n{'='*65}")
    logger.info(f"  再平衡期数   : {len(weight_df)}")
    logger.info(f"  失败期数     : {stats.fail_count}")
    logger.info(
        f"  Alpha 陈旧   : 最大 {stats.alpha_max_staleness} 日  "
        f"告警期数 {stats.alpha_stale_periods}  不可用跳过 {stats.alpha_skipped_periods}  "
        f"零方差 {stats.alpha_zero_variance_periods}"
    )
    if stats.alpha_stale_periods or stats.alpha_skipped_periods:
        logger.warning(
            "  ⚠️  存在 Alpha 质量异常，请检查面板覆盖、新鲜度和截面方差；"
            "异常期已按配置告警或跳过，必须结合落盘 alpha_quality 审计业绩"
        )
    logger.info(f"  平均持仓数   : {(weight_df > 1e-6).sum(axis=1).mean():.0f} 只")
    turnover_summary = _target_turnover_summary(stats)
    logger.info(
        f"  平均目标换手 : 净 {turnover_summary['net_mean']*100:.1f}%"
        f"（股票间调仓，与 max_turnover 同口径）  "
        f"含现金重投 {turnover_summary['gross_mean']*100:.1f}%"
        f"（其中现金缺口 {turnover_summary['cash_gap_mean']*100:.1f}%）"
    )
    logger.info(
        f"  平均耗时     : {np.mean(stats.solve_times):.2f}s  "
        f"总耗时: {outcome.elapsed:.1f}s"
    )


def _publish_outputs(inputs: _BatchInputs, outcome: _PeriodOutcome) -> pd.DataFrame:
    """汇总权重矩阵，原子发布 bundle（weights + sidecar + 统计 + 清单）。"""
    if not outcome.weight_records:
        raise RuntimeError("所有期均求解失败，请检查配置")

    weight_df = pd.DataFrame(outcome.weight_records).T.fillna(0.0)
    weight_df.index.name = "date"
    _log_run_summary(weight_df, outcome)

    sell_only_df = (
        pd.DataFrame(outcome.sell_only_records)
        .T.reindex(index=weight_df.index, columns=weight_df.columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    sell_only_df.index.name = "date"

    ledger = inputs.ledger
    batch_execution_stats = {
        "expired_order_count": ledger.expired_order_count,
        "expired_notional": float(ledger.expired_notional),
        "target_pending": ledger.pending_target is not None,
        "final_cash": float(ledger.cash),
        "final_nav": float(ledger.nav),
        "final_shares": dict(ledger.shares),
        "order_states": {
            ticker: state.value for ticker, state in ledger.order_states.items()
        },
        "optimization": {
            "candidate_period_count": len(inputs.rebal_dates),
            "successful_period_count": len(outcome.weight_records),
            "failed_period_count": outcome.stats.fail_count,
            "target_turnover": {
                "definition": (
                    "双边目标换手。gross=Σ|w_target − w_prev_actual|，w_prev 取自"
                    "成交账本实际持仓（不含现金，故 Σ<1）；cash_gap=1−Σw_prev 是"
                    "把留存现金买回股票所必需的买入量；net=gross−cash_gap 为股票间"
                    "调仓强度，与 max_turnover 约束同口径。首期无上期故不计入。"
                    "注意这是**目标**换手，与回测实际成交换手"
                    "（turnover.parquet）因涨跌停/停牌/现金不足而不同。"
                ),
                **_target_turnover_summary(outcome.stats),
                "by_period": outcome.stats.target_turnover_by_period,
            },
            "post_solve_validation": {
                "absolute_tolerance": POST_SOLVE_ABS_TOL,
                "failure_count": outcome.stats.postcheck_failure_count,
                "max_observed_violation": outcome.stats.postcheck_max_violation,
                "max_violation_by_period": (
                    outcome.stats.postcheck_max_violation_by_period
                ),
                "failures_by_period": outcome.stats.postcheck_failures_by_period,
            },
        },
        "benchmark_quality": (
            inputs.benchmark.audit_summary()
            if inputs.benchmark is not None
            and hasattr(inputs.benchmark, "audit_summary")
            else {"source": None}
        ),
        "alpha_quality": {
            "synthetic": inputs.synthetic_alpha,
            "standardized": inputs.alpha_policy.standardize,
            "max_staleness_days_limit": inputs.alpha_policy.max_staleness_days,
            "max_observed_staleness_days": outcome.stats.alpha_max_staleness,
            "stale_warning_period_count": outcome.stats.alpha_stale_periods,
            "skipped_period_count": outcome.stats.alpha_skipped_periods,
            "zero_variance_period_count": (
                outcome.stats.alpha_zero_variance_periods
            ),
            "as_of_by_period": outcome.stats.alpha_as_of_by_period,
            "staleness_days_by_period": (
                outcome.stats.alpha_staleness_days_by_period
            ),
        },
    }
    (
        out_path,
        sell_only_path,
        batch_stats_path,
        sell_only_manifest_path,
    ) = publish_batch_bundle(
        weight_df,
        sell_only_df,
        batch_execution_stats,
        inputs.output_path,
    )

    # 水印按权重 stem 隔离：真实 Alpha bundle 的清除动作绝不能摘掉同目录内
    # 其他合成 bundle 的水印。
    warning_path = synthetic_alpha_warning_path_for_weights(out_path)
    if inputs.synthetic_alpha:
        warning_path.write_text(SYNTHETIC_ALPHA_WARNING_TEXT, encoding="utf-8")
    else:
        warning_path.unlink(missing_ok=True)

    logger.info(f"\n  权重矩阵已保存：{out_path}")
    logger.info(f"  只卖元数据已保存：{sell_only_path}")
    logger.info(f"  内容绑定清单已保存：{sell_only_manifest_path}")
    logger.info(
        f"  成交统计已保存：{batch_stats_path}  "
        f"过期={ledger.expired_order_count}笔/"
        f"{ledger.expired_notional:,.2f}元"
    )
    logger.info(f"\n{'='*65}\n")
    return weight_df
