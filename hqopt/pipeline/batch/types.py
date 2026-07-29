"""阶段间传递的数据结构。

这些 dataclass 是三个阶段之间的契约：``_prepare_inputs`` 产出 ``_BatchInputs``，
``_run_periods`` 产出 ``_PeriodOutcome``，``_publish_outputs`` 消费两者。字段一律
只读（``frozen=True``），累计型统计除外。

此处只做类型标注，不构造任何可被测试 monkeypatch 的对象（见 ``__init__`` 说明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from hqopt.backtest.execution import ExecutionLedger
from hqopt.data.benchmark import IndexBenchmarkWeights
from hqopt.data.real_adapter import RealMarketAdapter
from hqopt.risk import CNE6RiskModel


@dataclass(frozen=True)
class _AlphaPolicy:
    """Alpha 可信度控制参数（见 universe.get_alpha_for_date）。"""

    max_staleness_days: int | None
    standardize: bool
    stale_warn_days: int


@dataclass(frozen=True)
class _RunConfig:
    """YAML 配置解析后的运行参数。"""

    strategy: str
    index: str
    start_date: date
    end_date: date
    rebal_freq: int
    initial_value: float
    universe_cfg: dict[str, Any]
    optimizer_cfg: dict[str, Any]
    alpha_cfg: dict[str, Any]
    execution_cfg: dict[str, Any]
    output_path: Path


@dataclass(frozen=True)
class _BatchInputs:
    """`_prepare_inputs` 的产物：逐期优化所需的全部只读输入与组件。"""

    strategy: str
    index: str
    universe_cfg: dict[str, Any]
    output_path: Path
    synthetic_alpha: bool
    panel: pl.DataFrame
    alpha_df: pd.DataFrame
    alpha_policy: _AlphaPolicy
    trade_dates: list[date]
    rebal_dates: list[date]
    execution_days: dict[date, pl.DataFrame]
    ledger: ExecutionLedger
    adapter: RealMarketAdapter
    risk_model: CNE6RiskModel
    optimizer: Any
    base_config: Any
    benchmark: IndexBenchmarkWeights | None
    min_risk_coverage: float
    use_cost_vector: bool


@dataclass(frozen=True)
class _PeriodContext:
    """单个调仓日通过前置校验后的优化输入。"""

    snapshot: Any
    risk_snapshot: Any
    style_loading: pd.DataFrame
    prev_weight: np.ndarray | None


@dataclass
class _RunStats:
    """逐期优化的累计统计。"""

    fail_count: int = 0
    solve_times: list[float] = field(default_factory=list)
    target_turnovers: list[float] = field(default_factory=list)
    alpha_stale_periods: int = 0
    alpha_skipped_periods: int = 0
    alpha_zero_variance_periods: int = 0
    alpha_max_staleness: int = 0
    alpha_as_of_by_period: dict[str, str] = field(default_factory=dict)
    alpha_staleness_days_by_period: dict[str, int] = field(default_factory=dict)
    postcheck_failure_count: int = 0
    postcheck_max_violation: float = 0.0
    postcheck_max_violation_by_period: dict[str, float] = field(default_factory=dict)
    postcheck_failures_by_period: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


@dataclass
class _PeriodOutcome:
    """`_run_periods` 的产物：逐期权重、只卖矩阵与统计。"""

    weight_records: dict
    sell_only_records: dict
    stats: _RunStats
    elapsed: float
