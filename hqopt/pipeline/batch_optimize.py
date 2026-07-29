"""
批量组合优化 pipeline。

支持 index_enhance（指数增强）和 alpha_max（量化多头）两种策略。
通过 YAML 配置文件驱动，不依赖具体 demo 脚本。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from hqopt.backtest.execution import ExecutionLedger, publish_batch_bundle
from hqopt.constants import (
    SYNTHETIC_ALPHA_WARNING_FILE,
    SYNTHETIC_ALPHA_WARNING_TEXT,
)
from hqopt.data.benchmark import (
    DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    IndexBenchmarkWeights,
)
from hqopt.data.real_adapter import RealMarketAdapter
from hqopt.io.data_panel import load_panel
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer
from hqopt.pipeline.universe import (
    AlphaZeroVarianceError,
    build_cost_vector,
    build_synthetic_alpha,
    filter_universe,
    get_alpha_for_date,
    load_alpha_panel,
)
from hqopt.risk import CNE6RiskModel

logger = logging.getLogger(__name__)

_INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}
_STRATEGIES = {"index_enhance", "alpha_max"}
_ALPHA_SOURCES = {"file", "synthetic"}

# 求解器数值粉尘阈值：仅当所有单票粉尘的总和也不超过该阈值时才清零，
# 避免微量持仓，同时防止累计清零破坏优化器硬约束。
_DUST_WEIGHT_TOL = 1e-6
# 滞留持仓"显著"阈值：仅用于告警分级展示，不影响处理逻辑。
_STUCK_WEIGHT_TOL = 1e-4
# Alpha 陈旧告警阈值下限（自然日），防止高频调仓时阈值过小而刷屏。
_MIN_ALPHA_STALENESS_WARN_DAYS = 7
# Alpha 对优化域的最低覆盖率；低于此值仅告警（不跳过），提示选股信息量不足。
_MIN_ALPHA_COVERAGE = 0.5

_EXECUTION_COLUMNS = (
    "date", "code", "adj_close", "adj_vwap", "close",
    "limit_up", "limit_down", "trade_status",
)


def _partition_execution_days(
    panel: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> dict[date, pl.DataFrame]:
    """按交易日切分成交所需字段，供逐期优化按日推进实际持仓账本。"""
    missing = [column for column in _EXECUTION_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"行情面板缺少实际成交所需字段：{missing}")
    execution_panel = panel.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    ).select(_EXECUTION_COLUMNS)
    return {
        key[0] if isinstance(key, tuple) else key: day
        for key, day in execution_panel.partition_by("date", as_dict=True).items()
    }


def _execution_day_frame(day: pl.DataFrame) -> pd.DataFrame:
    """把一个 polars 日截面转换为以股票代码为索引的成交数据。"""
    return day.drop("date").to_pandas().set_index("code")


def _mark_execution_day(ledger: ExecutionLedger, day: pl.DataFrame) -> None:
    """仅更新收盘估值，不执行当前 pending 目标。"""
    ledger.mark_to_market(_execution_day_frame(day)["adj_close"])


def _advance_execution_day(ledger: ExecutionLedger, day: pl.DataFrame) -> None:
    """把一个 polars 日截面送入共享成交账本。"""
    pdf = _execution_day_frame(day)
    ledger.step(
        adj_close=pdf["adj_close"],
        adj_vwap=pdf["adj_vwap"],
        close_raw=pdf["close"],
        limit_up=pdf["limit_up"],
        limit_down=pdf["limit_down"],
        trade_status=pdf["trade_status"],
    )


def _signal_day_suspended_tickers(day: pl.DataFrame) -> list[str]:
    """返回信号日原始行情中停牌的股票代码。"""
    return (
        day.filter(pl.col("trade_status") == "停牌")
        .get_column("code")
        .cast(pl.Utf8)
        .to_list()
    )


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


def _parse_style_bound(v: Any) -> float | dict[str, float]:
    """解析风格约束：dict（按因子分别约束）或标量（统一）。"""
    if isinstance(v, dict):
        return {str(k): float(val) for k, val in v.items()}
    return float(v)


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    """解析可空数值配置；显式 0.0 必须保留。"""
    value = config.get(key)
    return None if value is None else float(value)


def _synthetic_alpha_enabled(alpha_cfg: dict[str, Any]) -> bool:
    """返回 Alpha 是否含前视；文件型 Alpha 也必须通过 synthetic 显式声明。"""
    synthetic = alpha_cfg.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise ValueError("alpha.synthetic 必须是布尔值 true/false")
    return synthetic or alpha_cfg.get("source") == "synthetic"


def _alpha_staleness_warn_days(rebalance_freq: int) -> int:
    """告警阈值：正常情况下每个调仓日都应有当期 Alpha，容忍约两个调仓间隔。

    交易日 → 自然日按 1.5 倍折算（含周末），下限 7 天避免高频调仓时过度告警。
    """
    return max(int(rebalance_freq * 2 * 1.5), _MIN_ALPHA_STALENESS_WARN_DAYS)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────
# 阶段间传递的数据结构
# ──────────────────────────────────────────────────────────────────


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


@dataclass
class _PeriodOutcome:
    """`_run_periods` 的产物：逐期权重、只卖矩阵与统计。"""

    weight_records: dict
    sell_only_records: dict
    stats: _RunStats
    elapsed: float


class _ExecutionWalker:
    """按交易日推进共享成交账本，把游标状态收敛在一处。

    优化阶段与回测阶段必须走完全一致的执行语义，因此推进顺序（调仓日只估值、
    未发布新目标的调仓日恢复旧目标成交、区间末尾补记）都固定在这里。
    """

    def __init__(
        self,
        ledger: ExecutionLedger,
        trade_dates: list[date],
        execution_days: dict[date, pl.DataFrame],
    ) -> None:
        self._ledger = ledger
        self._trade_dates = trade_dates
        self._days = execution_days
        self._cursor = 0

    def _execution_day(self, execution_date: date) -> pl.DataFrame:
        day = self._days.get(execution_date)
        if day is None:
            raise RuntimeError(f"{execution_date} 缺少成交行情截面")
        return day

    def open_signal_day(self, rebal_date: date) -> pl.DataFrame:
        """推进到调仓日：补齐此前交易日的成交，信号日仅更新估值。

        候选调仓日先暂停旧目标——成功的新目标会直接替换，失败则由
        ``replay_signal_day`` 恢复旧目标的当日尝试。
        """
        while (
            self._cursor < len(self._trade_dates)
            and self._trade_dates[self._cursor] < rebal_date
        ):
            _advance_execution_day(
                self._ledger, self._execution_day(self._trade_dates[self._cursor])
            )
            self._cursor += 1

        if (
            self._cursor >= len(self._trade_dates)
            or self._trade_dates[self._cursor] != rebal_date
        ):
            raise RuntimeError(f"{rebal_date} 不在成交交易日序列中")

        signal_day = self._days.get(rebal_date)
        if signal_day is None:
            raise RuntimeError(f"{rebal_date} 缺少信号日行情截面")
        _mark_execution_day(self._ledger, signal_day)
        self._cursor += 1
        return signal_day

    def replay_signal_day(self, signal_day: pl.DataFrame) -> None:
        """该调仓日未发布新目标：恢复旧目标在当日的正常成交尝试。"""
        _advance_execution_day(self._ledger, signal_day)

    def finish(self) -> None:
        """推进到区间末日，补记最后一批 T+1/T+2/T+3 成交或过期。"""
        while self._cursor < len(self._trade_dates):
            _advance_execution_day(
                self._ledger, self._execution_day(self._trade_dates[self._cursor])
            )
            self._cursor += 1


# ──────────────────────────────────────────────────────────────────
# 阶段一：准备输入
# ──────────────────────────────────────────────────────────────────


def _load_market_panel(
    panel: pl.DataFrame | None,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """加载行情面板；多取到当年年初，保证首期 ADV/VWAP 有历史窗口。"""
    if panel is not None:
        logger.info(
            f"\n[1] 使用预加载行情数据  交易日={panel['date'].n_unique()}  "
            f"股票={panel['code'].n_unique()}"
        )
        return panel

    data_start = date(start_date.year, 1, 1)
    logger.info(f"\n[1] 加载行情数据（{data_start} ~ {end_date}）...")
    panel = load_panel(
        data_start, end_date,
        columns=[
            "code", "date", "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")
    return panel


def _load_alpha_matrix(
    alpha_cfg: dict[str, Any],
    alpha_df: pd.DataFrame | None,
    panel: pl.DataFrame,
) -> pd.DataFrame:
    """按 alpha.source 加载外部因子或生成合成因子。"""
    if alpha_df is not None:
        logger.info("\n[2] 使用预加载 Alpha 矩阵")
        return alpha_df

    if "source" not in alpha_cfg:
        raise ValueError(
            "配置缺少 alpha.source 字段。必须显式指定 'file'（外部真实/DLF alpha）"
            "或 'synthetic'（含未来信息的标定用合成因子，仅供流程验证，"
            "回测业绩不可信）——不接受默认兜底，防止漏配时误用前视信号。"
        )
    alpha_source = alpha_cfg["source"]
    if alpha_source not in _ALPHA_SOURCES:
        raise ValueError(
            f"alpha.source 须为 {sorted(_ALPHA_SOURCES)} 之一，"
            f"当前为 {alpha_source!r}"
        )
    if alpha_source == "file":
        logger.info(f"\n[2] 读取外部 Alpha：{alpha_cfg['path']}")
        return load_alpha_panel(alpha_cfg["path"])
    return build_synthetic_alpha(
        panel,
        fwd_days=int(alpha_cfg["fwd_days"]),
        ic_mean=float(alpha_cfg["ic_mean"]),
        ic_std=float(alpha_cfg["ic_std"]),
        decay=float(alpha_cfg["decay"]),
        seed=int(alpha_cfg["seed"]),
    )


def _build_alpha_policy(alpha_cfg: dict[str, Any], rebal_freq: int) -> _AlphaPolicy:
    """解析 Alpha 可信度控制参数。

    陈旧信号会让回测"继续赚钱"却毫无依据；量纲未标准化则使 risk_aversion /
    turnover_penalty 的标定失真。
    """
    default_staleness = 15 if alpha_cfg.get("source") == "file" else None
    max_staleness_days = alpha_cfg.get("max_staleness_days", default_staleness)
    if max_staleness_days is not None:
        if isinstance(max_staleness_days, bool) or not isinstance(
            max_staleness_days, int
        ):
            raise ValueError("alpha.max_staleness_days 必须是非负整数或 null")
        if max_staleness_days < 0:
            raise ValueError("alpha.max_staleness_days 必须是非负整数或 null")

    policy = _AlphaPolicy(
        max_staleness_days=max_staleness_days,
        standardize=bool(alpha_cfg.get("standardize", True)),
        stale_warn_days=_alpha_staleness_warn_days(rebal_freq),
    )
    hard_skip = (
        "关闭" if policy.max_staleness_days is None
        else f">{policy.max_staleness_days}日"
    )
    logger.info(
        f"  Alpha 截面标准化={'是（z-score）' if policy.standardize else '否（原始量纲）'}  "
        f"陈旧告警>{policy.stale_warn_days}日  硬跳过={hard_skip}"
    )
    return policy


def _build_optimizer(
    strategy: str,
    opt_cfg: dict[str, Any],
    risk_aversion: float | None,
) -> tuple[Any, Any]:
    """按策略构造优化器与其基础配置。"""
    if strategy == "index_enhance":
        base_config = IndexEnhanceConfig(
            weight_upper=float(opt_cfg["weight_upper"]),
            min_constituent_ratio=float(opt_cfg["min_constituent_ratio"]),
            industry_active_bound=float(opt_cfg["industry_active_bound"]),
            style_active_bound=_parse_style_bound(opt_cfg["style_active_bound"]),
            tracking_penalty=float(opt_cfg["tracking_penalty"]),
            max_turnover=_optional_float(opt_cfg, "max_turnover"),
            turnover_penalty=float(opt_cfg.get("turnover_penalty", 0.0)),
            active_weight_upper=_optional_float(opt_cfg, "active_weight_upper"),
            weight_diff_l2_bound=_optional_float(opt_cfg, "weight_diff_l2_bound"),
            risk_aversion=risk_aversion,
        )
        return IndexEnhanceOptimizer(base_config), base_config

    base_config = AlphaMaxConfig(
        weight_upper=float(opt_cfg["weight_upper"]),
        industry_upper=float(opt_cfg.get("industry_upper", 0.20)),
        min_constituent_ratio=float(opt_cfg.get("min_constituent_ratio", 0.0)),
        diversification_penalty=float(opt_cfg.get("diversification_penalty", 0.05)),
        style_bound=(
            _parse_style_bound(opt_cfg["style_bound"])
            if opt_cfg.get("style_bound") is not None else None
        ),
        max_turnover=_optional_float(opt_cfg, "max_turnover"),
        turnover_penalty=float(opt_cfg.get("turnover_penalty", 0.0)),
        risk_aversion=risk_aversion,
    )
    return AlphaMaxOptimizer(base_config), base_config


def _parse_run_config(cfg: dict[str, Any]) -> _RunConfig:
    """校验策略并解析回测区间，打印运行头。"""
    strategy = cfg["strategy"]          # "index_enhance" | "alpha_max"
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"strategy 须为 {sorted(_STRATEGIES)} 之一，当前为 {strategy!r}"
        )
    index    = cfg["index"]
    bt_cfg   = cfg["backtest"]
    uni_cfg  = cfg["universe"]
    run_cfg = _RunConfig(
        strategy=strategy,
        index=index,
        start_date=date.fromisoformat(bt_cfg["start_date"]),
        end_date=date.fromisoformat(bt_cfg["end_date"]),
        rebal_freq=int(bt_cfg["rebalance_freq"]),
        initial_value=float(bt_cfg["initial_value"]),
        universe_cfg=uni_cfg,
        optimizer_cfg=cfg["optimizer"],
        alpha_cfg=cfg["alpha"],
        execution_cfg=cfg.get("execution", {}),
        output_path=Path(cfg["output"]["weights"]),
    )

    index_name = _INDEX_NAMES.get(index, index.upper())
    logger.info(f"\n{'='*65}")
    logger.info(
        f"  {index_name} {strategy} 批量优化  "
        f"{run_cfg.start_date} ~ {run_cfg.end_date}"
    )
    logger.info(f"  调仓={run_cfg.rebal_freq}日  候选池: 剔除北交所+ST"
          + (f"  TOP_N={uni_cfg['top_n']}" if uni_cfg.get("top_n") else "  全市场"))
    logger.info(f"{'='*65}")
    return run_cfg


def _build_risk_model(
    opt_cfg: dict[str, Any],
    rebal_dates: list[date],
) -> tuple[CNE6RiskModel, float | None, float]:
    """加载 CNE6 风险模型，返回 (模型, risk_aversion, min_risk_coverage)。

    CNE6 因子风险模型恒为风格源：S的20个或L的16个风格因子暴露用于
    style_active_bound 约束。
    risk_aversion 设置时，因子协方差 λ·active'Σactive 进目标（真跟踪误差）；
    不设时退回 L2 偏离惩罚 tracking_penalty。
    cne6_data_dir：风险面板来源目录，默认 None → CNE6RiskModel 默认路径
    （短周期 CNE6S，data/barra_cne6_S/）；传 "data/barra_cne6_L" 则改用长周期
    CNE6L 面板（hl=252，月度以上策略）。
    """
    risk_aversion = _optional_float(opt_cfg, "risk_aversion")
    min_risk_coverage = float(opt_cfg.get("min_risk_coverage", 0.5))
    if not 0.0 <= min_risk_coverage <= 1.0:
        raise ValueError("optimizer.min_risk_coverage 必须位于 [0, 1]")

    cne6_data_dir = opt_cfg.get("cne6_data_dir") or None
    # 传入调仓日：只把这些日期 as-of 命中的暴露截面读进内存（完整面板整表
    # 常驻约需 2.7GB，按需加载约减半）。
    risk_model = CNE6RiskModel(data_dir=cne6_data_dir, query_dates=rebal_dates)
    cov0, cov1 = risk_model.coverage
    tag = Path(cne6_data_dir).name if cne6_data_dir else "barra_cne6_S(默认/短周期S)"
    mode = f"λ={risk_aversion}" if risk_aversion is not None else "L2 偏离惩罚"
    logger.info(f"\n[3a] CNE6 风险模型[{tag}]  覆盖={cov0}~{cov1}  目标风险项={mode}")
    return risk_model, risk_aversion, min_risk_coverage


def _build_benchmark(
    run_cfg: _RunConfig,
    panel: pl.DataFrame,
) -> IndexBenchmarkWeights | None:
    """指数增强预计算基准权重；量化多头无自然基准，返回 None。"""
    if run_cfg.strategy != "index_enhance":
        return None
    bm_source = str(
        run_cfg.optimizer_cfg.get("benchmark_weight_source", "official_drift")
    )
    max_snapshot_age_days = run_cfg.optimizer_cfg.get(
        "benchmark_max_snapshot_age_days",
        DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    )
    logger.info(
        f"\n[3] 预计算 {run_cfg.index.upper()} 基准权重（来源={bm_source}，"
        f"快照最大陈旧={max_snapshot_age_days}自然日）..."
    )
    benchmark = IndexBenchmarkWeights(
        index=run_cfg.index,
        panel=panel,
        source=bm_source,
        max_snapshot_age_days=max_snapshot_age_days,
    )
    benchmark.precompute(run_cfg.start_date, run_cfg.end_date, panel=panel)
    return benchmark


def _prepare_inputs(
    config: str | Path | dict[str, Any],
    panel: pl.DataFrame | None,
    alpha_df: pd.DataFrame | None,
) -> _BatchInputs:
    """解析配置、加载数据、构造优化器与风险模型。"""
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    run_cfg = _parse_run_config(cfg)
    opt_cfg = run_cfg.optimizer_cfg

    panel = _load_market_panel(panel, run_cfg.start_date, run_cfg.end_date)

    synthetic_alpha = _synthetic_alpha_enabled(run_cfg.alpha_cfg)
    alpha_df = _load_alpha_matrix(run_cfg.alpha_cfg, alpha_df, panel)
    if synthetic_alpha:
        logger.warning(
            "\n%s\n  ⚠️  合成 Alpha 警告\n  %s%s",
            "=" * 70,
            SYNTHETIC_ALPHA_WARNING_TEXT.replace("\n", "\n  "),
            "=" * 70,
        )
    logger.info(
        f"  Alpha 矩阵: {alpha_df.shape}  "
        f"日期 {alpha_df.index.min().date()}~{alpha_df.index.max().date()}"
    )
    alpha_policy = _build_alpha_policy(run_cfg.alpha_cfg, run_cfg.rebal_freq)

    trade_dates = (
        panel.filter(
            (pl.col("date") >= run_cfg.start_date)
            & (pl.col("date") <= run_cfg.end_date)
        ).select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::run_cfg.rebal_freq]
    logger.info(f"\n  回测交易日数={len(trade_dates)}  再平衡日数={len(rebal_dates)}")

    ledger = ExecutionLedger(
        initial_value=run_cfg.initial_value,
        cost_buy=float(run_cfg.execution_cfg.get("cost_buy", 0.001)),
        cost_sell=float(run_cfg.execution_cfg.get("cost_sell", 0.002)),
    )
    risk_model, risk_aversion, min_risk_coverage = _build_risk_model(
        opt_cfg, rebal_dates
    )
    benchmark = _build_benchmark(run_cfg, panel)
    optimizer, base_config = _build_optimizer(
        run_cfg.strategy, opt_cfg, risk_aversion
    )

    return _BatchInputs(
        strategy=run_cfg.strategy,
        index=run_cfg.index,
        universe_cfg=run_cfg.universe_cfg,
        output_path=run_cfg.output_path,
        synthetic_alpha=synthetic_alpha,
        panel=panel,
        alpha_df=alpha_df,
        alpha_policy=alpha_policy,
        trade_dates=trade_dates,
        rebal_dates=rebal_dates,
        execution_days=_partition_execution_days(
            panel, run_cfg.start_date, run_cfg.end_date
        ),
        ledger=ledger,
        adapter=RealMarketAdapter(
            new_listing_days=int(run_cfg.universe_cfg.get("new_listing_days", 120))
        ),
        risk_model=risk_model,
        optimizer=optimizer,
        base_config=base_config,
        benchmark=benchmark,
        min_risk_coverage=min_risk_coverage,
        use_cost_vector=(
            float(opt_cfg.get("turnover_penalty", 0.0)) > 0
            and bool(opt_cfg.get("liquidity_weighted_cost", True))
        ),
    )


# ──────────────────────────────────────────────────────────────────
# 阶段二：逐期优化
# ──────────────────────────────────────────────────────────────────


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

    任何一期在求解前后失败都不中断整段回测：不发布新权重行，由
    ``replay_signal_day`` 恢复旧目标在该日的正常成交尝试。
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
            walker.replay_signal_day(signal_day)
            continue

        alpha = _resolve_period_alpha(inputs, ctx, rebal_date, outcome.stats)
        if alpha is None:
            walker.replay_signal_day(signal_day)
            continue

        result = _solve_period(inputs, ctx, alpha, rebal_date, outcome.stats)
        if result is None:
            walker.replay_signal_day(signal_day)
            continue

        elapsed = time.time() - t0
        outcome.stats.solve_times.append(elapsed)
        if not result.is_feasible:
            outcome.stats.fail_count += 1
            logger.info(f"  [{rebal_date}] ✗ 求解失败：{result.status}")
            # 未发布新目标时恢复旧目标在当日的正常执行，与重放权重日期保持一致。
            walker.replay_signal_day(signal_day)
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


# ──────────────────────────────────────────────────────────────────
# 阶段三：汇总与发布
# ──────────────────────────────────────────────────────────────────


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
    avg_turnover = (
        float(np.mean(stats.target_turnovers)) if stats.target_turnovers
        else float("nan")
    )
    logger.info(f"  平均目标换手 : {avg_turnover*100:.1f}%（相对实际持仓）")
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

    warning_path = out_path.parent / SYNTHETIC_ALPHA_WARNING_FILE
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


def run_batch_optimize(
    config: str | Path | dict[str, Any],
    panel: pl.DataFrame | None = None,
    alpha_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    读取 YAML 配置，执行批量组合优化，保存权重并返回权重矩阵。

    分三段：``_prepare_inputs`` 解析配置并装配数据与组件，``_run_periods``
    逐期推进成交账本并求解，``_publish_outputs`` 汇总并原子发布 bundle。

    Parameters
    ----------
    config : str | Path | dict
        YAML 配置文件路径（参见 configs/ 目录），或已解析的配置 dict
    panel : pl.DataFrame | None
        预加载行情面板；为 None 时按配置日期范围加载（默认行为）。
        多次调用（如扫描多个 Alpha）时传入同一面板可避免重复加载。
    alpha_df : pd.DataFrame | None
        预加载 Alpha 矩阵（index=date, columns=ticker）；为 None 时按
        配置 alpha.source 加载/生成（默认行为）。

    Returns
    -------
    pd.DataFrame  index=date, columns=ticker, values=weight
    """
    inputs = _prepare_inputs(config, panel, alpha_df)
    outcome = _run_periods(inputs)
    return _publish_outputs(inputs, outcome)
