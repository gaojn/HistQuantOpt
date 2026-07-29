"""
批量组合优化 pipeline。

支持 index_enhance（指数增强）和 alpha_max（量化多头）两种策略。
通过 YAML 配置文件驱动，不依赖具体 demo 脚本。

三段式：``_prepare_inputs``（本模块）→ ``_run_periods``（``batch.periods``）
→ ``_publish_outputs``（``batch.publish``）。

**阶段一为什么留在本模块**：测试对 ``CNE6RiskModel`` / ``AlphaMaxOptimizer`` /
``IndexEnhanceOptimizer`` / ``IndexBenchmarkWeights`` / ``ExecutionLedger`` 的
monkeypatch 打在本模块命名空间上（46 处）。构造这些对象的代码若移入子模块，
会在子模块命名空间解析这些名字，patch 全部静默失效——失败模式是测试变空转却
依旧显示通过。拆分边界与取舍见 docs/design.md §6.1。

文件末尾另有 4 个只为兼容而 re-export 的符号，见那里的说明。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from hqopt.backtest.execution import ExecutionLedger
from hqopt.constants import SYNTHETIC_ALPHA_WARNING_TEXT
from hqopt.data.benchmark import (
    DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    IndexBenchmarkWeights,
)
from hqopt.data.real_adapter import RealMarketAdapter
from hqopt.io.data_panel import load_panel
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer
from hqopt.pipeline.batch.config import (
    _ALPHA_SOURCES,
    _build_alpha_policy,
    _optional_float,
    _parse_run_config,
    _parse_style_bound,
    _synthetic_alpha_enabled,
    load_config,
)
from hqopt.pipeline.batch.execution_walk import _partition_execution_days
from hqopt.pipeline.batch.periods import _run_periods
from hqopt.pipeline.batch.publish import _publish_outputs
from hqopt.pipeline.batch.types import _BatchInputs, _RunConfig
from hqopt.pipeline.universe import build_synthetic_alpha, load_alpha_panel
from hqopt.risk import CNE6RiskModel

logger = logging.getLogger(__name__)

__all__ = ["load_config", "run_batch_optimize"]


# ──────────────────────────────────────────────────────────────────
# 阶段一：准备输入（含全部可 monkeypatch 的构造点，勿外移）
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


# ──────────────────────────────────────────────────────────────────
# 兼容 re-export
# ──────────────────────────────────────────────────────────────────
# 这 4 个符号已随拆分移入 batch/ 子模块，本模块自身不再使用，但既有测试仍按
# ``batch_optimize.<name>`` 访问（``import hqopt.pipeline.batch_optimize as batch``）。
# 保留导入以维持对外契约；F401 为此显式豁免，删除会破坏调用方。
#
# 注意：真正**不能**外移的是构造 CNE6RiskModel / AlphaMaxOptimizer /
# IndexEnhanceOptimizer / IndexBenchmarkWeights / ExecutionLedger 的代码，
# 它们被 monkeypatch 打在本模块命名空间上——那些留在了上方阶段一，不在此列。
from hqopt.constants import SYNTHETIC_ALPHA_WARNING_FILE  # noqa: F401,E402
from hqopt.pipeline.batch.execution_walk import _ExecutionWalker  # noqa: F401,E402
from hqopt.pipeline.batch.periods import (  # noqa: E402
    _DUST_WEIGHT_TOL,  # noqa: F401
    _clean_target_weights,  # noqa: F401
)
