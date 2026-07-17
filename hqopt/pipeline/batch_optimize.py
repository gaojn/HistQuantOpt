"""
批量组合优化 pipeline。

支持 index_enhance（指数增强）和 alpha_max（量化多头）两种策略。
通过 YAML 配置文件驱动，不依赖具体 demo 脚本。
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from hqopt.data.benchmark import IndexBenchmarkWeights
from hqopt.data.real_adapter import RealMarketAdapter
from hqopt.backtest.execution import ExecutionLedger
from hqopt.constants import (
    SYNTHETIC_ALPHA_WARNING_FILE,
    SYNTHETIC_ALPHA_WARNING_TEXT,
)
from hqopt.io.data_panel import load_panel
from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer
from hqopt.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer
from hqopt.pipeline.universe import (
    build_cost_vector, build_synthetic_alpha, filter_universe, get_alpha_for_date,
    load_alpha_panel,
)
from hqopt.risk import CNE6RiskModel

logger = logging.getLogger(__name__)

_INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}
_STRATEGIES = {"index_enhance", "alpha_max"}
_ALPHA_SOURCES = {"file", "synthetic"}

# 求解器数值粉尘阈值：低于该权重的目标置零后重归一，避免账本按 min_notional
# 买入微量仓位（此类仓位一旦退市即成为永久滞留资产）。
_DUST_WEIGHT_TOL = 1e-6
# 滞留持仓"显著"阈值：仅用于告警分级展示，不影响处理逻辑。
_STUCK_WEIGHT_TOL = 1e-4

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


def _advance_execution_day(ledger: ExecutionLedger, day: pl.DataFrame) -> None:
    """把一个 polars 日截面送入共享成交账本。"""
    pdf = day.drop("date").to_pandas().set_index("code")
    ledger.step(
        adj_close=pdf["adj_close"],
        adj_vwap=pdf["adj_vwap"],
        close_raw=pdf["close"],
        limit_up=pdf["limit_up"],
        limit_down=pdf["limit_down"],
        trade_status=pdf["trade_status"],
    )


def _parse_style_bound(v: Any) -> "float | dict[str, float]":
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


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch_optimize(
    config: str | Path | dict[str, Any],
    panel: pl.DataFrame | None = None,
    alpha_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    读取 YAML 配置，执行批量组合优化，保存权重并返回权重矩阵。

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
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    strategy  = cfg["strategy"]          # "index_enhance" | "alpha_max"
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"strategy 须为 {sorted(_STRATEGIES)} 之一，当前为 {strategy!r}"
        )
    index     = cfg["index"]
    bt_cfg    = cfg["backtest"]
    uni_cfg   = cfg["universe"]
    opt_cfg   = cfg["optimizer"]
    alpha_cfg = cfg["alpha"]
    out_cfg   = cfg["output"]

    start_date = date.fromisoformat(bt_cfg["start_date"])
    end_date   = date.fromisoformat(bt_cfg["end_date"])
    rebal_freq = int(bt_cfg["rebalance_freq"])
    port_val   = float(bt_cfg["initial_value"])

    index_name = _INDEX_NAMES.get(index, index.upper())
    logger.info(f"\n{'='*65}")
    logger.info(f"  {index_name} {strategy} 批量优化  {start_date} ~ {end_date}")
    logger.info(f"  调仓={rebal_freq}日  候选池: 剔除北交所+ST"
          + (f"  TOP_N={uni_cfg['top_n']}" if uni_cfg.get("top_n") else "  全市场"))
    logger.info(f"{'='*65}")

    # ── 加载行情 ─────────────────────────────────────────────
    data_start = date(start_date.year, 1, 1)
    if panel is None:
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
    else:
        logger.info(f"\n[1] 使用预加载行情数据  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── Alpha ────────────────────────────────────────────────
    synthetic_alpha = _synthetic_alpha_enabled(alpha_cfg)
    if alpha_df is not None:
        logger.info("\n[2] 使用预加载 Alpha 矩阵")
    else:
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
            alpha_df = load_alpha_panel(alpha_cfg["path"])
        else:
            alpha_df = build_synthetic_alpha(
                panel,
                fwd_days=int(alpha_cfg["fwd_days"]),
                ic_mean=float(alpha_cfg["ic_mean"]),
                ic_std=float(alpha_cfg["ic_std"]),
                decay=float(alpha_cfg["decay"]),
                seed=int(alpha_cfg["seed"]),
            )
    if synthetic_alpha:
        logger.warning(
            "\n%s\n  ⚠️  合成 Alpha 警告\n  %s%s",
            "=" * 70,
            SYNTHETIC_ALPHA_WARNING_TEXT.replace("\n", "\n  "),
            "=" * 70,
        )
    logger.info(f"  Alpha 矩阵: {alpha_df.shape}  日期 {alpha_df.index.min().date()}~{alpha_df.index.max().date()}")

    # ── 再平衡日 ─────────────────────────────────────────────
    trade_dates = (
        panel.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        ).select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::rebal_freq]
    logger.info(f"\n  回测交易日数={len(trade_dates)}  再平衡日数={len(rebal_dates)}")

    execution_cfg = cfg.get("execution", {})
    execution_ledger = ExecutionLedger(
        initial_value=port_val,
        cost_buy=float(execution_cfg.get("cost_buy", 0.001)),
        cost_sell=float(execution_cfg.get("cost_sell", 0.002)),
    )
    execution_days = _partition_execution_days(panel, start_date, end_date)
    execution_cursor = 0

    # ── 优化器 ───────────────────────────────────────────────
    adapter = RealMarketAdapter()

    # CNE6 因子风险模型恒为风格源：16 风格因子暴露用于 style_active_bound 约束。
    # risk_aversion 设置时，因子协方差 λ·active'Σactive 进目标（真跟踪误差）；
    # 不设时退回 L2 偏离惩罚 tracking_penalty。
    # cne6_data_dir：风险面板来源目录，默认 None → CNE6RiskModel 默认路径
    # （短周期 CNE6S，data/barra_cne6/）；传 "data/barra_cne6_L" 则改用长周期
    # CNE6L 面板（hl=252，月度以上策略）。
    risk_aversion = _optional_float(opt_cfg, "risk_aversion")
    min_risk_coverage = float(opt_cfg.get("min_risk_coverage", 0.90))
    if not 0.0 <= min_risk_coverage <= 1.0:
        raise ValueError("optimizer.min_risk_coverage 必须位于 [0, 1]")
    cne6_data_dir = opt_cfg.get("cne6_data_dir") or None
    cne6_rm = CNE6RiskModel(data_dir=cne6_data_dir)
    cov0, cov1 = cne6_rm.coverage
    tag = Path(cne6_data_dir).name if cne6_data_dir else "barra_cne6(默认/短周期S)"
    mode = f"λ={risk_aversion}" if risk_aversion is not None else "L2 偏离惩罚"
    logger.info(f"\n[3a] CNE6 风险模型[{tag}]  覆盖={cov0}~{cov1}  目标风险项={mode}")

    if strategy == "index_enhance":
        bm_source = str(opt_cfg.get("benchmark_weight_source", "official"))
        logger.info(f"\n[3] 预计算 {index.upper()} 基准权重（来源={bm_source}）...")
        bm = IndexBenchmarkWeights(index=index, panel=panel, source=bm_source)
        bm.precompute(start_date, end_date, panel=panel)

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
        optimizer = IndexEnhanceOptimizer(base_config)

    else:  # alpha_max
        base_config = AlphaMaxConfig(
            weight_upper=float(opt_cfg["weight_upper"]),
            industry_upper=float(opt_cfg.get("industry_upper", 0.20)),
            min_constituent_ratio=float(opt_cfg.get("min_constituent_ratio", 0.0)),
            diversification_penalty=float(opt_cfg.get("diversification_penalty", 0.05)),
            style_bound=_parse_style_bound(opt_cfg["style_bound"]) if opt_cfg.get("style_bound") is not None else None,
            max_turnover=_optional_float(opt_cfg, "max_turnover"),
            turnover_penalty=float(opt_cfg.get("turnover_penalty", 0.0)),
            risk_aversion=risk_aversion,
        )
        optimizer = AlphaMaxOptimizer(base_config)

    use_cost_vector = (
        float(opt_cfg.get("turnover_penalty", 0.0)) > 0
        and bool(opt_cfg.get("liquidity_weighted_cost", True))
    )

    # ── 逐期优化 ─────────────────────────────────────────────
    logger.info("\n[4] 逐期优化...")
    t_total = time.time()
    weight_records: dict = {}
    has_prior_target = False
    fail_count = 0
    solve_times = []
    target_turnovers: list[float] = []

    for i, rebal_date in enumerate(rebal_dates):
        t0 = time.time()

        # 先推进到本调仓日收盘：上一目标的 T+1 成交及所有延期订单均已反映在账本中。
        while execution_cursor < len(trade_dates) and trade_dates[execution_cursor] <= rebal_date:
            execution_date = trade_dates[execution_cursor]
            day = execution_days.get(execution_date)
            if day is None:
                raise RuntimeError(f"{execution_date} 缺少成交行情截面")
            _advance_execution_day(execution_ledger, day)
            execution_cursor += 1

        actual_holdings = execution_ledger.actual_weights()

        try:
            snap_full = adapter.build_snapshot_from_panel(
                panel=panel, target_date=rebal_date,
                index=index, portfolio_value=execution_ledger.nav,
            )
        except ValueError as e:
            logger.info(f"  [{rebal_date}] 跳过（快照失败：{e}）")
            continue

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

        snapshot = filter_universe(
            snap_full, panel, rebal_date,
            exclude_bj=bool(uni_cfg.get("exclude_bj", True)),
            exclude_st=bool(uni_cfg.get("exclude_st", True)),
            top_n=int(uni_cfg["top_n"]) if uni_cfg.get("top_n") else None,
            prev_holdings=actual_holdings if has_prior_target else None,
        )

        # 风格载荷 + 风险模型：均来自 CNE6 面板（16 风格 + 行业）。
        # 暴露用于 style_active_bound 约束；risk_aversion 设置时协方差进目标。
        # 无 CNE6 覆盖的调仓日跳过。
        risk_snap = cne6_rm.at(rebal_date, snapshot.tickers)
        if risk_snap is None:
            logger.info(f"  [{rebal_date}] 跳过（CNE6 风险面板无覆盖）")
            continue
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
        style_loading = risk_snap.style_loading()

        alpha = get_alpha_for_date(alpha_df, rebal_date, snapshot.tickers)

        # 上期权重来自实际成交账本，现金保留为权重缺口，不做重新归一化。
        if has_prior_target:
            ps = actual_holdings.reindex(snapshot.tickers).fillna(0.0).values
        else:
            ps = None

        # 个股冲击成本权重（仅在启用换手软惩罚时计算）
        cost_vec = None
        if use_cost_vector and ps is not None:
            cost_vec = build_cost_vector(
                tickers=snapshot.tickers,
                panel=panel,
                target_date=rebal_date,
            )

        # 优化
        if strategy == "index_enhance":
            bm_series = bm.get_weights(rebal_date, tickers=snapshot.tickers)
            bm_weight = bm_series.values
            bm_total = float(bm_series.sum())
            covered_weight = (
                float(bm_series.values[risk_snap.covered_mask].sum()) / bm_total
                if bm_total > 1e-12 else 0.0
            )
            if covered_weight < min_risk_coverage:
                # 覆盖率异常多为面板数据阶段性缺口（如 2020 年初），跳过该期
                # 并保留账本原目标继续执行；不中断整段回测。
                fail_count += 1
                logger.warning(
                    f"  [{rebal_date}] 跳过优化：基准风险覆盖率 {covered_weight:.2%} "
                    f"低于阈值 {min_risk_coverage:.2%}（零暴露填充不可信）"
                )
                continue

            cfg_this = base_config if ps is not None else IndexEnhanceConfig(
                **{**base_config.__dict__, "max_turnover": None}
            )
            optimizer.config = cfg_this
            result = optimizer.optimize(
                alpha=alpha, snapshot=snapshot,
                benchmark_weight=bm_weight,
                style_loading=style_loading,
                prev_weight=ps,
                cost_vector=cost_vec,
                risk_snapshot=risk_snap,
            )
        else:
            result = optimizer.optimize(
                alpha, snapshot,
                style_loading=style_loading,
                prev_weight=ps,
                cost_vector=cost_vec,
                risk_snapshot=risk_snap,
            )

        elapsed = time.time() - t0
        solve_times.append(elapsed)

        if result.is_feasible:
            w = pd.Series(result.weights, index=snapshot.tickers)
            # 清除求解器数值粉尘再重归一：微量权重会被账本按 min_notional 真实
            # 买入，退市后成为永久滞留资产。
            w = w.where(w >= _DUST_WEIGHT_TOL, 0.0)
            w_sum = float(w.sum())
            if w_sum > 1e-8:
                w = w / w_sum
            weight_records[rebal_date] = w
            execution_ledger.submit_target(w)
            has_prior_target = True

            turnover = float(np.abs(result.weights - ps).sum()) if ps is not None else float("nan")
            if ps is not None:
                target_turnovers.append(turnover)
            if i % 10 == 0 or i == len(rebal_dates) - 1:
                extra = ""
                if strategy == "index_enhance":
                    const_w = result.weights[snapshot.constituent_mask].sum()
                    te_l2 = result.tracking_error_l2()
                    extra = f"  {index.upper()}={const_w*100:.1f}%  TE_L2={te_l2:.4f}"
                logger.info(f"  [{i+1:3d}/{len(rebal_dates)}] {rebal_date}  "
                      f"持仓={result.n_positions:3d}  换手={turnover*100:>5.1f}%{extra}  耗时={elapsed:.2f}s")
        else:
            fail_count += 1
            logger.info(f"  [{rebal_date}] ✗ 求解失败：{result.status}")
            # 不提交新目标：成交账本继续追踪上一个尚未完成的目标，避免用失败回退
            # 覆盖掉真实的延期订单状态。

    # ── 汇总 & 保存 ──────────────────────────────────────────
    if not weight_records:
        raise RuntimeError("所有期均求解失败，请检查配置")

    weight_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weight_df.index.name = "date"
    logger.info(f"\n{'='*65}\n  批量优化汇总\n{'='*65}")
    logger.info(f"  再平衡期数   : {len(weight_df)}")
    logger.info(f"  失败期数     : {fail_count}")
    logger.info(f"  平均持仓数   : {(weight_df > 1e-6).sum(axis=1).mean():.0f} 只")
    avg_target_turnover = float(np.mean(target_turnovers)) if target_turnovers else float("nan")
    logger.info(f"  平均目标换手 : {avg_target_turnover*100:.1f}%（相对实际持仓）")
    logger.info(f"  平均耗时     : {np.mean(solve_times):.2f}s  总耗时: {time.time()-t_total:.1f}s")

    out_path = Path(out_cfg["weights"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weight_df.to_parquet(out_path)
    warning_path = out_path.parent / SYNTHETIC_ALPHA_WARNING_FILE
    if synthetic_alpha:
        warning_path.write_text(SYNTHETIC_ALPHA_WARNING_TEXT, encoding="utf-8")
    else:
        warning_path.unlink(missing_ok=True)
    logger.info(f"\n  权重矩阵已保存：{out_path}")
    logger.info(f"\n{'='*65}\n")

    return weight_df
