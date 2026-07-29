"""调仓权重 → CNE6 因子收益归因 → 归因报告。

供 CLI（`hqopt attribute`）复用。权重文件支持：
  - 长表 parquet：列 [date, code, weight]
  - 宽表 parquet：index=date, columns=ticker
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqopt.analysis.attribution import AttributionResult, ReturnAttributor
from hqopt.analysis.report import generate_attribution_html
from hqopt.backtest.engine import RealisticBacktester
from hqopt.backtest.run import (
    _align_sell_only_metadata,
    _load_execution_bundle,
)
from hqopt.backtest.run import (
    _load_weights as _load_weights,
)
from hqopt.data.benchmark import (
    DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    IndexBenchmarkWeights,
    equal_weight_benchmark_weights,
)
from hqopt.io.data_panel import load_panel
from hqopt.risk import CNE6RiskModel, FactorReturnLoader

logger = logging.getLogger(__name__)

# 有官方成分权重的指数；全市场等权必须显式使用 equal_weight，禁止把
# csiall 等指数名称静默解释为另一种基准。
_CONSTITUENT_INDICES = {"hs300", "zz500", "zz1000"}
_EQUAL_WEIGHT_BENCHMARK = "equal_weight"

# 覆盖率低于此阈值的交易日会被单独提示（见 attribution.py 顶部"已知局限"）
_LOW_COVERAGE_WARN = 0.8


def _to_wide(panel: pl.DataFrame, col: str) -> pd.DataFrame:
    """从 polars 面板 pivot 指定列为宽表，index=DatetimeIndex。"""
    wide = (
        panel.select(["date", "code", col]).to_pandas()
        .pivot(index="date", columns="code", values=col)
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide


def _parse_date(s: str | date | pd.Timestamp) -> date:
    if isinstance(s, date):
        return s
    return pd.Timestamp(s).date()


def run_attribution(
    weight_path: str | Path,
    start_date: str | date,
    end_date: str | date,
    index: str = "zz1000",
    benchmark_weight_source: str = "official_drift",
    benchmark_max_snapshot_age_days: int | None = (
        DEFAULT_MAX_SNAPSHOT_AGE_DAYS
    ),
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cne6_data_dir: str | Path | None = None,
    cost_buy: float = 0.001,
    cost_sell: float = 0.002,
    initial_value: float = 1e8,
) -> AttributionResult:
    """
    加载权重文件，执行 CNE6 因子收益归因。

    Parameters
    ----------
    weight_path : 权重 parquet（长表或宽表）
    start_date / end_date : 归因区间
    index       : 基准。equal_weight 用全市场等权；hs300/zz500/zz1000
        用指数成分权重。
    benchmark_weight_source : official_drift（默认）/official_frozen/reconstruct。
    benchmark_max_snapshot_age_days : official_drift 快照最大陈旧自然日数。
    out_dir     : 输出目录，None 则不落地文件
    cache_dir   : 行情 parquet 缓存目录，None 用默认
    cne6_data_dir : CNE6 风险面板目录，None 用默认短周期 S

    Returns
    -------
    AttributionResult
    """
    t1 = _parse_date(start_date)
    t2 = _parse_date(end_date)

    # ── 1. 权重 ──────────────────────────────────────────────
    logger.info(f"\n[1] 加载权重：{weight_path}")
    all_weight_df, sell_only_metadata = _load_execution_bundle(weight_path)
    weight_df = all_weight_df[
        (all_weight_df.index >= pd.Timestamp(t1))
        & (all_weight_df.index <= pd.Timestamp(t2))
    ]
    if weight_df.empty:
        raise ValueError(f"权重文件在 {t1}~{t2} 无数据，请检查日期区间")
    sell_only_df = _align_sell_only_metadata(
        sell_only_metadata,
        all_weight_df,
        weight_path,
    )
    if sell_only_df is not None:
        sell_only_df = sell_only_df.loc[weight_df.index]
    logger.info(
        f"  调仓日={len(weight_df)}  "
        f"区间={weight_df.index.min().date()}~{weight_df.index.max().date()}"
    )

    # ── 2. 行情面板 ──────────────────────────────────────────
    data_start = date(t1.year, 1, 1)
    logger.info(f"\n[2] 加载行情面板（{data_start} ~ {t2}）...")
    execution_columns = [
        "code", "date", "adj_close", "adj_vwap", "close",
        "limit_up", "limit_down", "trade_status",
    ]
    benchmark_columns = [
        "free_mv", "total_mv", "is_hs300", "is_zz500", "is_zz1000",
    ]
    panel = load_panel(
        data_start,
        t2,
        columns=execution_columns + benchmark_columns,
        cache_dir=cache_dir,
    )
    market = {
        column: _to_wide(panel, column)
        for column in execution_columns[2:]
    }
    adj_close = market["adj_close"]
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # 复用真实回测引擎重放目标权重，得到停牌/涨跌停/费用后的逐日实际持仓。
    execution_result, execution_stats = RealisticBacktester(
        cost_buy=cost_buy, cost_sell=cost_sell, risk_free=0.0
    ).run(
        weight_df=weight_df,
        adj_close=adj_close,
        adj_vwap=market["adj_vwap"],
        close_raw=market["close"],
        limit_up_df=market["limit_up"],
        limit_down_df=market["limit_down"],
        trade_status_df=market["trade_status"],
        benchmark_ret=pd.Series(0.0, index=adj_close.index),
        initial_value=initial_value,
        sell_only_df=sell_only_df,
    )
    logger.info(
        "  成交重放：过期订单=%s笔  过期未成交金额=%.2f元",
        execution_stats["expired_order_count"],
        execution_stats["expired_notional"],
    )

    # ── 3. 基准权重 ──────────────────────────────────────────
    logger.info(f"\n[3] 构建基准权重（{index}）...")
    if index in _CONSTITUENT_INDICES:
        bm = IndexBenchmarkWeights(
            index=index,
            panel=panel,
            source=benchmark_weight_source,
            max_snapshot_age_days=benchmark_max_snapshot_age_days,
        )
        bm.precompute(t1, t2, panel=panel)
        bm_matrix = bm.get_weights_matrix(list(weight_df.index.date), tickers=None)
        bm_matrix.index = pd.to_datetime(bm_matrix.index)
    elif index == _EQUAL_WEIGHT_BENCHMARK:
        logger.info("  使用全市场等权基准")
        bm_matrix = equal_weight_benchmark_weights(adj_close)
    else:
        supported = sorted(_CONSTITUENT_INDICES | {_EQUAL_WEIGHT_BENCHMARK})
        raise ValueError(
            f"归因基准 {index!r} 不支持；可选 {supported}。"
            "CSIALL 缺少官方成分权重，不能静默退回等权。"
        )

    # ── 4. CNE6 风险模型 + 因子收益 ──────────────────────────
    tag = Path(cne6_data_dir).name if cne6_data_dir else "barra_cne6_S(默认/短周期S)"
    logger.info(
        f"\n[4] 加载 CNE6 风险模型[{tag}] / "
        "因子收益(test_barra_cne6_gao，同目录同模型)..."
    )
    # 归因按调仓（信号）日取暴露快照，只需读这些日期对应的截面
    risk_model = CNE6RiskModel(
        data_dir=cne6_data_dir,
        query_dates=list(weight_df.index.date),
    )
    factor_loader = FactorReturnLoader(data_dir=cne6_data_dir)

    # ── 5. 归因 ──────────────────────────────────────────────
    logger.info("\n[5] 执行归因（风格/行业/Country/特质分解 + Carino 多期链接）...")
    result = ReturnAttributor(risk_model, factor_loader).run(
        weight_df,
        bm_matrix,
        adj_close,
        actual_weight_df=execution_result.actual_weights,
        realized_portfolio_return=execution_result.daily_ret,
    )

    logger.info(f"\n{'='*60}\n  收益归因  {t1}~{t2}  基准={index}\n{'='*60}")
    logger.info(f"\n{result}")

    low_cov = result.daily[result.daily["coverage_pct"] < _LOW_COVERAGE_WARN]
    if len(low_cov):
        logger.info(
            f"\n  ⚠️ {len(low_cov)} 个交易日风险模型覆盖率 <{_LOW_COVERAGE_WARN*100:.0f}%，"
            f"对应期间归因结果（尤其残差占比）需谨慎解读"
        )

    # ── 6. 输出 ──────────────────────────────────────────────
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(out_path / "attribution_summary.csv", encoding="utf-8-sig")
        result.daily.to_parquet(out_path / "attribution_daily.parquet")
        result.factor_daily.to_parquet(out_path / "attribution_factor_daily.parquet")
        (out_path / "attribution_execution_stats.json").write_text(
            json.dumps(
                execution_stats,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=lambda value: value.item(),
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = generate_attribution_html(
            result,
            out_path / "attribution_report.html",
            benchmark=index,
            model_name=tag,
        )
        logger.info(f"\n  归因汇总：{out_path / 'attribution_summary.csv'}")
        logger.info(f"  逐日明细：{out_path / 'attribution_daily.parquet'}")
        logger.info(
            f"  因子明细：{out_path / 'attribution_factor_daily.parquet'}"
        )
        logger.info(f"  成交统计：{out_path / 'attribution_execution_stats.json'}")
        logger.info(f"  HTML 报告：{report_path}")

    return result


__all__ = ["run_attribution"]
