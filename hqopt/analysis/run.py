"""调仓权重 → CNE6 因子收益归因 → 归因报告。

供 CLI（`hqopt attribute`）复用。权重文件支持：
  - 长表 parquet：列 [date, code, weight]
  - 宽表 parquet：index=date, columns=ticker
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from hqopt.analysis.attribution import AttributionResult, ReturnAttributor
from hqopt.data.benchmark import IndexBenchmarkWeights
from hqopt.io.data_panel import load_panel
from hqopt.risk import CNE6RiskModel, FactorReturnLoader

logger = logging.getLogger(__name__)

# 有官方成分权重的指数；其余 index 值（如 all/csiall）退回全市场等权基准
_CONSTITUENT_INDICES = {"hs300", "zz500", "zz1000"}

# 覆盖率低于此阈值的交易日会被单独提示（见 attribution.py 顶部"已知局限"）
_LOW_COVERAGE_WARN = 0.8


def _load_weights(path: str | Path) -> pd.DataFrame:
    """加载权重，统一输出宽表（index=DatetimeIndex，columns=ticker）。"""
    df = pd.read_parquet(path)
    if {"date", "code", "weight"}.issubset(df.columns):       # 长表
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="code", values="weight").sort_index()
    else:                                                      # 宽表
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    df.index.name = "date"
    return df


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


def _equal_weight_benchmark(adj_close: pd.DataFrame) -> pd.DataFrame:
    """全市场等权基准：与 RealisticBacktester 无基准时的默认口径一致
    （见 backtest/engine.py 的 ``benchmark_ret is None`` 分支）。"""
    valid = (adj_close.notna() & (adj_close > 0)).astype(float)
    row_sum = valid.sum(axis=1)
    return valid.div(row_sum.replace(0, pd.NA), axis=0).fillna(0.0)


def _parse_date(s: str | date | pd.Timestamp) -> date:
    if isinstance(s, date):
        return s
    return pd.Timestamp(s).date()


def run_attribution(
    weight_path: str | Path,
    start_date: str | date,
    end_date: str | date,
    index: str = "zz1000",
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cne6_data_dir: str | Path | None = None,
) -> AttributionResult:
    """
    加载权重文件，执行 CNE6 因子收益归因。

    Parameters
    ----------
    weight_path : 权重 parquet（长表或宽表）
    start_date / end_date : 归因区间
    index       : 基准。hs300/zz500/zz1000 用官方成分权重（同 backtest 口径）；
                  其余（如 all/csiall，量化多头常用）退回全市场等权基准。
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
    weight_df = _load_weights(weight_path)
    weight_df = weight_df[
        (weight_df.index >= pd.Timestamp(t1)) & (weight_df.index <= pd.Timestamp(t2))
    ]
    if weight_df.empty:
        raise ValueError(f"权重文件在 {t1}~{t2} 无数据，请检查日期区间")
    logger.info(
        f"  调仓日={len(weight_df)}  "
        f"区间={weight_df.index.min().date()}~{weight_df.index.max().date()}"
    )

    # ── 2. 行情面板 ──────────────────────────────────────────
    data_start = date(t1.year, 1, 1)
    logger.info(f"\n[2] 加载行情面板（{data_start} ~ {t2}）...")
    panel = load_panel(
        data_start, t2, columns=["code", "date", "adj_close"], cache_dir=cache_dir,
    )
    adj_close = _to_wide(panel, "adj_close")
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 3. 基准权重 ──────────────────────────────────────────
    logger.info(f"\n[3] 构建基准权重（{index}）...")
    if index in _CONSTITUENT_INDICES:
        bm = IndexBenchmarkWeights(index=index)
        bm_matrix = bm.get_weights_matrix(list(weight_df.index.date), tickers=None)
        bm_matrix.index = pd.to_datetime(bm_matrix.index)
    else:
        logger.info("  非成分指数，退回全市场等权基准")
        bm_matrix = _equal_weight_benchmark(adj_close)

    # ── 4. CNE6 风险模型 + 因子收益 ──────────────────────────
    tag = Path(cne6_data_dir).name if cne6_data_dir else "barra_cne6(默认/短周期S)"
    logger.info(f"\n[4] 加载 CNE6 风险模型[{tag}] / 因子收益(cne6_risk)...")
    risk_model = CNE6RiskModel(data_dir=cne6_data_dir)
    factor_loader = FactorReturnLoader()

    # ── 5. 归因 ──────────────────────────────────────────────
    logger.info("\n[5] 执行归因（风格/行业/Country/特质分解 + Carino 多期链接）...")
    result = ReturnAttributor(risk_model, factor_loader).run(weight_df, bm_matrix, adj_close)

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
        logger.info(f"\n  归因汇总：{out_path / 'attribution_summary.csv'}")
        logger.info(f"  逐日明细：{out_path / 'attribution_daily.parquet'}")

    return result


__all__ = ["run_attribution"]
