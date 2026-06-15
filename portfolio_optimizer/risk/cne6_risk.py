"""CNE6 因子风险模型加载器。

消费 BarraCNE6 产出的逐日（防前视）风险面板：
    data/barra_cne6/exposure_panel.parquet   —— 逐调仓日 × 个股 × 47 因子暴露 X + spec_var
    data/barra_cne6/factor_cov_panel.parquet —— 逐调仓日 × 47×47 因子协方差 F

47 因子 = 16 风格 + Country + 30 行业（CITIC L1），由
scripts/export_cne6_panels.py 从 ClickHouse the_quant.cne6_risk 导出。

组合风险：V = X F Xᵀ + diag(δ)，其中 X=暴露(N×K)，F=因子协方差(K×K)，δ=特质方差(N,)。

设计要点：
- HistQuantOpt 不 import BarraCNE6 代码，只读其导出的 parquet（两项目解耦，
  对应"中心化发布 → 只读消费"）。
- as-of 对齐：给定 target_date，取 ≤ target_date 的最近调仓日的风险模型（防前视）；
  早于面板最早日则返回 None，优化器自动退回 L2 惩罚。
- 股票对齐：暴露缺失填 0；特质方差缺失填当期截面中位数（避免 δ=0 让优化器
  把"零特质风险"的票配到极端权重）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# 16 个 CNE6 风格因子（命名与 ClickHouse cne6_risk.factor_exposure 一致）。
# Country 因子（全市场恒为 1）不计入风格，归入非风格因子（与行业同处理，
# 不受 style_active_bound 约束）。
STYLE_FACTORS: tuple[str, ...] = (
    "Size", "MidCap", "Beta", "Momentum", "ResidualVolatility", "LongTermReversal",
    "Liquidity", "Value", "EarningsYield", "Growth", "Profitability",
    "InvestmentQuality", "EarningsQuality", "EarningsVariability", "Leverage",
    "DividendYield",
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "barra_cne6"


@dataclass(frozen=True)
class RiskSnapshot:
    """某调仓日、对齐到给定 tickers 的因子风险模型。"""
    as_of: date                 # 实际取用的面板调仓日（≤ 请求日）
    tickers: list[str]
    factor_names: list[str]     # 50：风格 + 行业，顺序与 X 列、F 行列一致
    X: np.ndarray               # (N, K) 因子暴露
    F: np.ndarray               # (K, K) 因子协方差（对称 PSD）
    delta: np.ndarray           # (N,)   特质方差

    @property
    def style_names(self) -> list[str]:
        return [f for f in self.factor_names if f in STYLE_FACTORS]

    @property
    def industry_names(self) -> list[str]:
        return [f for f in self.factor_names if f not in STYLE_FACTORS]

    def style_loading(self) -> pd.DataFrame:
        """风格暴露子矩阵（N×19，index=tickers），兼容现有风格约束接口。"""
        sidx = [self.factor_names.index(s) for s in self.style_names]
        return pd.DataFrame(
            self.X[:, sidx], index=self.tickers, columns=self.style_names
        )


@lru_cache(maxsize=4)
def _load_panels(data_dir: str) -> tuple[pl.DataFrame, dict, list[date]]:
    """加载并预处理风险面板（按 data_dir 缓存，避免重复 IO）。

    Returns:
        exposure: polars DF（rebal_date, code, <50因子>, spec_var）
        cov_by_date: {rebal_date: (factor_names, F 矩阵 K×K)}
        rebal_dates: 升序调仓日列表
    """
    d = Path(data_dir)
    exposure = pl.read_parquet(d / "exposure_panel.parquet").with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )
    cov = pl.read_parquet(d / "factor_cov_panel.parquet").with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )

    factor_names = [c for c in cov.columns if c not in ("rebal_date", "factor")]
    cov_by_date: dict[date, tuple[list[str], np.ndarray]] = {}
    for rdate, sub in cov.group_by("rebal_date"):
        rdate = rdate[0] if isinstance(rdate, tuple) else rdate
        order = sub["factor"].to_list()
        mat = sub.select(order).to_numpy().astype(np.float64)
        # 对齐到统一 factor_names 顺序
        pos = [order.index(f) for f in factor_names]
        F = mat[np.ix_(pos, pos)]
        F = 0.5 * (F + F.T)  # 数值对称化
        cov_by_date[rdate] = (factor_names, F)

    rebal_dates = sorted(cov_by_date.keys())
    return exposure, cov_by_date, rebal_dates


class CNE6RiskModel:
    """CNE6 因子风险模型查询器：加载一次，按调仓日多次查询。"""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
        self._exposure, self._cov_by_date, self._rebal_dates = _load_panels(
            self.data_dir
        )

    @property
    def rebal_dates(self) -> list[date]:
        return self._rebal_dates

    @property
    def coverage(self) -> tuple[date, date]:
        return self._rebal_dates[0], self._rebal_dates[-1]

    def _asof_date(self, target_date: date) -> date | None:
        """≤ target_date 的最近调仓日（防前视）；早于最早日返回 None。"""
        eligible = [d for d in self._rebal_dates if d <= target_date]
        return eligible[-1] if eligible else None

    def at(self, target_date: date, tickers: list[str]) -> RiskSnapshot | None:
        """取对齐到 tickers 的风险模型；无可用历史则返回 None。"""
        asof = self._asof_date(target_date)
        if asof is None:
            return None

        factor_names, F = self._cov_by_date[asof]

        day = (
            self._exposure.filter(pl.col("rebal_date") == asof)
            .to_pandas()
            .set_index("code")
        )
        # 当期截面特质方差中位数，用于填补 universe 差异导致的缺失
        spec_median = float(np.nanmedian(day["spec_var"].to_numpy()))

        aligned = day.reindex(tickers)
        X = aligned[list(factor_names)].fillna(0.0).to_numpy().astype(np.float64)
        delta = aligned["spec_var"].fillna(spec_median).to_numpy().astype(np.float64)

        return RiskSnapshot(
            as_of=asof,
            tickers=list(tickers),
            factor_names=list(factor_names),
            X=X,
            F=F,
            delta=delta,
        )
