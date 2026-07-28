"""因子收益 / 特质收益加载器（收益归因用）。

消费 scripts/export_factor_attribution.py 从 ClickHouse test_barra_cne6_gao
对应 S/L 模型导出的：
    data/barra_cne6[_L]/factor_return.parquet    —— trade_date, factor_name, ret
    data/barra_cne6[_L]/specific_return.parquet  —— trade_date, code, u

与 hqopt.risk.cne6_risk.CNE6RiskModel 消费的 exposure_panel.parquet
同源（均来自 test_barra_cne6_gao 的对应 S/L 模型），因子名对齐，供 hqopt.analysis.attribution
做「暴露 X × 因子收益 f」+「特质收益 u」的归因分解。
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "barra_cne6"


@lru_cache(maxsize=4)
def _load_factor_return(data_dir: str) -> pd.DataFrame:
    """加载并 pivot 为宽表：index=trade_date, columns=factor_name。"""
    df = pl.read_parquet(Path(data_dir) / "factor_return.parquet").with_columns(
        pl.col("trade_date").cast(pl.Date)
    )
    wide = df.to_pandas().pivot(index="trade_date", columns="factor_name", values="ret")
    wide.index = pd.to_datetime(wide.index).date
    return wide.sort_index()


@lru_cache(maxsize=4)
def _load_specific_return(data_dir: str) -> pl.DataFrame:
    """加载特质收益长表（按 trade_date 过滤，不整体 pivot，避免稀疏大宽表）。"""
    return pl.read_parquet(Path(data_dir) / "specific_return.parquet").with_columns(
        pl.col("trade_date").cast(pl.Date)
    )


class FactorReturnLoader:
    """按交易日查询因子收益 f 与个股特质收益 u（均为已实现值）。"""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
        self._factor_ret = _load_factor_return(self.data_dir)
        self._spec_ret = _load_specific_return(self.data_dir)

    @property
    def trading_dates(self) -> list[date]:
        return list(self._factor_ret.index)

    @property
    def factor_names(self) -> list[str]:
        return list(self._factor_ret.columns)

    def factor_return(self, trade_date: date, factor_names: list[str] | None = None) -> pd.Series:
        """某日各因子已实现收益，index=factor_name。无该日数据返回全 0。

        factor_names 提供时对齐到该顺序（缺失填 0，如「未知」行业不在此列出）。
        """
        if trade_date in self._factor_ret.index:
            row = self._factor_ret.loc[trade_date]
        else:
            row = pd.Series(0.0, index=self._factor_ret.columns)
        row = row.fillna(0.0)
        if factor_names is not None:
            row = row.reindex(factor_names).fillna(0.0)
        return row

    def specific_return(self, trade_date: date, tickers: list[str]) -> pd.Series:
        """某日个股特质收益，对齐到 tickers（缺失填 0：视为当日无特质信息）。"""
        day = (
            self._spec_ret.filter(pl.col("trade_date") == trade_date)
            .to_pandas()
            .set_index("code")["u"]
        )
        return day.reindex(tickers).fillna(0.0).astype(np.float64)


__all__ = ["FactorReturnLoader", "DEFAULT_DATA_DIR"]
