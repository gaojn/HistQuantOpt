"""
聚源 Barra 风格因子加载器。

从本地 parquet 文件读取已计算好的截面 z-score 风格因子，
与 RealBarraFactors 接口完全兼容，可直接替换用于优化器。

因子列映射（9个风格因子）：
    siz   → Size
    vol   → ResidualVolatility
    liq   → Liquidity
    mom   → Momentum
    qua   → EarningsYield
    val   → BookToPrice
    gro   → Growth
    sen   → Sentiment
    divid → Dividend

行业虚拟变量从行情面板 industry_l1 字段构建。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.data.generator import MarketSnapshot

# 文件内列名 → 框架因子名
_COL_MAP = {
    "siz":   "Size",
    "vol":   "ResidualVolatility",
    "liq":   "Liquidity",
    "mom":   "Momentum",
    "qua":   "EarningsYield",
    "val":   "BookToPrice",
    "gro":   "Growth",
    "sen":   "Sentiment",
    "divid": "Dividend",
}

STYLE_FACTORS = list(_COL_MAP.values())   # 顺序固定


class JYBarraFactors:
    """
    聚源 Barra 因子加载器，接口与 RealBarraFactors 完全一致。

    Parameters
    ----------
    snapshot : MarketSnapshot
        目标日市场快照（提供 tickers / industry）
    target_date : date
        目标日期
    factor_path : Path | str
        聚源风格因子 parquet 文件路径
    panel : pl.DataFrame | None
        行情面板（用于补充 industry_l1；已在 snapshot 中时可不传）
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        target_date: date,
        factor_path: Path | str,
        panel: pl.DataFrame | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.target_date = target_date
        self._factor_path = Path(factor_path)
        self._loading = self._build_loading(snapshot, panel)

    # ------------------------------------------------------------------
    # Public interface（与 RealBarraFactors 一致）
    # ------------------------------------------------------------------

    @property
    def factor_names(self) -> list[str]:
        return self._loading.columns.tolist()

    @property
    def style_names(self) -> list[str]:
        return STYLE_FACTORS

    @property
    def industry_names(self) -> list[str]:
        return [c for c in self._loading.columns if c not in STYLE_FACTORS]

    @property
    def loading(self) -> pd.DataFrame:
        return self._loading

    @property
    def style_loading(self) -> pd.DataFrame:
        return self._loading[STYLE_FACTORS]

    @property
    def industry_loading(self) -> pd.DataFrame:
        return self._loading[self.industry_names]

    # ------------------------------------------------------------------
    # 内部构建
    # ------------------------------------------------------------------

    def _build_loading(
        self,
        snapshot: MarketSnapshot,
        panel: pl.DataFrame | None,
    ) -> pd.DataFrame:
        tickers = snapshot.tickers

        # ---- 读取目标日风格因子 ----
        target_ts = pd.Timestamp(self.target_date)
        raw = pd.read_parquet(self._factor_path)
        day_df = (
            raw[raw["trade_dt"] == target_ts]
            .set_index("s_info_windcode")
            [list(_COL_MAP.keys())]
            .rename(columns=_COL_MAP)
        )

        # 对齐到 tickers（缺失的股票风格因子填0）
        style_df = day_df.reindex(tickers).fillna(0.0)

        # ---- 行业虚拟变量 ----
        ind_df = self._build_industry_dummies(tickers, snapshot, panel)

        return pd.concat([style_df, ind_df], axis=1)

    @staticmethod
    def _build_industry_dummies(
        tickers: list[str],
        snapshot: MarketSnapshot,
        panel: pl.DataFrame | None,
    ) -> pd.DataFrame:
        """优先用 snapshot.industry，缺失时从 panel 补充。"""
        ind = snapshot.industry.reindex(tickers)

        # 若 snapshot.industry 覆盖不全，从 panel 补
        if panel is not None and ind.isna().any():
            missing = ind[ind.isna()].index.tolist()
            today = (
                panel
                .filter(pl.col("code").is_in(missing))
                .filter(pl.col("date") == snapshot.industry.name or True)
                .select(["code", "industry_l1"])
                .unique(subset=["code"])
                .to_pandas()
                .set_index("code")["industry_l1"]
            )
            ind = ind.fillna(today)

        ind = ind.fillna("未知")
        dummies = pd.get_dummies(ind, dtype=float)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        return dummies
