"""
指数基准权重计算器（分级靠档加权）。

依据中证指数有限公司《指数编制通则》分级靠档规则：

    Step 1  计算自由流通比例
        f = free_shares / total_shares
          ≈ free_mv / total_mv  （价格相消，数值等价）

    Step 2  向上靠档到最近的 10% 整数倍，得到调整系数 A
        f ∈ (0, 10%]  → A = 10%
        f ∈ (10%, 20%] → A = 20%
        ...
        f ∈ (70%, 80%] → A = 80%
        f > 80%         → A = 100%（按全流通处理）

        公式：A = ceil(f × 10) / 10，其中 f > 80% 时取 1.0

    Step 3  计算调整后市值
        adjusted_mv_i = total_mv_i × A_i

    Step 4  截面归一化
        w_i = adjusted_mv_i / Σ_j adjusted_mv_j，j 为全部成分股

    停牌处理：停牌日 total_mv 与 free_mv 均前向填充（价格冻结）。
    成分股：直接使用 Wind 数据中的 is_zz500 / is_hs300 / is_zz1000 字段。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel


class IndexBenchmarkWeights:
    """
    指数基准权重计算器（支持 zz500 / hs300 / zz1000）。

    预计算全期每日权重并缓存，避免重复计算。

    Parameters
    ----------
    index : str
        指数代码，'zz500' / 'hs300' / 'zz1000'
    panel : pl.DataFrame | None
        已预加载的行情面板；None 时按需加载。
    cache_dir : Path | str | None
        parquet 缓存目录
    """

    _INDEX_COL = {"zz500": "is_zz500", "hs300": "is_hs300", "zz1000": "is_zz1000"}

    def __init__(
        self,
        index: str = "zz500",
        panel: pl.DataFrame | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        if index not in self._INDEX_COL:
            raise ValueError(f"index 须为 {list(self._INDEX_COL)} 之一")
        self.index = index
        self._col = self._INDEX_COL[index]
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._panel = panel
        self._weight_cache: pd.DataFrame | None = None   # (date, ticker) → weight

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_weights(
        self,
        target_date: date,
        tickers: list[str] | None = None,
    ) -> pd.Series:
        """
        获取目标日基准权重。

        Parameters
        ----------
        target_date : date
            目标交易日
        tickers : list[str] | None
            若提供，返回结果对齐到此列表（缺失填 0）

        Returns
        -------
        pd.Series
            index=ticker，value=权重（合计≈1）
        """
        cache = self._get_or_build_cache()

        if target_date not in cache.index:
            # 找最近一个早于 target_date 的日期
            avail = cache.index[cache.index <= target_date]
            if len(avail) == 0:
                raise ValueError(f"无 {target_date} 之前的基准权重数据")
            target_date = avail[-1]

        w = cache.loc[target_date].dropna()
        w = w[w > 0]

        if tickers is not None:
            w = w.reindex(tickers).fillna(0.0)

        # 强制归一化：部分成分股 free_mv 缺失时权重之和可能 < 1
        total = w.sum()
        if total > 1e-8:
            w = w / total

        return w.rename("bm_weight")

    def get_weights_matrix(
        self,
        dates: list[date],
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        批量获取多日基准权重矩阵。

        Returns
        -------
        pd.DataFrame
            index=date，columns=ticker，values=权重
        """
        result = {}
        for d in dates:
            result[d] = self.get_weights(d, tickers)
        df = pd.DataFrame(result).T
        df.index.name = "date"
        return df.fillna(0.0)

    def precompute(
        self,
        start_date: date,
        end_date: date,
        panel: pl.DataFrame | None = None,
    ) -> None:
        """
        预计算指定日期区间的基准权重并缓存到内存。

        在批量优化前调用，避免每次 get_weights() 触发 I/O。
        """
        if panel is not None:
            self._panel = panel
        self._weight_cache = self._build_weight_matrix(start_date, end_date)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_or_build_cache(self) -> pd.DataFrame:
        if self._weight_cache is not None:
            return self._weight_cache
        # 未预计算时：从已有 panel 临时构建（日期范围宽泛）
        if self._panel is not None:
            dates = (
                self._panel.select("date").unique().sort("date")["date"].to_list()
            )
            start, end = dates[0], dates[-1]
            self._weight_cache = self._build_weight_matrix(start, end)
        else:
            raise RuntimeError(
                "请先调用 precompute() 或在构造函数中传入 panel，"
                "以避免按需 I/O 时的日期范围未知问题。"
            )
        return self._weight_cache

    def _build_weight_matrix(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        构建完整日期 × 股票的权重矩阵（分级靠档加权）。

        Step 1  f = free_mv / total_mv（自由流通比例）
        Step 2  A = ceil(f×10)/10，f>80% 时取 1.0
        Step 3  adjusted_mv = total_mv × A
        Step 4  w_i = adjusted_mv_i / Σ adjusted_mv_j
        停牌日 total_mv / free_mv 均前向填充。
        """
        panel = self._panel
        if panel is None:
            panel = load_panel(
                start_date, end_date,
                columns=["code", "date", "free_mv", "total_mv",
                         "trade_status", self._col],
                cache_dir=self._cache_dir,
            )

        # 筛选成分股
        const = panel.filter(pl.col(self._col) == 1)

        # ---- Step 1-3：分级靠档计算 adjusted_mv ----
        # 仅过滤市值异常（<=0）为 None，停牌时保留 total_mv/free_mv（价格冻结）
        # 后续在宽表中前向填充；若整个窗口停牌，停牌日市值已由数据源前向填充，直接可用
        const = const.with_columns([
            pl.when(
                (pl.col("total_mv") <= 0) | pl.col("total_mv").is_null()
            )
            .then(None)
            .otherwise(pl.col("total_mv"))
            .alias("total_mv_adj"),

            pl.when(
                (pl.col("free_mv") <= 0) | pl.col("free_mv").is_null()
            )
            .then(None)
            .otherwise(pl.col("free_mv"))
            .alias("free_mv_adj"),
        ])

        # 转 pandas 做分级靠档（polars ceil 对 None 友好，但 pandas 更直观）
        pdf = (
            const
            .select(["date", "code", "total_mv_adj", "free_mv_adj"])
            .to_pandas()
        )

        # 计算自由流通比例
        pdf["free_ratio"] = pdf["free_mv_adj"] / pdf["total_mv_adj"]

        # 分级靠档系数 A：f>80% → 1.0，否则 ceil(f×10)/10
        pdf["A"] = np.where(
            pdf["free_ratio"] > 0.8,
            1.0,
            np.ceil(pdf["free_ratio"] * 10) / 10,
        )
        # A 最小取 10%（f 极小时 ceil 可能为 0）
        pdf["A"] = pdf["A"].clip(lower=0.1)

        # 调整后市值
        pdf["adj_mv"] = pdf["total_mv_adj"] * pdf["A"]

        # ---- 转宽表，前向填充，归一化 ----
        adj_mv_wide = pdf.pivot(index="date", columns="code", values="adj_mv")
        adj_mv_wide.index = pd.to_datetime(adj_mv_wide.index).date
        adj_mv_wide = adj_mv_wide.sort_index()

        # 前向填充：停牌股保持上一有效日 adjusted_mv（价格冻结）
        adj_mv_wide = adj_mv_wide.ffill()

        # 截面归一化
        row_sum = adj_mv_wide.sum(axis=1).replace(0, np.nan)
        weight_df = adj_mv_wide.div(row_sum, axis=0)

        return weight_df
