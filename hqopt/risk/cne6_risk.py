"""CNE6 因子风险模型加载器。

消费 BarraCNE6 产出的逐日（防前视）风险面板：
    data/barra_cne6/exposure_panel.parquet   —— CNE6S：20 风格 + Country + 30 行业
    data/barra_cne6_L/exposure_panel.parquet —— CNE6L：16 风格 + Country + 30 行业
    各目录 factor_cov_panel.parquet          —— 与暴露同维的因子协方差 F

CNE6S 共 51 因子，CNE6L 共 47 因子；均由 scripts/export_cne6_panels.py
从 ClickHouse test_barra_cne6_gao 导出。源库中的空行业不进入面板。

组合风险：V = X F Xᵀ + diag(δ)，其中 X=暴露(N×K)，F=因子协方差(K×K)，δ=特质方差(N,)。

设计要点：
- HistQuantOpt 不 import BarraCNE6 代码，只读其导出的 parquet（两项目解耦，
  对应"中心化发布 → 只读消费"）。
- as-of 对齐：给定 target_date，取 ≤ target_date 的最近调仓日的风险模型（防前视）；
  早于面板最早日则返回 None，优化器自动退回 L2 惩罚。
- 股票对齐：保留原始覆盖掩码；暴露缺失填 0、特质方差缺失填当期截面中位数仅
  用于矩阵数值稳定。pipeline 会禁止未覆盖股票新开仓，不能把填充值视为真实暴露。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# CNE6L 16 个核心风格；CNE6S 增加 4 个快策略风格。STYLE_FACTORS 保持为
# 默认短周期模型的完整风格集合，RiskSnapshot 会按实际 factor_names 取交集，
# 因而加载 L 面板时仍只返回 16 个风格。
# Country 因子（全市场恒为 1）不计入风格，归入非风格因子（与行业同处理，
# 不受 style_active_bound 约束）。
STYLE_FACTORS_L: tuple[str, ...] = (
    "Size", "MidCap", "Beta", "Momentum", "ResidualVolatility", "LongTermReversal",
    "Liquidity", "Value", "EarningsYield", "Growth", "Profitability",
    "InvestmentQuality", "EarningsQuality", "EarningsVariability", "Leverage",
    "DividendYield",
)
STYLE_FACTORS_S: tuple[str, ...] = (
    *STYLE_FACTORS_L,
    "AnalystSentiment", "IndustryMomentum", "Seasonality", "ShortTermReversal",
)
STYLE_FACTORS: tuple[str, ...] = STYLE_FACTORS_S

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "barra_cne6"

# 单期 pandas 暴露截面的 LRU 上限（每期约数十 MB，不设上限会在长回测里
# 把整个面板再以 pandas 形式复制一份）。
_PANDAS_CACHE_SIZE = 4


@dataclass(frozen=True)
class RiskSnapshot:
    """某调仓日、对齐到给定 tickers 的因子风险模型。"""
    as_of: date                 # 实际取用的面板调仓日（≤ 请求日）
    tickers: list[str]
    factor_names: list[str]     # S=51 / L=47，顺序与 X 列、F 行列一致
    X: np.ndarray               # (N, K) 因子暴露
    F: np.ndarray               # (K, K) 因子协方差（对称 PSD）
    delta: np.ndarray           # (N,)   特质方差
    covered_mask: np.ndarray    # (N,)   暴露与特质方差均完整的股票

    @property
    def style_names(self) -> list[str]:
        return [f for f in self.factor_names if f in STYLE_FACTORS]

    @property
    def industry_names(self) -> list[str]:
        return [f for f in self.factor_names if f not in STYLE_FACTORS]

    def style_loading(self) -> pd.DataFrame:
        """风格暴露子矩阵（S为N×20，L为N×16，index=tickers）。"""
        sidx = [self.factor_names.index(s) for s in self.style_names]
        return pd.DataFrame(
            self.X[:, sidx], index=self.tickers, columns=self.style_names
        )


@lru_cache(maxsize=4)
def _load_cov_panel(data_dir: str) -> tuple[dict, list[date]]:
    """加载因子协方差面板（~16MB，整表常驻），按 data_dir 缓存。

    暴露面板另行按需加载——它可达 ~1GB，整表常驻会吃掉数 GB 内存。

    Returns:
        cov_by_date: {rebal_date: (factor_names, F 矩阵 K×K)}
        rebal_dates: 升序调仓日列表
    """
    cov = pl.read_parquet(Path(data_dir) / "factor_cov_panel.parquet").with_columns(
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
        F = 0.5 * (F + F.T)           # 数值对称化
        F += 1e-8 * np.eye(F.shape[0])  # 保证严格 PSD，防微小负特征值破坏凸性
        cov_by_date[rdate] = (factor_names, F)

    return cov_by_date, sorted(cov_by_date.keys())


def _scan_exposure(data_dir: str) -> pl.LazyFrame:
    return pl.scan_parquet(Path(data_dir) / "exposure_panel.parquet").with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )


def _load_exposure_partitions(
    data_dir: str,
    wanted_dates: list[date],
) -> dict[date, pl.DataFrame]:
    """只读指定调仓日的暴露面板，并按调仓日预先分区。

    暴露面板可达 ~700 万行 × 50 列（约 1GB parquet）。整表 eager 读取的进程
    峰值内存约 2.7GB，且每期还要再做一次全表 filter。实测 155 期回测下，
    只读所需日期 + 预分区把峰值内存降到 ~1.5GB、单期查询从 ~30ms 降到 ~2ms。
    """
    frame = _scan_exposure(data_dir).filter(
        pl.col("rebal_date").is_in(wanted_dates)
    ).collect()
    return {
        (key[0] if isinstance(key, tuple) else key): part
        for key, part in frame.partition_by("rebal_date", as_dict=True).items()
    }


class CNE6RiskModel:
    """CNE6 因子风险模型查询器：加载一次，按调仓日多次查询。

    Parameters
    ----------
    data_dir : str | Path | None
        风险面板目录，None 用默认短周期 CNE6S（``data/barra_cne6/``）。
    query_dates : Iterable[date] | None
        将要查询的调仓日。**强烈建议传入**：只有这些日期 as-of 命中的
        暴露截面会被读入并预先分区，实测 155 期回测下峰值内存由 ~2.7GB
        降到 ~1.5GB、单期查询由 ~30ms 降到 ~2ms。None 时退回整表加载
        （行为与优化前一致），仅供测试和临时脚本使用。
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        query_dates: Iterable[date] | None = None,
    ) -> None:
        self.data_dir = str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
        self._cov_by_date, self._rebal_dates = _load_cov_panel(self.data_dir)

        wanted = self._asof_dates_for(query_dates)
        self._exposure_by_date: dict[date, pl.DataFrame] = {}
        self._exposure_full: pl.DataFrame | None = None
        if wanted is None:
            # 未声明查询范围：整表常驻并按需 filter，与优化前行为一致
            self._exposure_full = _scan_exposure(self.data_dir).collect()
        else:
            self._exposure_by_date = _load_exposure_partitions(self.data_dir, wanted)
        self._pandas_cache: OrderedDict[date, pd.DataFrame] = OrderedDict()

    def _asof_dates_for(
        self, query_dates: Iterable[date] | None
    ) -> list[date] | None:
        """把待查询日期映射成实际需要的面板调仓日（as-of ≤ 查询日）。"""
        if query_dates is None:
            return None
        wanted = {
            asof
            for asof in (self._asof_date(d) for d in query_dates)
            if asof is not None
        }
        return sorted(wanted)

    @property
    def rebal_dates(self) -> list[date]:
        return self._rebal_dates

    @property
    def coverage(self) -> tuple[date, date]:
        return self._rebal_dates[0], self._rebal_dates[-1]

    def _asof_date(self, target_date: date) -> date | None:
        """≤ target_date 的最近调仓日（防前视）；早于最早日返回 None。"""
        position = bisect_right(self._rebal_dates, target_date)
        return self._rebal_dates[position - 1] if position else None

    def _exposure_at(self, asof: date) -> pd.DataFrame | None:
        """取某调仓日的暴露截面（pandas，按 code 索引），带缓存。

        调用方传了 query_dates 却查询区间外日期时，回退到按需单期读取，
        保证查询语义不因内存优化而改变。
        """
        cached = self._pandas_cache.get(asof)
        if cached is not None:
            self._pandas_cache.move_to_end(asof)
            return cached

        partition = self._exposure_by_date.get(asof)
        if partition is None and self._exposure_full is not None:
            partition = self._exposure_full.filter(pl.col("rebal_date") == asof)
        if partition is None:
            partition = _scan_exposure(self.data_dir).filter(
                pl.col("rebal_date") == asof
            ).collect()
            self._exposure_by_date[asof] = partition
        if partition.is_empty():
            return None

        frame = partition.to_pandas().set_index("code")
        # 有界 LRU：面板频率低于调仓频率时相邻调仓日常复用同一 as-of 截面，
        # 但缓存必须有上限，否则长回测会把整个面板以 pandas 形式再存一份。
        self._pandas_cache[asof] = frame
        while len(self._pandas_cache) > _PANDAS_CACHE_SIZE:
            self._pandas_cache.popitem(last=False)
        return frame

    def at(self, target_date: date, tickers: list[str]) -> RiskSnapshot | None:
        """取对齐到 tickers 的风险模型；无可用历史则返回 None。"""
        asof = self._asof_date(target_date)
        if asof is None:
            return None

        factor_names, F = self._cov_by_date[asof]

        day = self._exposure_at(asof)
        if day is None:
            return None
        # 当期截面特质方差中位数，用于填补 universe 差异导致的缺失
        spec_median = float(np.nanmedian(day["spec_var"].to_numpy()))

        aligned = day.reindex(tickers)
        covered_mask = (
            aligned[list(factor_names)].notna().all(axis=1)
            & aligned["spec_var"].notna()
        ).to_numpy(dtype=bool)
        X = aligned[list(factor_names)].fillna(0.0).to_numpy().astype(np.float64)
        delta = aligned["spec_var"].fillna(spec_median).to_numpy().astype(np.float64)

        return RiskSnapshot(
            as_of=asof,
            tickers=list(tickers),
            factor_names=list(factor_names),
            X=X,
            F=F,
            delta=delta,
            covered_mask=covered_mask,
        )
