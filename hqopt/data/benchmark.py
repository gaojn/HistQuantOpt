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

import logging
from bisect import bisect_right
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hqopt.io.data_panel import load_panel

# 官方成分权重 parquet（由 scripts/export_index_weight.py 从 wind_db 导出）
DEFAULT_OFFICIAL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "index_weight" / "official_weight.parquet"
)

logger = logging.getLogger(__name__)

OFFICIAL_DRIFT_SOURCE = "official_drift"
OFFICIAL_FROZEN_SOURCE = "official_frozen"
RECONSTRUCT_SOURCE = "reconstruct"
DEFAULT_MAX_SNAPSHOT_AGE_DAYS = 30
_SOURCE_ALIASES = {"official": OFFICIAL_DRIFT_SOURCE}
_VALID_SOURCES = {
    OFFICIAL_DRIFT_SOURCE,
    OFFICIAL_FROZEN_SOURCE,
    RECONSTRUCT_SOURCE,
    *_SOURCE_ALIASES,
}


def equal_weight_benchmark_weights(adj_close: pd.DataFrame) -> pd.DataFrame:
    """按每个交易日的有效复权收盘价股票构造全市场等权权重。"""
    prices = adj_close.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    valid = (prices.notna() & prices.gt(0)).astype(float)
    row_sum = valid.sum(axis=1).replace(0.0, np.nan)
    return valid.div(row_sum, axis=0).fillna(0.0)


def benchmark_returns_from_rebalance_weights(
    benchmark_weight_df: pd.DataFrame,
    adj_close: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.Series:
    """用调仓日基准权重计算 ``(T_i, T_{i+1}]`` 持有期收益。

    权重在每个信号日收盘确定，从下一交易日开始生效；持有期内保持不变。
    该口径与 :class:`ReturnAttributor` 一致，用于让回测与归因共享同一基准。
    """
    weights = benchmark_weight_df.copy()
    weights.index = pd.to_datetime(weights.index)
    weights = weights.sort_index()
    prices = adj_close.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    signal_dates = list(pd.DatetimeIndex(pd.to_datetime(rebalance_dates)).sort_values())
    if not signal_dates:
        raise ValueError("rebalance_dates 为空")

    daily_ret = prices.pct_change(fill_method=None)
    rows: dict[pd.Timestamp, float] = {}
    for i, signal_date in enumerate(signal_dates):
        next_signal = signal_dates[i + 1] if i + 1 < len(signal_dates) else None
        period_dates = [
            day
            for day in prices.index
            if day > signal_date and (next_signal is None or day <= next_signal)
        ]
        if not period_dates:
            continue
        available = weights.index[weights.index <= signal_date]
        if available.empty:
            continue
        benchmark_weight = weights.loc[available[-1]].dropna()
        benchmark_weight = benchmark_weight[benchmark_weight != 0.0]
        for day in period_dates:
            returns = daily_ret.loc[day].reindex(benchmark_weight.index).fillna(0.0)
            rows[day] = float(benchmark_weight @ returns)

    return pd.Series(rows, dtype=float, name="benchmark_return").sort_index()


class BenchmarkPriceCoverageError(ValueError):
    """官方快照成分缺少锚点或目标日价格，无法安全漂移。"""


def drift_official_weights(
    snapshot_weights: pd.Series,
    anchor_prices: pd.Series,
    target_prices: pd.Series,
) -> pd.Series:
    """按后复权价格比把官方快照权重漂移到目标日。

    本函数只做确定性数学变换，不读取数据、不选择快照。运行时和每日导出必须
    共用它，避免形成两套漂移口径。调用方负责保证价格均来自 ``<= target``。
    """
    base = pd.to_numeric(snapshot_weights, errors="coerce").dropna()
    base = base[base > 0]
    if base.empty:
        raise ValueError("官方快照没有正权重")
    base = base / base.sum()

    p0 = pd.to_numeric(anchor_prices.reindex(base.index), errors="coerce")
    p1 = pd.to_numeric(target_prices.reindex(base.index), errors="coerce")
    valid = p0.gt(0) & p1.gt(0) & np.isfinite(p0) & np.isfinite(p1)
    if not bool(valid.all()):
        missing = base.index[~valid].tolist()
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise BenchmarkPriceCoverageError(
            f"{len(missing)} 只官方成分缺少有效漂移价格：{preview}{suffix}"
        )

    drifted = base * (p1 / p0)
    total = float(drifted.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise BenchmarkPriceCoverageError("漂移后权重和无效")
    return (drifted / total).rename("bm_weight")


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
        source: str = OFFICIAL_DRIFT_SOURCE,
        official_path: Path | str | None = None,
        max_snapshot_age_days: int | None = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    ) -> None:
        """
        Parameters
        ----------
        source : str
            基准权重来源：
            - "official_drift"（默认）：取 ≤目标日最近官方快照，并按后复权
              收盘价漂移；成分变化、价格缺失或官方未覆盖时回退重构。
            - "official_frozen"：旧口径，月度快照之间保持权重不变。
            - "official"：兼容别名，等同 "official_drift"。
            - "reconstruct"：始终用 free_mv 重构（不读官方权重）。
        official_path : Path | str | None
            官方权重 parquet 路径，None 用默认 data/index_weight/official_weight.parquet。
        max_snapshot_age_days : int | None
            ``official_drift`` 允许的官方快照最大陈旧自然日数，默认 30。
            超过后告警并回退重构；None 显式关闭限制。兼容用
            ``official_frozen`` 不应用此限制。
        """
        if index not in self._INDEX_COL:
            raise ValueError(f"index 须为 {list(self._INDEX_COL)} 之一")
        if source not in _VALID_SOURCES:
            raise ValueError(
                "source 须为 'official_drift'、'official_frozen' 或 'reconstruct'"
            )
        if (
            max_snapshot_age_days is not None
            and (
                isinstance(max_snapshot_age_days, bool)
                or not isinstance(max_snapshot_age_days, int)
                or max_snapshot_age_days < 0
            )
        ):
            raise ValueError("max_snapshot_age_days 必须是非负整数或 None")
        self.index = index
        self._col = self._INDEX_COL[index]
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._panel = panel
        self._weight_cache: pd.DataFrame | None = None   # (date, ticker) → weight，free_mv 重构
        self.source = _SOURCE_ALIASES.get(source, source)
        self.max_snapshot_age_days = max_snapshot_age_days
        self._official_path = Path(official_path) if official_path else DEFAULT_OFFICIAL_PATH
        self._official_cache: pd.DataFrame | None = None  # date × code，官方权重
        self._official_loaded = False
        self._drift_price_cache: pd.DataFrame | None = None
        self._roster_cache: pd.DataFrame | None = None
        self._prepared_range: tuple[date, date] | None = None
        self._trading_dates: list[date] = []
        self._audit_method_by_period: dict[str, str] = {}
        self._audit_anchor_by_period: dict[str, str] = {}
        self._audit_snapshot_age_by_period: dict[str, int] = {}
        self._audit_effective_date_by_period: dict[str, str] = {}
        self._audit_fallback_by_period: dict[str, str] = {}

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
        # 优先官方权重：旧模式冻结，新模式按后复权价格漂移。
        if self.source in (OFFICIAL_DRIFT_SOURCE, OFFICIAL_FROZEN_SOURCE):
            oc = self._get_official_cache()
            if oc is not None and len(oc):
                anchor = self._official_anchor(target_date, oc)
                if anchor is not None:
                    snapshot = oc.loc[anchor].dropna()
                    snapshot_age = (target_date - anchor).days
                    if (
                        self.source == OFFICIAL_DRIFT_SOURCE
                        and self.max_snapshot_age_days is not None
                        and snapshot_age > self.max_snapshot_age_days
                    ):
                        fallback_reason = (
                            "snapshot_stale: "
                            f"age_days={snapshot_age}, "
                            f"limit_days={self.max_snapshot_age_days}"
                        )
                        self._warn_fallback(target_date, fallback_reason)
                        self._record_audit(
                            target_date,
                            RECONSTRUCT_SOURCE,
                            anchor,
                            fallback_reason,
                        )
                        return self._reconstructed_weights(target_date, tickers)

                    if self.source == OFFICIAL_FROZEN_SOURCE:
                        self._record_audit(
                            target_date, OFFICIAL_FROZEN_SOURCE, anchor
                        )
                        return self._finalize(snapshot, tickers)

                    fallback_reason = self._drift_fallback_reason(
                        target_date, snapshot
                    )
                    if fallback_reason is None:
                        if anchor == target_date:
                            self._record_audit(
                                target_date, "official_snapshot", anchor
                            )
                            return self._finalize(snapshot, tickers)
                        try:
                            w = self._drift_weight(target_date, anchor, snapshot)
                        except BenchmarkPriceCoverageError as exc:
                            fallback_reason = f"price_coverage: {exc}"
                        else:
                            self._record_audit(
                                target_date, OFFICIAL_DRIFT_SOURCE, anchor
                            )
                            return self._finalize(w, tickers)

                    self._warn_fallback(target_date, fallback_reason)
                    self._record_audit(
                        target_date, RECONSTRUCT_SOURCE, anchor, fallback_reason
                    )
                    return self._reconstructed_weights(target_date, tickers)
            # 官方未覆盖（早于起点 / 无该指数）→ 回退 free_mv 重构
            reason = "official_snapshot_unavailable"
            self._record_audit(target_date, RECONSTRUCT_SOURCE, None, reason)

        if self.source == RECONSTRUCT_SOURCE:
            self._record_audit(target_date, RECONSTRUCT_SOURCE, None)
        return self._reconstructed_weights(target_date, tickers)

    @staticmethod
    def _finalize(w: pd.Series, tickers: list[str] | None) -> pd.Series:
        """对齐 tickers 并强制归一化（成分股权重缺失时和可能 < 1）。"""
        w = w[w > 0]
        if tickers is not None:
            w = w.reindex(tickers).fillna(0.0)
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
        if panel is not None and panel is not self._panel:
            self._panel = panel
            self._weight_cache = None
            self._drift_price_cache = None
            self._roster_cache = None
            self._prepared_range = None
        if self.source in (OFFICIAL_DRIFT_SOURCE, OFFICIAL_FROZEN_SOURCE):
            self._get_official_cache()
            if self.source == OFFICIAL_DRIFT_SOURCE:
                self._prepare_drift_inputs(start_date, end_date)
        else:
            self._weight_cache = self._build_weight_matrix(start_date, end_date)

    def audit_summary(self) -> dict[str, object]:
        """返回可直接写入运行统计 JSON 的基准权重审计摘要。"""
        fallback = self._audit_fallback_by_period
        roster_dates = [d for d, reason in fallback.items() if reason == "roster_changed"]
        stale_dates = [
            d for d, reason in fallback.items()
            if reason.startswith("snapshot_stale:")
        ]
        return {
            "source": self.source,
            "price_field": "adj_close" if self.source == OFFICIAL_DRIFT_SOURCE else None,
            "max_snapshot_age_days": self.max_snapshot_age_days,
            "fallback_period_count": len(fallback),
            "roster_change_period_count": len(roster_dates),
            "stale_snapshot_period_count": len(stale_dates),
            "method_by_period": dict(self._audit_method_by_period),
            "snapshot_as_of_by_period": dict(self._audit_anchor_by_period),
            "snapshot_age_days_by_period": dict(
                self._audit_snapshot_age_by_period
            ),
            "effective_date_by_period": dict(
                self._audit_effective_date_by_period
            ),
            "fallback_reason_by_period": dict(fallback),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_official_cache(self) -> pd.DataFrame | None:
        """加载该指数官方权重（date × code），文件缺失/无该指数返回 None。"""
        if self._official_loaded:
            return self._official_cache
        self._official_loaded = True
        self._official_cache = None
        if not self._official_path.exists():
            return None
        df = pl.read_parquet(self._official_path).filter(pl.col("index") == self.index)
        if df.is_empty():
            return None
        pdf = df.select(["date", "code", "weight"]).to_pandas()
        wide = pdf.pivot(index="date", columns="code", values="weight")
        wide.index = pd.to_datetime(wide.index).date
        self._official_cache = wide.sort_index()
        return self._official_cache

    @staticmethod
    def _official_anchor(target_date: date, cache: pd.DataFrame) -> date | None:
        dates = list(cache.index)
        pos = bisect_right(dates, target_date) - 1
        return dates[pos] if pos >= 0 else None

    def _prepare_drift_inputs(self, start_date: date, end_date: date) -> None:
        """构建 PIT 价格与每日成分缓存；只允许向前填充。"""
        if self._prepared_range == (start_date, end_date):
            return
        if self._panel is None:
            return
        required = {"date", "code", "adj_close", self._col}
        missing = required - set(self._panel.columns)
        if missing:
            raise KeyError(f"官方权重漂移所需行情列缺失：{sorted(missing)}")

        panel = self._panel.select(
            ["date", "code", "adj_close", self._col]
        ).with_columns(pl.col("date").cast(pl.Date))
        self._trading_dates = (
            panel.select("date").unique().sort("date")["date"].to_list()
        )
        price_pdf = panel.select(["date", "code", "adj_close"]).to_pandas()

        oc = self._get_official_cache()
        if oc is not None and len(oc):
            anchor = self._official_anchor(start_date, oc)
            known_dates = set(price_pdf["date"])
            if anchor is not None and anchor not in known_dates:
                try:
                    extra = load_panel(
                        anchor,
                        anchor,
                        columns=["adj_close"],
                        cache_dir=self._cache_dir,
                        add_adj_vwap=False,
                    ).select(["date", "code", "adj_close"])
                except (FileNotFoundError, KeyError):
                    extra = pl.DataFrame()
                if not extra.is_empty():
                    price_pdf = pd.concat(
                        [price_pdf, extra.to_pandas()], ignore_index=True
                    )

        prices = price_pdf.pivot(index="date", columns="code", values="adj_close")
        prices.index = pd.to_datetime(prices.index).date
        self._drift_price_cache = prices.sort_index().ffill()

        roster_pdf = panel.select(["date", "code", self._col]).to_pandas()
        roster = roster_pdf.pivot(index="date", columns="code", values=self._col)
        roster.index = pd.to_datetime(roster.index).date
        self._roster_cache = roster.sort_index().fillna(0).astype(bool)
        self._prepared_range = (start_date, end_date)

    def _ensure_drift_inputs(self, target_date: date) -> None:
        if self._drift_price_cache is not None and self._roster_cache is not None:
            return
        if self._panel is None:
            raise RuntimeError(
                "official_drift 需要传入含 adj_close 与指数成分标记的行情 panel，"
                "或先调用 precompute()"
            )
        dates = self._panel.select("date").unique().sort("date")["date"].to_list()
        self._prepare_drift_inputs(dates[0], max(dates[-1], target_date))

    def _drift_fallback_reason(
        self,
        target_date: date,
        snapshot: pd.Series,
    ) -> str | None:
        self._ensure_drift_inputs(target_date)
        assert self._roster_cache is not None
        if target_date not in self._roster_cache.index:
            return "target_roster_unavailable"
        target_row = self._roster_cache.loc[target_date]
        target_roster = set(target_row.index[target_row.values])
        anchor_roster = set(snapshot[snapshot > 0].index)
        if not target_roster:
            return "target_roster_empty"
        if target_roster != anchor_roster:
            return "roster_changed"
        return None

    def _drift_weight(
        self,
        target_date: date,
        anchor: date,
        snapshot: pd.Series,
    ) -> pd.Series:
        self._ensure_drift_inputs(target_date)
        assert self._drift_price_cache is not None
        prices = self._drift_price_cache
        if anchor not in prices.index:
            raise BenchmarkPriceCoverageError(f"缺少官方锚点日 {anchor} 行情")
        avail = prices.index[prices.index <= target_date]
        if len(avail) == 0:
            raise BenchmarkPriceCoverageError(f"缺少目标日 {target_date} 之前行情")
        price_date = avail[-1]
        return drift_official_weights(
            snapshot,
            prices.loc[anchor],
            prices.loc[price_date],
        )

    def _reconstructed_weights(
        self,
        target_date: date,
        tickers: list[str] | None,
    ) -> pd.Series:
        cache = self._get_or_build_cache()
        resolved = target_date
        if resolved not in cache.index:
            avail = cache.index[cache.index <= resolved]
            if len(avail) == 0:
                raise ValueError(f"无 {target_date} 之前的基准权重数据")
            resolved = avail[-1]
        return self._finalize(cache.loc[resolved].dropna(), tickers)

    def _record_audit(
        self,
        target_date: date,
        method: str,
        anchor: date | None,
        fallback_reason: str | None = None,
    ) -> None:
        key = target_date.isoformat()
        self._audit_method_by_period[key] = method
        if anchor is not None:
            self._audit_anchor_by_period[key] = anchor.isoformat()
            self._audit_snapshot_age_by_period[key] = (
                target_date - anchor
            ).days
        effective_date = self._next_trading_date(target_date)
        if effective_date is not None:
            self._audit_effective_date_by_period[key] = (
                effective_date.isoformat()
            )
        if fallback_reason is not None:
            self._audit_fallback_by_period[key] = fallback_reason

    def _next_trading_date(self, target_date: date) -> date | None:
        """返回面板中严格晚于 T 的首个交易日，用于标记 T 收盘权重的生效日。"""
        if not self._trading_dates and self._panel is not None:
            self._trading_dates = (
                self._panel.select("date")
                .unique()
                .sort("date")["date"]
                .to_list()
            )
        pos = bisect_right(self._trading_dates, target_date)
        if pos >= len(self._trading_dates):
            return None
        return self._trading_dates[pos]

    def _warn_fallback(self, target_date: date, reason: str) -> None:
        key = target_date.isoformat()
        if key not in self._audit_fallback_by_period:
            logger.warning(
                "[%s] 官方指数权重漂移回退 reconstruct：%s", target_date, reason
            )

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
                         self._col],
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

        # ---- 转宽表，前向填充（仅在册日），归一化 ----
        adj_mv_wide = pdf.pivot(index="date", columns="code", values="adj_mv")
        adj_mv_wide.index = pd.to_datetime(adj_mv_wide.index).date
        adj_mv_wide = adj_mv_wide.sort_index()

        roster = (
            panel.select(["date", "code", self._col])
            .to_pandas()
            .pivot(index="date", columns="code", values=self._col)
        )
        roster.index = pd.to_datetime(roster.index).date
        roster = roster.reindex(
            index=adj_mv_wide.index, columns=adj_mv_wide.columns, fill_value=0
        ).fillna(0).astype(bool)
        # 停牌日在册时保留前值；任何退出区间（包括之后重新进入）都立即清零。
        adj_mv_wide = adj_mv_wide.ffill().where(roster)

        # 截面归一化
        row_sum = adj_mv_wide.sum(axis=1).replace(0, np.nan)
        weight_df = adj_mv_wide.div(row_sum, axis=0)

        return weight_df
