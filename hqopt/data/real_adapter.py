"""
真实市场数据适配器。

从本地 parquet 缓存读取数据，构建 MarketSnapshot，
替换 MarketDataGenerator 的随机生成逻辑。

字段单位说明（来自 Wind / schema.py）：
    amount   : 千元  →  ADV = rolling(20).mean() * 1000（元）
    float_mv : 万元  →  market_cap = float_mv * 10000（元）
    turnover / free_turnover : %（百分比，非小数）
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hqopt.constants import LIMIT_TOL
from hqopt.data.generator import MarketSnapshot, TradingStatus
from hqopt.io.data_panel import load_panel


class RealMarketAdapter:
    """
    从 parquet 缓存构建 MarketSnapshot。

    Parameters
    ----------
    cache_dir : Path | str | None
        parquet 缓存目录，None 时使用 data_panel 默认路径
    adv_window : int
        ADV 计算窗口（交易日数），默认 20
    new_listing_days : int
        上市不足此自然日数的股票视为次新，默认 120
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        adv_window: int = 20,
        new_listing_days: int = 120,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.adv_window = adv_window
        self.new_listing_days = new_listing_days
        # ADV 预计算缓存（惰性初始化）：date × code 宽表（pandas，已 ffill），
        # as-of 查询退化为一次行定位。此前每期对 ~190 万行长表做
        # filter+sort+group_by（9~13ms/期），6 年回测合计约 3.8s。
        self._adv_wide: pd.DataFrame | None = None
        self._adv_cache_panel_id: int | None = None  # 用 id(panel) 判断缓存是否有效

    def build_snapshot(
        self,
        target_date: date,
        index: str = "zz500",
        prev_weight: pd.Series | None = None,
        portfolio_value: float = 1e8,
    ) -> MarketSnapshot:
        """
        构建目标日期的市场快照。

        Parameters
        ----------
        target_date : date
            目标交易日
        index : str
            成分股指数，可选 'zz500' / 'hs300' / 'zz1000'
        prev_weight : pd.Series | None
            上期持仓权重（index=ticker）。None 时取指数成分股等权。
        portfolio_value : float
            组合总市值（元）

        Returns
        -------
        MarketSnapshot
        """
        panel = self._load_panel(target_date)
        return self.build_snapshot_from_panel(panel, target_date, index, prev_weight, portfolio_value)

    def build_snapshot_from_panel(
        self,
        panel: pl.DataFrame,
        target_date: date,
        index: str = "zz500",
        prev_weight: pd.Series | None = None,
        portfolio_value: float = 1e8,
    ) -> MarketSnapshot:
        """从已加载的面板数据构建快照（批量优化专用，避免重复 I/O）。"""
        # 目标日截面
        today = (
            panel
            .filter(pl.col("date") == target_date)
            .to_pandas()
            .set_index("code")
        )

        if today.empty:
            raise ValueError(f"{target_date} 无数据，请确认该日为交易日且缓存存在")

        tickers = today.index.tolist()

        # ---- ADV（元）----
        adv = self._compute_adv(panel, target_date, tickers)

        # ---- 交易状态 ----
        status = self._compute_status(today)

        # ---- 市值（元）----
        market_cap = pd.Series(
            today["float_mv"].values * 1e4,   # 万元 → 元
            index=tickers,
            name="market_cap",
        )

        # ---- 行业 ----
        industry = today["industry_l1"].rename("industry")

        # ---- 成分股标记 ----
        col_map = {"zz500": "is_zz500", "hs300": "is_hs300", "zz1000": "is_zz1000"}
        if index in ("all", "winda", "csiall", "market"):
            # 全市场选股（alpha_max）：「成分」= 主流宽基（HS300∪ZZ500∪ZZ1000）。
            # 此前把 universe 内全部股票标为 True，使 min_constituent_ratio 的约束
            # 退化成 sum(全部 w) >= R，在预算约束 sum(w)==1 下恒成立——一个被配置
            # 注释和文档背书、实际完全空转的流动性/容量下限。空转的风控比没有风控
            # 更危险：使用者以为组合有 R 的权重落在主流宽基内，实则毫无约束。
            is_constituent = (
                today["is_hs300"].astype(bool)
                | today["is_zz500"].astype(bool)
                | today["is_zz1000"].astype(bool)
            ).rename("is_constituent")
        elif index in col_map:
            is_constituent = today[col_map[index]].astype(bool).rename("is_constituent")
        else:
            raise ValueError(
                f"index 须为 {list(col_map.keys())} 或 'all'(全市场) 之一，当前：{index!r}"
            )

        # ---- 上期持仓权重 ----
        if prev_weight is None:
            prev_weight = self._default_prev_weight(tickers, is_constituent, status)
        else:
            # 对齐 ticker，缺失股票补 0，重新归一化
            prev_weight = prev_weight.reindex(tickers).fillna(0.0)
            if prev_weight.sum() > 1e-10:
                prev_weight = prev_weight / prev_weight.sum()
            prev_weight = prev_weight.rename("prev_weight")

        return MarketSnapshot(
            tickers=tickers,
            industry=industry,
            adv=adv,
            status=status,
            prev_weight=prev_weight,
            market_cap=market_cap,
            portfolio_value=portfolio_value,
            is_constituent=is_constituent,
        )

    def filter_universe(
        self,
        snapshot: MarketSnapshot,
        mode: str = "constituent_only",
        n_off_benchmark: int = 0,
    ) -> MarketSnapshot:
        """
        缩减投资域，减少优化变量数量，提升求解速度。

        Parameters
        ----------
        snapshot : MarketSnapshot
            原始全市场快照
        mode : str
            过滤模式：
            - 'constituent_only'  : 只保留指数成分股（标准指数增强）
            - 'constituent_plus'  : 成分股 + ADV 最大的 n_off_benchmark 只非成分股
        n_off_benchmark : int
            'constituent_plus' 模式下额外纳入的非成分股数量

        Returns
        -------
        MarketSnapshot
            过滤后的快照（is_constituent 自动更新）
        """
        const_mask = snapshot.constituent_mask

        if mode == "constituent_only":
            keep = const_mask
        elif mode == "constituent_plus":
            non_const_adv = snapshot.adv.copy()
            non_const_adv[const_mask] = -1   # 排除成分股
            top_off = non_const_adv.nlargest(n_off_benchmark).index.tolist()
            keep = const_mask | snapshot.adv.index.isin(top_off)
        else:
            raise ValueError(f"mode 须为 'constituent_only' 或 'constituent_plus'，当前：{mode!r}")

        tickers_sub = [t for t, k in zip(snapshot.tickers, keep, strict=True) if k]
        w_sub = snapshot.prev_weight[tickers_sub]
        total = w_sub.sum()
        w_sub = (w_sub / total) if total > 1e-10 else w_sub

        return MarketSnapshot(
            tickers=tickers_sub,
            industry=snapshot.industry[tickers_sub],
            adv=snapshot.adv[tickers_sub],
            status=snapshot.status[tickers_sub],
            prev_weight=w_sub.rename("prev_weight"),
            market_cap=snapshot.market_cap[tickers_sub],
            portfolio_value=snapshot.portfolio_value,
            is_constituent=snapshot.is_constituent[tickers_sub]
            if snapshot.is_constituent is not None else None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_panel(self, target_date: date) -> pl.DataFrame:
        """加载历史 + 目标日数据（用于滚动 ADV 计算）。"""
        # ADV 需要 adv_window 个交易日，实际用 adv_window * 2 个日历日作缓冲
        t1 = target_date - timedelta(days=self.adv_window * 2 + 10)
        return load_panel(
            t1,
            target_date,
            columns=[
                "code", "date",
                "close", "limit_up", "limit_down",
                "amount", "float_mv",
                "free_turnover", "trade_status",
                "industry_l1", "list_days",
                "is_hs300", "is_zz500", "is_zz1000", "is_st",
            ],
            cache_dir=self.cache_dir,
        )

    def _build_adv_cache(self, panel: pl.DataFrame) -> None:
        """
        预计算全面板 ADV 长表（惰性，首次调用时执行）。

        语义 = "截至某日，最近 ≤adv_window 个非停牌日 amount 的均值 × 1000（元）"。
        逐 code 在非停牌非空 amount 的子序列上做 rolling_mean(window=adv_window, min_periods=1)，
        再存为 (code, date, adv) 长表供后续按日 as-of 取值。

        口径说明（2026-07-16 有意变更）：缓存数据中停牌日 amount 记为 0（非 null），
        旧实现 dropna().tail(w).mean() 会把这些 0 计入均值、稀释停牌股的 ADV；
        本实现按 trade_status != "停牌" 剔除停牌日，ADV 只在真实成交日上计算。
        ADV 仅用于非成分股候选池的流动性排序，该口径对此用途更合理。
        """
        adv_long = (
            panel
            .select(["code", "date", "amount", "trade_status"])
            .filter(
                (pl.col("trade_status") != "停牌") &
                pl.col("amount").is_not_null() &
                (pl.col("amount") > 0)
            )
            .sort(["code", "date"])
            .with_columns(
                (
                    pl.col("amount")
                    .rolling_mean(window_size=self.adv_window, min_samples=1)
                    .over("code")
                    * 1000  # 千元 → 元
                ).alias("adv")
            )
            .select(["code", "date", "adv"])
        )
        # 一次性 pivot 成 date × code 宽表并前向填充：每行即"截至该日
        # 各股最近一次成交日的 ADV"，与逐期 as-of group_by(last) 语义一致。
        adv_wide = (
            adv_long.to_pandas()
            .pivot(index="date", columns="code", values="adv")
            .sort_index()
            .ffill()
        )
        self._adv_wide = adv_wide
        self._adv_cache_panel_id = id(panel)

    def _compute_adv(
        self, panel: pl.DataFrame, target_date: date, tickers: list[str]
    ) -> pd.Series:
        """
        计算近 adv_window 交易日的平均成交额（元）。

        停牌日 amount 不计入，取最近 ≤adv_window 个非停牌日均值。
        首次调用时预计算全面板 ADV 并缓存；后续按 target_date 做 as-of 查询。
        """
        # 惰性初始化 / 面板变化时重建缓存
        if self._adv_wide is None or self._adv_cache_panel_id != id(panel):
            self._build_adv_cache(panel)
        # _build_adv_cache 无条件填充缓存；断言把这个后置条件显式化。
        assert self._adv_wide is not None

        # as-of：宽表已按日 ffill，定位 ≤ target_date 的最后一行即可
        wide = self._adv_wide
        key = (
            pd.Timestamp(target_date)
            if isinstance(wide.index, pd.DatetimeIndex)
            else target_date
        )
        position = wide.index.searchsorted(key, side="right")
        row = pd.Series(dtype=float) if position == 0 else wide.iloc[position - 1]
        adv_series = (
            row.reindex(tickers)
            .fillna(1e5)   # 数据缺失时给极小 ADV（相当于限制该股换手）
            .astype(float)
        )
        adv_series.name = "adv"
        adv_series.index.name = "code"
        return adv_series

    def _compute_status(self, today: pd.DataFrame) -> pd.Series:
        """
        根据 trade_status、涨跌停价、ST、上市天数推断 TradingStatus。

        优先级：停牌 > 次新/ST > 涨停 > 跌停 > 正常
        """
        n = len(today)
        status = np.full(n, TradingStatus.NORMAL, dtype=object)
        idx = today.index

        ts = today["trade_status"].values
        close = today["close"].values
        limit_up = today["limit_up"].values
        limit_down = today["limit_down"].values
        list_days = today["list_days"].values
        is_st = today["is_st"].values.astype(bool)

        # 1. 涨停（在停牌之前判断，防止停牌日 close/limit 为 NaN）
        #    前置条件用 ts != "停牌"，覆盖 XD/XR/N 等可交易非"交易"状态
        hit_up = (
            (ts != "停牌") &
            (limit_up > 0) &
            (np.abs(close - limit_up) / np.where(limit_up > 0, limit_up, 1) < LIMIT_TOL)
        )
        status[hit_up] = TradingStatus.LIMIT_UP

        # 2. 跌停（同上，ts != "停牌" 覆盖 XD/XR/N）
        hit_down = (
            (ts != "停牌") &
            (limit_down > 0) &
            (np.abs(close - limit_down) / np.where(limit_down > 0, limit_down, 1) < LIMIT_TOL)
        )
        status[hit_down] = TradingStatus.LIMIT_DOWN

        # 3. 次新股（上市不足 new_listing_days 个自然日）
        status[list_days < self.new_listing_days] = TradingStatus.NEW_LISTING

        # 4. ST / *ST（禁止持仓，独立状态）
        status[is_st] = TradingStatus.ST

        # 5. 停牌优先级最高：停牌持仓必须冻结股数，不能被 ST/次新覆盖。
        status[ts == "停牌"] = TradingStatus.SUSPENDED

        return pd.Series(status, index=idx, name="status")

    @staticmethod
    def _default_prev_weight(
        tickers: list[str],
        is_constituent: pd.Series,
        status: pd.Series,
    ) -> pd.Series:
        """默认上期持仓：指数成分股等权，非成分股权重为 0。"""
        w = pd.Series(0.0, index=tickers, name="prev_weight")
        banned = {TradingStatus.NEW_LISTING, TradingStatus.ST, TradingStatus.SUSPENDED}
        eligible = is_constituent & ~status.isin(banned)
        n_eligible = eligible.sum()
        if n_eligible > 0:
            w[eligible] = 1.0 / n_eligible
        return w
