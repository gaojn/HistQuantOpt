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

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.generator import MarketSnapshot, TradingStatus

# 涨跌停判断容差：close 与 limit 价之差在此比例内视为封板
_LIMIT_TOL = 5e-4   # 0.05%，处理浮点误差


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
        上市不足此自然日数的股票视为次新，默认 60
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        adv_window: int = 20,
        new_listing_days: int = 60,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.adv_window = adv_window
        self.new_listing_days = new_listing_days

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
            # 全市场选股（量化选股 alpha_max）：universe 内全部视为"成分"
            is_constituent = pd.Series(True, index=tickers, name="is_constituent")
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

        tickers_sub = [t for t, k in zip(snapshot.tickers, keep) if k]
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

    def _compute_adv(
        self, panel: pl.DataFrame, target_date: date, tickers: list[str]
    ) -> pd.Series:
        """
        计算近 adv_window 交易日的平均成交额（元）。

        停牌日 amount=0，视为 NaN 剔除后取均值。
        """
        df = (
            panel
            .filter(pl.col("date") <= target_date)
            .filter(pl.col("code").is_in(tickers))
            .select(["code", "date", "amount", "trade_status"])
            .sort(["code", "date"])
            .to_pandas()
        )

        # 停牌日 amount 置 NaN
        df.loc[df["trade_status"] == "停牌", "amount"] = np.nan

        # 每只股票取最近 adv_window 个非停牌日均值，单位：千元 → 元
        adv = (
            df.groupby("code")["amount"]
            .apply(lambda s: s.dropna().tail(self.adv_window).mean() * 1000)
            .reindex(tickers)
            .fillna(1e5)   # 数据缺失时给极小 ADV（相当于限制该股换手）
        )
        adv.name = "adv"
        return adv

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
        hit_up = (
            (ts == "交易") &
            (limit_up > 0) &
            (np.abs(close - limit_up) / np.where(limit_up > 0, limit_up, 1) < _LIMIT_TOL)
        )
        status[hit_up] = TradingStatus.LIMIT_UP

        # 2. 跌停
        hit_down = (
            (ts == "交易") &
            (limit_down > 0) &
            (np.abs(close - limit_down) / np.where(limit_down > 0, limit_down, 1) < _LIMIT_TOL)
        )
        status[hit_down] = TradingStatus.LIMIT_DOWN

        # 3. 停牌（覆盖涨跌停，停牌日以固定仓位处理）
        status[ts == "停牌"] = TradingStatus.SUSPENDED

        # 4. 次新股（上市不足 new_listing_days 个自然日）
        status[list_days < self.new_listing_days] = TradingStatus.NEW_LISTING

        # 5. ST / *ST（禁止持仓，与次新同等处理）
        status[is_st] = TradingStatus.NEW_LISTING

        return pd.Series(status, index=idx, name="status")

    @staticmethod
    def _default_prev_weight(
        tickers: list[str],
        is_constituent: pd.Series,
        status: pd.Series,
    ) -> pd.Series:
        """默认上期持仓：指数成分股等权，非成分股权重为 0。"""
        w = pd.Series(0.0, index=tickers, name="prev_weight")
        eligible = is_constituent & (status != TradingStatus.NEW_LISTING)
        n_eligible = eligible.sum()
        if n_eligible > 0:
            w[eligible] = 1.0 / n_eligible
        return w
