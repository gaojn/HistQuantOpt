"""
随机市场数据生成器，模拟 A 股截面数据。

生成内容：
- 股票基本信息（代码、行业）
- 近 20 日平均成交额（ADV）
- 交易状态（正常 / 停牌 / 涨停 / 跌停 / 上市首日）
- 当前持仓权重
- 指数成分股标记（is_constituent）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class TradingStatus(Enum):
    """A 股交易状态枚举。"""
    NORMAL = "normal"           # 正常交易
    SUSPENDED = "suspended"     # 停牌（不可买卖）
    LIMIT_UP = "limit_up"       # 涨停（不可买入）
    LIMIT_DOWN = "limit_down"   # 跌停（不可卖出）
    NEW_LISTING = "new_listing" # 上市首日 / 次新（禁止持仓）


# CITIC 一级行业（简化为 30 个）
CITIC_INDUSTRIES = [
    "Coal", "NonFerrous", "Steel", "Petrochemical", "BasicChemical",
    "Building", "BuildingMaterial", "LightManufacturing", "Machinery",
    "Defense", "Power", "Electronics", "Automotive", "Retail",
    "Agriculture", "Food", "Textile", "Pharmaceutical", "Utility",
    "Transportation", "RealEstate", "Commercial", "BankFinance",
    "NonBankFinance", "Computer", "Media", "Telecom", "Leisure",
    "Environment", "Comprehensive",
]

N_INDUSTRIES = len(CITIC_INDUSTRIES)


@dataclass
class MarketSnapshot:
    """单截面市场数据快照。"""
    tickers: list[str]
    industry: pd.Series           # index=ticker, value=行业名称
    adv: pd.Series                # index=ticker, value=近20日均成交额（元）
    status: pd.Series             # index=ticker, value=TradingStatus
    prev_weight: pd.Series        # index=ticker, value=上期持仓权重（已归一化）
    market_cap: pd.Series         # index=ticker, value=总市值（元）
    portfolio_value: float        # 组合总市值（元）
    is_constituent: pd.Series | None = None  # index=ticker, value=bool，是否为指数成分股

    @property
    def n_stocks(self) -> int:
        return len(self.tickers)

    @property
    def suspended_mask(self) -> np.ndarray:
        return (self.status == TradingStatus.SUSPENDED).values

    @property
    def limit_up_mask(self) -> np.ndarray:
        return (self.status == TradingStatus.LIMIT_UP).values

    @property
    def limit_down_mask(self) -> np.ndarray:
        return (self.status == TradingStatus.LIMIT_DOWN).values

    @property
    def new_listing_mask(self) -> np.ndarray:
        return (self.status == TradingStatus.NEW_LISTING).values

    @property
    def tradable_mask(self) -> np.ndarray:
        """可正常参与优化的股票（排除停牌和上市首日）。"""
        return ~(self.suspended_mask | self.new_listing_mask)

    @property
    def constituent_mask(self) -> np.ndarray:
        """指数成分股布尔掩码，shape=(N,)。未设置时返回全 True。"""
        if self.is_constituent is None:
            return np.ones(self.n_stocks, dtype=bool)
        return self.is_constituent.values.astype(bool)


class MarketDataGenerator:
    """
    随机生成 A 股截面市场数据。

    Parameters
    ----------
    n_stocks : int
        股票池总数（含成分股和非成分股）
    portfolio_value : float
        组合总市值（元），默认 1 亿
    seed : int
        随机种子
    suspended_ratio : float
        停牌比例，默认 3%
    limit_up_ratio : float
        涨停比例，默认 2%
    limit_down_ratio : float
        跌停比例，默认 1%
    new_listing_ratio : float
        次新股比例，默认 1%
    n_constituents : int | None
        指数成分股数量。None 表示所有股票均为成分股。
        例如模拟中证500增强时，传入 500，其余股票为非成分股（泛化选股空间）。
    """

    def __init__(
        self,
        n_stocks: int = 600,
        portfolio_value: float = 1e8,
        seed: int = 42,
        suspended_ratio: float = 0.03,
        limit_up_ratio: float = 0.02,
        limit_down_ratio: float = 0.01,
        new_listing_ratio: float = 0.01,
        n_constituents: int | None = None,
    ) -> None:
        self.n_stocks = n_stocks
        self.portfolio_value = portfolio_value
        self.rng = np.random.default_rng(seed)
        self.suspended_ratio = suspended_ratio
        self.limit_up_ratio = limit_up_ratio
        self.limit_down_ratio = limit_down_ratio
        self.new_listing_ratio = new_listing_ratio
        self.n_constituents = n_constituents

    def generate(self) -> MarketSnapshot:
        tickers = [f"SZ{i:06d}" for i in range(1, self.n_stocks + 1)]
        industry = self._generate_industry(tickers)
        adv = self._generate_adv(tickers)
        status = self._generate_status(tickers)
        prev_weight = self._generate_prev_weight(tickers, status)
        market_cap = self._generate_market_cap(tickers)
        is_constituent = self._generate_constituent(tickers)

        return MarketSnapshot(
            tickers=tickers,
            industry=industry,
            adv=adv,
            status=status,
            prev_weight=prev_weight,
            market_cap=market_cap,
            portfolio_value=self.portfolio_value,
            is_constituent=is_constituent,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_industry(self, tickers: list[str]) -> pd.Series:
        n = self.n_stocks
        # 保证每个行业至少有一只股票，剩余随机分配
        base = list(range(N_INDUSTRIES))
        extra = self.rng.integers(0, N_INDUSTRIES, size=max(0, n - N_INDUSTRIES)).tolist()
        idx = np.array(base + extra)
        self.rng.shuffle(idx)
        return pd.Series(
            [CITIC_INDUSTRIES[i] for i in idx[:n]],
            index=tickers,
            name="industry",
        )

    def _generate_adv(self, tickers: list[str]) -> pd.Series:
        # 对数正态分布，均值约 5000 万，长尾
        log_mean = np.log(5e7)
        log_std = 1.2
        adv = self.rng.lognormal(log_mean, log_std, size=self.n_stocks)
        return pd.Series(adv, index=tickers, name="adv")

    def _generate_status(self, tickers: list[str]) -> pd.Series:
        n = self.n_stocks
        status = np.full(n, TradingStatus.NORMAL, dtype=object)

        def _assign(ratio: float, value: TradingStatus, mask: np.ndarray) -> np.ndarray:
            available = np.where(mask)[0]
            k = max(0, int(ratio * n))
            chosen = self.rng.choice(available, size=min(k, len(available)), replace=False)
            mask[chosen] = False
            status[chosen] = value
            return mask

        available_mask = np.ones(n, dtype=bool)
        available_mask = _assign(self.suspended_ratio, TradingStatus.SUSPENDED, available_mask)
        available_mask = _assign(self.limit_up_ratio, TradingStatus.LIMIT_UP, available_mask)
        available_mask = _assign(self.limit_down_ratio, TradingStatus.LIMIT_DOWN, available_mask)
        available_mask = _assign(self.new_listing_ratio, TradingStatus.NEW_LISTING, available_mask)

        return pd.Series(status, index=tickers, name="status")

    def _generate_prev_weight(
        self, tickers: list[str], status: pd.Series
    ) -> pd.Series:
        # 上期持仓：次新股权重为 0，其余 Dirichlet 分布
        raw = self.rng.dirichlet(np.ones(self.n_stocks) * 0.5)
        new_listing_idx = status[status == TradingStatus.NEW_LISTING].index
        raw_series = pd.Series(raw, index=tickers)
        raw_series[new_listing_idx] = 0.0
        raw_series = raw_series / raw_series.sum()
        return raw_series.rename("prev_weight")

    def _generate_market_cap(self, tickers: list[str]) -> pd.Series:
        log_mean = np.log(1e10)   # 100 亿
        log_std = 1.0
        cap = self.rng.lognormal(log_mean, log_std, size=self.n_stocks)
        return pd.Series(cap, index=tickers, name="market_cap")

    def _generate_constituent(self, tickers: list[str]) -> pd.Series | None:
        """
        生成成分股标记。

        成分股随机选取前 n_constituents 只（模拟指数权重筛选后的结果）。
        非成分股仍可进入优化，但受 ConstituentConstraint 限制其权重合计上限。
        """
        if self.n_constituents is None:
            return None

        n = self.n_stocks
        k = min(self.n_constituents, n)
        is_const = np.zeros(n, dtype=bool)
        chosen = self.rng.choice(n, size=k, replace=False)
        is_const[chosen] = True
        return pd.Series(is_const, index=tickers, name="is_constituent")
