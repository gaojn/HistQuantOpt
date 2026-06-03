"""
Alpha 因子容器。

框架假设外部已生成合成 Alpha 因子（预期超额收益信号），
本模块负责：
1. 接收并验证 Alpha 向量
2. 截面标准化（z-score + 缩尾）
3. 对停牌/次新股的 Alpha 置零（避免信号污染优化）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimizer.data.generator import MarketSnapshot, TradingStatus


class AlphaFactors:
    """
    Alpha 因子处理器。

    Parameters
    ----------
    snapshot : MarketSnapshot
        市场快照（提供股票列表和交易状态）
    raw_alpha : array-like, optional
        外部传入的原始 Alpha 向量（年化超额收益，单位：小数）。
        若为 None，则随机生成（仅用于演示）。
    winsor_pct : float
        缩尾百分位，默认 1%（即截断 [1%, 99%]）
    seed : int
        随机种子（仅在 raw_alpha=None 时使用）
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        raw_alpha: np.ndarray | None = None,
        winsor_pct: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.snapshot = snapshot
        self.winsor_pct = winsor_pct
        n = snapshot.n_stocks

        if raw_alpha is None:
            rng = np.random.default_rng(seed)
            # 模拟：信息系数 IC ≈ 0.05，年化 Alpha 约 ±20%
            raw_alpha = rng.normal(loc=0.0, scale=0.15, size=n)

        if len(raw_alpha) != n:
            raise ValueError(
                f"raw_alpha 长度 {len(raw_alpha)} 与股票数量 {n} 不匹配"
            )

        self._raw = np.asarray(raw_alpha, dtype=float)
        self._alpha = self._process()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """处理后的 Alpha 向量（z-score），shape=(N,)。"""
        return self._alpha

    def to_return_alpha(
        self,
        ic: float,
        stock_vol: np.ndarray | float,
    ) -> np.ndarray:
        """
        将 z-score alpha 转化为预期超额收益率（Grinold-Kahn 公式）。

        α_return_i = IC × σ_i × z_i

        参数确定参考：
            IC   - 因子预期信息系数，典型值 0.03~0.08（可用历史 IC 均值）
            σ_i  - 个股年化波动率，典型 0.25~0.40

        转化后优化器中 γ（L2惩罚）约为 0.3~2.0，
        λ（换手惩罚）约等于单次往返成本 × 年换仓次数（如 0.003~0.015）。

        Parameters
        ----------
        ic : float
            预期信息系数（Information Coefficient）
        stock_vol : array-like or float
            个股年化波动率，shape (N,) 或标量（所有股票相同）

        Returns
        -------
        np.ndarray, shape (N,)
            预期超额年化收益率（小数，如 0.05 表示 5%）
        """
        vol = np.broadcast_to(np.asarray(stock_vol, dtype=float), self._alpha.shape)
        return ic * vol * self._alpha


    @property
    def series(self) -> pd.Series:
        return pd.Series(self._alpha, index=self.snapshot.tickers, name="alpha")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _process(self) -> np.ndarray:
        alpha = self._raw.copy()

        # 1. 缩尾（winsorize）
        lo = np.nanpercentile(alpha, self.winsor_pct * 100)
        hi = np.nanpercentile(alpha, (1 - self.winsor_pct) * 100)
        alpha = np.clip(alpha, lo, hi)

        # 2. 截面 z-score 标准化
        mu, sigma = alpha.mean(), alpha.std()
        if sigma > 1e-10:
            alpha = (alpha - mu) / sigma

        # 3. 停牌 / 次新股 Alpha 置零（不参与收益贡献）
        mask_zero = self.snapshot.suspended_mask | self.snapshot.new_listing_mask
        alpha[mask_zero] = 0.0

        return alpha
