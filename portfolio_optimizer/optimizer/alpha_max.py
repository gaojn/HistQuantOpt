"""
量化多头 Alpha 最大化优化器（轻量 QP）。

目标函数：
    max  w'α  -  γ · ‖w‖²

    w'α       : 组合 alpha 收益
    γ·‖w‖²   : L2 分散惩罚（无需 Sigma，等价于隐式对角风险）
               γ 越大 → 持仓越分散，单票集中度越低

约束条件：
    sum(w)              = 1                      预算约束
    w_i                >= 0                      纯多头
    w_i                <= weight_upper           个股权重上限
    sum(w[ind==k])      <= industry_upper        行业绝对权重上限
    sum(w[const])       >= min_constituent_ratio 成分股权重下限（可选）
    |B_style[:,k]' w|   <= style_bound           风格因子绝对暴露约束
    ‖w - w_prev‖₁       <= max_turnover          双边换手率约束
    w[停牌/ST/次新]      = 0                      禁止持仓

求解器：CLARABEL
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_optimizer.data.generator import MarketSnapshot, TradingStatus


@dataclass
class AlphaMaxConfig:
    """
    量化多头优化参数。

    Parameters
    ----------
    weight_upper : float
        单票权重上限，默认 2%
    industry_upper : float
        单行业权重绝对上限，默认 20%
    min_constituent_ratio : float
        成分股权重下限，0 表示不约束，默认 0.0
    diversification_penalty : float
        L2 分散惩罚系数 γ，默认 0.05。
        调参参考：
            0.01 ~ 0.05  轻度分散，持仓 50-100 只
            0.05 ~ 0.20  中度分散，持仓 100-200 只
            > 0.20       接近等权，持仓 200+ 只
    style_bound : float | None
        风格因子绝对暴露上限，None 表示不约束，默认 1.0
        即 |B_style[:,k]' w| <= style_bound  对所有风格因子 k
    max_turnover : float | None
        双边换手率上限（0~2），None 表示不约束，默认 None
        例：0.5 表示单期最多换 50% 仓位
    """
    weight_upper: float = 0.02
    industry_upper: float = 0.20
    min_constituent_ratio: float = 0.0
    diversification_penalty: float = 0.05
    style_bound: float | None = 1.0
    max_turnover: float | None = None


class AlphaMaxOptimizer:
    """
    量化多头 Alpha 最大化优化器。

    Parameters
    ----------
    config : AlphaMaxConfig
    """

    def __init__(self, config: AlphaMaxConfig) -> None:
        self.config = config

    def optimize(
        self,
        alpha: np.ndarray,
        snapshot: MarketSnapshot,
        style_loading: pd.DataFrame | None = None,
        prev_weight: np.ndarray | None = None,
    ) -> "AlphaMaxResult":
        """
        执行优化。

        Parameters
        ----------
        alpha : np.ndarray, shape (N,)
            Alpha 向量，与 snapshot.tickers 对齐
        snapshot : MarketSnapshot
            市场快照
        style_loading : pd.DataFrame | None
            风格因子载荷矩阵，shape (N, K)，index=tickers。
            传入时启用风格约束，None 则跳过。
        prev_weight : np.ndarray | None
            上期权重，shape (N,)，用于换手率约束。
            None 时跳过换手约束。

        Returns
        -------
        AlphaMaxResult
        """
        cfg = self.config
        tickers = snapshot.tickers
        n = len(tickers)

        alpha = np.array(alpha, dtype=float)

        # 禁止持仓的股票：停牌 / ST / 次新
        banned_status = {TradingStatus.SUSPENDED, TradingStatus.NEW_LISTING}
        banned_mask = np.array(
            [s in banned_status for s in snapshot.status.values], dtype=bool
        )
        # 禁止股票 alpha 清零，避免干扰目标函数方向
        alpha[banned_mask] = 0.0

        # 涨停：不可加仓；跌停：不可减仓
        lup_mask = snapshot.limit_up_mask
        ldn_mask = snapshot.limit_down_mask

        w = cp.Variable(n, name="w", nonneg=True)
        constraints = []

        # ---- 1. 预算约束 ----
        constraints.append(cp.sum(w) == 1.0)

        # ---- 2. 个股上限 ----
        constraints.append(w <= cfg.weight_upper)

        # ---- 3. 禁止持仓 ----
        banned_idx = np.where(banned_mask)[0]
        for i in banned_idx:
            constraints.append(w[i] == 0.0)

        # ---- 3b. 涨跌停约束（依赖上期权重）----
        if prev_weight is not None:
            w_prev_arr = np.array(prev_weight, dtype=float)
            for i in np.where(lup_mask)[0]:
                constraints.append(w[i] <= float(w_prev_arr[i]))   # 涨停：不可加仓
            for i in np.where(ldn_mask)[0]:
                constraints.append(w[i] >= float(w_prev_arr[i]))   # 跌停：不可减仓

        # ---- 4. 行业绝对权重上限 ----
        industries = snapshot.industry.reindex(tickers).fillna("未知")
        for ind_name in industries.unique():
            idx = np.where(industries.values == ind_name)[0]
            if len(idx) > 0:
                constraints.append(cp.sum(w[idx]) <= cfg.industry_upper)

        # ---- 5. 成分股权重下限（可选）----
        if (
            cfg.min_constituent_ratio > 0
            and snapshot.is_constituent is not None
        ):
            const_idx = np.where(snapshot.constituent_mask)[0]
            if len(const_idx) > 0:
                constraints.append(
                    cp.sum(w[const_idx]) >= cfg.min_constituent_ratio
                )

        # ---- 6. 风格因子绝对暴露约束 ----
        if style_loading is not None and cfg.style_bound is not None:
            B = style_loading.reindex(tickers).fillna(0.0).values  # (N, K)
            # |B[:,k]' w| <= style_bound  逐因子
            exposure = B.T @ w   # (K,)
            constraints.append(exposure <= cfg.style_bound)
            constraints.append(exposure >= -cfg.style_bound)

        # ---- 7. 换手率约束 ----
        if prev_weight is not None and cfg.max_turnover is not None:
            w_prev = np.array(prev_weight, dtype=float)
            # 双边换手：sum |w_i - w_prev_i| <= max_turnover
            turnover = cp.sum(cp.abs(w - w_prev))
            constraints.append(turnover <= cfg.max_turnover)

        # ---- 目标：max w'α - γ·‖w‖² ----
        objective = cp.Maximize(
            alpha @ w - cfg.diversification_penalty * cp.sum_squares(w)
        )

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            return AlphaMaxResult.infeasible(tickers, str(e))

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return AlphaMaxResult.infeasible(tickers, prob.status)

        weights = np.clip(np.array(w.value, dtype=float), 0.0, None)
        if weights.sum() > 1e-8:
            weights /= weights.sum()

        return AlphaMaxResult(
            tickers=tickers,
            weights=weights,
            status=prob.status,
            objective_value=float(prob.value),
            snapshot=snapshot,
        )


@dataclass
class AlphaMaxResult:
    """优化结果。"""
    tickers: list[str]
    weights: np.ndarray
    status: str
    objective_value: float
    snapshot: MarketSnapshot | None

    @classmethod
    def infeasible(cls, tickers: list[str], reason: str) -> "AlphaMaxResult":
        return cls(
            tickers=tickers,
            weights=np.zeros(len(tickers)),
            status=f"infeasible: {reason}",
            objective_value=float("nan"),
            snapshot=None,
        )

    @property
    def is_feasible(self) -> bool:
        return "optimal" in self.status

    @property
    def n_positions(self) -> int:
        return int((self.weights > 1e-6).sum())

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.tickers, name="weight")

    def top_holdings(self, n: int = 10) -> pd.DataFrame:
        s = self.to_series().sort_values(ascending=False).head(n)
        df = s.to_frame("weight")
        df["weight_pct"] = df["weight"] * 100
        if self.snapshot is not None:
            df["industry"] = self.snapshot.industry.reindex(s.index)
            if self.snapshot.is_constituent is not None:
                df["is_constituent"] = self.snapshot.is_constituent.reindex(s.index)
        return df

    def industry_weights(self) -> pd.Series:
        if self.snapshot is None:
            return pd.Series(dtype=float)
        ind = self.snapshot.industry.reindex(self.tickers).fillna("未知")
        return self.to_series().groupby(ind.values).sum().sort_values(ascending=False)

    def style_exposures(self, style_loading: pd.DataFrame) -> pd.Series:
        """计算组合风格因子暴露。"""
        B = style_loading.reindex(self.tickers).fillna(0.0)
        return B.T @ pd.Series(self.weights, index=self.tickers)

    def summary(self) -> str:
        lines = [
            f"状态       : {self.status}",
            f"持仓数     : {self.n_positions}",
            f"权重和     : {self.weights.sum():.6f}",
            f"最大单票   : {self.weights.max()*100:.3f}%",
            f"HHI        : {(self.weights**2).sum():.6f}",
        ]
        if self.snapshot is not None and self.snapshot.is_constituent is not None:
            const_w = self.weights[self.snapshot.constituent_mask].sum()
            lines.append(f"成分股权重 : {const_w*100:.2f}%")
        return "\n".join(lines)
