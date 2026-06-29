"""
量化多头 Alpha 最大化优化器（轻量 QP）。

目标函数：
    max  w'α  -  γ · ‖w‖²  -  λ · Σ c_i |w_i - w_prev_i|

    w'α                      : 组合 alpha 收益
    γ·‖w‖²                  : L2 分散惩罚（无需 Sigma，等价于隐式对角风险）
    λ·Σ c_i|w_i - w_prev_i| : 加权换手惩罚（软约束），c_i 为个股成本权重

约束条件：
    sum(w)              = 1                      预算约束
    w_i                >= 0                      纯多头
    w_i                <= weight_upper           个股权重上限
    sum(w[ind==k])      <= industry_upper        行业绝对权重上限
    sum(w[const])       >= min_constituent_ratio 成分股权重下限（可选）
    |B_style[:,k]' w|   <= style_bound           风格因子绝对暴露约束
    ‖w - w_prev‖₁       <= max_turnover          双边换手率硬上限（可选）
    w[停牌/ST/次新]      = 0                      禁止持仓

求解器：CLARABEL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio_optimizer.data.generator import MarketSnapshot, TradingStatus

if TYPE_CHECKING:
    from portfolio_optimizer.risk.cne6_risk import RiskSnapshot

_UNBOUNDED = 1e6


def _resolve_style_bounds(
    bound: "float | dict[str, float]", factor_names: "pd.Index | list[str]"
) -> np.ndarray:
    """把 style_bound（float 或 dict）展开成与因子列对齐的 K 维上限向量。"""
    if isinstance(bound, dict):
        default = bound.get("default", _UNBOUNDED)
        return np.array([float(bound.get(f, default)) for f in factor_names], dtype=float)
    return np.full(len(factor_names), float(bound), dtype=float)


def _industry_group_matrix(ind_values: np.ndarray, n: int) -> np.ndarray:
    """构造行业分组矩阵 G (K×n)，G[k,i]=1 表示股票 i 属于第 k 个行业。

    用 G @ w 一次性表达全部行业权重和，替代逐行业 append 的标量约束。
    """
    uniq = pd.unique(ind_values)
    G = np.zeros((len(uniq), n), dtype=float)
    for k, name in enumerate(uniq):
        G[k, ind_values == name] = 1.0
    return G


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
        L2 分散惩罚系数 γ，默认 0.05。仅在未启用因子风险模型
        （risk_aversion=None）时生效。
        调参参考：
            0.01 ~ 0.05  轻度分散，持仓 50-100 只
            0.05 ~ 0.20  中度分散，持仓 100-200 只
            > 0.20       接近等权，持仓 200+ 只
    style_bound : float | dict[str, float] | None
        风格因子绝对暴露上限，None 表示不约束，默认 1.0
        - float：所有风格因子统一上限（如 1.0 → 每个因子 ±1σ）
        - dict：按因子名分别约束，可含 "default" 键作为未列出因子的兜底，
          例：{"default": 0.8, "Size": 0.3, "Beta": 0.25}
          （未列出且无 default 时该因子不约束）
    max_turnover : float | None
        双边换手率硬上限（0~2），None 表示不约束，默认 None
        例：0.5 表示单期最多换 50% 仓位
    turnover_penalty : float
        换手惩罚系数 λ（软约束），默认 0.0（不惩罚）。
        与 max_turnover 可同时使用：先用软惩罚自然压制换手，
        再用硬上限兜底。
        调参参考：
            0.005 ~ 0.02  轻度惩罚，换手下降 20-40%
            0.02  ~ 0.10  中度惩罚，换手下降 40-70%
            > 0.10        强惩罚，组合趋向保持不变
    risk_aversion : float | None
        因子风险厌恶系数 λ，默认 None。
        - None：退回 L2 分散惩罚 γ·‖w‖²（向后兼容旧行为）
        - 提供且 optimize() 传入 risk_snapshot 时：启用真因子风险模型
          λ·(w'XFX'w + δ'w²)，刻画因子相关性与个股特质风险差异
        与 turnover_penalty 正交：风险项控组合风险，成本项控换手。
    """
    weight_upper: float = 0.02
    industry_upper: float = 0.20
    min_constituent_ratio: float = 0.0
    diversification_penalty: float = 0.05
    style_bound: float | dict[str, float] | None = 1.0
    max_turnover: float | None = None
    turnover_penalty: float = 0.0
    risk_aversion: float | None = None


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
        cost_vector: np.ndarray | None = None,
        risk_snapshot: "RiskSnapshot | None" = None,
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
            上期权重，shape (N,)，用于换手约束/惩罚。
            None 时跳过所有换手相关处理。
        cost_vector : np.ndarray | None, shape (N,)
            个股成本权重，用于加权换手惩罚 Σ c_i|Δw_i|。
            None 时等权（所有股票成本相同）。
            典型用法：传入相对冲击成本，如 1/sqrt(ADV_ratio)，
            使流动性差的股票换手惩罚更强。
        risk_snapshot : RiskSnapshot | None
            CNE6 因子风险模型（X/F/δ，已对齐 snapshot.tickers）。
            与 config.risk_aversion 同时提供时，目标函数用真因子风险
            λ·(w'XFX'w + δ'w²) 替代 L2 分散惩罚。

        Returns
        -------
        AlphaMaxResult
        """
        cfg = self.config
        tickers = snapshot.tickers
        n = len(tickers)

        alpha = np.array(alpha, dtype=float)

        # 禁止持仓的股票：停牌 / 次新 / ST
        banned_status = {
            TradingStatus.SUSPENDED, TradingStatus.NEW_LISTING, TradingStatus.ST,
        }
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

        # ---- 3. 禁止持仓（向量化：一条等式覆盖全部禁持仓票）----
        banned_idx = np.where(banned_mask)[0]
        if len(banned_idx) > 0:
            constraints.append(w[banned_idx] == 0.0)

        # ---- 3b. 涨跌停约束（依赖上期权重，向量化）----
        if prev_weight is not None:
            w_prev_arr = np.array(prev_weight, dtype=float)
            lup_idx = np.where(lup_mask)[0]
            ldn_idx = np.where(ldn_mask)[0]
            if len(lup_idx) > 0:
                constraints.append(w[lup_idx] <= w_prev_arr[lup_idx])   # 涨停：不可加仓
            if len(ldn_idx) > 0:
                constraints.append(w[ldn_idx] >= w_prev_arr[ldn_idx])   # 跌停：不可减仓

        # ---- 4. 行业绝对权重上限（向量化：G @ w <= 上限）----
        industries = snapshot.industry.reindex(tickers).fillna("未知")
        G_ind = _industry_group_matrix(industries.values, n)
        constraints.append(G_ind @ w <= cfg.industry_upper)

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
            bound_vec = _resolve_style_bounds(cfg.style_bound, style_loading.columns)
            constraints.append(exposure <= bound_vec)
            constraints.append(exposure >= -bound_vec)

        # ---- 7. 换手约束与惩罚 ----
        turnover_penalty_term = 0.0
        if prev_weight is not None:
            w_prev = np.array(prev_weight, dtype=float)
            delta_w = cp.abs(w - w_prev)

            # 7a. 软约束：加权换手惩罚进目标函数
            if cfg.turnover_penalty > 0:
                if cost_vector is not None:
                    c = np.array(cost_vector, dtype=float)
                    c = np.clip(c, 0.0, None)
                    turnover_penalty_term = cfg.turnover_penalty * cp.sum(cp.multiply(c, delta_w))
                else:
                    turnover_penalty_term = cfg.turnover_penalty * cp.sum(delta_w)

            # 7b. 硬上限（可与软惩罚同时存在）
            if cfg.max_turnover is not None:
                constraints.append(cp.sum(delta_w) <= cfg.max_turnover)

        # ---- 目标：max w'α - 风险惩罚 - 成本惩罚 ----
        # 风险项：优先 CNE6 因子风险模型 λ·(w'XFX'w + δ'w²)，
        #         未提供时退回 L2 分散惩罚 γ·‖w‖²（向后兼容）
        # 成本项：加权换手惩罚 turnover_penalty_term（软约束，见上方 step 7）
        if cfg.risk_aversion is not None and risk_snapshot is not None:
            X = risk_snapshot.X                       # (N, K)
            F = risk_snapshot.F                       # (K, K)
            delta = risk_snapshot.delta               # (N,)
            factor_risk = cp.quad_form(X.T @ w, cp.psd_wrap(F))
            specific_risk = cp.sum(cp.multiply(delta, cp.square(w)))
            risk_penalty = cfg.risk_aversion * (factor_risk + specific_risk)
        else:
            risk_penalty = cfg.diversification_penalty * cp.sum_squares(w)

        objective = cp.Maximize(alpha @ w - risk_penalty - turnover_penalty_term)

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
