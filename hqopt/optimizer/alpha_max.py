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
    w[停牌且有持仓]       = w_prev                 冻结权重
    w[停牌且无持仓]       = 0                      禁止新开仓
    T 日涨跌停不限制目标方向，是否成交由 T+1 行情决定

交易状态掩码、换手项与求解降级由 :mod:`hqopt.optimizer._common` 与
`index_enhance` 共用。

求解器：CLARABEL（失败降级 SCS）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd

from hqopt.data.generator import MarketSnapshot
from hqopt.optimizer._common import (
    BasePortfolioResult,
    build_trading_masks,
    finalize_weights,
    industry_matrix_for,
    neutralize_alpha,
    resolve_style_bounds,
    solve_with_fallback,
    state_constraints,
    turnover_terms,
    weight_upper_vector,
)

if TYPE_CHECKING:
    from hqopt.risk.cne6_risk import RiskSnapshot


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
        risk_snapshot: RiskSnapshot | None = None,
    ) -> AlphaMaxResult:
        """
        执行优化。

        Parameters
        ----------
        alpha : np.ndarray, shape (N,)
            Alpha 向量，与 snapshot.tickers 对齐。必须是**截面 z-score**：
            α 与 γ/risk_aversion/turnover_penalty 量纲耦合，同一因子排序乘 100 倍
            即可把分散组合压成单票全仓。走 pipeline 时由 `alpha.standardize` 保证。
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

        # 三类互斥交易状态掩码（优先级：frozen > zero > sell_only）
        masks = build_trading_masks(snapshot, prev_weight)
        alpha = neutralize_alpha(alpha, masks)

        w = cp.Variable(n, name="w", nonneg=True)
        constraints = []

        # ---- 1. 预算约束 ----
        constraints.append(cp.sum(w) == 1.0)

        # ---- 2. 个股上限（向量化，冻结票豁免）----
        constraints.append(w <= weight_upper_vector(masks, cfg.weight_upper))

        # ---- 3. 三类状态约束（冻结 / 强制零 / 只卖不买）----
        constraints.extend(state_constraints(w, masks))

        # ---- 4. 行业绝对权重上限（向量化：G @ w <= 上限）----
        G_ind = industry_matrix_for(snapshot, n)
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
            bound_vec = resolve_style_bounds(cfg.style_bound, style_loading.columns)
            constraints.append(exposure <= bound_vec)
            constraints.append(exposure >= -bound_vec)

        # ---- 7. 换手硬上限 + 软惩罚 ----
        # 注：冻结票因 w==w_prev 等式约束，|w-w_prev|=0，自动不消耗换手预算。
        turnover_constraints, turnover_penalty_term = turnover_terms(
            w, masks,
            max_turnover=cfg.max_turnover,
            turnover_penalty=cfg.turnover_penalty,
            cost_vector=cost_vector,
        )
        constraints.extend(turnover_constraints)

        # ---- 目标：max w'α - 风险惩罚 - 成本惩罚 ----
        # 风险项：优先 CNE6 因子风险模型 λ·(w'XFX'w + δ'w²)，
        #         未提供时退回 L2 分散惩罚 γ·‖w‖²（向后兼容）
        if cfg.risk_aversion is not None and risk_snapshot is not None:
            X = risk_snapshot.X                       # (N, K)
            F = risk_snapshot.F                       # (K, K)
            delta = risk_snapshot.delta               # (N,)
            factor_risk = cp.quad_form(X.T @ w, cp.psd_wrap(F))
            specific_risk = cp.sum(cp.multiply(delta, cp.square(w)))
            risk_penalty = cfg.risk_aversion * (factor_risk + specific_risk)
        elif cfg.risk_aversion == 0.0:
            # 显式设 risk_aversion=0：完全关闭风险惩罚（无论有无 risk_snapshot）
            risk_penalty = 0.0
        else:
            risk_penalty = cfg.diversification_penalty * cp.sum_squares(w)

        objective = cp.Maximize(alpha @ w - risk_penalty - turnover_penalty_term)

        prob = cp.Problem(objective, constraints)
        failure = solve_with_fallback(prob)
        if failure is not None:
            return AlphaMaxResult.infeasible(tickers, failure)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return AlphaMaxResult.infeasible(tickers, prob.status)

        return AlphaMaxResult(
            tickers=tickers,
            weights=finalize_weights(np.array(w.value, dtype=float), masks),
            status=prob.status,
            objective_value=float(prob.value),
            snapshot=snapshot,
        )


@dataclass
class AlphaMaxResult(BasePortfolioResult):
    """优化结果。"""

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
