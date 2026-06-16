"""
指数增强组合优化器（QP）。

目标函数：
    max  w'α  -  γ · ‖w − w_bm‖²  -  λ · Σ c_i |w_i - w_prev_i|

    w'α                      : 组合预期收益
    γ·‖w−w_bm‖²              : 跟踪误差代理（L2 偏离基准的惩罚）
    λ·Σ c_i|w_i - w_prev_i| : 加权换手惩罚（软约束），c_i 为个股成本权重

约束体系（相对基准）：
    sum(w)                  = 1
    0 ≤ w_i                 ≤ W_max
    sum(w[const])           ≥ R_min                      成分股下限（如 HS300 ≥ 80%）
    |Σ_{i∈ind_k}(w_i-w_bm_i)| ≤ I_active                行业主动偏离 (±5%)
    |B_style[:,k]'(w-w_bm)| ≤ S_active                  风格主动暴露 (±0.3σ)
    ‖w − w_prev‖₁           ≤ T_max                     双边换手率硬上限（可选）
    w[停牌/ST/次新]          = 0                         交易状态

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

# 未列出且无 default 时，用极大值表示"不约束"（避免 cvxpy 的 inf 问题）
_UNBOUNDED = 1e6


def _resolve_style_bounds(
    bound: "float | dict[str, float]", factor_names: "pd.Index | list[str]"
) -> np.ndarray:
    """把 style_active_bound（float 或 dict）展开成与因子列对齐的 K 维上限向量。"""
    if isinstance(bound, dict):
        default = bound.get("default", _UNBOUNDED)
        return np.array([float(bound.get(f, default)) for f in factor_names], dtype=float)
    return np.full(len(factor_names), float(bound), dtype=float)


@dataclass
class IndexEnhanceConfig:
    """
    指数增强参数（推荐 HS300 默认）。

    Parameters
    ----------
    weight_upper : float
        单票绝对权重上限（默认 5%，容纳基准重仓股如茅台/宁德 ~5%）
    weight_lower : float
        单票权重下限，默认 0
    min_constituent_ratio : float
        成分股权重下限（HS300 ≥ 80%）
    industry_active_bound : float
        行业相对基准偏离上限（±5%）
    style_active_bound : float | dict[str, float]
        风格因子主动暴露上限（±x σ）。
        - float：所有风格因子统一上限（如 0.30 → 每个因子 ±0.3σ）
        - dict：按因子名分别约束，可含 "default" 键作为未列出因子的兜底，
          例：{"default": 0.3, "Momentum": 0.2, "Size": 0.5}
          （未列出且无 default 时该因子不约束）
    tracking_penalty : float
        跟踪误差惩罚系数 γ（越大越像基准）。仅在未启用因子风险模型
        （risk_aversion=None）时生效。
    max_turnover : float | None
        双边换手率硬上限（默认 20%）
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
        - None：退回 L2 偏离惩罚 γ·‖w−w_bm‖²（向后兼容旧行为）
        - 提供且 optimize() 传入 risk_snapshot 时：用真跟踪误差
          λ·(active'XFX'active + δ'active²)，active=w−w_bm，
          刻画相对基准的真实主动风险（因子相关性 + 特质风险）
        与 turnover_penalty 正交：风险项控主动风险，成本项控换手。
    """
    weight_upper: float = 0.05
    weight_lower: float = 0.0
    min_constituent_ratio: float = 0.80
    industry_active_bound: float = 0.05
    style_active_bound: float | dict[str, float] = 0.30
    tracking_penalty: float = 10.0
    max_turnover: float | None = 0.20
    turnover_penalty: float = 0.0
    weight_diff_l2_bound: float | None = None   # ‖w-w_bm‖₂ 硬约束上限
    risk_aversion: float | None = None


class IndexEnhanceOptimizer:
    """指数增强优化器。"""

    def __init__(self, config: IndexEnhanceConfig) -> None:
        self.config = config

    def optimize(
        self,
        alpha: np.ndarray,
        snapshot: MarketSnapshot,
        benchmark_weight: np.ndarray,
        style_loading: pd.DataFrame | None = None,
        prev_weight: np.ndarray | None = None,
        cost_vector: np.ndarray | None = None,
        risk_snapshot: "RiskSnapshot | None" = None,
    ) -> "IndexEnhanceResult":
        cfg = self.config
        tickers = snapshot.tickers
        n = len(tickers)

        alpha = np.array(alpha, dtype=float)
        w_bm = np.array(benchmark_weight, dtype=float)
        # 基准权重归一化（防浮点偏差）
        bm_sum = w_bm.sum()
        if bm_sum > 1e-8:
            w_bm = w_bm / bm_sum

        # 禁止持仓
        banned_status = {TradingStatus.SUSPENDED, TradingStatus.NEW_LISTING}
        banned_mask = np.array(
            [s in banned_status for s in snapshot.status.values], dtype=bool
        )
        alpha[banned_mask] = 0.0

        # 涨停：不可加仓（w_i ≤ w_prev_i）；跌停：不可减仓（w_i ≥ w_prev_i）
        lup_mask = snapshot.limit_up_mask   # bool array, shape (n,)
        ldn_mask = snapshot.limit_down_mask

        w = cp.Variable(n, name="w", nonneg=True)
        constraints = []

        # 1. 预算
        constraints.append(cp.sum(w) == 1.0)

        # 2. 个股区间
        constraints.append(w <= cfg.weight_upper)

        # 3. 禁止持仓
        for i in np.where(banned_mask)[0]:
            constraints.append(w[i] == 0.0)

        # 3b. 涨跌停约束（依赖上期权重）
        if prev_weight is not None:
            w_prev = np.array(prev_weight, dtype=float)
            for i in np.where(lup_mask)[0]:
                constraints.append(w[i] <= float(w_prev[i]))   # 涨停：不可加仓
            for i in np.where(ldn_mask)[0]:
                constraints.append(w[i] >= float(w_prev[i]))   # 跌停：不可减仓

        # 4. 成分股权重下限
        if snapshot.is_constituent is not None and cfg.min_constituent_ratio > 0:
            const_idx = np.where(snapshot.constituent_mask)[0]
            if len(const_idx) > 0:
                constraints.append(
                    cp.sum(w[const_idx]) >= cfg.min_constituent_ratio
                )

        # 5. 行业相对基准偏离
        industries = snapshot.industry.reindex(tickers).fillna("未知")
        for ind_name in industries.unique():
            idx = np.where(industries.values == ind_name)[0]
            if len(idx) == 0:
                continue
            bm_ind = float(w_bm[idx].sum())
            active_ind = cp.sum(w[idx]) - bm_ind
            constraints.append(active_ind <= cfg.industry_active_bound)
            constraints.append(active_ind >= -cfg.industry_active_bound)

        # 6. 风格因子主动暴露（支持按因子分别约束）
        if style_loading is not None:
            B = style_loading.reindex(tickers).fillna(0.0).values  # (N, K)
            active_exp = B.T @ (w - w_bm)
            bound_vec = _resolve_style_bounds(cfg.style_active_bound, style_loading.columns)
            constraints.append(active_exp <= bound_vec)
            constraints.append(active_exp >= -bound_vec)

        # 7. 换手约束与惩罚
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

        # 8. L2 偏离基准硬约束（TE 代理）
        if cfg.weight_diff_l2_bound is not None:
            constraints.append(cp.norm(w - w_bm, 2) <= cfg.weight_diff_l2_bound)

        # 目标函数：max w'α - 主动风险惩罚 - λ·Σ c_i|Δw_i|
        # 风险项：优先 CNE6 因子风险模型 λ·(active'XFX'active + δ'active²)，
        #         未提供时退回 L2 偏离惩罚 γ·‖w-w_bm‖²（向后兼容）
        # 成本项：加权换手惩罚 turnover_penalty_term（软约束，见上方 step 7）
        if cfg.risk_aversion is not None and risk_snapshot is not None:
            active = w - w_bm
            X = risk_snapshot.X                       # (N, K)
            F = risk_snapshot.F                       # (K, K)
            delta = risk_snapshot.delta               # (N,)
            factor_te = cp.quad_form(X.T @ active, cp.psd_wrap(F))
            specific_te = cp.sum(cp.multiply(delta, cp.square(active)))
            risk_penalty = cfg.risk_aversion * (factor_te + specific_te)
        else:
            risk_penalty = cfg.tracking_penalty * cp.sum_squares(w - w_bm)

        objective = cp.Maximize(alpha @ w - risk_penalty - turnover_penalty_term)

        prob = cp.Problem(objective, constraints)
        # 优先 CLARABEL（max_iter 提至 500，应对大规模候选池）；
        # 失败时降级 SCS 兜底
        # 优先 CLARABEL，失败再尝试 SCS 兜底
        clarabel_ok = False
        try:
            prob.solve(solver=cp.CLARABEL, max_iter=500, verbose=False)
            clarabel_ok = prob.status in ("optimal", "optimal_inaccurate")
        except Exception:
            clarabel_ok = False

        if not clarabel_ok:
            try:
                prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
            except Exception as e:
                return IndexEnhanceResult.infeasible(tickers, f"both solvers failed: {e}")

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return IndexEnhanceResult.infeasible(tickers, prob.status)

        weights = np.clip(np.array(w.value, dtype=float), 0.0, None)
        if weights.sum() > 1e-8:
            weights /= weights.sum()

        return IndexEnhanceResult(
            tickers=tickers,
            weights=weights,
            status=prob.status,
            objective_value=float(prob.value),
            snapshot=snapshot,
            benchmark_weight=w_bm,
        )


@dataclass
class IndexEnhanceResult:
    """指数增强优化结果。"""
    tickers: list[str]
    weights: np.ndarray
    status: str
    objective_value: float
    snapshot: MarketSnapshot | None
    benchmark_weight: np.ndarray | None = None

    @classmethod
    def infeasible(cls, tickers: list[str], reason: str) -> "IndexEnhanceResult":
        return cls(
            tickers=tickers,
            weights=np.zeros(len(tickers)),
            status=f"infeasible: {reason}",
            objective_value=float("nan"),
            snapshot=None,
            benchmark_weight=None,
        )

    @property
    def is_feasible(self) -> bool:
        return "optimal" in self.status

    @property
    def n_positions(self) -> int:
        return int((self.weights > 1e-6).sum())

    @property
    def active_weight(self) -> np.ndarray | None:
        if self.benchmark_weight is None:
            return None
        return self.weights - self.benchmark_weight

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
        if self.benchmark_weight is not None:
            bm_s = pd.Series(self.benchmark_weight * 100, index=self.tickers)
            df["bm_weight_pct"] = bm_s.reindex(s.index)
            df["active_pct"]    = df["weight_pct"] - df["bm_weight_pct"]
        return df

    def style_active_exposure(self, style_loading: pd.DataFrame) -> pd.Series:
        """组合相对基准的风格主动暴露。"""
        if self.benchmark_weight is None:
            return pd.Series(dtype=float)
        B = style_loading.reindex(self.tickers).fillna(0.0)
        active = pd.Series(
            self.weights - self.benchmark_weight, index=self.tickers
        )
        return B.T @ active

    def industry_active_weights(self) -> pd.Series:
        """各行业相对基准的主动权重偏离（正=超配，负=低配）。"""
        if self.snapshot is None or self.benchmark_weight is None:
            return pd.Series(dtype=float)
        ind = self.snapshot.industry.reindex(self.tickers).fillna("未知")
        port = pd.Series(self.weights, index=self.tickers).groupby(ind.values).sum()
        bm   = pd.Series(self.benchmark_weight, index=self.tickers).groupby(ind.values).sum()
        return (port - bm).sort_values(ascending=False)

    def industry_weights(self) -> pd.Series:
        """各行业绝对权重。"""
        if self.snapshot is None:
            return pd.Series(dtype=float)
        ind = self.snapshot.industry.reindex(self.tickers).fillna("未知")
        return pd.Series(self.weights, index=self.tickers) \
            .groupby(ind.values).sum().sort_values(ascending=False)

    def tracking_error_l2(self) -> float:
        """跟踪误差 L2 范数（粗略代理）。"""
        if self.benchmark_weight is None:
            return float("nan")
        return float(np.linalg.norm(self.weights - self.benchmark_weight))

    def summary(self) -> str:
        lines = [
            f"状态           : {self.status}",
            f"持仓数         : {self.n_positions}",
            f"权重和         : {self.weights.sum():.6f}",
            f"最大单票       : {self.weights.max()*100:.3f}%",
        ]
        if self.snapshot is not None and self.snapshot.is_constituent is not None:
            const_w = self.weights[self.snapshot.constituent_mask].sum()
            lines.append(f"HS300 权重     : {const_w*100:.2f}%")
        if self.benchmark_weight is not None:
            active_l1 = float(np.abs(self.weights - self.benchmark_weight).sum())
            lines.append(f"主动权重 L1    : {active_l1:.4f}")
            lines.append(f"主动权重 L2    : {self.tracking_error_l2():.4f}")
        return "\n".join(lines)
