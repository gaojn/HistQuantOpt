"""
指数增强组合优化器（QP）。

候选池为全市场（剔北交所+ST），**不限定宽基成分**；靠 min_constituent_ratio
把目标指数成分权重托到下限（默认 80%），剩余 ≤20% 可配全市场——属「泛化指增」。

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
    |w_i − w_bm_i|          ≤ δ                          单票主动偏离上限（可选）
    ‖w − w_prev‖₁           ≤ T_max                     双边换手率硬上限（可选）
    w[停牌且有持仓]           = w_prev                    冻结权重
    w[停牌且无持仓]           = 0                         禁止新开仓
    T 日涨跌停不限制目标方向，是否成交由 T+1 行情决定

交易状态掩码、换手项与求解降级由 :mod:`hqopt.optimizer._common` 与
`alpha_max` 共用。

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
    annualized_variance_cap,
    build_trading_masks,
    common_constraint_violations,
    finalize_weights,
    industry_matrix_for,
    max_positive_violation,
    neutralize_alpha,
    post_solve_failure_reason,
    post_solve_failures,
    realized_annual_vol,
    resolve_style_bounds,
    risk_quadratic_form,
    solve_with_fallback,
    state_constraints,
    turnover_terms,
    weight_upper_vector,
)

if TYPE_CHECKING:
    from hqopt.risk.cne6_risk import RiskSnapshot


@dataclass
class IndexEnhanceConfig:
    """
    指数增强参数（推荐 HS300 默认）。

    Parameters
    ----------
    weight_upper : float
        单票绝对权重上限（默认 5%，容纳基准重仓股如茅台/宁德 ~5%）
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
    active_weight_upper : float | None
        单票主动偏离硬上限 |w_i - w_bm_i| ≤ δ，None=不约束（默认）。
        典型值：0.01（±1%）～ 0.03（±3%）。
    te_upper : float | None
        **年化跟踪误差硬上限**（小数，0.05 = 5%），None=不约束（默认）。

        启用后目标函数的风险惩罚项自动关闭，退化为
        ``max α'w − 换手惩罚  s.t.  TE(w) ≤ te_upper``——
        即由"软惩罚调 λ 试出 TE"改为"直接指定 TE"，两者是同一有效前沿
        的两种参数化，因此**不可与 risk_aversion 同时设置**（会报错）。

        要求 optimize() 传入 risk_snapshot（需要真实因子协方差），
        否则报错；不会退回 L2 代理（那是 weight_diff_l2_bound，语义不同）。

        注意：仅支持上限。下限 ``TE ≥ x`` 是反凸约束，无法在凸优化框架内
        表达；且关闭惩罚项后目标为线性，最优解通常落在可行域边界上，
        TE 会自然贴近上限。
    """
    weight_upper: float = 0.05
    min_constituent_ratio: float = 0.80
    industry_active_bound: float = 0.05
    style_active_bound: float | dict[str, float] = 0.30
    tracking_penalty: float = 10.0
    max_turnover: float | None = 0.20
    turnover_penalty: float = 0.0
    weight_diff_l2_bound: float | None = None   # ‖w-w_bm‖₂ 硬约束上限
    active_weight_upper: float | None = None    # 单票主动偏离上限 |w_i - w_bm_i| ≤ δ
    risk_aversion: float | None = None
    te_upper: float | None = None               # 年化跟踪误差硬上限（0.05=5%）

    def __post_init__(self) -> None:
        if self.te_upper is not None:
            if self.te_upper <= 0:
                raise ValueError(f"te_upper 必须为正，收到 {self.te_upper}")
            if self.risk_aversion is not None:
                raise ValueError(
                    "te_upper 与 risk_aversion 不可同时设置："
                    "前者以硬约束直接指定跟踪误差，后者以软惩罚间接控制，"
                    "二者是同一权衡的两种参数化。请只保留一个。"
                )


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
        risk_snapshot: RiskSnapshot | None = None,
    ) -> IndexEnhanceResult:
        """执行优化。

        ``alpha`` 必须是截面 z-score（理由同 :meth:`AlphaMaxOptimizer.optimize`）。
        """
        cfg = self.config
        tickers = snapshot.tickers
        n = len(tickers)

        w_bm = np.array(benchmark_weight, dtype=float)
        # 基准权重归一化（防浮点偏差）
        bm_sum = w_bm.sum()
        if bm_sum > 1e-8:
            w_bm = w_bm / bm_sum

        # 三类互斥交易状态掩码（优先级：frozen > zero > sell_only）
        masks = build_trading_masks(snapshot, prev_weight)
        alpha = neutralize_alpha(alpha, masks)

        w = cp.Variable(n, name="w", nonneg=True)
        constraints = []

        # 1. 预算
        constraints.append(cp.sum(w) == 1.0)

        # 2. 个股上界（向量化，冻结票豁免：漂移后持仓可能超绝对上限）
        constraints.append(w <= weight_upper_vector(masks, cfg.weight_upper))

        # 2b. 单票主动偏离上限 |w_i - w_bm_i| ≤ δ（冻结票豁免，其 w 已被等式固定）
        # 下界（w_bm - w <= delta_ub）额外豁免 zero/sellonly：
        #   这类票被制度强制 w=0 或 w<=w_prev，若 w_bm > delta_ub 会因下界导致 infeasible
        #   它们的负主动偏离是被动制度结果，非主动超配，应豁免下界约束
        if cfg.active_weight_upper is not None:
            delta_ub = cfg.active_weight_upper
            non_frozen_idx = np.where(~masks.frozen)[0]
            if len(non_frozen_idx) > 0:
                constraints.append(w[non_frozen_idx] - w_bm[non_frozen_idx] <= delta_ub)
            # 下界豁免 zero/sellonly（制度被迫负偏离，不施加下界约束）
            lower_active_idx = np.where(~masks.restricted)[0]
            if len(lower_active_idx) > 0:
                constraints.append(w_bm[lower_active_idx] - w[lower_active_idx] <= delta_ub)

        # 3. 三类状态约束（冻结 / 强制零 / 只卖不买）
        constraints.extend(state_constraints(w, masks))

        # 4. 成分股权重下限
        if snapshot.is_constituent is not None and cfg.min_constituent_ratio > 0:
            const_idx = np.where(snapshot.constituent_mask)[0]
            if len(const_idx) > 0:
                constraints.append(
                    cp.sum(w[const_idx]) >= cfg.min_constituent_ratio
                )

        # 5. 行业相对基准偏离（向量化：|G@w - G@w_bm| <= 上限）
        G_ind = industry_matrix_for(snapshot, n)
        bm_ind = G_ind @ w_bm                       # (K,) 基准各行业权重
        active_ind = G_ind @ w - bm_ind             # (K,) 主动偏离
        constraints.append(active_ind <= cfg.industry_active_bound)
        constraints.append(active_ind >= -cfg.industry_active_bound)

        # 6. 风格因子主动暴露（支持按因子分别约束）
        if style_loading is not None:
            B = style_loading.reindex(tickers).fillna(0.0).values  # (N, K)
            active_exp = B.T @ (w - w_bm)
            bound_vec = resolve_style_bounds(cfg.style_active_bound, style_loading.columns)
            constraints.append(active_exp <= bound_vec)
            constraints.append(active_exp >= -bound_vec)

        # 7. 换手硬上限 + 软惩罚
        # 注：冻结票因 w==w_prev 等式约束，|w-w_prev|=0，自动不消耗换手预算。
        turnover_constraints, turnover_penalty_term = turnover_terms(
            w, masks,
            max_turnover=cfg.max_turnover,
            turnover_penalty=cfg.turnover_penalty,
            cost_vector=cost_vector,
        )
        constraints.extend(turnover_constraints)

        # 8. L2 偏离基准硬约束（TE 代理）
        if cfg.weight_diff_l2_bound is not None:
            constraints.append(cp.norm(w - w_bm, 2) <= cfg.weight_diff_l2_bound)

        # 9. 跟踪误差硬上限（年化）。启用后关闭目标函数的风险惩罚项：
        #    TE 已被硬约束限定，再加软惩罚等于对同一风险重复收费，
        #    会把解拉到上限以内、失去"直接指定 TE"的语义。
        te_constrained = cfg.te_upper is not None
        if te_constrained:
            if risk_snapshot is None:
                raise ValueError(
                    "启用 te_upper 需要 optimize(risk_snapshot=...) 提供 CNE6 风险模型；"
                    "若只想约束权重偏离，请改用 weight_diff_l2_bound。"
                )
            te_var = risk_quadratic_form(
                w - w_bm, risk_snapshot.X, risk_snapshot.F, risk_snapshot.delta
            )
            assert cfg.te_upper is not None      # te_constrained 的定义即此
            constraints.append(te_var <= annualized_variance_cap(cfg.te_upper))

        # 目标函数：max w'α - 主动风险惩罚 - λ·Σ c_i|Δw_i|
        # 风险项：优先 CNE6 因子风险模型 λ·(active'XFX'active + δ'active²)，
        #         未提供时退回 L2 偏离惩罚 γ·‖w-w_bm‖²（向后兼容）
        if te_constrained:
            risk_penalty = 0.0
        elif cfg.risk_aversion is not None and risk_snapshot is not None:
            risk_penalty = cfg.risk_aversion * risk_quadratic_form(
                w - w_bm, risk_snapshot.X, risk_snapshot.F, risk_snapshot.delta
            )
        elif cfg.risk_aversion == 0.0:
            # 显式设 risk_aversion=0：完全关闭风险惩罚（无论有无 risk_snapshot）
            risk_penalty = 0.0
        else:
            risk_penalty = cfg.tracking_penalty * cp.sum_squares(w - w_bm)

        objective = cp.Maximize(alpha @ w - risk_penalty - turnover_penalty_term)

        prob = cp.Problem(objective, constraints)
        failure = solve_with_fallback(prob)
        # TE 上限与成分股下限/行业偏离/换手上限叠加时很容易不可行，
        # 笼统的 "infeasible" 无法指向病因，故在原因里点名 TE 约束。
        te_hint = (
            f"（已启用 te_upper={cfg.te_upper:.2%} 硬约束，"
            "与成分股/行业/换手约束叠加可能不可行，可先放宽本项排查）"
            if te_constrained
            else ""
        )
        if failure is not None:
            return IndexEnhanceResult.infeasible(tickers, f"{failure}{te_hint}")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return IndexEnhanceResult.infeasible(tickers, f"{prob.status}{te_hint}")

        weights = finalize_weights(np.array(w.value, dtype=float), masks)
        violations = common_constraint_violations(
            weights,
            masks,
            weight_upper=cfg.weight_upper,
            max_turnover=cfg.max_turnover,
        )
        if cfg.active_weight_upper is not None:
            non_frozen_idx = np.where(~masks.frozen)[0]
            violations["active_weight_upper_positive"] = max_positive_violation(
                weights[non_frozen_idx]
                - w_bm[non_frozen_idx]
                - cfg.active_weight_upper
            )
            lower_active_idx = np.where(~masks.restricted)[0]
            violations["active_weight_upper_negative"] = max_positive_violation(
                w_bm[lower_active_idx]
                - weights[lower_active_idx]
                - cfg.active_weight_upper
            )
        if snapshot.is_constituent is not None and cfg.min_constituent_ratio > 0:
            post_const_idx = np.where(snapshot.constituent_mask)[0]
            if len(post_const_idx) > 0:
                violations["constituent_minimum"] = max_positive_violation(
                    cfg.min_constituent_ratio - weights[post_const_idx].sum()
                )
        violations["industry_active"] = max_positive_violation(
            np.abs(G_ind @ weights - bm_ind) - cfg.industry_active_bound
        )
        if style_loading is not None:
            post_style = style_loading.reindex(tickers).fillna(0.0).values
            post_bound = resolve_style_bounds(
                cfg.style_active_bound, style_loading.columns
            )
            violations["style_active"] = max_positive_violation(
                np.abs(post_style.T @ (weights - w_bm)) - post_bound
            )
        if cfg.weight_diff_l2_bound is not None:
            violations["weight_diff_l2"] = max_positive_violation(
                np.linalg.norm(weights - w_bm) - cfg.weight_diff_l2_bound
            )
        if te_constrained:
            # 与其他硬约束一视同仁做求解后校验：数值求解可能轻微越界，
            # 超出容差即拒绝发布，避免"声称 TE≤5% 实则更高"的组合流入生产。
            # te_constrained 蕴含二者非 None（上方构建约束时已校验并报错）。
            assert risk_snapshot is not None and cfg.te_upper is not None
            realized_te = realized_annual_vol(
                weights - w_bm,
                risk_snapshot.X,
                risk_snapshot.F,
                risk_snapshot.delta,
            )
            violations["tracking_error_upper"] = max_positive_violation(
                realized_te - cfg.te_upper
            )
        failures = post_solve_failures(violations)
        if failures:
            return IndexEnhanceResult.infeasible(
                tickers,
                post_solve_failure_reason(failures),
                constraint_violations=violations,
            )

        return IndexEnhanceResult(
            tickers=tickers,
            weights=weights,
            status=prob.status,
            objective_value=float(prob.value),
            snapshot=snapshot,
            constraint_violations=violations,
            benchmark_weight=w_bm,
        )


@dataclass
class IndexEnhanceResult(BasePortfolioResult):
    """指数增强优化结果。"""

    benchmark_weight: np.ndarray | None = None

    @property
    def active_weight(self) -> np.ndarray | None:
        if self.benchmark_weight is None:
            return None
        return self.weights - self.benchmark_weight

    def top_holdings(self, n: int = 10) -> pd.DataFrame:
        df = super().top_holdings(n)
        if self.benchmark_weight is not None:
            bm_s = pd.Series(self.benchmark_weight * 100, index=self.tickers)
            df["bm_weight_pct"] = bm_s.reindex(df.index)
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
        ind = self._industry_series()
        if ind is None or self.benchmark_weight is None:
            return pd.Series(dtype=float)
        port = pd.Series(self.weights, index=self.tickers).groupby(ind.values).sum()
        bm   = pd.Series(self.benchmark_weight, index=self.tickers).groupby(ind.values).sum()
        return (port - bm).sort_values(ascending=False)

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
