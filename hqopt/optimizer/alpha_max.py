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
    vol_upper : float | None
        **年化组合波动率硬上限**（小数，0.20 = 20%），None=不约束（默认）。

        启用后目标函数的风险惩罚项自动关闭，退化为
        ``max α'w − 换手惩罚  s.t.  σ(w) ≤ vol_upper``——
        即由"软惩罚调 λ 试出波动率"改为"直接指定波动率"，两者是同一
        有效前沿的两种参数化，因此**不可与 risk_aversion 同时设置**（会报错）。

        要求 optimize() 传入 risk_snapshot，否则报错。

        注意：仅支持上限。下限 ``σ ≥ x`` 是反凸约束，无法在凸优化框架内表达。
    """
    weight_upper: float = 0.02
    industry_upper: float = 0.20
    min_constituent_ratio: float = 0.0
    diversification_penalty: float = 0.05
    style_bound: float | dict[str, float] | None = 1.0
    max_turnover: float | None = None
    turnover_penalty: float = 0.0
    risk_aversion: float | None = None
    vol_upper: float | None = None              # 年化组合波动率硬上限（0.20=20%）

    def __post_init__(self) -> None:
        if self.vol_upper is not None:
            if self.vol_upper <= 0:
                raise ValueError(f"vol_upper 必须为正，收到 {self.vol_upper}")
            if self.risk_aversion is not None:
                raise ValueError(
                    "vol_upper 与 risk_aversion 不可同时设置："
                    "前者以硬约束直接指定组合波动率，后者以软惩罚间接控制，"
                    "二者是同一权衡的两种参数化。请只保留一个。"
                )


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
            const_mask = snapshot.constituent_mask
            # 防退化：成分掩码全 True 时该约束是 sum(全部 w) >= R，在预算约束
            # sum(w)==1 下恒成立——配置里写着下限、实际毫无作用。空转的风控比
            # 没有风控更危险（使用者据此以为容量/流动性有保障），故直接报错而非
            # 静默接受。曾真实发生：index='all' 把全部股票标为成分（见
            # RealMarketAdapter.build_snapshot_from_panel）。
            if const_mask.all():
                raise ValueError(
                    f"min_constituent_ratio={cfg.min_constituent_ratio} 但成分掩码"
                    f"全为 True（{len(const_mask)} 只全是成分），该约束在预算约束下"
                    "恒成立、不起任何作用。请检查 snapshot.is_constituent 的口径，"
                    "或把 min_constituent_ratio 设为 0 显式表示不约束。"
                )
            const_idx = np.where(const_mask)[0]
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

        # ---- 8. 组合波动率硬上限（年化）----
        # 启用后关闭目标函数的风险惩罚项：波动率已被硬约束限定，
        # 再加软惩罚等于对同一风险重复收费，会让解退到上限以内。
        vol_constrained = cfg.vol_upper is not None
        if vol_constrained:
            if risk_snapshot is None:
                raise ValueError(
                    "启用 vol_upper 需要 optimize(risk_snapshot=...) 提供 CNE6 风险模型；"
                    "若只想控制集中度，请改用 diversification_penalty 或 weight_upper。"
                )
            vol_var = risk_quadratic_form(
                w, risk_snapshot.X, risk_snapshot.F, risk_snapshot.delta
            )
            assert cfg.vol_upper is not None     # vol_constrained 的定义即此
            constraints.append(vol_var <= annualized_variance_cap(cfg.vol_upper))

        # ---- 目标：max w'α - 风险惩罚 - 成本惩罚 ----
        # 风险项：优先 CNE6 因子风险模型 λ·(w'XFX'w + δ'w²)，
        #         未提供时退回 L2 分散惩罚 γ·‖w‖²（向后兼容）
        if vol_constrained:
            risk_penalty = 0.0
        elif cfg.risk_aversion is not None and risk_snapshot is not None:
            risk_penalty = cfg.risk_aversion * risk_quadratic_form(
                w, risk_snapshot.X, risk_snapshot.F, risk_snapshot.delta
            )
        elif cfg.risk_aversion == 0.0:
            # 显式设 risk_aversion=0：完全关闭风险惩罚（无论有无 risk_snapshot）
            risk_penalty = 0.0
        else:
            risk_penalty = cfg.diversification_penalty * cp.sum_squares(w)

        objective = cp.Maximize(alpha @ w - risk_penalty - turnover_penalty_term)

        prob = cp.Problem(objective, constraints)
        failure = solve_with_fallback(prob)
        # 波动率上限与个股/行业/换手约束叠加时可能不可行，原因里点名本约束
        vol_hint = (
            f"（已启用 vol_upper={cfg.vol_upper:.2%} 硬约束，"
            "与个股/行业/换手约束叠加可能不可行，可先放宽本项排查）"
            if vol_constrained
            else ""
        )
        if failure is not None:
            return AlphaMaxResult.infeasible(tickers, f"{failure}{vol_hint}")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return AlphaMaxResult.infeasible(tickers, f"{prob.status}{vol_hint}")

        weights = finalize_weights(np.array(w.value, dtype=float), masks)
        violations = common_constraint_violations(
            weights,
            masks,
            weight_upper=cfg.weight_upper,
            max_turnover=cfg.max_turnover,
        )
        violations["industry_upper"] = max_positive_violation(
            G_ind @ weights - cfg.industry_upper
        )
        if (
            cfg.min_constituent_ratio > 0
            and snapshot.is_constituent is not None
        ):
            post_const_idx = np.where(snapshot.constituent_mask)[0]
            if len(post_const_idx) > 0:
                violations["constituent_minimum"] = max_positive_violation(
                    cfg.min_constituent_ratio - weights[post_const_idx].sum()
                )
        if style_loading is not None and cfg.style_bound is not None:
            post_style = style_loading.reindex(tickers).fillna(0.0).values
            post_bound = resolve_style_bounds(
                cfg.style_bound, style_loading.columns
            )
            violations["style_absolute"] = max_positive_violation(
                np.abs(post_style.T @ weights) - post_bound
            )
        if vol_constrained:
            # 硬约束一律做求解后校验，避免"声称 σ≤20% 实则更高"的组合流入生产
            # vol_constrained 蕴含二者非 None（上方构建约束时已校验并报错）。
            assert risk_snapshot is not None and cfg.vol_upper is not None
            realized_vol = realized_annual_vol(
                weights, risk_snapshot.X, risk_snapshot.F, risk_snapshot.delta
            )
            violations["volatility_upper"] = max_positive_violation(
                realized_vol - cfg.vol_upper
            )
        failures = post_solve_failures(violations)
        if failures:
            return AlphaMaxResult.infeasible(
                tickers,
                post_solve_failure_reason(failures),
                constraint_violations=violations,
            )

        return AlphaMaxResult(
            tickers=tickers,
            weights=weights,
            status=prob.status,
            objective_value=float(prob.value),
            snapshot=snapshot,
            constraint_violations=violations,
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
