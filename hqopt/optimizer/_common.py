"""两个 QP 优化器共用的约束构造、求解与结果基类。

`alpha_max`（绝对收益）与 `index_enhance`（相对基准）只在**目标函数**和
**基准相关约束**上不同；交易状态处理、个股上限、换手预算、求解器降级和结果
展示完全一致。这些共性集中在本模块，避免执行语义修正只改了一侧——回测框架里
"改一处漏一处"是最典型的隐性 bug 源。

各优化器仍在自己的 `optimize()` 里按顺序 append 约束，约束体系保持线性可读。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from hqopt.data.generator import MarketSnapshot

# 未列出且无 default 时，用极大值表示"不约束"（避免 cvxpy 的 inf 问题）
UNBOUNDED = 1e6

# 求解器参数：CLARABEL 为主（max_iter 提至 500 应对大规模候选池），SCS 兜底
_CLARABEL_MAX_ITER = 500
_SCS_MAX_ITERS = 10000
_OPTIMAL_STATUSES = ("optimal", "optimal_inaccurate")


def resolve_style_bounds(
    bound: float | dict[str, float],
    factor_names: pd.Index | list[str],
) -> np.ndarray:
    """把风格约束（float 或 dict）展开成与因子列对齐的 K 维上限向量。

    dict 形式可含 "default" 键作为未列出因子的兜底；未列出且无 default 时
    该因子不约束（取 UNBOUNDED）。
    """
    if isinstance(bound, dict):
        default = bound.get("default", UNBOUNDED)
        return np.array([float(bound.get(f, default)) for f in factor_names], dtype=float)
    return np.full(len(factor_names), float(bound), dtype=float)


def industry_group_matrix(ind_values: np.ndarray, n: int) -> np.ndarray:
    """构造行业分组矩阵 G (K×n)，G[k,i]=1 表示股票 i 属于第 k 个行业。

    用 G @ w 一次性表达全部行业权重和，替代逐行业 append 的标量约束。
    """
    uniq = pd.unique(ind_values)
    G = np.zeros((len(uniq), n), dtype=float)
    for k, name in enumerate(uniq):
        G[k, ind_values == name] = 1.0
    return G


def industry_matrix_for(snapshot: MarketSnapshot, n: int) -> np.ndarray:
    """按 snapshot 的行业字段构造分组矩阵，缺失行业归入「未知」。"""
    industries = snapshot.industry.reindex(snapshot.tickers).fillna("未知")
    return industry_group_matrix(industries.values, n)


def finalize_weights(
    raw_weights: np.ndarray,
    masks: TradingMasks,
) -> np.ndarray:
    """求解后清理数值误差，并精确保留冻结股票权重。

    不做二次归一化；否则可能把 sell_only、个股上限等硬约束重新放大突破。
    求解器/清零产生的极小预算缺口保留为现金。
    """
    weights = np.clip(np.asarray(raw_weights, dtype=float), 0.0, None)
    weights[masks.zero] = 0.0
    weights[masks.frozen] = masks.prev_weight[masks.frozen]
    excess = max(0.0, float(weights.sum()) - 1.0)
    if excess > 0:
        # 只向下扣减可调整股票，绝不放大任何硬约束；冻结值保持精确不变。
        for idx in np.argsort(weights * ~masks.frozen)[::-1]:
            if masks.frozen[idx] or weights[idx] <= 0 or excess <= 0:
                continue
            reduction = min(float(weights[idx]), excess)
            weights[idx] -= reduction
            excess -= reduction
    return weights


@dataclass(frozen=True)
class TradingMasks:
    """三类互斥的交易状态掩码（优先级：frozen > zero > sell_only）。

    frozen    : 停牌且有持仓 → w == w_prev（不计换手、不释放资金）
    zero      : 停牌/次新/ST 且无持仓 → w == 0（禁止新开仓）
    sell_only : 掉池持仓票 + 次新/ST 持仓票 → w <= w_prev（只卖不买）
    """

    frozen: np.ndarray
    zero: np.ndarray
    sell_only: np.ndarray
    held: np.ndarray
    prev_weight: np.ndarray      # 上期权重；调用方未提供时为全零
    has_prev: bool               # 调用方是否真的提供了上期权重（决定是否施加换手约束）

    @property
    def restricted(self) -> np.ndarray:
        """三类受限股票的并集——其 alpha 不应驱动优化方向。"""
        return self.frozen | self.zero | self.sell_only


def build_trading_masks(
    snapshot: MarketSnapshot,
    prev_weight: np.ndarray | None,
) -> TradingMasks:
    """由市场快照与上期权重推导三类互斥掩码。"""
    n = len(snapshot.tickers)
    prev = (
        np.array(prev_weight, dtype=float) if prev_weight is not None
        else np.zeros(n)
    )
    held = prev > 1e-12

    suspended = snapshot.suspended_mask
    new_listing = snapshot.new_listing_mask
    st = snapshot.st_mask

    frozen = suspended & held
    zero = (suspended | new_listing | st) & ~held
    sell_only = (
        snapshot.sell_only_mask | ((new_listing | st) & held)
    ) & ~frozen & ~zero

    return TradingMasks(
        frozen=frozen,
        zero=zero,
        sell_only=sell_only,
        held=held,
        prev_weight=prev,
        has_prev=prev_weight is not None,
    )


def neutralize_alpha(alpha: np.ndarray, masks: TradingMasks) -> np.ndarray:
    """把受限股票的 alpha 清零，返回新数组（不改写调用方的输入）。

    冻结/强零/只卖票的权重已被约束固定或单向限制，让 alpha 继续驱动方向
    只会干扰目标函数。
    """
    neutralized = np.array(alpha, dtype=float)
    neutralized[masks.restricted] = 0.0
    return neutralized


def weight_upper_vector(masks: TradingMasks, weight_upper: float) -> np.ndarray:
    """个股绝对上限向量；冻结票豁免，因为漂移后的持仓可能已超过上限。"""
    upper = np.full(len(masks.prev_weight), float(weight_upper))
    frozen_idx = np.where(masks.frozen)[0]
    if len(frozen_idx) > 0:
        upper[frozen_idx] = np.maximum(weight_upper, masks.prev_weight[frozen_idx])
    return upper


def state_constraints(w: cp.Variable, masks: TradingMasks) -> list:
    """三类交易状态的硬约束（向量化）。"""
    constraints = []
    frozen_idx = np.where(masks.frozen)[0]
    if len(frozen_idx) > 0:
        # 等式固定 → |Δw|=0，自动不消耗换手预算
        constraints.append(w[frozen_idx] == masks.prev_weight[frozen_idx])
    zero_idx = np.where(masks.zero)[0]
    if len(zero_idx) > 0:
        constraints.append(w[zero_idx] == 0.0)
    sellonly_idx = np.where(masks.sell_only)[0]
    if len(sellonly_idx) > 0:
        # 卖出正常计换手
        constraints.append(w[sellonly_idx] <= masks.prev_weight[sellonly_idx])
    return constraints


def turnover_terms(
    w: cp.Variable,
    masks: TradingMasks,
    *,
    max_turnover: float | None,
    turnover_penalty: float,
    cost_vector: np.ndarray | None,
) -> tuple[list, Any]:
    """换手硬上限约束与软惩罚项。

    Returns
    -------
    (constraints, penalty_term)
        未提供上期权重时返回 ``([], 0.0)``——首期建仓没有换手概念。
    """
    if not masks.has_prev:
        return [], 0.0

    w_prev = masks.prev_weight
    delta_w = cp.abs(w - w_prev)

    penalty_term: Any = 0.0
    if turnover_penalty > 0:
        if cost_vector is not None:
            c = np.clip(np.array(cost_vector, dtype=float), 0.0, None)
            penalty_term = turnover_penalty * cp.sum(cp.multiply(c, delta_w))
        else:
            penalty_term = turnover_penalty * cp.sum(delta_w)

    constraints = []
    if max_turnover is not None:
        # 实际成交可能因涨停/停牌留下现金，使股票权重和 < 1。
        # 现金重新投入不挤占原本用于股票间调仓的换手预算。
        cash_gap = max(0.0, 1.0 - float(w_prev.sum()))
        constraints.append(cp.sum(delta_w) <= max_turnover + cash_gap)

    return constraints, penalty_term


def solve_with_fallback(problem: cp.Problem) -> str | None:
    """先 CLARABEL 后 SCS 求解。

    Returns
    -------
    str | None
        两个求解器都抛异常时返回失败原因字符串，否则返回 None（此时用
        ``problem.status`` 判断是否取到最优解）。
    """
    try:
        problem.solve(solver=cp.CLARABEL, max_iter=_CLARABEL_MAX_ITER, verbose=False)
        if problem.status in _OPTIMAL_STATUSES:
            return None
    except Exception:  # noqa: BLE001 - 求解器异常类型不稳定，一律降级重试
        pass

    try:
        problem.solve(solver=cp.SCS, max_iters=_SCS_MAX_ITERS, verbose=False)
    except Exception as exc:  # noqa: BLE001
        return f"both solvers failed: {exc}"
    return None


@dataclass
class BasePortfolioResult:
    """两个优化器结果的公共字段与展示方法。"""

    tickers: list[str]
    weights: np.ndarray
    status: str
    objective_value: float
    snapshot: MarketSnapshot | None

    @classmethod
    def infeasible(cls, tickers: list[str], reason: str):
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

    def _industry_series(self) -> pd.Series | None:
        if self.snapshot is None:
            return None
        return self.snapshot.industry.reindex(self.tickers).fillna("未知")

    def industry_weights(self) -> pd.Series:
        """各行业绝对权重。"""
        ind = self._industry_series()
        if ind is None:
            return pd.Series(dtype=float)
        return self.to_series().groupby(ind.values).sum().sort_values(ascending=False)

    def top_holdings(self, n: int = 10) -> pd.DataFrame:
        s = self.to_series().sort_values(ascending=False).head(n)
        df = s.to_frame("weight")
        df["weight_pct"] = df["weight"] * 100
        if self.snapshot is not None:
            df["industry"] = self.snapshot.industry.reindex(s.index)
            if self.snapshot.is_constituent is not None:
                df["is_constituent"] = self.snapshot.is_constituent.reindex(s.index)
        return df
