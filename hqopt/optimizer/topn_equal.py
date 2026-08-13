"""Top-N 等权持有策略（规则式，不走凸优化）。

思路与 qlib 的 ``TopkDropoutStrategy`` 同源：每期持有 alpha 排名前 N 只、
逐票等权；调仓时用「最差持仓 ↔ 最优候选」的配对交换逐步逼近理想组合，
而不是每期整体重建。在此基础上加两条 A 股实盘友好的规则：

1. **双边换手硬上限** ``max_turnover``：单期 Σ|Δw| ≤ max_turnover + cash_gap
   （现金重投不挤占股票间调仓预算，口径与 QP 优化器的
   :func:`hqopt.optimizer._common.turnover_terms` 完全一致）。预算不足时
   优先保证换股（alpha 捕获），其次才做等权再平衡。
2. **免交易带** ``no_trade_band``：继续持有的股票若与等权目标的差异
   |w_prev − w_eq| 小于该带宽，则保持原权重不动——尽量减少交易只数。

交易状态语义复用 :mod:`hqopt.optimizer._common` 的三类掩码：
停牌且有持仓 → 权重冻结；停牌/次新/ST 且无持仓 → 禁止开仓；
掉池/次新/ST 持仓票 → 只卖不买。

与 ``alpha_max`` / ``index_enhance`` 共用 ``optimize()`` 签名与
``BasePortfolioResult`` 结果契约，pipeline 无需特殊分支；
``style_loading`` / ``cost_vector`` / ``risk_snapshot`` 参数接受但不使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from hqopt.data.generator import MarketSnapshot
from hqopt.optimizer._common import (
    BasePortfolioResult,
    TradingMasks,
    build_trading_masks,
    finalize_weights,
    max_positive_violation,
    post_solve_failure_reason,
    post_solve_failures,
)

if TYPE_CHECKING:
    from hqopt.risk.cne6_risk import RiskSnapshot

# 判定"有持仓/有交易"的权重阈值，与 _common.build_trading_masks 一致
_HOLDING_TOL = 1e-12


@dataclass
class TopNEqualConfig:
    """Top-N 等权持有参数。

    Parameters
    ----------
    top_n : int
        目标持仓只数 N，每只目标权重约 1/N。
    max_turnover : float | None
        双边换手率硬上限（0~2），None 表示不限制，默认 0.40。
        实际预算为 ``max_turnover + cash_gap``：上期留存现金的重投
        不计入股票间调仓预算（与 QP 优化器口径一致）。
    no_trade_band : float
        免交易带（绝对权重）。继续持有的股票与等权目标的差异小于该值时
        不交易，默认 0.005。注意这是**绝对**权重差：N=100 时等权 1%，
        0.005 意味着漂到 0.5%~1.5% 都不动。
    """

    top_n: int = 100
    max_turnover: float | None = 0.40
    no_trade_band: float = 0.005

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError(f"top_n 必须 ≥ 1，收到 {self.top_n}")
        if self.max_turnover is not None and not 0.0 < self.max_turnover <= 2.0:
            raise ValueError(
                f"max_turnover 必须位于 (0, 2] 或 None，收到 {self.max_turnover}"
            )
        if self.no_trade_band < 0:
            raise ValueError(f"no_trade_band 必须 ≥ 0，收到 {self.no_trade_band}")


@dataclass
class TopNEqualResult(BasePortfolioResult):
    """Top-N 等权持有结果；``n_trades`` 为本期实际交易只数。"""

    n_trades: int = 0

    def summary(self) -> str:
        return "\n".join([
            f"状态       : {self.status}",
            f"持仓数     : {self.n_positions}",
            f"交易只数   : {self.n_trades}",
            f"权重和     : {self.weights.sum():.6f}",
            f"最大单票   : {self.weights.max()*100:.3f}%",
        ])


class TopNEqualOptimizer:
    """Top-N 等权持有策略（规则式）。

    Parameters
    ----------
    config : TopNEqualConfig
    """

    def __init__(self, config: TopNEqualConfig) -> None:
        self.config = config

    def optimize(
        self,
        alpha: np.ndarray,
        snapshot: MarketSnapshot,
        style_loading: pd.DataFrame | None = None,
        prev_weight: np.ndarray | None = None,
        cost_vector: np.ndarray | None = None,
        risk_snapshot: RiskSnapshot | None = None,
    ) -> TopNEqualResult:
        """构建当期目标权重。

        Parameters
        ----------
        alpha : np.ndarray, shape (N,)
            Alpha 向量，与 snapshot.tickers 对齐；只用其**排序**，
            量纲无关（与 QP 优化器不同）。NaN 视为最差。
        snapshot : MarketSnapshot
            市场快照（交易状态掩码来源）。
        prev_weight : np.ndarray | None
            上期实际权重；None 表示首期建仓（不施加换手限制）。
        style_loading, cost_vector, risk_snapshot
            为保持与 QP 优化器相同的调用签名而接受，本策略不使用。

        Returns
        -------
        TopNEqualResult
        """
        del style_loading, cost_vector, risk_snapshot
        tickers = snapshot.tickers
        masks = build_trading_masks(snapshot, prev_weight)
        # NaN alpha 置为 -inf：排序时排在最后，绝不入选
        rank_alpha = np.where(np.isfinite(alpha), alpha, -np.inf)

        if not masks.has_prev:
            weights = self._initial_weights(rank_alpha, masks)
        else:
            weights = self._rebalance_weights(rank_alpha, masks)

        weights = finalize_weights(weights, masks)
        violations = self._violations(weights, masks)
        failures = post_solve_failures(violations)
        if failures:
            return TopNEqualResult.infeasible(
                tickers,
                post_solve_failure_reason(failures),
                constraint_violations=violations,
            )

        n_trades = int(
            (np.abs(weights - masks.prev_weight) > _HOLDING_TOL).sum()
        ) if masks.has_prev else int((weights > _HOLDING_TOL).sum())
        return TopNEqualResult(
            tickers=tickers,
            weights=weights,
            status="optimal",
            objective_value=float(np.nan_to_num(alpha, nan=0.0) @ weights),
            snapshot=snapshot,
            constraint_violations=violations,
            n_trades=n_trades,
        )

    # ------------------------------------------------------------------
    # 首期建仓
    # ------------------------------------------------------------------

    def _initial_weights(
        self, rank_alpha: np.ndarray, masks: TradingMasks
    ) -> np.ndarray:
        """无上期持仓：直接买入可开仓票中 alpha 前 N 只，逐票等权。"""
        buyable = ~masks.restricted & np.isfinite(rank_alpha)
        candidates = np.where(buyable)[0]
        if len(candidates) == 0:
            return np.zeros(len(rank_alpha))
        order = candidates[np.argsort(-rank_alpha[candidates], kind="stable")]
        chosen = order[: self.config.top_n]
        weights = np.zeros(len(rank_alpha))
        weights[chosen] = 1.0 / len(chosen)
        return weights

    # ------------------------------------------------------------------
    # 调仓：配对交换 + 免交易带 + 换手预算
    # ------------------------------------------------------------------

    def _rebalance_weights(
        self, rank_alpha: np.ndarray, masks: TradingMasks
    ) -> np.ndarray:
        cfg = self.config
        prev = masks.prev_weight
        held = masks.held

        cash_gap = max(0.0, 1.0 - float(prev.sum()))
        budget = (
            float("inf") if cfg.max_turnover is None
            else cfg.max_turnover + cash_gap
        )

        # 冻结票强制保留并占用 N 的名额；其余名额在「继续可持有 ∪ 可买入」中
        # 按 alpha 取满。sell_only 持仓票可继续持有（计入名额）但不得增持。
        frozen_idx = np.where(masks.frozen)[0]
        slots = max(cfg.top_n - len(frozen_idx), 0)
        holdable = (held | ~masks.restricted) & ~masks.frozen & ~masks.zero
        cand_idx = np.where(holdable)[0]
        order = cand_idx[np.argsort(-rank_alpha[cand_idx], kind="stable")]
        ideal = set(order[:slots].tolist())

        entries = [i for i in order[:slots] if not held[i]]      # alpha 降序
        exit_pool = [i for i in np.where(held & ~masks.frozen)[0] if i not in ideal]
        exits = sorted(exit_pool, key=lambda i: rank_alpha[i])   # alpha 升序（最差先卖）

        # 数量缺口：执行损耗（买单涨停/停牌未成交）会让实际持仓少于 N，
        # 净买入补足不需要卖出配对，是恢复"等权持有 N 只"的最高优先级。
        # 反向（持仓多于 N，如调小 N）则需净卖出。
        n_current = int(masks.frozen.sum()) + int(
            (held & ~masks.frozen & ~masks.zero).sum()
        )
        net_gap = cfg.top_n - n_current

        def build(n_net: int, n_swaps: int, n_rebalance: int) -> np.ndarray:
            return self._build_target(
                masks, entries, exits, net_gap, n_net, n_swaps, n_rebalance
            )

        def gross(target: np.ndarray) -> float:
            return float(np.abs(target - prev).sum())

        # 三级贪心：数量补足 → 换股（alpha 捕获）→ 等权再平衡（带外偏差
        # 从大到小）。全零时 target==prev（零交易），恒在预算内，是安全兜底。
        tol = 1e-9
        best_net = 0
        # 补仓除预算外还要求持仓数确实增加：满仓且无带外超重票可卖时，
        # 新买票会因资金无来源被残差打回零（幽灵补仓），必须就此停止，
        # 否则虚占名额还会挤掉后面换股/再平衡阶段的预算判定。
        n_pos_prev = int((build(0, 0, 0) > _HOLDING_TOL).sum())
        for d in range(1, abs(net_gap) + 1):
            candidate = build(d, 0, 0)
            n_pos = int((candidate > _HOLDING_TOL).sum())
            if gross(candidate) > budget + tol:
                break
            if net_gap > 0 and n_pos <= n_pos_prev:
                break
            best_net = d
            n_pos_prev = n_pos

        best_swaps = 0
        for s in range(1, max(len(entries), len(exits)) + 1):
            if gross(build(best_net, s, 0)) > budget + tol:
                break
            best_swaps = s

        n_rebal_max = len(self._rebalance_candidates(
            masks, entries, exits, net_gap, best_net, best_swaps
        ))
        best_rebal = 0
        for r in range(1, n_rebal_max + 1):
            if gross(build(best_net, best_swaps, r)) > budget + tol:
                break
            best_rebal = r

        return build(best_net, best_swaps, best_rebal)

    @staticmethod
    def _apply_counts(
        net_gap: int, n_net: int, n_swaps: int
    ) -> tuple[int, int]:
        """换算（净数量修正, 换股数）→（买入应用数, 卖出应用数）。

        净买入（net_gap>0）只增加买入端；净卖出（net_gap<0）只增加卖出端；
        换股两端各加一。
        """
        n_entry = n_swaps + (n_net if net_gap > 0 else 0)
        n_exit = n_swaps + (n_net if net_gap < 0 else 0)
        return n_entry, n_exit

    def _hold_state(
        self,
        masks: TradingMasks,
        entries: list[int],
        exits: list[int],
        net_gap: int,
        n_net: int,
        n_swaps: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """给定交易计数，返回（继续持有掩码, 新入选掩码, 等权目标, 冻结权重和）。"""
        n = len(masks.prev_weight)
        n_entry, n_exit = self._apply_counts(net_gap, n_net, n_swaps)
        entry_mask = np.zeros(n, dtype=bool)
        entry_mask[entries[: min(n_entry, len(entries))]] = True
        exit_mask = np.zeros(n, dtype=bool)
        exit_mask[exits[: min(n_exit, len(exits))]] = True

        continuing = masks.held & ~masks.frozen & ~exit_mask
        frozen_sum = float(masks.prev_weight[masks.frozen].sum())
        k = int(continuing.sum() + entry_mask.sum())
        w_eq = (1.0 - frozen_sum) / k if k > 0 else 0.0
        return continuing, entry_mask, w_eq, frozen_sum

    def _rebalance_candidates(
        self,
        masks: TradingMasks,
        entries: list[int],
        exits: list[int],
        net_gap: int,
        n_net: int,
        n_swaps: int,
    ) -> list[int]:
        """带外（|w_prev − w_eq| ≥ no_trade_band）的继续持有票，偏差降序。

        sell_only 且低于等权的票被排除：对它的"再平衡"是加仓，会被只卖不买
        约束封顶回原权重，纯属浪费再平衡名额。
        """
        continuing, _, w_eq, _ = self._hold_state(
            masks, entries, exits, net_gap, n_net, n_swaps
        )
        prev = masks.prev_weight
        outside = (
            continuing
            & (np.abs(prev - w_eq) >= self.config.no_trade_band)
            & ~(masks.sell_only & (prev < w_eq))
        )
        return sorted(
            np.where(outside)[0].tolist(),
            key=lambda i: -abs(prev[i] - w_eq),
        )

    def _build_target(
        self,
        masks: TradingMasks,
        entries: list[int],
        exits: list[int],
        net_gap: int,
        n_net: int,
        n_swaps: int,
        n_rebalance: int,
    ) -> np.ndarray:
        """按（净数量修正, 换股数, 再平衡数）构造目标权重。

        - 卖出票 → 0，新入选票 → w_eq；
        - 继续持有票默认保持原权重，仅前 ``n_rebalance`` 个带外偏差票调回 w_eq；
        - sell_only 票封顶于原权重（只卖不买）；
        - 预算缺口/盈余只在**买方向**的交易票（新入选、加仓再平衡）上均摊：
          带内票绝不因此被触碰，卖出方向的票也不回补（把刚卖出的钱还给同一只
          票等于撤销交易）。正残差无处可放时留作现金，下期经 cash_gap 重投。
        """
        prev = masks.prev_weight
        continuing, entry_mask, w_eq, _ = self._hold_state(
            masks, entries, exits, net_gap, n_net, n_swaps
        )
        _, n_exit = self._apply_counts(net_gap, n_net, n_swaps)

        target = prev.copy()
        target[masks.zero] = 0.0
        exit_applied = np.zeros(len(prev), dtype=bool)
        exit_applied[exits[: min(n_exit, len(exits))]] = True
        target[exit_applied] = 0.0
        target[entry_mask] = w_eq

        rebal = self._rebalance_candidates(
            masks, entries, exits, net_gap, n_net, n_swaps
        )
        rebal_applied = np.zeros(len(prev), dtype=bool)
        rebal_applied[rebal[:n_rebalance]] = True
        target[rebal_applied] = w_eq

        # 只卖不买：继续持有的 sell_only 票不得高于原权重
        sellonly_cap = masks.sell_only & (continuing | entry_mask)
        target[sellonly_cap] = np.minimum(target[sellonly_cap], prev[sellonly_cap])

        # 买入资金缺口先从带外超重的继续持有票上卖出补足（朝等权方向、
        # 不低于 w_eq，带内票仍不动）——满仓补仓（如调大 N）需要卖出融资，
        # 否则新买票会被下方残差分摊直接打回零。
        shortfall = float(target.sum()) - 1.0
        if shortfall > 0:
            fund = (
                continuing & ~rebal_applied
                & (prev - w_eq >= self.config.no_trade_band)
            )
            for i in sorted(
                np.where(fund)[0].tolist(), key=lambda j: -(prev[j] - w_eq)
            ):
                cut = min(target[i] - w_eq, shortfall)
                target[i] -= cut
                shortfall -= cut
                if shortfall <= 0:
                    break

        # 残差均摊：卖出释放的资金 + 留存现金 与 买入需求不会恰好相等，
        # 差额只在买方向的交易票（新入选 + 加仓再平衡，非 sell_only）上均摊。
        buy_side = (
            (entry_mask | rebal_applied) & ~masks.sell_only & (target >= prev)
        )
        buy_idx = np.where(buy_side)[0]
        if len(buy_idx) > 0:
            residual = 1.0 - float(target.sum())
            target[buy_idx] = np.maximum(
                target[buy_idx] + residual / len(buy_idx), 0.0
            )
        return target

    # ------------------------------------------------------------------
    # 求解后校验（与 QP 优化器同一门禁口径）
    # ------------------------------------------------------------------

    def _violations(
        self, weights: np.ndarray, masks: TradingMasks
    ) -> dict[str, float]:
        """规则式构造理应零违约；门禁兜住实现回归。

        与 :func:`hqopt.optimizer._common.common_constraint_violations` 的差异：
        预算约束只查**上限**（Σw ≤ 1）。免交易带允许目标不满仓——零交易期
        Σw = Σw_prev < 1 是本策略的正常状态，不是违约。
        """
        prev = masks.prev_weight
        violations = {
            "long_only": max_positive_violation(-weights),
            "budget_upper": max_positive_violation(float(weights.sum()) - 1.0),
            "frozen": max_positive_violation(
                np.abs(weights[masks.frozen] - prev[masks.frozen])
            ),
            "forced_zero": max_positive_violation(np.abs(weights[masks.zero])),
            "sell_only": max_positive_violation(
                weights[masks.sell_only] - prev[masks.sell_only]
            ),
        }
        if masks.has_prev and self.config.max_turnover is not None:
            cash_gap = max(0.0, 1.0 - float(prev.sum()))
            violations["turnover"] = max_positive_violation(
                np.abs(weights - prev).sum()
                - (self.config.max_turnover + cash_gap)
            )
        return violations
