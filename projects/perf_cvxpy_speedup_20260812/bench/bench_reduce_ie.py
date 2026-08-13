"""index_enhance：候选池瘦身 + PIQP 组合实验（对比生产基线）。

缩减规则：alpha top-M ∪ 当前持仓 ∪ 基准权重非零（保基准可复制）∪ 成分股容量。
"""
import sys
import time
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/guoguo/Documents/HistQuant/HistQuantOpt")

import hqopt.optimizer.alpha_max as am
import hqopt.optimizer.index_enhance as ie
from hqopt.pipeline.batch_optimize import load_config, _prepare_inputs
from hqopt.pipeline.batch.periods import _prepare_period, _resolve_period_alpha
from hqopt.pipeline.batch.types import _RunStats
from hqopt.pipeline.batch.execution_walk import _ExecutionWalker


def solve_piqp(problem):
    try:
        problem.solve(solver="PIQP", eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return None
    except Exception:
        pass
    try:
        problem.solve(solver=cp.CLARABEL, max_iter=500, verbose=False)
    except Exception as e:
        return f"both failed: {e}"
    return None


cfg = load_config("/tmp/hqopt_bench/index_enhance_default_profile.yaml")
inputs = _prepare_inputs(cfg, None, None)
stats = _RunStats()


def slice_snapshot(snap, keep_idx):
    keep_tickers = [snap.tickers[i] for i in keep_idx]
    return replace(
        snap,
        tickers=keep_tickers,
        industry=snap.industry.iloc[keep_idx],
        adv=snap.adv.iloc[keep_idx],
        status=snap.status.iloc[keep_idx],
        prev_weight=snap.prev_weight.iloc[keep_idx],
        market_cap=snap.market_cap.iloc[keep_idx],
        is_constituent=None if snap.is_constituent is None else snap.is_constituent.iloc[keep_idx],
        sell_only=None if snap.sell_only is None else snap.sell_only.iloc[keep_idx],
    )


walker = _ExecutionWalker(inputs.ledger, inputs.trade_dates, inputs.execution_days)
has_prior = False
results = []

for pi, rd in enumerate(inputs.rebal_dates[:6]):
    signal_day = walker.open_signal_day(rd)
    actual = inputs.ledger.actual_weights()
    ctx = _prepare_period(inputs, rd, actual, has_prior)
    if ctx is None:
        continue
    alpha = _resolve_period_alpha(inputs, ctx, rd, stats)
    if alpha is None:
        continue

    bm = inputs.benchmark.get_weights(rd, tickers=ctx.snapshot.tickers)
    inputs.optimizer.config = inputs.base_config if ctx.prev_weight is not None else \
        ie.IndexEnhanceConfig(**{**inputs.base_config.__dict__, "max_turnover": None})

    if pi >= 2:
        snap, prev = ctx.snapshot, ctx.prev_weight
        N = len(snap.tickers)
        for variant, M, use_piqp in (
            ("baseline_clarabel", None, False),
            ("piqp_full", None, True),
            ("piqp_top1000", 1000, True),
            ("clarabel_top1000", 1000, False),
        ):
            if M is None:
                keep = np.arange(N)
            else:
                rank = pd.Series(alpha).rank(ascending=False).values
                keep_mask = rank <= M
                if prev is not None:
                    keep_mask |= prev > 1e-12
                keep_mask |= bm.values > 1e-10          # 基准权重非零全保留
                cmask = snap.constituent_mask
                crank = pd.Series(np.where(cmask, alpha, -np.inf)).rank(ascending=False).values
                keep_mask |= (crank <= int(M * 0.6)) & cmask
                keep = np.where(keep_mask)[0]
            tag = f"{variant}_N{len(keep)}"
            sub_snap = slice_snapshot(snap, keep)
            sub_alpha = alpha[keep]
            sub_prev = None if prev is None else prev[keep]
            sub_bm = bm.values[keep]
            sub_risk = inputs.risk_model.at(rd, sub_snap.tickers)
            solver = solve_piqp if use_piqp else am.solve_with_fallback.__globals__.get("solve_with_fallback")
            import hqopt.optimizer._common as common
            ie.solve_with_fallback = solve_piqp if use_piqp else common.solve_with_fallback
            t0 = time.time()
            res = inputs.optimizer.optimize(
                alpha=sub_alpha, snapshot=sub_snap, benchmark_weight=sub_bm,
                style_loading=ctx.style_loading, prev_weight=sub_prev,
                cost_vector=None, risk_snapshot=sub_risk,
            )
            wall = time.time() - t0
            w_series = pd.Series(res.weights, index=sub_snap.tickers)
            results.append({
                "period": str(rd), "tag": tag, "variant": variant,
                "wall_s": round(wall, 3), "status": res.status,
                "obj": res.objective_value, "n_pos": res.n_positions,
                "weights": w_series[w_series > 1e-6],
            })
            print(str(rd), tag, f"{wall:.2f}s", res.status[:30],
                  f"obj={res.objective_value:.6f}", f"npos={res.n_positions}")

    import hqopt.optimizer._common as common
    ie.solve_with_fallback = common.solve_with_fallback
    result = inputs.optimizer.optimize(
        alpha=alpha, snapshot=ctx.snapshot, benchmark_weight=bm.values,
        style_loading=ctx.style_loading, prev_weight=ctx.prev_weight,
        cost_vector=None, risk_snapshot=ctx.risk_snapshot,
    )
    if result.is_feasible:
        w = pd.Series(result.weights, index=ctx.snapshot.tickers)
        inputs.ledger.submit_target(w)
        has_prior = True

print("\n===== 解质量对比（vs baseline_clarabel）=====")
df = pd.DataFrame(results)
for period, grp in df.groupby("period"):
    base = grp[grp.variant == "baseline_clarabel"].iloc[0]
    for _, row in grp.iterrows():
        if row.variant == "baseline_clarabel":
            continue
        wf, wr = base["weights"], row["weights"]
        union = wf.index.union(wr.index)
        l1 = float((wf.reindex(union, fill_value=0) - wr.reindex(union, fill_value=0)).abs().sum())
        overlap = len(wf.index.intersection(wr.index)) / max(len(wf), 1)
        print(f"{period} {row.tag:28s} 提速 {base.wall_s/row.wall_s:4.1f}x  "
              f"obj差 {row.obj - base.obj:+.2e}  权重L1差 {l1:.4f}  持仓重叠 {overlap:.1%}")
