"""候选池瘦身实验：真实 alpha_max 问题，全池 vs 缩减池 的耗时与解质量。

缩减规则：alpha 截面排名前 M ∪ 当前持仓 ∪（成分股按 alpha 排名前若干，保证
成分下限有足够容量）。对比全池解与缩减池解的目标值 / 持仓重叠 / 权重差。
"""
import sys
import time
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/guoguo/Documents/HistQuant/HistQuantOpt")

import hqopt.optimizer._common as common
from hqopt.pipeline.batch_optimize import load_config, _prepare_inputs
from hqopt.pipeline.batch.periods import _prepare_period, _resolve_period_alpha
from hqopt.pipeline.batch.types import _RunStats

cfg = load_config("/tmp/hqopt_bench/alpha_max_default_profile.yaml")
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

# 用第 3 期（有 prev_weight，贴近稳态）：先按生产流程推进 2 期
from hqopt.pipeline.batch.execution_walk import _ExecutionWalker
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

    if pi >= 2:
        snap, prev = ctx.snapshot, ctx.prev_weight
        N = len(snap.tickers)
        for M in (None, 2500, 1500, 1000):
            if M is None:
                keep = np.arange(N)
                tag = f"full_N{N}"
            else:
                rank = pd.Series(alpha).rank(ascending=False).values
                keep_mask = rank <= M
                if prev is not None:
                    keep_mask |= prev > 1e-12
                # 成分股容量：按 alpha 取前 60%·M，确保 40% 下限可行
                cmask = snap.constituent_mask
                crank = pd.Series(np.where(cmask, alpha, -np.inf)).rank(ascending=False).values
                keep_mask |= (crank <= int(M * 0.6)) & cmask
                keep = np.where(keep_mask)[0]
                tag = f"top{M}_N{len(keep)}"
            sub_snap = slice_snapshot(snap, keep)
            sub_alpha = alpha[keep]
            sub_prev = None if prev is None else prev[keep]
            sub_style = ctx.style_loading  # reindex 按 ticker 自动对齐
            sub_risk = inputs.risk_model.at(rd, sub_snap.tickers)
            t0 = time.time()
            res = inputs.optimizer.optimize(
                sub_alpha, sub_snap, style_loading=sub_style,
                prev_weight=sub_prev, cost_vector=None, risk_snapshot=sub_risk,
            )
            wall = time.time() - t0
            w_series = pd.Series(res.weights, index=sub_snap.tickers)
            results.append({
                "period": str(rd), "tag": tag, "wall_s": round(wall, 3),
                "status": res.status, "obj": res.objective_value,
                "alpha_dot_w": float(np.dot(sub_alpha, res.weights)),
                "n_pos": res.n_positions,
                "weights": w_series[w_series > 1e-6],
            })
            print(results[-1]["period"], tag, f"{wall:.2f}s", res.status,
                  f"obj={res.objective_value:.6f}", f"npos={res.n_positions}")

    # 生产路径继续推进账本
    result = inputs.optimizer.optimize(
        alpha, ctx.snapshot, style_loading=ctx.style_loading,
        prev_weight=ctx.prev_weight, cost_vector=None,
        risk_snapshot=ctx.risk_snapshot,
    )
    if result.is_feasible:
        w = pd.Series(result.weights, index=ctx.snapshot.tickers)
        inputs.ledger.submit_target(w)
        has_prior = True

# 汇总：解质量对比（同期 full vs reduced）
print("\n===== 解质量对比 =====")
df = pd.DataFrame([r for r in results])
for period, grp in df.groupby("period"):
    full = grp[grp.tag.str.startswith("full")].iloc[0]
    for _, row in grp.iterrows():
        if row.tag.startswith("full"):
            continue
        wf, wr = full["weights"], row["weights"]
        union = wf.index.union(wr.index)
        l1 = float((wf.reindex(union, fill_value=0) - wr.reindex(union, fill_value=0)).abs().sum())
        overlap = len(wf.index.intersection(wr.index)) / max(len(wf), 1)
        print(f"{period} {row.tag:16s} 提速 {full.wall_s/row.wall_s:4.1f}x  "
              f"obj差 {row.obj - full.obj:+.2e}  权重L1差 {l1:.4f}  持仓重叠 {overlap:.1%}")
