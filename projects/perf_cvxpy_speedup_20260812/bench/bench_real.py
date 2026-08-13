"""在真实流水线问题上对比求解方案。

跑 index_enhance / alpha_max 的 6 个月窗口，monkeypatch solve_with_fallback：
每期把同一个 cvxpy Problem 用多种求解设置各解一遍，记录 wall/solve 时间、
迭代数、目标值；最后再用基线 CLARABEL 解一遍，保证流水线后续轨迹与生产一致。
"""
import json
import sys
import time

import cvxpy as cp
import numpy as np

sys.path.insert(0, "/Users/guoguo/Documents/HistQuant/HistQuantOpt")

import hqopt.optimizer._common as common
import hqopt.optimizer.alpha_max as am
import hqopt.optimizer.index_enhance as ie

RECORDS = []

VARIANTS = [
    ("clarabel_default", dict(solver=cp.CLARABEL, max_iter=500)),
    ("clarabel_tol1e-6", dict(solver=cp.CLARABEL, max_iter=500,
                              tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-6)),
    ("clarabel_tol1e-7", dict(solver=cp.CLARABEL, max_iter=500,
                              tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7)),
    ("piqp", dict(solver="PIQP", eps_abs=1e-7, eps_rel=1e-7)),
    ("piqp_loose", dict(solver="PIQP", eps_abs=1e-6, eps_rel=1e-6)),
]

def bench_solve_with_fallback(problem):
    period = {"variants": {}}
    for tag, kw in VARIANTS:
        t0 = time.time()
        try:
            problem.solve(verbose=False, **kw)
            wall = time.time() - t0
            st = problem.solver_stats
            period["variants"][tag] = {
                "wall_s": round(wall, 3),
                "solve_s": round(st.solve_time, 3) if st and st.solve_time else None,
                "status": problem.status,
                "iters": st.num_iters if st else None,
                "obj": float(problem.value) if problem.value is not None else None,
            }
        except Exception as e:
            period["variants"][tag] = {"error": str(e)[:120],
                                       "wall_s": round(time.time() - t0, 3)}
    RECORDS.append(period)
    # 最后按生产路径再解一遍，流水线拿到与基线一致的解
    return common.solve_with_fallback.__wrapped__(problem) \
        if hasattr(common.solve_with_fallback, "__wrapped__") else _orig(problem)

_orig = common.solve_with_fallback
def patched(problem):
    r = bench_solve_with_fallback(problem)
    return r

am.solve_with_fallback = patched
ie.solve_with_fallback = patched

from hqopt.pipeline.batch_optimize import load_config, run_batch_optimize

which = sys.argv[1]  # alpha_max / index_enhance
cfg = load_config(f"/tmp/hqopt_bench/{which}_default_profile.yaml")
run_batch_optimize(cfg)

out = f"/tmp/hqopt_bench/real_{which}.json"
json.dump(RECORDS, open(out, "w"), indent=1)
print("saved", out)

# 汇总
import statistics
tags = [t for t, _ in VARIANTS]
print(f"\n{'variant':22s} {'mean_wall':>9s} {'median':>8s} {'ok':>3s}  mean_obj")
base_obj = None
for tag in tags:
    walls = [r["variants"][tag].get("wall_s") for r in RECORDS if "error" not in r["variants"][tag]]
    objs = [r["variants"][tag].get("obj") for r in RECORDS
            if r["variants"][tag].get("obj") is not None]
    ok = sum(1 for r in RECORDS if r["variants"][tag].get("status") in ("optimal", "optimal_inaccurate"))
    if walls:
        print(f"{tag:22s} {statistics.mean(walls):9.3f} {statistics.median(walls):8.3f} "
              f"{ok:3d}  {statistics.mean(objs) if objs else float('nan'):.6f}")
    else:
        errs = {r["variants"][tag].get("error") for r in RECORDS}
        print(f"{tag:22s}  FAILED: {list(errs)[0]}")
