"""HistQuantOpt cvxpy 瓶颈剖析基准。

用真实 CNE6 数据（最后一个调仓日，N≈5366，K=51）复刻 alpha_max / index_enhance
的优化问题，分别测量：
  1. cvxpy 建模 + 编译（canonicalization）时间 vs 求解器内部时间
  2. 各优化方案的加速比
"""
import json
import time
import numpy as np
import pandas as pd
import polars as pl
import cvxpy as cp
import scipy.sparse as sp

RNG = np.random.default_rng(42)
ROOT = "/Users/guoguo/Documents/HistQuant/HistQuantOpt"

# ---------- 真实数据 ----------
t0 = time.time()
expo = pl.read_parquet(f"{ROOT}/data/barra_cne6_S/exposure_panel.parquet")
last = expo["rebal_date"].max()
sub = expo.filter(pl.col("rebal_date") == last).to_pandas()
cov = pl.read_parquet(f"{ROOT}/data/barra_cne6_S/factor_cov_panel.parquet").to_pandas()
cov["rebal_date"] = pd.to_datetime(cov["rebal_date"])
covlast = cov[cov["rebal_date"] == cov["rebal_date"].max()]
factor_names = [c for c in covlast.columns if c not in ("rebal_date", "factor")]
F = covlast.set_index("factor")[factor_names].loc[factor_names].to_numpy(float)
F = 0.5 * (F + F.T)

sub = sub.dropna(subset=factor_names)
N = len(sub)
K = len(factor_names)
X = sub[factor_names].to_numpy(float)
delta = sub["spec_var"].fillna(sub["spec_var"].median()).to_numpy(float)
delta = np.clip(delta, 1e-8, None)
print(f"data loaded {time.time()-t0:.1f}s  N={N} K={K}")

# 风格列（20 个风格因子做暴露约束）
STYLE = [f for f in factor_names if f in (
    "Size","MidCap","Beta","Momentum","ResidualVolatility","LongTermReversal",
    "Liquidity","Value","Growth","Leverage","EarningsYield","EarningsQuality",
    "EarningsVariability","InvestmentQuality","Profitability","DividendYield",
    "AnalystSentiment","Industry Momentum","Seasonality","ShortTermReversal")]
B = sub[STYLE].to_numpy(float)                     # (N, Ks)
Ks = B.shape[1]

alpha = RNG.standard_normal(N)
alpha = (alpha - alpha.mean()) / alpha.std()

# 行业：30 组随机分配（真实为中信30行业，规模一致）
ind = RNG.integers(0, 30, N)
G = np.zeros((30, N)); G[ind, np.arange(N)] = 1.0
G_sp = sp.csr_matrix(G)

const_mask = np.zeros(N, bool); const_mask[RNG.choice(N, 1800, replace=False)] = True
w_bm = np.zeros(N)
bm_idx = RNG.choice(np.where(const_mask)[0], 1000, replace=False)
w_bm[bm_idx] = RNG.random(1000); w_bm /= w_bm.sum()

# 上期权重：300 只随机持仓
prev = np.zeros(N)
h = RNG.choice(N, 300, replace=False)
prev[h] = RNG.random(300); prev = prev / prev.sum() * 0.99
cost_vec = np.clip(RNG.lognormal(0, 0.5, N), 0.2, 5.0)

WEIGHT_UPPER, IND_UPPER, MIN_CONST = 0.01, 0.20, 0.40
STYLE_BOUND = np.full(Ks, 0.5)
MAX_TO, TO_PEN = 0.40, 0.01
RISK_AVERSION = 10.0

results = {"N": N, "K": K, "variants": []}

def timed_solve(prob, tag, **kw):
    t0 = time.time()
    prob.solve(**kw)
    wall = time.time() - t0
    st = prob.solver_stats
    entry = {
        "tag": tag,
        "wall_s": round(wall, 3),
        "solve_s": round(st.solve_time, 3) if st and st.solve_time else None,
        "setup_s": round(st.setup_time, 4) if st and st.setup_time else None,
        "compile_s": round(prob.compilation_time, 3) if prob.compilation_time else None,
        "status": prob.status,
        "iters": st.num_iters if st else None,
        "obj": round(float(prob.value), 6) if prob.value is not None else None,
    }
    results["variants"].append(entry)
    print(entry)
    return entry


def build_alpha_max(use_factor_risk, sparse_ind=False, chol=False):
    """复刻 alpha_max.optimize 的问题结构。"""
    w = cp.Variable(N, nonneg=True)
    Gm = G_sp if sparse_ind else G
    cons = [cp.sum(w) == 1.0, w <= WEIGHT_UPPER, Gm @ w <= IND_UPPER,
            cp.sum(w[const_mask]) >= MIN_CONST]
    expos = B.T @ w
    cons += [expos <= STYLE_BOUND, expos >= -STYLE_BOUND]
    dw = cp.abs(w - prev)
    cash_gap = max(0.0, 1.0 - prev.sum())
    cons.append(cp.sum(dw) <= MAX_TO + cash_gap)
    pen = TO_PEN * cp.sum(cp.multiply(cost_vec, dw))
    if use_factor_risk:
        if chol:
            L = np.linalg.cholesky(F + 1e-12 * np.eye(K))
            risk = RISK_AVERSION * (
                cp.sum_squares(L.T @ (X.T @ w))
                + cp.sum(cp.multiply(delta, cp.square(w))))
        else:
            risk = RISK_AVERSION * (
                cp.quad_form(X.T @ w, cp.psd_wrap(F))
                + cp.sum(cp.multiply(delta, cp.square(w))))
    else:
        risk = 0.05 * cp.sum_squares(w)
    return cp.Problem(cp.Maximize(alpha @ w - risk - pen), cons), w


# ============ A. 现状基线：L2 惩罚（alpha_max 默认，risk_aversion=null） ============
print("\n=== A. alpha_max 默认（L2 分散惩罚，CLARABEL）===")
prob, _ = build_alpha_max(False)
timed_solve(prob, "A1_baseline_L2_clarabel", solver=cp.CLARABEL, max_iter=500)

prob, _ = build_alpha_max(False)
timed_solve(prob, "A2_L2_osqp", solver=cp.OSQP, max_iter=20000, eps_abs=1e-6, eps_rel=1e-6)

# ============ B. 因子风险（index_enhance 默认 risk_aversion=10）============
print("\n=== B. 因子风险 quad_form vs Cholesky 重构 ===")
prob, _ = build_alpha_max(True, chol=False)
timed_solve(prob, "B1_quadform_clarabel", solver=cp.CLARABEL, max_iter=500)

prob, _ = build_alpha_max(True, chol=True)
timed_solve(prob, "B2_cholesky_clarabel", solver=cp.CLARABEL, max_iter=500)

prob, _ = build_alpha_max(True, chol=True, sparse_ind=True)
timed_solve(prob, "B3_cholesky_sparse_clarabel", solver=cp.CLARABEL, max_iter=500)

# ============ C. SCS 兜底路径成本 ============
print("\n=== C. SCS（降级路径）===")
prob, _ = build_alpha_max(True, chol=False)
timed_solve(prob, "C1_quadform_scs", solver=cp.SCS, max_iters=10000)

json.dump(results, open("/tmp/hqopt_bench/results_a.json", "w"), indent=1)
print("saved results_a.json")
