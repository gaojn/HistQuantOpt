"""
指数增强批量优化（HS300 / ZZ500 / ZZ1000 通用）。

Alpha   : 合成因子 IC=0.08, IC_Std=0.10, decay=0.80（与多头一致）
回测区间 : 2024-06-01 ~ 2026-05-22
调仓频率 : 每5个交易日
候选池   : 全市场 - 北交所 - ST（约4500只）
基准     : 沪深300（分级靠档加权）

约束：
  单票绝对 ≤5%（容纳基准重仓股）
  目标指数成分股 ≥80%
  行业相对基准偏离 ±5%
  风格主动暴露 ±0.3σ
  跟踪误差惩罚 γ=10
  双边换手 ≤20%

运行：
    python examples/demo_batch_index_enhance.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dataclasses import replace
from datetime import date
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.index_enhance import (
    IndexEnhanceConfig, IndexEnhanceOptimizer,
)

FACTOR_PATH    = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")
INDEX          = "zz1000"              # 基准指数：hs300 / zz500 / zz1000
UNIVERSE_SIZE  = None                  # None=全市场；整数=按市值取前N只
BACKTEST_START = date(2023, 6, 1)
BACKTEST_END   = date(2026, 5, 22)
REBAL_FREQ     = 5
PORTFOLIO_VAL  = 1e8

IC_MEAN  = 0.08
IC_STD   = 0.10
DECAY    = 0.80
SEED     = 42


# ────────────────────────────────────────────────────────────────
# Alpha 生成（与 demo_batch_alpha_max 一致）
# ────────────────────────────────────────────────────────────────

def build_synthetic_alpha(
    panel: pl.DataFrame,
    fwd_days: int  = 5,
    ic_mean: float = IC_MEAN,
    ic_std: float  = IC_STD,
    decay: float   = DECAY,
    seed: int      = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    adj = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    fwd_ret = adj.shift(-fwd_days) / adj - 1
    dates = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    rows: dict = {}
    f_prev: pd.Series | None = None
    for dt in dates:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50: continue
        mu, sig = r.mean(), r.std()
        if sig < 1e-8: continue
        z_r = (r - mu) / sig
        rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = pd.Series(
            rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps,
            index=r.index,
        )
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        if f_prev is None or decay == 0.0:
            f = new_sig
        else:
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    decay * f_prev[common]
                    + np.sqrt(max(1 - decay**2, 0)) * new_sig[common]
                )
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f
        rows[dt] = f

    return pd.DataFrame(rows).T


def get_alpha_for_date(
    alpha_df: pd.DataFrame, target_date: date, tickers: list[str],
) -> np.ndarray:
    ts = pd.Timestamp(target_date)
    avail = alpha_df.index[alpha_df.index <= ts]
    if len(avail) == 0:
        return np.zeros(len(tickers))
    return alpha_df.loc[avail[-1]].reindex(tickers).fillna(0.0).values.astype(float)


# ────────────────────────────────────────────────────────────────
# 候选池过滤
# ────────────────────────────────────────────────────────────────

def filter_universe(snapshot, panel: pl.DataFrame, target_date: date):
    """候选池：全市场 - 北交所(.BJ) - ST 股票，可选按市值截取 TOP_N。"""
    today = (
        panel.filter(pl.col("date") == target_date)
        .select(["code", "is_st"])
        .to_pandas().set_index("code")
    )
    is_bj = today.index.str.endswith(".BJ")
    is_st = today["is_st"] == 1
    keep_mask = (~is_bj) & (~is_st)
    keep_set = set(today.index[keep_mask].tolist())
    keep = [t for t in snapshot.tickers if t in keep_set]

    # 按市值截取 TOP_N
    if UNIVERSE_SIZE is not None and len(keep) > UNIVERSE_SIZE:
        cap = snapshot.market_cap.reindex(keep).fillna(0.0)
        keep = cap.nlargest(UNIVERSE_SIZE).index.tolist()

    return replace(
        snapshot,
        tickers=keep,
        industry=snapshot.industry.reindex(keep),
        adv=snapshot.adv.reindex(keep),
        status=snapshot.status.reindex(keep),
        prev_weight=snapshot.prev_weight.reindex(keep).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(keep),
        is_constituent=(
            snapshot.is_constituent.reindex(keep)
            if snapshot.is_constituent is not None else None
        ),
    )


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    print(f"\n{'='*65}")
    print(f"  {INDEX.upper()} 指数增强批量优化  {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  调仓={REBAL_FREQ}日  候选池=全市场（剔除北交所+ST）")
    print(f"  Alpha: IC={IC_MEAN}, IC_Std={IC_STD}, decay={DECAY}")
    print(f"{'='*65}")

    # 1. 加载行情
    # 数据加载起始：使用同年1月1日（确保缓存可用，留出 alpha 暖启动空间）
    data_start = date(BACKTEST_START.year, 1, 1)
    print(f"\n[1] 加载行情数据（{data_start} ~ {BACKTEST_END}）...")
    panel = load_panel(
        data_start, BACKTEST_END,
        columns=[
            "code", "date", "adj_close", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # 2. 生成合成 Alpha（全市场，后续按需子集）
    print(f"\n[2] 生成合成 Alpha（IC={IC_MEAN}, IC_Std={IC_STD}, decay={DECAY}）...")
    alpha_df = build_synthetic_alpha(panel)
    print(f"  Alpha 矩阵: {alpha_df.shape}")

    # 3. 预计算基准权重
    print(f"\n[3] 预计算 {INDEX.upper()} 基准权重...")
    t0 = time.time()
    bm = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm.precompute(BACKTEST_START, BACKTEST_END, panel=panel)
    print(f"  基准权重预计算完成 ({time.time()-t0:.1f}s)")

    # 4. 再平衡日
    trade_dates = (
        panel.filter(
            (pl.col("date") >= BACKTEST_START) & (pl.col("date") <= BACKTEST_END)
        ).select("date").unique().sort("date")["date"].to_list()
    )
    rebal_dates = trade_dates[::REBAL_FREQ]
    print(f"\n  回测交易日数={len(trade_dates)}  再平衡日数={len(rebal_dates)}")

    # 5. 优化器配置
    # weight_upper：HS300建议 0.05（茅台~5%）；ZZ500建议 0.02（最大成分~1.5%）；ZZ1000建议 0.01
    weight_upper = {"hs300": 0.05, "zz500": 0.02, "zz1000": 0.01}.get(INDEX, 0.03)
    # 全市场候选池较大（~5000只），约束适度放宽以保证可行性
    config = IndexEnhanceConfig(
        weight_upper=weight_upper,
        min_constituent_ratio=0.80,
        industry_active_bound=0.07,    # 5% → 7%
        style_active_bound=0.50,       # 0.3 → 0.5
        tracking_penalty=10.0,
        max_turnover=0.40,
    )
    optimizer = IndexEnhanceOptimizer(config)
    adapter   = RealMarketAdapter()

    # 6. 逐期优化
    print(f"\n[4] 逐期优化...")
    weight_records: dict = {}
    prev_w_arr, prev_tickers = None, None
    fail_count = 0
    solve_times = []

    for i, rebal_date in enumerate(rebal_dates):
        t0 = time.time()

        # 快照
        try:
            snap_full = adapter.build_snapshot_from_panel(
                panel=panel, target_date=rebal_date,
                index=INDEX, portfolio_value=PORTFOLIO_VAL,
            )
        except ValueError as e:
            print(f"  [{rebal_date}] 跳过（快照失败：{e}）")
            continue

        # 过滤候选池
        snapshot = filter_universe(snap_full, panel, rebal_date)

        # Barra（基于过滤后的 snapshot）
        barra = JYBarraFactors(
            snapshot=snapshot, target_date=rebal_date,
            factor_path=FACTOR_PATH, panel=panel,
        )

        # Alpha
        alpha = get_alpha_for_date(alpha_df, rebal_date, snapshot.tickers)

        # 基准权重
        bm_series = bm.get_weights(rebal_date, tickers=snapshot.tickers)
        bm_weight = bm_series.values

        # 上期权重对齐
        if prev_w_arr is not None and prev_tickers is not None:
            ps = pd.Series(prev_w_arr, index=prev_tickers) \
                .reindex(snapshot.tickers).fillna(0.0).values
            s = ps.sum()
            ps = ps / s if s > 1e-8 else ps
        else:
            ps = None

        # 首期无换手约束
        if ps is None:
            optimizer.config = IndexEnhanceConfig(
                **{**config.__dict__, "max_turnover": None}
            )
        else:
            optimizer.config = config

        # 优化
        result = optimizer.optimize(
            alpha=alpha, snapshot=snapshot,
            benchmark_weight=bm_weight,
            style_loading=barra.style_loading,
            prev_weight=ps,
        )

        elapsed = time.time() - t0
        solve_times.append(elapsed)

        if result.is_feasible:
            w = pd.Series(result.weights, index=snapshot.tickers)
            weight_records[rebal_date] = w
            prev_w_arr, prev_tickers = result.weights, snapshot.tickers

            const_w = result.weights[snapshot.constituent_mask].sum()
            turnover = (
                float(np.abs(result.weights - ps).sum())
                if ps is not None else float("nan")
            )
            te_l2 = result.tracking_error_l2()

            if i % 10 == 0 or i == len(rebal_dates) - 1:
                print(f"  [{i+1:3d}/{len(rebal_dates)}] {rebal_date}  "
                      f"持仓={result.n_positions:3d}  {INDEX.upper()}={const_w*100:.1f}%  "
                      f"换手={turnover*100:>5.1f}%  TE_L2={te_l2:.4f}  "
                      f"耗时={elapsed:.2f}s")
        else:
            fail_count += 1
            print(f"  [{rebal_date}] ✗ 求解失败：{result.status}")
            if prev_w_arr is not None:
                w = pd.Series(prev_w_arr, index=prev_tickers) \
                    .reindex(snapshot.tickers).fillna(0.0)
                weight_records[rebal_date] = w

    # 7. 汇总
    if not weight_records:
        print("所有期均失败")
        return

    weight_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weight_df.index.name = "date"

    # 双边换手
    turnover_arr = weight_df.diff().abs().sum(axis=1).dropna()

    print(f"\n{'='*65}\n  批量优化汇总\n{'='*65}")
    print(f"  再平衡期数     : {len(weight_df)}")
    print(f"  失败期数       : {fail_count}")
    print(f"  股票池大小     : {weight_df.shape[1]}")
    print(f"  平均持仓数     : {(weight_df > 1e-6).sum(axis=1).mean():.0f} 只")
    print(f"  平均双边换手   : {turnover_arr.mean()*100:.1f}%")
    print(f"  最大双边换手   : {turnover_arr.max()*100:.1f}%")
    print(f"  平均求解耗时   : {np.mean(solve_times):.2f}s")
    print(f"  总耗时         : {time.time()-t_total:.1f}s")

    # 保存
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{INDEX}_enhance_weights.parquet"
    weight_df.to_parquet(out)
    print(f"\n  权重矩阵已保存：{out}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
