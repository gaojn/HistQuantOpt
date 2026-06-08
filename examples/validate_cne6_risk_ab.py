"""方向1 验证：CNE6 因子风险模型 vs L2 惩罚 的 A/B 对照回测。

同回测期（CNE6 覆盖 2025-01~2026-05）、同合成 alpha，对比：
    A 组  use_cne6_risk=false  —— 旧 L2 偏离惩罚 γ‖w−w_bm‖²
    B 组  use_cne6_risk=true   —— CNE6 真因子风险 λ·active'Σactive（扫多个 λ）

输出每组：年化超额 / 实际TE / IR / 超额最大回撤 / 平均换手，
用于判断"真风险模型是否在同等 TE 下取得不逊于 L2 的 IR、且风险更可控"。

运行：python examples/validate_cne6_risk_ab.py
"""

from __future__ import annotations

import sys
import os
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio_optimizer.pipeline.batch_optimize import run_batch_optimize
from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester

BASE_CONFIG = "configs/zz500_enhance_cne6.yaml"
INDEX = "zz500"

# 合成 alpha 强度：5日IC。真实优秀选股因子约 0.03~0.05，原 0.08 偏强会掩盖
# 风险模型差异，这里调低到 0.03 以更接近实盘场景。
IC_MEAN = 0.05

# (标签, use_cne6_risk, risk_aversion λ) —— B 组 λ 网格用于标定 λ↔TE 前沿
GROUPS = [
    ("A_L2",        False, None),
    ("B_cne6_λ2",   True,  2.0),
    ("B_cne6_λ5",   True,  5.0),
    ("B_cne6_λ10",  True,  10.0),
    ("B_cne6_λ20",  True,  20.0),
    ("B_cne6_λ40",  True,  40.0),
    ("B_cne6_λ80",  True,  80.0),
]


def make_config(label: str, use_cne6: bool, risk_aversion: float | None) -> str:
    """基于 BASE_CONFIG 生成一组临时 config，返回路径。"""
    cfg = yaml.safe_load(open(BASE_CONFIG, encoding="utf-8"))
    cfg["optimizer"]["use_cne6_risk"] = use_cne6
    if use_cne6:
        cfg["optimizer"]["risk_aversion"] = risk_aversion
    else:
        cfg["optimizer"]["risk_aversion"] = None  # 退回 tracking_penalty
    cfg["alpha"]["ic_mean"] = IC_MEAN
    cfg["output"]["weights"] = f"output/_ab_{label}_weights.parquet"
    tmp = Path(tempfile.gettempdir()) / f"_ab_{label}.yaml"
    yaml.dump(cfg, open(tmp, "w", encoding="utf-8"), allow_unicode=True)
    return str(tmp)


def load_market(start_date, end_date):
    """加载回测期行情 + 构建基准日收益（一次，所有组复用）。"""
    panel = load_panel(
        start_date, end_date,
        columns=[
            "code", "date", "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "trade_status",
            "free_mv", "total_mv", "is_hs300", "is_zz500", "is_zz1000",
        ],
    )

    def to_wide(col: str) -> pd.DataFrame:
        df = (panel.select(["date", "code", col]).to_pandas()
              .pivot(index="date", columns="code", values=col).sort_index())
        df.index = pd.to_datetime(df.index)
        return df

    wides = {c: to_wide(c) for c in
             ["adj_close", "adj_vwap", "close", "limit_up", "limit_down", "trade_status"]}

    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)

    daily_ret_all = wides["adj_close"].pct_change(fill_method=None).fillna(0.0)
    w_lag = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)
    bm_ret.name = INDEX.upper()
    return wides, bm_ret


def backtest_metrics(weight_df: pd.DataFrame, wides: dict, bm_ret: pd.Series) -> dict:
    """回测一组权重，返回结构化绩效指标。"""
    weight_df = weight_df.copy()
    weight_df.index = pd.to_datetime(weight_df.index)

    bt = RealisticBacktester(cost_buy=0.001, cost_sell=0.002, risk_free=0.02)
    result, _ = bt.run(
        weight_df=weight_df,
        adj_close=wides["adj_close"], adj_vwap=wides["adj_vwap"],
        close_raw=wides["close"],
        limit_up_df=wides["limit_up"], limit_down_df=wides["limit_down"],
        trade_status_df=wides["trade_status"],
        benchmark_ret=bm_ret, initial_value=1e8,
    )

    exc = (result.daily_ret - result.bm_ret).dropna()
    n = len(exc)
    excess_nav = result.excess_nav.dropna()
    ann_excess = excess_nav.iloc[-1] ** (252 / n) - 1
    te = exc.std() * np.sqrt(252)
    ir = ann_excess / te if te > 1e-9 else np.nan
    exc_mdd = float((excess_nav / excess_nav.cummax() - 1).min())
    return {
        "年化超额": ann_excess,
        "实际TE": te,
        "IR": ir,
        "超额MDD": exc_mdd,
        "平均换手": result.turnover.mean(),
    }


def main() -> None:
    # 回测期取自 base config
    base = yaml.safe_load(open(BASE_CONFIG, encoding="utf-8"))
    from datetime import date
    sd = date.fromisoformat(base["backtest"]["start_date"])
    ed = date.fromisoformat(base["backtest"]["end_date"])

    print(f"\n{'='*70}\n  方向1 A/B 验证  {sd} ~ {ed}  ({INDEX.upper()})\n{'='*70}")
    print("\n[行情] 加载 + 基准（一次，所有组复用）...")
    wides, bm_ret = load_market(sd, ed)

    rows = {}
    for label, use_cne6, la in GROUPS:
        print(f"\n{'─'*70}\n>>> {label}  use_cne6_risk={use_cne6}  λ={la}\n{'─'*70}")
        cfg_path = make_config(label, use_cne6, la)
        weight_df = run_batch_optimize(cfg_path)
        rows[label] = backtest_metrics(weight_df, wides, bm_ret)

    # ── 汇总对比 ──
    table = pd.DataFrame(rows).T
    pct = table.copy()
    for c in ["年化超额", "实际TE", "超额MDD", "平均换手"]:
        pct[c] = (pct[c] * 100).map(lambda x: f"{x:+.2f}%")
    pct["IR"] = table["IR"].map(lambda x: f"{x:.3f}")

    print(f"\n\n{'='*70}\n  A/B 汇总对比\n{'='*70}")
    print(pct.to_string())
    print(f"\n{'='*70}")
    print("判读：B 组若在相近实际TE下 IR ≥ A 组、且超额MDD更小，则风险模型有效。")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
