"""
基于未来 H=5 日 VWAP 涨跌幅，批量生成不同 IC / ICIR / 换手率的合成因子。

⚠️ 警告：本脚本生成的因子使用了未来信息（前视），仅用于因子合成/优化
   管线的输入样本测试，绝不可用于实盘或真实回测评估。

构造方法（与 research/signal_grid.SignalGridRunner 一致）：
    对每个截面日 t，取真实未来5日 adj_vwap 涨跌幅 r_{t+5}，生成：
        ρ_t  ~ clip( N(ic_mean, ic_std²), -0.95, 0.95 )
        sig_t = ρ_t · zscore(r_{t+5}) + √(1-ρ_t²) · ε
        f_t   = decay · f_{t-1} + √(1-decay²) · sig_t   (截面 z-score)
    decay 越高，因子跨期自相关越高、换手率越低。

网格：复用 signal_grid 默认网格
    Sweep A: IC × ICIR （decay 固定）
    Sweep B: decay 扫描（IC / ICIR 固定）
重复参数组合去重后生成。

输出：alphas/alpha_vwap5_ic{IC}_icir{ICIR}_decay{DECAY}.parquet
      列：date, code, alpha
      alphas/_summary.csv —— 各因子的输入参数与实测 IC/ICIR/自相关/换手

运行：
    cd /Users/guoguo/Desktop/HistQuantOpt
    python scripts/build_alphas_vwap5.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from pathlib import Path

import pandas as pd

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.research.signal_grid import (
    SignalGridRunner,
    DEFAULT_IC_LIST, DEFAULT_ICIR_LIST, DEFAULT_DECAY_LIST,
)

DATA_START = date(2020, 1, 1)
DATA_END   = date(2026, 5, 31)
FWD_DAYS   = 5
REBAL_FREQ = 5
SEED       = 42

DECAY_FIXED = 0.8
IC_FIXED, ICIR_FIXED = 0.08, 0.8

OUT_DIR = Path("alphas")


def build_grid() -> list[dict]:
    """组装 sweep A（IC×ICIR）+ sweep B（decay）网格，按 (ic, ic_std, decay) 去重。"""
    points: list[dict] = []
    for ic in DEFAULT_IC_LIST:
        for icir in DEFAULT_ICIR_LIST:
            points.append({"ic_mean": ic, "ic_std": ic / icir, "decay": DECAY_FIXED})
    for decay in DEFAULT_DECAY_LIST:
        points.append({"ic_mean": IC_FIXED, "ic_std": IC_FIXED / ICIR_FIXED, "decay": decay})

    seen: set[tuple] = set()
    unique: list[dict] = []
    for p in points:
        key = (round(p["ic_mean"], 4), round(p["ic_std"], 4), round(p["decay"], 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def alpha_to_long(alpha_df: pd.DataFrame) -> pd.DataFrame:
    long = alpha_df.stack().reset_index()
    long.columns = ["date", "code", "alpha"]
    return long


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"\n[1] 加载行情（{DATA_START} ~ {DATA_END}，adj_vwap）...")
    panel = load_panel(
        DATA_START, DATA_END,
        columns=["code", "date", "adj_vwap", "adj_close", "adj_factor", "vwap"],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    runner = SignalGridRunner(
        panel, fwd_days=FWD_DAYS, rebal_freq=REBAL_FREQ, price_col="adj_vwap",
    )
    print(f"  生成日={len(runner.gen_dates)}  取样日={len(runner.rebal_dates)}")

    grid = build_grid()
    print(f"\n[2] 生成 {len(grid)} 个因子（去重后）...\n")

    summary_rows = []
    for i, p in enumerate(grid, 1):
        ic_mean, ic_std, decay = p["ic_mean"], p["ic_std"], p["decay"]
        icir_in = ic_mean / ic_std

        alpha_df = runner.gen_alpha(ic_mean, ic_std, decay, seed=SEED)
        stats = runner.evaluate(alpha_df)

        fname = (
            f"alpha_vwap5_ic{ic_mean:.2f}_icir{icir_in:.1f}_decay{decay:.2f}.parquet"
        )
        alpha_to_long(alpha_df).to_parquet(OUT_DIR / fname)

        summary_rows.append({
            "filename": fname,
            "in_ic": ic_mean, "in_ic_std": ic_std, "in_icir": icir_in, "in_decay": decay,
            "ic": stats["ic"], "ic_std": stats["ic_std"], "icir": stats["icir"],
            "ir_theory": stats["ir_theory"], "ls_ir": stats["ls_ir"],
            "autocorr": stats["autocorr"], "turnover": stats["turnover"],
        })

        print(f"  [{i:2d}/{len(grid)}] {fname}")
        print(f"          实测IC={stats['ic']:.4f} ICIR={stats['icir']:.2f} "
              f"自相关={stats['autocorr']:.2f} 换手={stats['turnover']:.2f}")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n[3] 完成：{len(grid)} 个因子 -> {OUT_DIR}/")
    print(f"  汇总表 -> {summary_path}")


if __name__ == "__main__":
    main()
