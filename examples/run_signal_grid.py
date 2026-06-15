"""
信号强度 → 策略IR 标定实验入口。

用未来5日收益反向构造已知 IC/ICIR/decay 的合成因子，扫描网格，
量化「因子信号强度 → 预期策略IR」的映射，并用已跑的真实回测锚点标定 TC。

⚠️ 合成因子含前视，仅作标定，不可交易。

运行：
    python examples/run_signal_grid.py
输出：
    output/signal_grid_results.parquet
    output/signal_grid_ir.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.research.signal_grid import (
    SignalGridRunner, run_grid,
    DEFAULT_IC_LIST, DEFAULT_ICIR_LIST, DEFAULT_DECAY_LIST,
)

# ── 配置 ────────────────────────────────────────────────────────
DATA_START = date(2023, 1, 1)
DATA_END   = date(2026, 5, 22)
FWD_DAYS   = 5
REBAL_FREQ = 5
SEEDS      = (42, 43, 44)
Q          = 0.2

# 真实回测锚点：IC=0.08 / ICIR=0.8 / decay=0.8 下，本项目已跑出的全管线 IR
# （来源：指数增强真实回测；用于反推转换系数 TC = full_IR / ir_theory）
KNOWN_ANCHORS = {
    "HS300 增强":  2.85,
    "ZZ500 增强":  3.01,
    "ZZ1000 增强": 3.30,
}
ANCHOR_IC, ANCHOR_ICIR, ANCHOR_DECAY = 0.08, 0.8, 0.8


def main() -> None:
    print(f"\n{'='*64}")
    print(f"  信号强度 → 策略IR 标定实验（合成因子含前视，仅作标定）")
    print(f"  前瞻={FWD_DAYS}日  取样={REBAL_FREQ}日  种子={SEEDS}")
    print(f"{'='*64}")

    print(f"\n[1] 加载行情（{DATA_START} ~ {DATA_END}）...")
    panel = load_panel(DATA_START, DATA_END, columns=["code", "date", "adj_close"])
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    print(f"\n[2] 预计算未来收益 / z-score 截面...")
    runner = SignalGridRunner(panel, fwd_days=FWD_DAYS, rebal_freq=REBAL_FREQ)
    print(f"  生成日={len(runner.gen_dates)}  取样日={len(runner.rebal_dates)}  "
          f"年化系数=√{runner.periods_per_year:.0f}={np.sqrt(runner.periods_per_year):.2f}")

    print(f"\n[3] 扫描网格...")
    df = run_grid(
        runner,
        ic_list=DEFAULT_IC_LIST, icir_list=DEFAULT_ICIR_LIST,
        decay_fixed=ANCHOR_DECAY,
        decay_list=DEFAULT_DECAY_LIST,
        ic_fixed=ANCHOR_IC, icir_fixed=ANCHOR_ICIR,
        seeds=SEEDS, q=Q,
    )

    # ── 标定 TC ──────────────────────────────────────────────────
    center = df[
        (df["sweep"] == "ic_icir")
        & (np.isclose(df["in_ic"], ANCHOR_IC))
        & (np.isclose(df["in_icir"], ANCHOR_ICIR))
    ]
    ir_theory_center = float(center["ir_theory"].iloc[0])
    ls_ir_center     = float(center["ls_ir"].iloc[0])
    full_ir_mean     = float(np.mean(list(KNOWN_ANCHORS.values())))
    tc_theory = full_ir_mean / ir_theory_center      # full / 理论上限
    tc_ls     = full_ir_mean / ls_ir_center          # full / 分位多空

    # 估算全管线 IR = 理论IR × TC
    df["est_full_ir"] = df["ir_theory"] * tc_theory

    print(f"\n{'='*64}")
    print(f"  TC 标定（锚点 IC={ANCHOR_IC} ICIR={ANCHOR_ICIR} decay={ANCHOR_DECAY}）")
    print(f"{'='*64}")
    for name, ir in KNOWN_ANCHORS.items():
        print(f"  {name:<12}: full_IR={ir:.2f}")
    print(f"  {'锚点均值':<12}: full_IR={full_ir_mean:.2f}")
    print(f"  中心点理论IR (ICIR×√{runner.periods_per_year:.0f}) = {ir_theory_center:.2f}")
    print(f"  中心点分位多空IR              = {ls_ir_center:.2f}")
    print(f"  → TC(理论) = {tc_theory:.3f}   TC(分位) = {tc_ls:.3f}")
    print(f"\n  经验法则:  年化IR ≈ ICIR × √{runner.periods_per_year:.0f} × {tc_theory:.2f} "
          f"≈ {np.sqrt(runner.periods_per_year)*tc_theory:.1f} × ICIR")

    # ── 查找表：IC × ICIR → 估算全管线 IR ────────────────────────
    sub = df[df["sweep"] == "ic_icir"]
    pivot = sub.pivot_table(index="in_ic", columns="in_icir", values="est_full_ir")
    print(f"\n{'─'*64}")
    print(f"  查找表：估算全管线年化 IR   (行=IC, 列=ICIR, decay={ANCHOR_DECAY})")
    print(f"{'─'*64}")
    hdr = "   IC\\ICIR " + "".join(f"{c:>8.1f}" for c in pivot.columns)
    print(hdr)
    for ic_val, row in pivot.iterrows():
        print(f"  {ic_val:>8.2f}  " + "".join(f"{v:>8.2f}" for v in row.values))

    # ── decay 衰减对照 ───────────────────────────────────────────
    dsub = df[df["sweep"] == "decay"].sort_values("in_decay")
    print(f"\n{'─'*64}")
    print(f"  decay 影响  (IC={ANCHOR_IC} ICIR={ANCHOR_ICIR})")
    print(f"{'─'*64}")
    print(f"  {'decay':>6} {'实测IC':>8} {'实测ICIR':>9} {'LS_IR':>7} "
          f"{'估算IR':>7} {'自相关':>7} {'换手':>6}")
    for _, r in dsub.iterrows():
        print(f"  {r['in_decay']:>6.2f} {r['ic']:>8.4f} {r['icir']:>9.2f} "
              f"{r['ls_ir']:>7.2f} {r['est_full_ir']:>7.2f} "
              f"{r['autocorr']:>7.2f} {r['turnover']:>6.2f}")

    # ── 保存 ─────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    res_path = out_dir / "signal_grid_results.parquet"
    df.to_parquet(res_path)

    _make_heatmap(pivot, tc_theory, runner.periods_per_year,
                  out_dir / "signal_grid_ir.html")

    print(f"\n  结果表  : {res_path}")
    print(f"  热图    : {out_dir / 'signal_grid_ir.html'}")
    print(f"\n{'='*64}\n")


def _make_heatmap(pivot: pd.DataFrame, tc: float, ppy: float, out_path: Path) -> None:
    z = pivot.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{c:.1f}" for c in pivot.columns],
        y=[f"{i:.2f}" for i in pivot.index],
        colorscale="RdYlGn",
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 13},
        colorbar=dict(title="估算IR"),
        hovertemplate="IC=%{y} ICIR=%{x}<br>估算全管线IR=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=(f"因子信号强度 → 估算全管线年化IR"
               f"（5日调仓，TC≈{tc:.2f}；公式 IR≈{np.sqrt(ppy)*tc:.1f}×ICIR）"),
        xaxis_title="ICIR（IC均值 / IC标准差）",
        yaxis_title="IC（截面信息系数）",
        template="plotly_white",
        height=480, width=720,
        margin=dict(l=70, r=40, t=70, b=60),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    main()
