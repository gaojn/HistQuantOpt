"""
量化多头 Alpha 最大化选股 Demo。

目标   : max w'α - γ·‖w‖²
约束   : 单票上限 2%、行业上限 20%、风格绝对暴露 ±1σ、换手率 ≤ 50%
日期   : 2026-05-21  |  指数成分股：沪深300 ≥ 40%（可选约束演示）

运行：
    cd /Users/guoguo/Desktop/HistQuantOpt
    python examples/demo_alpha_max.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.factors.jy_barra import JYBarraFactors
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer

TARGET_DATE  = date(2026, 5, 21)
INDEX        = "hs300"
FACTOR_PATH  = Path("data/jy_stylefactor_000985_CSI_20230209_20260522.parquet")


def build_alpha(barra: JYBarraFactors) -> np.ndarray:
    """因子合成 Alpha（示例权重，替换为真实 IC 加权即可）。"""
    weights = {
        "Momentum":      0.30,
        "EarningsYield": 0.25,
        "BookToPrice":   0.20,
        "Growth":        0.15,
        "Size":         -0.10,
    }
    style = barra.style_loading
    alpha = sum(w * style[f].values for f, w in weights.items() if f in style.columns)
    return np.array(alpha, dtype=float)


def run_and_print(
    label: str,
    config: AlphaMaxConfig,
    alpha: np.ndarray,
    snapshot,
    style_loading: pd.DataFrame,
    prev_weight: np.ndarray | None = None,
) -> np.ndarray:
    optimizer = AlphaMaxOptimizer(config)
    result = optimizer.optimize(
        alpha, snapshot,
        style_loading=style_loading,
        prev_weight=prev_weight,
    )

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(result.summary())

    # 风格暴露
    exp = result.style_exposures(style_loading)
    print(f"\n  风格因子绝对暴露：")
    for fname, val in exp.items():
        bar = "█" * min(int(abs(val) * 10), 20)
        sign = "+" if val >= 0 else ""
        print(f"    {fname:<22} {sign}{val:+.3f}  {bar}")

    print(f"\n  行业权重（前6）：")
    for ind, w in result.industry_weights().head(6).items():
        print(f"    {str(ind):<18} {w*100:>5.2f}%  {'▓'*int(w*100)}")

    print(f"\n  前10大持仓：")
    top = result.top_holdings(10)
    for ticker, row in top.iterrows():
        ic = "✓" if row.get("is_constituent", False) else " "
        ind = str(row.get("industry", ""))[:12]
        print(f"    {ticker:<14} {row['weight_pct']:>5.3f}%  {ind:<14} {ic}")

    return result.weights


def main() -> None:
    print(f"\n{'='*55}")
    print(f"  量化多头选股  {TARGET_DATE}")
    print(f"{'='*55}")

    # ── 数据准备 ──────────────────────────────────────────
    print("\n[准备] 加载数据...")
    panel = load_panel(
        date(2026, 4, 1), TARGET_DATE,
        columns=[
            "code", "date", "close", "limit_up", "limit_down",
            "amount", "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status", "industry_l1",
            "list_days", "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    adapter = RealMarketAdapter()
    snapshot = adapter.build_snapshot_from_panel(
        panel=panel, target_date=TARGET_DATE,
        index=INDEX, portfolio_value=1e8,
    )
    barra = JYBarraFactors(
        snapshot=snapshot, target_date=TARGET_DATE,
        factor_path=FACTOR_PATH, panel=panel,
    )
    alpha = build_alpha(barra)
    style_loading = barra.style_loading
    print(f"  全市场={len(snapshot.tickers)}  HS300成分股={snapshot.constituent_mask.sum()}")

    # ── 场景1：基础配置（轻度分散，无换手约束）─────────────
    cfg1 = AlphaMaxConfig(
        weight_upper=0.02,
        industry_upper=0.20,
        min_constituent_ratio=0.40,
        diversification_penalty=0.02,   # 轻度分散
        style_bound=1.0,
        max_turnover=None,
    )
    w1 = run_and_print("场景1：轻度分散  γ=0.02  无换手约束", cfg1, alpha, snapshot, style_loading)

    # ── 场景2：中度分散 + 换手约束 50% ──────────────────────
    cfg2 = AlphaMaxConfig(
        weight_upper=0.02,
        industry_upper=0.20,
        min_constituent_ratio=0.40,
        diversification_penalty=0.10,   # 中度分散
        style_bound=1.0,
        max_turnover=0.50,
    )
    # 用场景1结果作为上期权重
    w2 = run_and_print("场景2：中度分散  γ=0.10  换手≤50%", cfg2, alpha, snapshot, style_loading,
                       prev_weight=w1)

    # ── 场景3：强分散 + 换手约束 30% ────────────────────────
    cfg3 = AlphaMaxConfig(
        weight_upper=0.02,
        industry_upper=0.20,
        min_constituent_ratio=0.40,
        diversification_penalty=0.30,   # 强分散，趋近等权
        style_bound=1.0,
        max_turnover=0.30,
    )
    run_and_print("场景3：强分散    γ=0.30  换手≤30%", cfg3, alpha, snapshot, style_loading,
                  prev_weight=w2)

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
