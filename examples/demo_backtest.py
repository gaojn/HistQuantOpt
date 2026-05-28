"""
量化回测 Demo。

输入  : examples/batch_weights_alpha_max.parquet（96期权重矩阵）
基准  : 沪深300 等权日收益
成本  : 单边 0.15%（佣金+冲击）
输出  : 净值曲线 + 绩效指标 + ASCII 图表

运行：
    cd /Users/guoguo/Desktop/HistQuantOpt
    python examples/demo_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.backtest.engine import Backtester
from portfolio_optimizer.backtest.report import generate_html_report

WEIGHT_PATH = Path("output/batch_weights_alpha_max.parquet")


# ─────────────────────────────────────────────────────────────────
# 辅助：ASCII 净值曲线
# ─────────────────────────────────────────────────────────────────

def ascii_chart(series: pd.Series, title: str, width: int = 60, height: int = 12) -> None:
    """在终端打印简单的 ASCII 折线图。"""
    vals = series.values.astype(float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-8:
        print(f"  {title}: 无变化")
        return

    # 降采样到 width 个点
    idx = np.linspace(0, len(vals) - 1, width, dtype=int)
    pts = vals[idx]

    print(f"\n  {title}")
    print(f"  最高: {hi:.4f}  最低: {lo:.4f}  末值: {vals[-1]:.4f}")
    print()

    rows = []
    for row in range(height, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = ""
        for v in pts:
            line += "█" if v >= threshold else " "
        label = f"{threshold:.3f} |" if row % (height // 4) == 0 else "       |"
        rows.append(f"  {label}{line}")
    print("\n".join(rows))

    # X 轴日期标签
    n_dates = len(series)
    date_pts = [series.index[i] for i in idx[::width // 5]]
    labels = "  " + " " * 8
    step = width // (len(date_pts) - 1) if len(date_pts) > 1 else width
    for d in date_pts:
        labels += str(d)[:7].ljust(step)
    print(labels)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*55}")
    print(f"  量化回测  AlphaMax 组合 vs 沪深300基准")
    print(f"{'='*55}")

    # ── 1. 加载权重矩阵 ──────────────────────────────────────
    print("\n[1] 加载权重矩阵...")
    weight_df = pd.read_parquet(WEIGHT_PATH)
    weight_df.index = pd.to_datetime(weight_df.index)
    start_date = weight_df.index[0].date()
    end_date   = weight_df.index[-1].date()
    print(f"  调仓期数={len(weight_df)}  股票池={weight_df.shape[1]}")
    print(f"  区间: {start_date} ~ {end_date}")

    # ── 2. 加载行情（含市值字段以便计算指数权重）──────────────
    print("\n[2] 加载行情数据...")
    panel = load_panel(
        start_date, end_date,
        columns=["code", "date", "adj_close",
                 "free_mv", "total_mv", "trade_status",
                 "is_hs300"],
    )
    adj_wide = (
        panel.select(["date", "code", "adj_close"])
        .to_pandas()
        .pivot(index="date", columns="code", values="adj_close")
        .sort_index()
    )
    adj_wide.index = pd.to_datetime(adj_wide.index)
    print(f"  交易日={len(adj_wide)}  股票={adj_wide.shape[1]}")

    # ── 3. 构建沪深300基准收益（分级靠档加权）────────────────
    print("\n[3] 构建沪深300基准（分级靠档加权）...")
    bm_calc = IndexBenchmarkWeights(index="hs300", panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache                # (date, ticker) 权重矩阵
    bm_weights.index = pd.to_datetime(bm_weights.index)
    bm_weights = bm_weights.sort_index()

    # 个股日收益（截面对齐）
    daily_ret_all = adj_wide.pct_change(fill_method=None).fillna(0.0)

    # 基准日收益 = 上一日权重 · 当日个股收益（避免前瞻）
    weights_lag = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common_cols = weights_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (
        (weights_lag[common_cols].fillna(0.0) *
         daily_ret_all[common_cols].fillna(0.0)).sum(axis=1)
    )
    bm_ret.name = "HS300_FreeFloat"
    print(f"  日均成分股数 = {(bm_weights > 0).sum(axis=1).mean():.0f}  "
          f"权重和均值 = {bm_weights.sum(axis=1).mean():.4f}")

    # ── 4. 回测 ───────────────────────────────────────────────
    print("\n[4] 执行回测（单边成本=0.15%）...")
    bt = Backtester(cost_one_way=0.0015, risk_free=0.02)
    result = bt.run(weight_df, adj_wide, benchmark_ret=bm_ret)

    # ── 5. 绩效汇总 ───────────────────────────────────────────
    print(f"\n{result.summary()}")

    # ── 6. 年度分解 ───────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  年度收益分解")
    print(f"{'─'*55}")
    print(f"  {'年份':<8} {'组合':>9}  {'基准':>9}  {'超额':>9}  {'最大回撤':>10}")
    print(f"  {'-'*50}")
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        port_y = (1 + grp).prod() - 1
        bm_y   = (1 + result.bm_ret.reindex(grp.index).fillna(0)).prod() - 1
        exc_y  = port_y - bm_y
        nav_y  = (1 + grp).cumprod()
        mdd_y  = (nav_y / nav_y.cummax() - 1).min()
        print(f"  {year:<8} {port_y*100:>+8.2f}%  {bm_y*100:>+8.2f}%  "
              f"{exc_y*100:>+8.2f}%  {mdd_y*100:>9.2f}%")

    # ── 7. 月度超额热图 ───────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  月度超额收益（组合 - 基准）")
    print(f"{'─'*55}")
    monthly_port = (1 + result.daily_ret).resample("ME").prod() - 1
    monthly_bm   = (1 + result.bm_ret).resample("ME").prod() - 1
    monthly_exc  = (monthly_port - monthly_bm) * 100

    years  = sorted(monthly_exc.index.year.unique())
    months = range(1, 13)
    header = f"  {'':>6}" + "".join(f"{m:>7}" for m in ["Jan","Feb","Mar","Apr","May","Jun",
                                                           "Jul","Aug","Sep","Oct","Nov","Dec"])
    print(header)
    for yr in years:
        row_str = f"  {yr:<6}"
        for m in months:
            key = monthly_exc[(monthly_exc.index.year == yr) & (monthly_exc.index.month == m)]
            if len(key) > 0:
                v = key.iloc[0]
                sign = "+" if v >= 0 else ""
                row_str += f" {sign}{v:.2f}%"[:7].rjust(7)
            else:
                row_str += f"{'':>7}"
        print(row_str)

    # ── 8. ASCII 净值曲线 ─────────────────────────────────────
    ascii_chart(result.nav,        "组合净值曲线")
    ascii_chart(result.bm_nav,     "基准净值曲线（HS300分级靠档）")
    ascii_chart(result.excess_nav, "超额净值曲线（组合/基准）")

    # ── 9. 换手统计 ───────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  换手率统计")
    print(f"{'─'*55}")
    to = result.turnover
    print(f"  平均双边换手  : {to.mean()*100:.1f}%")
    print(f"  最大双边换手  : {to.max()*100:.1f}%")
    print(f"  最小双边换手  : {to.min()*100:.1f}%")
    print(f"  年化双边换手  : {to.mean()*100 * 52:.0f}%"
          f"  （按每年约52次调仓估算）")

    # ── 保存 ─────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 净值曲线
    nav_out = out_dir / "nav.parquet"
    pd.DataFrame({
        "nav": result.nav,
        "bm_nav": result.bm_nav,
        "excess_nav": result.excess_nav,
        "port_ret": result.daily_ret,
        "bm_ret": result.bm_ret,
        "excess_ret": result.excess_ret,
    }).to_parquet(nav_out)

    # 换手记录
    to_out = out_dir / "turnover.parquet"
    result.turnover.to_frame("bilateral_turnover").to_parquet(to_out)

    print(f"\n  净值曲线已保存  : {nav_out}")
    print(f"  换手记录已保存  : {to_out}")

    # ── HTML 报告 ────────────────────────────────────────────
    report_path = generate_html_report(
        result,
        output_path=out_dir / "backtest_report.html",
        title="量化多头组合回测报告 (AlphaMax)",
    )
    print(f"  HTML 报告已生成 : {report_path}")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
