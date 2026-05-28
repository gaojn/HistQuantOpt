"""
指数增强回测 + HTML 报告生成（HS300 / ZZ500 / ZZ1000 通用）。

输入  : output/{INDEX}_enhance_weights.parquet
基准  : 同 INDEX（分级靠档加权）
成本  : 单边 0.15%
输出  : output/{INDEX}_enhance_report.html

运行：
    python examples/demo_index_enhance_backtest.py
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

INDEX = "zz500"   # 与批量优化保持一致：hs300 / zz500 / zz1000
WEIGHT_PATH = Path(f"output/{INDEX}_enhance_weights.parquet")
INDEX_NAME = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}[INDEX]


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  {INDEX_NAME} 指数增强回测")
    print(f"{'='*60}")

    # 权重
    print("\n[1] 加载权重矩阵...")
    weight_df = pd.read_parquet(WEIGHT_PATH)
    weight_df.index = pd.to_datetime(weight_df.index)
    start_date = weight_df.index[0].date()
    end_date   = weight_df.index[-1].date()
    print(f"  调仓期数={len(weight_df)}  股票池={weight_df.shape[1]}")
    print(f"  区间: {start_date} ~ {end_date}")

    # 行情
    print("\n[2] 加载行情...")
    panel = load_panel(
        start_date, end_date,
        columns=["code", "date", "adj_close",
                 "free_mv", "total_mv", "trade_status",
                 "is_hs300", "is_zz500", "is_zz1000"],
    )
    adj_wide = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    adj_wide.index = pd.to_datetime(adj_wide.index)
    print(f"  交易日={len(adj_wide)}  股票={adj_wide.shape[1]}")

    # 基准（分级靠档）
    print(f"\n[3] 构建 {INDEX_NAME} 基准（分级靠档加权）...")
    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache
    bm_weights.index = pd.to_datetime(bm_weights.index)
    bm_weights = bm_weights.sort_index()

    daily_ret_all = adj_wide.pct_change(fill_method=None).fillna(0.0)
    w_lag = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (
        w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)
    ).sum(axis=1)
    bm_ret.name = INDEX.upper()
    print(f"  日均成分股={int((bm_weights>0).sum(axis=1).mean())}  "
          f"权重和均值={bm_weights.sum(axis=1).mean():.4f}")

    # 回测
    print("\n[4] 执行回测（单边成本=0.15%）...")
    bt = Backtester(cost_one_way=0.0015, risk_free=0.02)
    result = bt.run(weight_df, adj_wide, benchmark_ret=bm_ret)

    # 控制台输出
    print(f"\n{result.summary()}")

    # 年度（区间累计；超额=几何方法 (1+组合)/(1+基准)-1）
    print(f"\n{'─'*60}\n  年度收益分解（区间累计；超额=几何）\n{'─'*60}")
    print(f"  {'年份':<10} {'组合':>9}  {'基准':>9}  {'超额':>9}  {'最大回撤':>10}")
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        port_y = (1 + grp).prod() - 1
        bm_y   = (1 + result.bm_ret.reindex(grp.index).fillna(0)).prod() - 1
        exc_y  = (1 + port_y) / (1 + bm_y) - 1 if (1 + bm_y) > 1e-8 else 0.0
        nav_y  = (1 + grp).cumprod()
        mdd_y  = (nav_y / nav_y.cummax() - 1).min()
        n_days = len(grp)
        print(f"  {year} ({n_days:>3d}d) {port_y*100:>+8.2f}%  {bm_y*100:>+8.2f}%  "
              f"{exc_y*100:>+8.2f}%  {mdd_y*100:>9.2f}%")

    # 换手
    to = result.turnover
    print(f"\n  平均双边换手  : {to.mean()*100:.1f}%")
    print(f"  年化双边换手  : {to.mean()*100 * 52:.0f}%")

    # 保存
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    nav_out = out_dir / f"{INDEX}_enhance_nav.parquet"
    pd.DataFrame({
        "nav": result.nav,
        "bm_nav": result.bm_nav,
        "excess_nav": result.excess_nav,
        "port_ret": result.daily_ret,
        "bm_ret": result.bm_ret,
        "excess_ret": result.excess_ret,
    }).to_parquet(nav_out)

    to_out = out_dir / f"{INDEX}_enhance_turnover.parquet"
    result.turnover.to_frame("bilateral_turnover").to_parquet(to_out)

    print(f"\n  净值已保存     : {nav_out}")
    print(f"  换手已保存     : {to_out}")

    # HTML 报告
    report_path = generate_html_report(
        result,
        output_path=out_dir / f"{INDEX}_enhance_report.html",
        title=f"{INDEX_NAME} 指数增强组合回测报告",
    )
    print(f"  HTML 报告      : {report_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
