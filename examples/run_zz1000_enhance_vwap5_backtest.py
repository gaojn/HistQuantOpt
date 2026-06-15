"""
中证1000 指数增强回测（真实执行模式，VWAP5 合成 Alpha 测试）。

⚠️ Alpha 来自 alphas/alpha_vwap5_ic0.08_icir0.8_decay0.80_wide.parquet，
   基于未来 H=5 日 adj_vwap 涨跌幅反向构造（含前视），仅用于模拟回测，
   不代表真实可交易因子。

执行规则：
  T+1 adj_vwap 成交，涨停不能买（留现金），跌停不能卖（T+2/T+3重试），停牌不可交易
  交易成本：买入 1/1000，卖出 2/1000

输入  : output/zz1000_enhance_vwap5_weights.parquet
输出  : output/zz1000_enhance_vwap5_report_realistic.html

运行：
    python examples/run_zz1000_enhance_vwap5_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester
from portfolio_optimizer.backtest.report import generate_html_report

INDEX       = "zz1000"
INDEX_NAME  = "中证1000"
WEIGHT_PATH = Path(f"output/{INDEX}_enhance_vwap5_weights.parquet")
TAG         = "vwap5"


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  {INDEX_NAME} 指数增强回测（真实执行模式，VWAP5 合成Alpha）")
    print(f"  T+1 VWAP 成交 | 涨跌停处理 | 买1‰ 卖2‰")
    print(f"{'='*60}")

    print("\n[1] 加载权重矩阵...")
    weight_df = pd.read_parquet(WEIGHT_PATH)
    weight_df.index = pd.to_datetime(weight_df.index)
    start_date = weight_df.index[0].date()
    end_date   = weight_df.index[-1].date()
    print(f"  调仓期数={len(weight_df)}  股票池={weight_df.shape[1]}")
    print(f"  区间: {start_date} ~ {end_date}")

    print("\n[2] 加载行情（adj_close / adj_vwap / limit / status）...")
    panel = load_panel(
        start_date, end_date,
        columns=[
            "code", "date",
            "adj_close", "adj_vwap",
            "close",
            "limit_up", "limit_down",
            "trade_status",
            "free_mv", "total_mv",
            "is_hs300", "is_zz500", "is_zz1000",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    def to_wide(col: str) -> pd.DataFrame:
        return (
            panel.select(["date", "code", col]).to_pandas()
            .pivot(index="date", columns="code", values=col)
            .sort_index()
        )

    adj_close_w    = to_wide("adj_close")
    adj_vwap_w     = to_wide("adj_vwap")
    close_raw_w    = to_wide("close")
    limit_up_w     = to_wide("limit_up")
    limit_down_w   = to_wide("limit_down")
    trade_status_w = to_wide("trade_status")

    for df in [adj_close_w, adj_vwap_w, close_raw_w, limit_up_w, limit_down_w, trade_status_w]:
        df.index = pd.to_datetime(df.index)

    print(f"\n[3] 构建 {INDEX_NAME} 基准（分级靠档）...")
    bm_calc = IndexBenchmarkWeights(index=INDEX, panel=panel)
    bm_calc.precompute(start_date, end_date, panel=panel)
    bm_weights = bm_calc._weight_cache.copy()
    bm_weights.index = pd.to_datetime(bm_weights.index)

    daily_ret_all = adj_close_w.pct_change(fill_method=None).fillna(0.0)
    w_lag  = bm_weights.shift(1).reindex(daily_ret_all.index).ffill()
    common = w_lag.columns.intersection(daily_ret_all.columns)
    bm_ret = (w_lag[common].fillna(0.0) * daily_ret_all[common].fillna(0.0)).sum(axis=1)
    bm_ret.name = INDEX.upper()

    print("\n[4] 执行真实回测（T+1 VWAP，涨跌停处理）...")
    bt = RealisticBacktester(cost_buy=0.001, cost_sell=0.002, risk_free=0.02)
    result, exec_stats = bt.run(
        weight_df      = weight_df,
        adj_close      = adj_close_w,
        adj_vwap       = adj_vwap_w,
        close_raw      = close_raw_w,
        limit_up_df    = limit_up_w,
        limit_down_df  = limit_down_w,
        trade_status_df= trade_status_w,
        benchmark_ret  = bm_ret,
        initial_value  = 1e8,
    )

    print(f"\n{result.summary()}")

    print(f"\n{'─'*60}")
    print("  执行质量统计")
    print(f"{'─'*60}")
    print(f"  涨停/停牌 无法买入次数  : {exec_stats['buy_fail_count']}")
    print(f"  跌停/停牌 延迟卖出次数  : {exec_stats['sell_defer_count']}")

    print(f"\n{'─'*60}")
    print("  年度收益分解（区间累计；超额=几何）")
    print(f"{'─'*60}")
    print(f"  {'年份':<10} {'组合':>9}  {'基准':>9}  {'超额':>9}  {'最大回撤':>10}")
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        port_y = (1 + grp).prod() - 1
        bm_y   = (1 + result.bm_ret.reindex(grp.index).fillna(0)).prod() - 1
        exc_y  = (1 + port_y) / (1 + bm_y) - 1 if abs(1 + bm_y) > 1e-8 else 0.0
        nav_y  = (1 + grp).cumprod()
        mdd_y  = float((nav_y / nav_y.cummax() - 1).min())
        print(f"  {year} ({len(grp):>3d}d) "
              f"{port_y*100:>+8.2f}%  {bm_y*100:>+8.2f}%  "
              f"{exc_y*100:>+8.2f}%  {mdd_y*100:>9.2f}%")

    to = result.turnover
    print(f"\n  平均双边换手  : {to.mean()*100:.1f}%")
    print(f"  年化双边换手  : {to.mean()*100 * 52:.0f}%")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    nav_out = out_dir / f"{INDEX}_enhance_{TAG}_nav_realistic.parquet"
    pd.DataFrame({
        "nav": result.nav, "bm_nav": result.bm_nav,
        "excess_nav": result.excess_nav,
        "port_ret": result.daily_ret, "bm_ret": result.bm_ret,
    }).to_parquet(nav_out)

    report_path = generate_html_report(
        result,
        output_path=out_dir / f"{INDEX}_enhance_{TAG}_report_realistic.html",
        title=f"{INDEX_NAME} 指数增强组合回测报告（VWAP5合成Alpha IC=0.08 ICIR=0.8，真实执行）",
    )
    print(f"\n  净值已保存     : {nav_out}")
    print(f"  HTML 报告      : {report_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
