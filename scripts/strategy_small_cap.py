"""
小市值策略：选择流通市值最小的 200 只股票等权持有。

选股规则（每期）：
  - 剔除 ST / *ST（is_st == 1）
  - 剔除停牌（trade_status == '停牌'）
  - 剔除北交所（code 以 .BJ 结尾）
  - 剔除上市不足 60 自然日（次新，流动性差）
  - 按流通市值（float_mv）升序，取最小 200 只
  - 等权持有（各占 1/200）

执行规则：
  - 10 个交易日换仓一次（T 日信号，T+1 VWAP 成交）
  - 买入费率 0.1‰，卖出费率 0.2‰

运行：
    python scripts/strategy_small_cap.py
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import polars as pl

from portfolio_optimizer.io.data_panel import load_panel
from scripts.run_backtest import run_backtest

# ── 路径配置 ──────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
CACHE_DIR        = PROJECT_ROOT / "data" / "cache"
INDEX_CLOSE_PATH = PROJECT_ROOT / "data" / "指数收盘价信息.csv"
OUT_DIR          = Path("output/small_cap_200")
WEIGHT_PATH      = OUT_DIR / "weights.parquet"

# ── 策略参数 ──────────────────────────────────────────────────────
START_DATE    = date(2020, 1, 1)
END_DATE      = date(2026, 5, 31)
REBAL_FREQ    = 10       # 每 N 个交易日换仓一次
TOP_N         = 200      # 选最小市值 N 只
MIN_LIST_DAYS = 60       # 上市不足此天数剔除（次新）
INDEX         = "zz1000" # 基准


def build_weights(
    panel: pl.DataFrame,
    trade_dates: list,
    rebal_freq: int = REBAL_FREQ,
    top_n: int = TOP_N,
    min_list_days: int = MIN_LIST_DAYS,
) -> pd.DataFrame:
    """
    对每个换仓日生成等权权重，返回长表 DataFrame（date, code, weight）。
    """
    rebal_dates = trade_dates[::rebal_freq]
    print(f"  换仓日共 {len(rebal_dates)} 期（首期={rebal_dates[0]}，末期={rebal_dates[-1]}）")

    records: list[dict] = []

    for rebal_date in rebal_dates:
        # 取当日截面
        day = (
            panel
            .filter(pl.col("date") == rebal_date)
            .select(["code", "float_mv", "trade_status", "is_st", "list_days"])
            .to_pandas()
        )

        if day.empty:
            continue

        # 过滤条件
        mask = (
            (day["is_st"] == 0) &
            # 仅剔除停牌；XD/XR/N（除息/除权/新股首日）均可正常交易，
            # 与 engine 一致（engine 只把 trade_status == "停牌" 视作不可成交）
            (day["trade_status"] != "停牌") &
            (~day["code"].str.endswith(".BJ")) &
            (day["list_days"] >= min_list_days) &
            (day["float_mv"] > 0)
        )
        universe = day[mask].copy()

        if len(universe) < top_n:
            print(f"  [{rebal_date}] 警告：有效股票 {len(universe)} < {top_n}，全部等权纳入")

        # 按流通市值升序，取最小 top_n 只
        selected = universe.nsmallest(top_n, "float_mv")
        n = len(selected)
        w = 1.0 / n

        for code in selected["code"]:
            records.append({"date": pd.Timestamp(rebal_date), "code": code, "weight": w})

    weight_long = pd.DataFrame(records)
    return weight_long


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  小市值 Top{TOP_N} 等权策略  |  {START_DATE} ~ {END_DATE}")
    print(f"  换仓频率={REBAL_FREQ}日  最小上市天数={MIN_LIST_DAYS}  基准={INDEX.upper()}")
    print(f"{'='*65}")

    # ── 1. 加载面板 ───────────────────────────────────────────────
    # 多取年初数据以保证 list_days 可计算
    data_start = date(START_DATE.year - 1, 1, 1)
    print(f"\n[1] 加载行情面板（{data_start} ~ {END_DATE}）...")
    panel = load_panel(
        data_start, END_DATE,
        columns=["code", "date", "float_mv", "trade_status", "is_st", "list_days"],
        cache_dir=CACHE_DIR,
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 2. 换仓日列表（回测区间内的所有交易日） ─────────────────
    trade_dates = (
        panel
        .filter(
            (pl.col("date") >= START_DATE) &
            (pl.col("date") <= END_DATE)
        )
        .select("date").unique().sort("date")["date"].to_list()
    )
    print(f"  回测区间交易日数={len(trade_dates)}")

    # ── 3. 生成权重 ───────────────────────────────────────────────
    print("\n[2] 生成换仓权重...")
    weight_long = build_weights(panel, trade_dates)
    print(f"  权重记录数={len(weight_long)}  换仓日={weight_long['date'].nunique()}")

    # 保存权重（长表）
    weight_long.to_parquet(WEIGHT_PATH, index=False)
    print(f"  权重已保存：{WEIGHT_PATH}")

    # ── 4. 回测 ───────────────────────────────────────────────────
    print("\n[3] 启动回测...")
    result, exec_stats = run_backtest(
        weight_path=WEIGHT_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        index=INDEX,
        cost_buy=0.001,
        cost_sell=0.002,
        initial_value=1e8,
        out_dir=OUT_DIR,
        title=f"小市值 Top{TOP_N} 等权策略（{START_DATE}~{END_DATE}，基准={INDEX.upper()}）",
        cache_dir=CACHE_DIR,
        index_close_path=INDEX_CLOSE_PATH,
    )

    # ── 5. 简要统计 ───────────────────────────────────────────────
    nav = result.nav
    print(f"\n{'='*65}")
    print(f"  净值区间：{nav.index[0].date()} ~ {nav.index[-1].date()}")
    print(f"  期末净值：{nav.iloc[-1]:.4f}")
    pm = result.portfolio_metrics
    print(f"  年化收益：{pm.annual_return*100:+.2f}%  Sharpe={pm.sharpe:.2f}")
    print(f"  最大回撤：{pm.max_drawdown*100:.2f}%  Calmar={pm.calmar:.2f}")
    print(f"  年化超额：{pm.annual_excess_return*100:+.2f}%  IR={pm.info_ratio:.2f}")
    print(f"  超额最大回撤：{pm.excess_max_drawdown*100:.2f}%")
    print(f"\n  报告：{OUT_DIR}/report.html")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
