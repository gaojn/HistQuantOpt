"""从 ClickHouse ``the_quant`` 同步行情 → ``data/cache/ashare_daily_<year>.parquet``。

口径与既有缓存严格一致（40 列，1:1 复刻 ``vw_ashare_daily_backtest`` 视图）：
行业=中信(CITICS)、复权=后复权、trade_status=中文字符串、单位口径不变。
delist_date '20991231'→保留原值、free_mv=close×free_shares、
list_days=自上市日历日数、adj_vwap=vwap×adj_factor。

用法：
    CLICKHOUSE_PASSWORD=... python scripts/sync_ashare_cache.py            # 默认: 当年
    CLICKHOUSE_PASSWORD=... python scripts/sync_ashare_cache.py --years 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from portfolio_optimizer.data.clickhouse_db import query_df

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

PRICE_SELECT = """
SELECT
  code,
  trade_dt                AS date,
  name,
  preclose                AS pre_close,
  open, high, low, close,
  limit_up, limit_down, pct_change, volume, amount,
  adj_preclose            AS adj_pre_close,
  adj_open, adj_high, adj_low, adj_close, adj_factor,
  trade_status,
  avg_price               AS vwap,
  total_mv, float_mv,
  turn                    AS turnover,
  free_turnover, total_shares, float_shares, free_shares,
  citics_l1               AS industry_l1,
  citics_l2               AS industry_l2,
  citics_l3               AS industry_l3,
  list_date, delist_date,
  in_hs300                AS is_hs300,
  in_zz500                AS is_zz500,
  in_zz1000               AS is_zz1000,
  in_st                   AS is_st
FROM vw_ashare_daily_backtest
WHERE trade_dt >= '{y}-01-01' AND trade_dt <= '{y}-12-31'
"""

# 原 ashare_daily 缓存 40 列顺序（严格复刻，保证下游任何脚本不缺列）
PRICE_COLUMNS = [
    "code", "date", "name", "pre_close", "open", "high", "low", "close",
    "limit_up", "limit_down", "pct_change", "volume", "amount",
    "adj_pre_close", "adj_open", "adj_high", "adj_low", "adj_close",
    "adj_vwap", "adj_factor", "trade_status", "vwap",
    "total_mv", "float_mv", "free_mv", "turnover", "free_turnover",
    "total_shares", "float_shares", "free_shares",
    "industry_l1", "industry_l2", "industry_l3",
    "list_date", "list_days", "delist_date",
    "is_hs300", "is_zz500", "is_zz1000", "is_st",
]


def sync_prices(years: list[int]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        print(f"  拉取 {y} 年行情 ...")
        df = query_df(PRICE_SELECT.format(y=y))
        if df.is_empty():
            print(f"  {y} 年无数据，跳过")
            continue
        df = df.with_columns(
            (pl.col("vwap") * pl.col("adj_factor")).alias("adj_vwap"),
            (pl.col("close") * pl.col("free_shares")).alias("free_mv"),
            (pl.col("date") - pl.col("list_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
            .dt.total_days().alias("list_days"),
        ).with_columns(
            pl.col("date").cast(pl.Datetime("ms")),  # 复刻原缓存 datetime[ms]
        ).select(PRICE_COLUMNS).sort(["date", "code"])
        out = CACHE_DIR / f"ashare_daily_{y}.parquet"
        df.write_parquet(out)
        print(f"  {y}: {df.height:,} 行 x {df.width} 列  日期 {df['date'].min()}~{df['date'].max()} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ClickHouse → data/cache/ashare_daily_<year>.parquet 同步")
    ap.add_argument("--years", nargs="*", type=int, help="指定年份（默认: 当年）")
    args = ap.parse_args()

    years = sorted(args.years) if args.years else [date.today().year]
    print(f"同步年份: {years}")
    sync_prices(years)
    print("完成。")


if __name__ == "__main__":
    main()
