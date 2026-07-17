"""从 ClickHouse ``wind_db.VW_ASHARE_STOCK_DAILY`` 同步行情 → ``data/cache/ashare_daily_<year>.parquet``。

源视图变更（2026-07-17）：旧 ``the_quant.vw_ashare_daily_backtest``（小写字段）已从服务器
下线，改用 ``wind_db.VW_ASHARE_STOCK_DAILY``（字段全大写），涨停价=S_DQ_LIMIT、
跌停价=S_DQ_STOPPING、vwap=S_DQ_AVGPRICE。缓存口径与既有文件严格一致（40 列不变）：
行业=中信(CITICS)、复权=后复权、trade_status=中文字符串、单位口径不变。
delist_date '20991231'→保留原值、free_mv=close×free_shares、
list_days=自上市日历日数、adj_vwap=vwap×adj_factor。

用法（单一只读凭证，覆盖全部库）：
    CLICKHOUSE_WIND_PASSWORD=... python scripts/sync_ashare_cache.py            # 默认: 当年
    CLICKHOUSE_WIND_PASSWORD=... python scripts/sync_ashare_cache.py --years 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from hqopt.data.clickhouse_db import query_df

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"

PRICE_SELECT = """
SELECT
  S_INFO_WINDCODE         AS code,
  TRADE_DT                AS date,
  S_INFO_NAME             AS name,
  S_DQ_PRECLOSE           AS pre_close,
  S_DQ_OPEN               AS open,
  S_DQ_HIGH               AS high,
  S_DQ_LOW                AS low,
  S_DQ_CLOSE              AS close,
  S_DQ_LIMIT              AS limit_up,
  S_DQ_STOPPING           AS limit_down,
  S_DQ_PCTCHANGE          AS pct_change,
  S_DQ_VOLUME             AS volume,
  S_DQ_AMOUNT             AS amount,
  S_DQ_ADJPRECLOSE        AS adj_pre_close,
  S_DQ_ADJOPEN            AS adj_open,
  S_DQ_ADJHIGH            AS adj_high,
  S_DQ_ADJLOW             AS adj_low,
  S_DQ_ADJCLOSE           AS adj_close,
  S_DQ_ADJFACTOR          AS adj_factor,
  S_DQ_TRADESTATUS        AS trade_status,
  S_DQ_AVGPRICE           AS vwap,
  S_VAL_MV                AS total_mv,
  S_DQ_MV                 AS float_mv,
  S_DQ_TURN               AS turnover,
  S_DQ_FREETURNOVER       AS free_turnover,
  TOT_SHR_TODAY           AS total_shares,
  FLOAT_A_SHR_TODAY       AS float_shares,
  FREE_SHARES_TODAY       AS free_shares,
  CITICS_IND_NAME_L1      AS industry_l1,
  CITICS_IND_NAME_L2      AS industry_l2,
  CITICS_IND_NAME_L3      AS industry_l3,
  S_INFO_LISTDATE         AS list_date,
  S_INFO_DELISTDATE       AS delist_date,
  IN_HS300                AS is_hs300,
  IN_ZZ500                AS is_zz500,
  IN_ZZ1000               AS is_zz1000,
  IS_ST                   AS is_st
FROM wind_db.VW_ASHARE_STOCK_DAILY
WHERE TRADE_DT >= '{y}-01-01' AND TRADE_DT <= '{y}-12-31'
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
