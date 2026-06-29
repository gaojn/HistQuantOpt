"""从 wind_db.AINDEXHS300FREEWEIGHT 导出指数官方成分权重（自由流通口径）。

输出长表 parquet：data/index_weight/official_weight.parquet
    列：index(hs300/zz500/zz1000), date(Date), code(成分股), weight(小数，∑≈1)

供 portfolio_optimizer.data.benchmark.IndexBenchmarkWeights 的 source="official"
模式读取（按调仓日 asof 取 ≤当日最近的官方快照；缺数据时回退 free_mv 重构）。

数据为官方**月度/调样快照**（非逐日）。数据源 wind_db（密码走
CLICKHOUSE_WIND_PASSWORD，绝不入代码/git），连接方式同 export_index_close.py。

运行：
    CLICKHOUSE_WIND_PASSWORD=... python scripts/export_index_weight.py
    # 可选：--start 2015-01-01 --end 2026-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

# 指数 Wind 代码 → 项目 key
IDX2KEY: dict[str, str] = {
    "000300.SH": "hs300",
    "000905.SH": "zz500",
    "000852.SH": "zz1000",
}

DEFAULT_OUT = Path("data/index_weight/official_weight.parquet")


def fetch(start: str, end: str) -> pl.DataFrame:
    pwd = os.environ.get("CLICKHOUSE_WIND_PASSWORD")
    if not pwd:
        raise RuntimeError(
            "请设置环境变量 CLICKHOUSE_WIND_PASSWORD（wind_db 只读密码，绝不入代码/git）。"
        )
    os.environ["CLICKHOUSE_PASSWORD"] = pwd
    os.environ["CLICKHOUSE_DB"] = "wind_db"
    from portfolio_optimizer.data.clickhouse_db import query_df

    codes = "','".join(IDX2KEY)
    sql = f"""
        SELECT s_info_windcode AS idx, trade_dt AS date,
               s_con_windcode AS code, i_weight
        FROM AINDEXHS300FREEWEIGHT
        WHERE s_info_windcode IN ('{codes}')
          AND trade_dt BETWEEN '{start}' AND '{end}'
          AND i_weight IS NOT NULL
    """
    df = query_df(sql)
    if df.is_empty():
        raise RuntimeError("查询为空，请检查日期区间")

    return df.with_columns(
        pl.col("idx").replace_strict(IDX2KEY).alias("index"),
        pl.col("date").cast(pl.Date),
        (pl.col("i_weight").cast(pl.Float64) / 100.0).alias("weight"),  # % → 小数
    ).select(["index", "date", "code", "weight"]).sort(["index", "date", "code"])


def main() -> None:
    p = argparse.ArgumentParser(description="导出指数官方成分权重 → parquet")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    print(f"[1/2] 拉取 {len(IDX2KEY)} 个指数官方权重（{args.start} ~ {args.end}）...")
    df = fetch(args.start, args.end)

    for key in IDX2KEY.values():
        sub = df.filter(pl.col("index") == key)
        ndays = sub["date"].n_unique()
        # 抽查最近一个快照的权重和
        last = sub["date"].max()
        wsum = sub.filter(pl.col("date") == last)["weight"].sum()
        print(f"      {key:7s}: {ndays} 个快照  {sub['date'].min()}~{last}  "
              f"末期∑权重={wsum:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"[2/2] 已写入 {out}  ({df.height:,} 行)")


if __name__ == "__main__":
    main()
