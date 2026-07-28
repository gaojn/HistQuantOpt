"""从 Wind 导出官方指数权重快照，可选生成 PIT 每日漂移权重。

默认输出月度/调样快照 ``data/index_weight/official_weight.parquet``。传 ``--daily``
后，使用 ``hqopt.data.benchmark.IndexBenchmarkWeights`` 的同一运行时实现生成
``official_weight_daily.parquet``，避免导出和优化两套公式漂移。每日记录的
``date`` 是 T 日收盘观测日，``effective_date`` 是下一交易日，外部执行必须使用
后者，不能把 T 收盘才知道的权重用于 T 日收益。
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import polars as pl

from hqopt.data.benchmark import (
    DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    IndexBenchmarkWeights,
)
from hqopt.io.data_panel import load_panel

IDX2KEY: dict[str, str] = {
    "000300.SH": "hs300",
    "000905.SH": "zz500",
    "000852.SH": "zz1000",
}

DEFAULT_OUT = Path("data/index_weight/official_weight.parquet")
DEFAULT_DAILY_OUT = Path("data/index_weight/official_weight_daily.parquet")


def fetch(start: str, end: str) -> pl.DataFrame:
    """读取区间内快照，并额外带上每个指数在 start 之前的最近锚点。"""
    pwd = os.environ.get("CLICKHOUSE_WIND_PASSWORD")
    if not pwd:
        raise RuntimeError(
            "请设置环境变量 CLICKHOUSE_WIND_PASSWORD（wind_db 只读密码，绝不入代码/git）。"
        )
    os.environ["CLICKHOUSE_PASSWORD"] = pwd
    os.environ["CLICKHOUSE_DB"] = "wind_db"
    from hqopt.data.clickhouse_db import query_df

    codes = "','".join(IDX2KEY)
    sql = f"""
        SELECT s_info_windcode AS idx, trade_dt AS date,
               s_con_windcode AS code, i_weight
        FROM AINDEXHS300FREEWEIGHT
        WHERE s_info_windcode IN ('{codes}')
          AND i_weight IS NOT NULL
          AND (
            trade_dt BETWEEN '{start}' AND '{end}'
            OR (s_info_windcode, trade_dt) IN (
              SELECT s_info_windcode, max(trade_dt)
              FROM AINDEXHS300FREEWEIGHT
              WHERE s_info_windcode IN ('{codes}')
                AND trade_dt < '{start}'
                AND i_weight IS NOT NULL
              GROUP BY s_info_windcode
            )
          )
    """
    df = query_df(sql)
    if df.is_empty():
        raise RuntimeError("查询为空，请检查日期区间")

    return df.with_columns(
        pl.col("idx").replace_strict(IDX2KEY).alias("index"),
        pl.col("date").cast(pl.Date),
        (pl.col("i_weight").cast(pl.Float64) / 100.0).alias("weight"),
    ).select(["index", "date", "code", "weight"]).sort(
        ["index", "date", "code"]
    )


def _atomic_write_parquet(frame: pl.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def build_daily_weights(
    official_path: Path,
    start: date,
    end: date,
    max_snapshot_age_days: int | None = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
) -> pl.DataFrame:
    """生成 T 日收盘权重，并标记下一交易日 ``effective_date``。"""
    panel = load_panel(
        start,
        end,
        columns=[
            "adj_close", "free_mv", "total_mv",
            "is_hs300", "is_zz500", "is_zz1000",
        ],
        add_adj_vwap=False,
    )
    dates = panel.select("date").unique().sort("date")["date"].to_list()
    outputs: list[pl.DataFrame] = []
    for index in IDX2KEY.values():
        benchmark = IndexBenchmarkWeights(
            index=index,
            panel=panel,
            source="official_drift",
            official_path=official_path,
            max_snapshot_age_days=max_snapshot_age_days,
        )
        benchmark.precompute(start, end, panel=panel)
        wide = benchmark.get_weights_matrix(dates)
        long = (
            wide.rename_axis("date")
            .stack()
            .rename("weight")
            .reset_index()
            .rename(columns={"level_1": "code"})
        )
        long = long[long["weight"] > 0].copy()
        audit = benchmark.audit_summary()
        methods = audit["method_by_period"]
        anchors = audit["snapshot_as_of_by_period"]
        ages = audit["snapshot_age_days_by_period"]
        effective_dates = audit["effective_date_by_period"]
        fallbacks = audit["fallback_reason_by_period"]
        date_keys = pd.to_datetime(long["date"]).dt.date.map(date.isoformat)
        long.insert(0, "index", index)
        long["effective_date"] = pd.to_datetime(
            date_keys.map(effective_dates), errors="coerce"
        ).dt.date
        long["anchor_date"] = pd.to_datetime(date_keys.map(anchors), errors="coerce").dt.date
        long["snapshot_age_days"] = date_keys.map(ages).astype("Int64")
        long["method"] = date_keys.map(methods)
        long["fallback_reason"] = date_keys.map(fallbacks)
        outputs.append(
            pl.from_pandas(long).with_columns(
                pl.col("method").cast(pl.String),
                pl.col("fallback_reason").cast(pl.String),
            )
        )
    return pl.concat(outputs).sort(["index", "date", "code"])


def _print_snapshot_summary(frame: pl.DataFrame) -> None:
    for key in IDX2KEY.values():
        sub = frame.filter(pl.col("index") == key)
        ndays = sub["date"].n_unique()
        last = sub["date"].max()
        wsum = sub.filter(pl.col("date") == last)["weight"].sum()
        print(
            f"      {key:7s}: {ndays} 个快照  {sub['date'].min()}~{last}  "
            f"末期∑权重={wsum:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出指数官方成分权重 → parquet")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--daily", action="store_true", help="同时生成 PIT 每日漂移权重")
    parser.add_argument("--daily-out", default=str(DEFAULT_DAILY_OUT))
    parser.add_argument(
        "--max-snapshot-age-days",
        type=int,
        default=DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
        help="official_drift 快照最大陈旧自然日数（默认 30）",
    )
    args = parser.parse_args()

    print(f"[1/3] 拉取 {len(IDX2KEY)} 个指数官方权重（{args.start} ~ {args.end}）...")
    snapshots = fetch(args.start, args.end)
    _print_snapshot_summary(snapshots)

    output = Path(args.out)
    _atomic_write_parquet(snapshots, output)
    print(f"[2/3] 已写入 {output}  ({snapshots.height:,} 行)")

    if not args.daily:
        print("[3/3] 未指定 --daily，跳过每日权重")
        return

    daily = build_daily_weights(
        output,
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        max_snapshot_age_days=args.max_snapshot_age_days,
    )
    daily_output = Path(args.daily_out)
    _atomic_write_parquet(daily, daily_output)
    print(f"[3/3] 已写入 {daily_output}  ({daily.height:,} 行)")


if __name__ == "__main__":
    main()
