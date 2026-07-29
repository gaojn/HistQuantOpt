"""导出 CNE6S/L 因子收益 / 特质收益：从 ClickHouse test_barra_cne6_gao
拉取，供 hqopt.analysis.attribution 做收益归因。

输出：
    data/barra_cne6_S/factor_return.parquet     —— trade_date, factor_name, ret
    data/barra_cne6_S/specific_return.parquet   —— trade_date, code, u
    data/barra_cne6_L/{factor_return,specific_return}.parquet —— CNE6L 同结构

与各目录 exposure_panel.parquet 同源（均来自 ClickHouse
test_barra_cne6_gao 的对应 S/L 模型），
保证归因用的因子暴露 X 与因子/特质收益 f、u 出自同一套模型，残差自检才对得上。

注意：新库的收益表按 S/L 模型分开，必须分别落入风险面板对应目录，不得混用。

数据源：ClickHouse test_barra_cne6_gao（环境变量 CLICKHOUSE_WIND_PASSWORD
必填，单一凭证覆盖全部库，见 hqopt/data/clickhouse_db.py）。

运行：CLICKHOUSE_WIND_PASSWORD=... python scripts/export_factor_attribution.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from hqopt.data.clickhouse_db import query_df
from hqopt.risk.cne6_risk import STYLE_FACTORS_L, STYLE_FACTORS_S

OUT_DIRS = {"S": Path("data/barra_cne6_S"), "L": Path("data/barra_cne6_L")}
SOURCE_DATABASE = "test_barra_cne6_gao"

# 与 export_cne6_panels.py 保持一致的提取范围；按需调整
START_DATE = "2020-01-01"
END_DATE = "2026-05-31"


def _validate_factor_return(frame: pl.DataFrame, variant: str) -> None:
    styles = STYLE_FACTORS_S if variant == "S" else STYLE_FACTORS_L
    expected_count = len(styles) + 31
    actual = set(frame["factor_name"].to_list())
    missing_styles = sorted(set(styles) - actual)
    if (
        frame.is_empty()
        or missing_styles
        or "Country" not in actual
        or "" in actual
        or len(actual) != expected_count
    ):
        raise ValueError(
            f"CNE6{variant} factor_return 因子合同失败："
            f"实际={len(actual)} 期望={expected_count} 缺风格={missing_styles}"
        )
    duplicates = (
        frame.group_by(["trade_date", "factor_name"])
        .len()
        .filter(pl.col("len") != 1)
    )
    required_daily = set(styles) | {"Country"}
    missing_daily = []
    for row in frame.group_by("trade_date").agg(pl.col("factor_name")).iter_rows(named=True):
        missing = sorted(required_daily - set(row["factor_name"]))
        if missing:
            missing_daily.append((row["trade_date"], missing))
    if not duplicates.is_empty() or missing_daily:
        raise ValueError(
            f"CNE6{variant} factor_return 每日核心因子不完整或有重复，"
            f"示例={missing_daily[:3]}"
        )
    if not np.isfinite(frame["ret"].to_numpy()).all():
        raise ValueError(f"CNE6{variant} factor_return 含 NaN/Inf")


def load_factor_return(variant: str) -> pl.DataFrame:
    sql = f"""
        SELECT trade_date, factor_name, ret
        FROM {SOURCE_DATABASE}.factor_return_{variant} FINAL
        WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND factor_name != ''
    """
    frame = query_df(sql).with_columns(pl.col("trade_date").cast(pl.Date))
    _validate_factor_return(frame, variant)
    return frame


def load_specific_return(variant: str) -> pl.DataFrame:
    sql = f"""
        SELECT trade_date, code, u
        FROM {SOURCE_DATABASE}.specific_return_{variant} FINAL
        WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    frame = query_df(sql).with_columns(pl.col("trade_date").cast(pl.Date))
    if frame.is_empty():
        raise ValueError(f"CNE6{variant} specific_return 查询为空")
    duplicates = (
        frame.group_by(["trade_date", "code"]).len().filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        raise ValueError(f"CNE6{variant} specific_return 存在重复键")
    if not np.isfinite(frame["u"].to_numpy()).all():
        raise ValueError(f"CNE6{variant} specific_return 含 NaN/Inf")
    return frame


def _atomic_write_parquet(frame: pl.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    for variant, out_dir in OUT_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[CNE6{variant} 1/2] 拉取因子收益 "
            f"{SOURCE_DATABASE}.factor_return_{variant} ..."
        )
        factor_ret = load_factor_return(variant)
        print(
            f"      {factor_ret.height:,} 行  "
            f"因子数={factor_ret['factor_name'].n_unique()}  "
            f"日期 {factor_ret['trade_date'].min()}~{factor_ret['trade_date'].max()}"
        )
        _atomic_write_parquet(factor_ret, out_dir / "factor_return.parquet")

        print(
            f"[CNE6{variant} 2/2] 拉取特质收益 "
            f"{SOURCE_DATABASE}.specific_return_{variant} ..."
        )
        spec_ret = load_specific_return(variant)
        print(
            f"      {spec_ret.height:,} 行  股票数={spec_ret['code'].n_unique()}  "
            f"日期 {spec_ret['trade_date'].min()}~{spec_ret['trade_date'].max()}"
        )
        _atomic_write_parquet(spec_ret, out_dir / "specific_return.parquet")

        print(f"      完成 -> {out_dir}/{{factor_return,specific_return}}.parquet")


if __name__ == "__main__":
    main()
