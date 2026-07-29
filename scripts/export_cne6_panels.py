"""导出 CNE6 风险模型面板：从 ClickHouse test_barra_cne6_gao 拉取因子暴露 /
因子协方差 / 特质风险，转换为 hqopt.risk.cne6_risk.CNE6RiskModel
消费的格式。

输出：
    data/barra_cne6_S/{exposure_panel,factor_cov_panel}.parquet    —— CNE6S（短周期 hl=63）
    data/barra_cne6_L/{exposure_panel,factor_cov_panel}.parquet  —— CNE6L（长周期 hl=252）

因子集合：
    CNE6S：51 = 20 风格 + Country + 30 行业
    CNE6L：47 = 16 风格 + Country + 30 行业
与 test_barra_cne6_gao schema 一致；与之配套的 STYLE_FACTORS 定义见
hqopt/risk/cne6_risk.py。

exposure 取自 test_barra_cne6_gao.factor_exposure 的 zscore，按 univ_flag==1 过滤（估计域：
当日可交易 + 上市满期）；Country 因子全市场暴露恒为 1；行业暴露由本地
data/cache/ashare_daily_<year>.parquet 的 industry_l1（中信一级）做 one-hot
（空值/""/"未知" 不计入 30 个行业因子，对应股票当日行业暴露为全 0）。
协方差表中的空行业因子会同时从 factor_i 行和 factor_j 列过滤，保证 S/L
分别严格输出 51×51 / 47×47 矩阵。

数据源：ClickHouse test_barra_cne6_gao（环境变量 CLICKHOUSE_WIND_PASSWORD
必填，单一凭证覆盖全部库，见 hqopt/data/clickhouse_db.py）。

运行：CLICKHOUSE_WIND_PASSWORD=... python scripts/export_cne6_panels.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from hqopt.data.clickhouse_db import query_df
from hqopt.risk.cne6_risk import STYLE_FACTORS_L, STYLE_FACTORS_S

CACHE_DIR = Path("data/cache")
OUT_DIRS = {"S": Path("data/barra_cne6_S"), "L": Path("data/barra_cne6_L")}
SOURCE_DATABASE = "test_barra_cne6_gao"

# 提取时间范围（含端点）；按需调整
# 2026-07-29 扩至全历史：源库 factor_exposure 自 2013-12-31、factor_return 自 2014-01-02，
# 本地行业缓存 data/cache/ashare_daily_*.parquet 覆盖 2012~2026，故 2014-01-01 起点可用。
START_DATE = "2014-01-01"
END_DATE = "2026-06-30"

# CNE6L 16 个核心风格；CNE6S 在此基础上增加 4 个快策略风格。
STYLE_FACTORS_BY_VARIANT = {"S": STYLE_FACTORS_S, "L": STYLE_FACTORS_L}
EXCLUDED_INDUSTRIES = {"", "未知"}


def validate_exposure_source() -> None:
    """确认估计域内每个股票日恰好具有 20 个有限风格暴露。"""
    factor_names = ", ".join(f"'{factor}'" for factor in STYLE_FACTORS_S)
    stats = query_df(
        f"""
        SELECT
            count() AS groups,
            min(factor_count) AS min_factor_count,
            max(factor_count) AS max_factor_count,
            min(row_count) AS min_row_count,
            max(row_count) AS max_row_count,
            countIf(
                factor_count != {len(STYLE_FACTORS_S)}
                OR row_count != {len(STYLE_FACTORS_S)}
            ) AS bad_groups,
            sum(invalid_values) AS invalid_values
        FROM (
            SELECT
                asof_date,
                code,
                uniqExact(factor_name) AS factor_count,
                count() AS row_count,
                countIf(NOT isFinite(zscore)) AS invalid_values
            FROM {SOURCE_DATABASE}.factor_exposure FINAL
            WHERE univ_flag = 1
              AND asof_date BETWEEN '{START_DATE}' AND '{END_DATE}'
              AND factor_name IN ({factor_names})
            GROUP BY asof_date, code
        )
        """
    )
    if stats.height != 1:
        raise ValueError("CNE6 factor_exposure 覆盖统计返回异常")
    row = stats.row(0, named=True)
    expected = len(STYLE_FACTORS_S)
    if (
        int(row["groups"] or 0) == 0
        or int(row["min_factor_count"] or 0) != expected
        or int(row["max_factor_count"] or 0) != expected
        or int(row["min_row_count"] or 0) != expected
        or int(row["max_row_count"] or 0) != expected
        or int(row["bad_groups"] or 0) != 0
        or int(row["invalid_values"] or 0) != 0
    ):
        raise ValueError(
            "CNE6 factor_exposure 内容合同失败："
            f"groups={row['groups']} "
            f"factor_count={row['min_factor_count']}~{row['max_factor_count']} "
            f"row_count={row['min_row_count']}~{row['max_row_count']} "
            f"bad_groups={row['bad_groups']} invalid_values={row['invalid_values']}"
        )


def _quarter_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        month_index = current.year * 12 + current.month - 1 + 3
        next_start = date(month_index // 12, month_index % 12 + 1, 1)
        chunk_end = min(end, next_start - timedelta(days=1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def load_exposure_base() -> pl.DataFrame:
    # 暴露结果约 7 百万行；按季度读取避免单个 GB 级 HTTP 响应中断。
    # 各块仍使用同一 FINAL 语义，合并后再统一构建 S/L 面板。
    pivot_cols = ",\n        ".join(
        f"sumIf(zscore, factor_name = '{f}') AS {f}" for f in STYLE_FACTORS_S
    )
    factor_names = ", ".join(f"'{factor}'" for factor in STYLE_FACTORS_S)
    frames = []
    for chunk_start, chunk_end in _quarter_ranges(
        date.fromisoformat(START_DATE),
        date.fromisoformat(END_DATE),
    ):
        sql = f"""
            SELECT asof_date, code,
            {pivot_cols}
            FROM {SOURCE_DATABASE}.factor_exposure FINAL
            WHERE univ_flag = 1
              AND asof_date BETWEEN '{chunk_start}' AND '{chunk_end}'
              AND factor_name IN ({factor_names})
            GROUP BY asof_date, code
        """
        frame = query_df(sql)
        if frame.is_empty():
            raise ValueError(
                f"CNE6 factor_exposure 查询为空：{chunk_start}~{chunk_end}"
            )
        print(f"      暴露分块 {chunk_start}~{chunk_end}: {frame.height:,} 行")
        frames.append(frame)
    df = pl.concat(frames, how="vertical")
    df = df.rename({"asof_date": "rebal_date"})
    output = df.with_columns(
        pl.col("rebal_date").cast(pl.Date),
        pl.lit(1.0).alias("Country"),
    )
    if not np.isfinite(output.select(STYLE_FACTORS_S).to_numpy()).all():
        raise ValueError("CNE6 factor_exposure 导出结果含 NaN/Inf")
    return output


def load_industry_dummies() -> tuple[pl.DataFrame, list[str]]:
    files = sorted(CACHE_DIR.glob("ashare_daily_*.parquet"))
    df = pl.concat(
        pl.read_parquet(f, columns=["code", "date", "industry_l1"]) for f in files
    ).unique(subset=["date", "code"]).with_columns(
        pl.col("industry_l1").cast(pl.String).str.strip_chars()
    )
    df = df.rename({"date": "rebal_date"}).with_columns(pl.col("rebal_date").cast(pl.Date))

    industries = sorted({
        c.strip()
        for c in df["industry_l1"].drop_nulls().unique().to_list()
        if c.strip() not in EXCLUDED_INDUSTRIES
    })
    dummy_cols = [(pl.col("industry_l1") == ind).cast(pl.Float64).alias(ind) for ind in industries]
    return df.select(["rebal_date", "code", *dummy_cols]), industries


def load_factor_cov(variant: str, factor_order: list[str]) -> pl.DataFrame:
    factor_names = ", ".join(f"'{factor}'" for factor in factor_order)
    cov = query_df(
        f"SELECT trade_date, factor_i, factor_j, cov "
        f"FROM {SOURCE_DATABASE}.factor_cov_{variant} FINAL "
        f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}' "
        f"AND factor_i IN ({factor_names}) AND factor_j IN ({factor_names})"
    ).filter(
        pl.col("factor_i").is_in(factor_order)
        & pl.col("factor_j").is_in(factor_order)
    )
    if cov.is_empty():
        raise ValueError(f"CNE6{variant} factor_cov 查询为空")
    duplicates = (
        cov.group_by(["trade_date", "factor_i", "factor_j"])
        .len()
        .filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        raise ValueError(f"CNE6{variant} factor_cov 存在重复键")
    expected_triangle = len(factor_order) * (len(factor_order) + 1) // 2
    bad_dates = (
        cov.group_by("trade_date")
        .agg(
            pl.len().alias("rows"),
            pl.col("factor_i").n_unique().alias("factor_i_count"),
            pl.col("factor_j").n_unique().alias("factor_j_count"),
        )
        .filter(
            (pl.col("rows") != expected_triangle)
            | (pl.col("factor_i_count") != len(factor_order))
            | (pl.col("factor_j_count") != len(factor_order))
        )
    )
    if not bad_dates.is_empty():
        raise ValueError(
            f"CNE6{variant} factor_cov 非完整三角矩阵，"
            f"示例={bad_dates.head(3).to_dicts()}"
        )
    if not np.isfinite(cov["cov"].to_numpy()).all():
        raise ValueError(f"CNE6{variant} factor_cov 含 NaN/Inf")

    swapped = cov.rename({"factor_i": "factor_j", "factor_j": "factor_i"}).select(cov.columns)
    sym = pl.concat([cov, swapped]).unique(subset=["trade_date", "factor_i", "factor_j"])
    wide = sym.pivot(index=["trade_date", "factor_i"], on="factor_j", values="cov")
    output = (
        wide.select(["trade_date", "factor_i", *factor_order])
        .rename({"trade_date": "rebal_date", "factor_i": "factor"})
        .with_columns(pl.col("rebal_date").cast(pl.Date))
    )
    if any(output.select(factor_order).null_count().row(0)):
        raise ValueError(f"CNE6{variant} factor_cov 对称化后仍有缺失值")
    return output


def load_spec_var(variant: str) -> pl.DataFrame:
    sr = query_df(
        f"SELECT trade_date, code, var FROM {SOURCE_DATABASE}.specific_risk_{variant} FINAL "
        f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'"
    )
    if sr.is_empty():
        raise ValueError(f"CNE6{variant} specific_risk 查询为空")
    if not np.isfinite(sr["var"].to_numpy()).all() or (sr["var"] <= 0).any():
        raise ValueError(f"CNE6{variant} specific_risk 含非正值或 NaN/Inf")
    duplicates = (
        sr.group_by(["trade_date", "code"]).len().filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        raise ValueError(f"CNE6{variant} specific_risk 存在重复键")
    return sr.rename({"trade_date": "rebal_date", "var": "spec_var"}).with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )


def _atomic_write_parquet(frame: pl.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    print("[0/3] 校验新库 factor_exposure 的 20 风格内容合同 ...")
    validate_exposure_source()
    print(
        "[1/3] 拉取因子暴露 "
        f"({SOURCE_DATABASE}, zscore, univ_flag==1, ClickHouse SQL 端 pivot) ..."
    )
    exposure_all = load_exposure_base()
    print(
        f"      {exposure_all.height:,} 行  "
        f"日期 {exposure_all['rebal_date'].min()}~{exposure_all['rebal_date'].max()}"
    )

    print("[2/3] 读取行业 one-hot (industry_l1, 本地 data/cache) ...")
    industry, industry_names = load_industry_dummies()
    if len(industry_names) != 30:
        raise ValueError(
            f"CITIC L1 行业应为 30 个，实际为 {len(industry_names)} 个: {industry_names}"
        )
    print(f"      {len(industry_names)} 个行业")

    for variant, out_dir in OUT_DIRS.items():
        style_factors = STYLE_FACTORS_BY_VARIANT[variant]
        factor_order = [*style_factors, "Country", *industry_names]
        exposure_base = (
            exposure_all.select(["rebal_date", "code", *style_factors, "Country"])
            .join(industry, on=["rebal_date", "code"], how="left")
            .with_columns([
                pl.col(c).fill_null(0.0)
                for c in [*style_factors, *industry_names]
            ])
        )
        print(f"\n[3/3] 构建 CNE6{variant} 面板 -> {out_dir} ...")

        spec = load_spec_var(variant)
        exposure = exposure_base.join(spec, on=["rebal_date", "code"], how="left")
        spec_median = exposure.group_by("rebal_date").agg(pl.col("spec_var").median().alias("_med"))
        exposure = (
            exposure.join(spec_median, on="rebal_date")
            .with_columns(pl.col("spec_var").fill_null(pl.col("_med")))
            .drop("_med")
            .select(["rebal_date", "code", *factor_order, "spec_var"])
        )

        cov = load_factor_cov(variant, factor_order)

        _atomic_write_parquet(exposure, out_dir / "exposure_panel.parquet")
        _atomic_write_parquet(cov, out_dir / "factor_cov_panel.parquet")
        print(
            f"      exposure: {exposure.height:,} 行   "
            f"cov: {cov.height:,} 行  日期 {cov['rebal_date'].min()}~{cov['rebal_date'].max()}"
        )

    print("\n完成。")


if __name__ == "__main__":
    main()
