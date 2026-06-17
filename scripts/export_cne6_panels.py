"""导出 CNE6 风险模型面板：从 ClickHouse the_quant.cne6_risk 拉取因子暴露 /
因子协方差 / 特质风险，转换为 portfolio_optimizer.risk.cne6_risk.CNE6RiskModel
消费的格式。

输出：
    data/barra_cne6/{exposure_panel,factor_cov_panel}.parquet    —— CNE6S（短周期 hl=63）
    data/barra_cne6_L/{exposure_panel,factor_cov_panel}.parquet  —— CNE6L（长周期 hl=252）

因子集合（47 = 16 风格 + Country + 30 行业），与 cne6_risk schema 一致；
与之配套的 STYLE_FACTORS 定义见 portfolio_optimizer/risk/cne6_risk.py。

exposure 取自 cne6_risk.factor_exposure 的 zscore，按 univ_flag==1 过滤（估计域：
当日可交易 + 上市满期）；Country 因子全市场暴露恒为 1；行业暴露由本地
data/cache/ashare_daily_<year>.parquet 的 industry_l1（中信一级）做 one-hot
（""/"未知" 不计入 30 个行业因子，对应股票当日行业暴露为全 0）。

数据源：ClickHouse the_quant.cne6_risk（环境变量 CLICKHOUSE_PASSWORD 必填，
见 portfolio_optimizer/data/clickhouse_db.py）。

运行：CLICKHOUSE_PASSWORD=... python scripts/export_cne6_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from portfolio_optimizer.data.clickhouse_db import query_df

CACHE_DIR = Path("data/cache")
OUT_DIRS = {"S": Path("data/barra_cne6"), "L": Path("data/barra_cne6_L")}

# 提取时间范围（含端点）；按需调整
START_DATE = "2020-01-01"
END_DATE = "2026-05-31"

# 16 个 CNE6 风格因子（命名与 cne6_risk.factor_exposure 一致）
STYLE_FACTORS: tuple[str, ...] = (
    "Size", "MidCap", "Beta", "Momentum", "ResidualVolatility", "LongTermReversal",
    "Liquidity", "Value", "EarningsYield", "Growth", "Profitability",
    "InvestmentQuality", "EarningsQuality", "EarningsVariability", "Leverage",
    "DividendYield",
)


def load_exposure_base() -> pl.DataFrame:
    # SQL 端按因子 pivot（16 个 sumIf），避免拉取长表全量行（16x 数据量）
    pivot_cols = ",\n        ".join(
        f"sumIf(zscore, factor_name = '{f}') AS {f}" for f in STYLE_FACTORS
    )
    sql = f"""
        SELECT asof_date, code,
        {pivot_cols}
        FROM cne6_risk.factor_exposure
        WHERE univ_flag = 1
          AND asof_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        GROUP BY asof_date, code
    """
    df = query_df(sql)
    df = df.rename({"asof_date": "rebal_date"})
    return df.with_columns(pl.col("rebal_date").cast(pl.Date), pl.lit(1.0).alias("Country"))


def load_industry_dummies() -> tuple[pl.DataFrame, list[str]]:
    files = sorted(CACHE_DIR.glob("ashare_daily_*.parquet"))
    df = pl.concat(
        pl.read_parquet(f, columns=["code", "date", "industry_l1"]) for f in files
    ).unique(subset=["date", "code"])
    df = df.rename({"date": "rebal_date"}).with_columns(pl.col("rebal_date").cast(pl.Date))

    industries = sorted(
        c for c in df["industry_l1"].unique().to_list() if c not in ("", "未知")
    )
    dummy_cols = [(pl.col("industry_l1") == ind).cast(pl.Float64).alias(ind) for ind in industries]
    return df.select(["rebal_date", "code", *dummy_cols]), industries


def load_factor_cov(variant: str) -> pl.DataFrame:
    cov = query_df(
        f"SELECT trade_date, factor_i, factor_j, cov FROM cne6_risk.factor_cov_{variant} "
        f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'"
    )
    swapped = cov.rename({"factor_i": "factor_j", "factor_j": "factor_i"}).select(cov.columns)
    sym = pl.concat([cov, swapped]).unique(subset=["trade_date", "factor_i", "factor_j"])
    wide = sym.pivot(index=["trade_date", "factor_i"], on="factor_j", values="cov")
    return wide.rename({"trade_date": "rebal_date", "factor_i": "factor"}).with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )


def load_spec_var(variant: str) -> pl.DataFrame:
    sr = query_df(
        f"SELECT trade_date, code, var FROM cne6_risk.specific_risk_{variant} "
        f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'"
    )
    return sr.rename({"trade_date": "rebal_date", "var": "spec_var"}).with_columns(
        pl.col("rebal_date").cast(pl.Date)
    )


def main() -> None:
    print("[1/3] 拉取因子暴露 (zscore, univ_flag==1, ClickHouse SQL 端 pivot) ...")
    exposure_base = load_exposure_base()
    print(f"      {exposure_base.height:,} 行  日期 {exposure_base['rebal_date'].min()}~{exposure_base['rebal_date'].max()}")

    print("[2/3] 读取行业 one-hot (industry_l1, 本地 data/cache) ...")
    industry, industry_names = load_industry_dummies()
    print(f"      {len(industry_names)} 个行业")

    factor_order = [*STYLE_FACTORS, "Country", *industry_names]

    exposure_base = exposure_base.join(industry, on=["rebal_date", "code"], how="left")
    fill_cols = [*STYLE_FACTORS, *industry_names]
    exposure_base = exposure_base.with_columns([pl.col(c).fill_null(0.0) for c in fill_cols])

    for variant, out_dir in OUT_DIRS.items():
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

        cov = load_factor_cov(variant).select(["rebal_date", "factor", *factor_order])

        out_dir.mkdir(parents=True, exist_ok=True)
        exposure.write_parquet(out_dir / "exposure_panel.parquet")
        cov.write_parquet(out_dir / "factor_cov_panel.parquet")
        print(
            f"      exposure: {exposure.height:,} 行   "
            f"cov: {cov.height:,} 行  日期 {cov['rebal_date'].min()}~{cov['rebal_date'].max()}"
        )

    print("\n完成。")


if __name__ == "__main__":
    main()
