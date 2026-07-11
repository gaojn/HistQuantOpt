"""导出因子收益 / 特质收益：从 ClickHouse cne6_risk.factor_return / specific_return
拉取，供 portfolio_optimizer.analysis.attribution 做收益归因。

输出：
    data/barra_cne6/factor_return.parquet     —— trade_date, factor_name, ret
    data/barra_cne6/specific_return.parquet   —— trade_date, code, u

与 data/barra_cne6/exposure_panel.parquet 同源（均来自 ClickHouse cne6_risk），
保证归因用的因子暴露 X 与因子/特质收益 f、u 出自同一套模型，残差自检才对得上。

注意：cne6_risk.factor_return / specific_return 不按 S/L 周期区分（周期只影响
协方差 F 与特质方差 δ 的估计窗口，不影响已实现的 f、u），因此只导出一份，
不像 exposure_panel 那样区分 barra_cne6 / barra_cne6_L 两个目录。

数据源：ClickHouse cne6_risk（环境变量 CLICKHOUSE_PASSWORD 必填，
见 portfolio_optimizer/data/clickhouse_db.py）。

运行：CLICKHOUSE_PASSWORD=... python scripts/export_factor_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from portfolio_optimizer.data.clickhouse_db import query_df

OUT_DIR = Path("data/barra_cne6")

# 与 export_cne6_panels.py 保持一致的提取范围；按需调整
START_DATE = "2020-01-01"
END_DATE = "2026-05-31"


def load_factor_return() -> pl.DataFrame:
    sql = f"""
        SELECT trade_date, factor_name, ret
        FROM cne6_risk.factor_return
        WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    return query_df(sql).with_columns(pl.col("trade_date").cast(pl.Date))


def load_specific_return() -> pl.DataFrame:
    sql = f"""
        SELECT trade_date, code, u
        FROM cne6_risk.specific_return
        WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    return query_df(sql).with_columns(pl.col("trade_date").cast(pl.Date))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] 拉取因子收益 cne6_risk.factor_return ...")
    factor_ret = load_factor_return()
    print(
        f"      {factor_ret.height:,} 行  因子数={factor_ret['factor_name'].n_unique()}  "
        f"日期 {factor_ret['trade_date'].min()}~{factor_ret['trade_date'].max()}"
    )
    factor_ret.write_parquet(OUT_DIR / "factor_return.parquet")

    print("[2/2] 拉取特质收益 cne6_risk.specific_return ...")
    spec_ret = load_specific_return()
    print(
        f"      {spec_ret.height:,} 行  股票数={spec_ret['code'].n_unique()}  "
        f"日期 {spec_ret['trade_date'].min()}~{spec_ret['trade_date'].max()}"
    )
    spec_ret.write_parquet(OUT_DIR / "specific_return.parquet")

    print(f"\n完成 -> {OUT_DIR}/{{factor_return,specific_return}}.parquet")


if __name__ == "__main__":
    main()
