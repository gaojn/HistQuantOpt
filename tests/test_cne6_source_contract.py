"""test_barra_cne6_gao 的 S/L 因子合同与空行业过滤测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

import scripts.export_cne6_panels as exporter
from hqopt.risk.cne6_risk import (
    STYLE_FACTORS,
    STYLE_FACTORS_L,
    STYLE_FACTORS_S,
)
from scripts.verify_data_bundle import load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_STYLE_FACTORS = {
    "AnalystSentiment",
    "IndustryMomentum",
    "Seasonality",
    "ShortTermReversal",
}


def test_s_and_l_style_contracts_stay_aligned() -> None:
    assert len(STYLE_FACTORS_L) == 16
    assert len(STYLE_FACTORS_S) == 20
    assert set(STYLE_FACTORS_S) - set(STYLE_FACTORS_L) == FAST_STYLE_FACTORS
    assert STYLE_FACTORS == STYLE_FACTORS_S
    assert exporter.STYLE_FACTORS_L == STYLE_FACTORS_L
    assert exporter.STYLE_FACTORS_S == STYLE_FACTORS_S


def test_data_manifest_requires_51_s_factors_and_47_l_factors() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data_manifest.yaml")
    column_sets = manifest["column_sets"]
    styles_l = set(column_sets["cne6_styles_l"])
    styles_s = styles_l | set(column_sets["cne6_styles_s_extra"])
    non_style = set(column_sets["cne6_non_style_factors"])

    assert styles_l == set(STYLE_FACTORS_L)
    assert styles_s == set(STYLE_FACTORS_S)
    assert len(non_style) == 31  # Country + 30 CITIC L1 industries
    assert "" not in non_style
    assert len(styles_s | non_style) == 51
    assert len(styles_l | non_style) == 47
    assert manifest["profiles"]["attribution"]["assets"][1:3] == [
        "cne6s_exposure",
        "cne6s_covariance",
    ]
    assert manifest["profiles"]["attribution_long"]["assets"][1:3] == [
        "cne6l_exposure",
        "cne6l_covariance",
    ]
    assert {
        "factor_return_l",
        "specific_return_l",
    } <= set(manifest["profiles"]["attribution_long"]["assets"])


def test_industry_dummies_drop_blank_unknown_and_null(tmp_path, monkeypatch) -> None:
    pl.DataFrame(
        {
            "code": ["A", "B", "C", "D", "E"],
            "date": [date(2024, 1, 2)] * 5,
            "industry_l1": ["银行", "", "未知", None, " 电子 "],
        }
    ).write_parquet(tmp_path / "ashare_daily_2024.parquet")
    monkeypatch.setattr(exporter, "CACHE_DIR", tmp_path)

    dummies, industries = exporter.load_industry_dummies()

    assert industries == ["电子", "银行"]
    assert dummies.columns == ["rebal_date", "code", "电子", "银行"]
    assert "" not in dummies.columns
    assert "未知" not in dummies.columns


def test_covariance_filters_empty_industry_on_both_axes(monkeypatch) -> None:
    captured_sql: list[str] = []

    def fake_query(sql: str) -> pl.DataFrame:
        captured_sql.append(sql)
        return pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2)] * 6,
                "factor_i": ["Size", "Size", "Size", "Country", "Country", ""],
                "factor_j": ["Size", "Country", "", "Country", "", ""],
                "cov": [1.0, 0.2, 9.0, 2.0, 8.0, 7.0],
            }
        )

    monkeypatch.setattr(exporter, "query_df", fake_query)

    panel = exporter.load_factor_cov("L", ["Size", "Country"]).sort("factor")

    assert exporter.SOURCE_DATABASE in captured_sql[0]
    assert "factor_i IN" in captured_sql[0]
    assert "factor_j IN" in captured_sql[0]
    assert panel.columns == ["rebal_date", "factor", "Size", "Country"]
    assert panel["factor"].to_list() == ["Country", "Size"]
    assert panel.select(["Size", "Country"]).to_numpy().tolist() == [
        [0.2, 2.0],
        [1.0, 0.2],
    ]
    assert "" not in panel["factor"].to_list()
