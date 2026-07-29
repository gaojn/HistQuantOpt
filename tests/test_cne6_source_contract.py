"""test_barra_cne6_gao 的 S/L 因子合同与空行业过滤测试。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

import scripts.export_cne6_panels as exporter
import scripts.export_factor_attribution as attribution_exporter
from hqopt.risk.cne6_risk import (
    STYLE_FACTORS,
    STYLE_FACTORS_L,
    STYLE_FACTORS_S,
    CNE6RiskModel,
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


def _write_risk_contract(data_dir: Path, factors: list[str]) -> None:
    data_dir.mkdir()
    rdate = date(2024, 1, 2)
    cov_data: dict[str, list[object]] = {
        "rebal_date": [rdate] * len(factors),
        "factor": factors,
    }
    for column_index, factor in enumerate(factors):
        cov_data[factor] = [
            1.0 if row_index == column_index else 0.0
            for row_index in range(len(factors))
        ]
    pl.DataFrame(cov_data).write_parquet(data_dir / "factor_cov_panel.parquet")

    exposure_data: dict[str, list[object]] = {
        "rebal_date": [rdate],
        "code": ["A"],
        "spec_var": [0.1],
    }
    exposure_data.update({factor: [0.0] for factor in factors})
    pl.DataFrame(exposure_data).write_parquet(data_dir / "exposure_panel.parquet")


def _contract_factors(styles: tuple[str, ...]) -> list[str]:
    return [*styles, "Country", *(f"Industry{i:02d}" for i in range(30))]


def test_default_s_directory_rejects_l_factor_contract(tmp_path) -> None:
    data_dir = tmp_path / "barra_cne6_S"
    _write_risk_contract(data_dir, _contract_factors(STYLE_FACTORS_L))

    with pytest.raises(ValueError, match="CNE6S 因子合同失败"):
        CNE6RiskModel(data_dir=data_dir, query_dates=[])


@pytest.mark.parametrize(
    ("directory", "styles", "variant"),
    [
        ("barra_cne6_S", STYLE_FACTORS_S, "S"),
        ("barra_cne6_L", STYLE_FACTORS_L, "L"),
        # 改名前的 S 目录旧名，仍须被识别为 S 并执行合同校验（不得静默跳过）
        ("barra_cne6", STYLE_FACTORS_S, "S"),
    ],
)
def test_s_and_l_directories_accept_exact_contracts(
    tmp_path,
    directory,
    styles,
    variant,
) -> None:
    data_dir = tmp_path / directory
    _write_risk_contract(data_dir, _contract_factors(styles))

    model = CNE6RiskModel(data_dir=data_dir, query_dates=[])

    assert model.variant == variant


def test_exposure_source_validation_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        exporter,
        "query_df",
        lambda _sql: pl.DataFrame(
            {
                "groups": [10],
                "min_factor_count": [19],
                "max_factor_count": [20],
                "min_row_count": [19],
                "max_row_count": [21],
                "bad_groups": [1],
                "invalid_values": [0],
            }
        ),
    )

    with pytest.raises(ValueError, match="factor_exposure 内容合同失败"):
        exporter.validate_exposure_source()


def test_covariance_rejects_incomplete_triangle(monkeypatch) -> None:
    monkeypatch.setattr(
        exporter,
        "query_df",
        lambda _sql: pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2)] * 2,
                "factor_i": ["Size", "Size"],
                "factor_j": ["Size", "Country"],
                "cov": [1.0, 0.2],
            }
        ),
    )

    with pytest.raises(ValueError, match="非完整三角矩阵"):
        exporter.load_factor_cov("L", ["Size", "Country"])


def _factor_return_frame(factors: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)] * len(factors),
            "factor_name": factors,
            "ret": [0.0] * len(factors),
        }
    )


@pytest.mark.parametrize(
    ("variant", "styles"),
    [("S", STYLE_FACTORS_S), ("L", STYLE_FACTORS_L)],
)
def test_factor_return_accepts_exact_s_and_l_contracts(variant, styles) -> None:
    attribution_exporter._validate_factor_return(
        _factor_return_frame(_contract_factors(styles)),
        variant,
    )


def test_factor_return_rejects_l_factors_in_s_table() -> None:
    with pytest.raises(ValueError, match="CNE6S factor_return 因子合同失败"):
        attribution_exporter._validate_factor_return(
            _factor_return_frame(_contract_factors(STYLE_FACTORS_L)),
            "S",
        )


def test_factor_return_allows_industry_without_constituents_for_one_day() -> None:
    factors = _contract_factors(STYLE_FACTORS_S)
    full_day = _factor_return_frame(factors)
    missing_industry_day = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3)] * (len(factors) - 1),
            "factor_name": factors[:-1],
            "ret": [0.0] * (len(factors) - 1),
        }
    )

    attribution_exporter._validate_factor_return(
        pl.concat([full_day, missing_industry_day]),
        "S",
    )


def test_exposure_quarter_ranges_cover_interval_without_overlap() -> None:
    """季度粒度（months=3）保持原切分语义。"""
    assert exporter._date_ranges(
        date(2024, 2, 15),
        date(2024, 8, 2),
        months=3,
    ) == [
        (date(2024, 2, 15), date(2024, 4, 30)),
        (date(2024, 5, 1), date(2024, 7, 31)),
        (date(2024, 8, 1), date(2024, 8, 2)),
    ]


def test_exposure_default_ranges_are_monthly() -> None:
    """默认按月切分：季度块在 2021 年后会触发服务端 MEMORY_LIMIT_EXCEEDED。"""
    assert exporter.CHUNK_MONTHS == 1
    assert exporter._date_ranges(date(2024, 2, 15), date(2024, 4, 10)) == [
        (date(2024, 2, 15), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 10)),
    ]


@pytest.mark.parametrize("months", [1, 3])
def test_exposure_ranges_tile_interval_exactly(months: int) -> None:
    """任意粒度下分块都必须无缺口、无重叠地恰好铺满区间。"""
    start, end = date(2014, 1, 1), date(2026, 6, 30)
    ranges = exporter._date_ranges(start, end, months=months)
    assert ranges[0][0] == start
    assert ranges[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        assert next_start == prev_end + timedelta(days=1)
    assert all(s <= e for s, e in ranges)
