"""行情面板加载（列解析 / 年份切片 / adj_vwap 派生）与 schema 契约测试。

`load_panel` 是全部策略流水线的数据入口：列名别名、年份缓存缺失、adj_vwap
派生一旦出错，后面所有优化与回测都建立在错误的行情上。
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from hqopt.io import schema
from hqopt.io.data_panel import (
    DEFAULT_PANEL_COLUMNS,
    _cache_files,
    _physical_columns,
    _resolve_columns,
    _years_in_range,
    load_panel,
)


def _write_cache(tmp_path, year: int, *, vwap_column: str = "vwap", with_adj_vwap: bool = False):
    """写一个最简年度缓存 parquet。"""
    days = [date(year, 1, 2), date(year, 1, 3)]
    rows = []
    for d in days:
        for code in ("000001.SZ", "600000.SH"):
            row = {
                "code": code,
                "date": d,
                "adj_close": 10.0,
                "close": 9.5,
                "adj_factor": 2.0,
                vwap_column: 5.0,
                "amount": 1000.0,
                "trade_status": "交易",
            }
            if with_adj_vwap:
                row["adj_vwap"] = 99.0
            rows.append(row)
    path = tmp_path / f"ashare_daily_{year}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


# ── 年份 / 文件解析 ──────────────────────────────────────────────


def test_years_in_range_is_inclusive():
    assert _years_in_range(date(2022, 12, 31), date(2024, 1, 1)) == [2022, 2023, 2024]


def test_years_in_range_rejects_inverted_range():
    with pytest.raises(ValueError, match="不能晚于"):
        _years_in_range(date(2024, 1, 2), date(2024, 1, 1))


def test_missing_cache_year_raises_with_actionable_message(tmp_path):
    _write_cache(tmp_path, 2024)

    with pytest.raises(FileNotFoundError, match="2025"):
        _cache_files([2024, 2025], tmp_path)


# ── 列解析 ───────────────────────────────────────────────────────


def test_resolve_columns_always_includes_code_and_date():
    resolved = _resolve_columns(["adj_close"], add_adj_vwap=False)
    assert resolved[:2] == ["code", "date"]


def test_resolve_columns_defaults_to_full_panel():
    assert _resolve_columns(None, add_adj_vwap=False) == list(DEFAULT_PANEL_COLUMNS)


def test_resolve_columns_pulls_in_adj_vwap_dependencies():
    resolved = _resolve_columns(["adj_close"], add_adj_vwap=True)
    assert {"adj_vwap", "vwap", "adj_factor"} <= set(resolved)


def test_physical_columns_maps_legacy_avg_price_alias():
    select, rename = _physical_columns(["avg_price"], {"vwap", "code"})
    assert select == ["vwap"]
    assert rename == {"vwap": "avg_price"}


def test_physical_columns_skips_derivable_adj_vwap():
    """缓存没有 adj_vwap 时不报错——它由 vwap × adj_factor 派生。"""
    select, _ = _physical_columns(["adj_vwap"], {"vwap", "adj_factor"})
    assert select == []


def test_physical_columns_rejects_truly_missing_column():
    with pytest.raises(KeyError, match="not_a_column"):
        _physical_columns(["not_a_column"], {"code", "date"})


# ── load_panel ───────────────────────────────────────────────────


def test_load_panel_slices_dates_and_sorts(tmp_path):
    _write_cache(tmp_path, 2024)

    panel = load_panel(
        date(2024, 1, 3), date(2024, 1, 3),
        columns=["adj_close"], cache_dir=tmp_path, add_adj_vwap=False,
    )

    assert panel["date"].unique().to_list() == [date(2024, 1, 3)]
    assert panel["code"].to_list() == sorted(panel["code"].to_list())


def test_load_panel_spans_multiple_years(tmp_path):
    _write_cache(tmp_path, 2023)
    _write_cache(tmp_path, 2024)

    panel = load_panel(
        date(2023, 1, 1), date(2024, 12, 31),
        columns=["adj_close"], cache_dir=tmp_path, add_adj_vwap=False,
    )

    assert {d.year for d in panel["date"].to_list()} == {2023, 2024}


def test_load_panel_derives_adj_vwap_when_absent(tmp_path):
    _write_cache(tmp_path, 2024)

    panel = load_panel(
        date(2024, 1, 1), date(2024, 12, 31),
        columns=["adj_close"], cache_dir=tmp_path, add_adj_vwap=True,
    )

    assert "adj_vwap" in panel.columns
    assert panel["adj_vwap"].to_list() == [10.0] * panel.height   # vwap 5.0 × adj_factor 2.0


def test_load_panel_prefers_cached_adj_vwap_over_derivation(tmp_path):
    _write_cache(tmp_path, 2024, with_adj_vwap=True)

    panel = load_panel(
        date(2024, 1, 1), date(2024, 12, 31),
        columns=["adj_vwap"], cache_dir=tmp_path, add_adj_vwap=True,
    )

    assert panel["adj_vwap"].to_list() == [99.0] * panel.height


def test_load_panel_renames_legacy_avg_price_cache(tmp_path):
    _write_cache(tmp_path, 2024, vwap_column="vwap")

    panel = load_panel(
        date(2024, 1, 1), date(2024, 12, 31),
        columns=["avg_price"], cache_dir=tmp_path, add_adj_vwap=False,
    )

    assert "avg_price" in panel.columns
    assert panel["avg_price"].to_list() == [5.0] * panel.height


def test_load_panel_casts_date_column(tmp_path):
    _write_cache(tmp_path, 2024)

    panel = load_panel(
        date(2024, 1, 1), date(2024, 12, 31),
        columns=["adj_close"], cache_dir=tmp_path, add_adj_vwap=False,
    )

    assert panel.schema["date"] == pl.Date


# ── schema 契约 ──────────────────────────────────────────────────


def test_schema_standard_names_are_unique():
    names = [std for _, std, _ in schema.SCHEMA]
    assert len(names) == len(set(names))


def test_schema_wind_names_are_unique():
    wind = [w for w, _, _ in schema.SCHEMA]
    assert len(wind) == len(set(wind))


def test_every_output_column_is_documented():
    assert set(schema.OUTPUT_COLUMNS) <= set(schema.COLUMN_DOCS)


def test_output_columns_cover_schema_and_derived_columns():
    known = {std for _, std, _ in schema.SCHEMA} | {n for n, _ in schema.DERIVED_SCHEMA}
    # OUTPUT_COLUMNS 不得引用 schema 之外的列
    assert set(schema.OUTPUT_COLUMNS) <= known


def test_backtest_presets_only_reference_output_columns():
    for preset, cols in schema.BACKTEST_PRESETS.items():
        unknown = set(cols) - set(schema.OUTPUT_COLUMNS)
        assert not unknown, f"{preset} 引用了未定义列: {unknown}"


def test_backtest_column_groups_only_reference_output_columns():
    for group, cols in schema.BACKTEST_COLUMN_GROUPS.items():
        unknown = set(cols) - set(schema.OUTPUT_COLUMNS)
        assert not unknown, f"{group} 引用了未定义列: {unknown}"


def test_excluded_columns_stay_out_of_presets():
    for cols in schema.BACKTEST_PRESETS.values():
        assert not (set(cols) & schema.BACKTEST_EXCLUDE_COLUMNS)


def test_trade_status_docs_cover_all_a_share_states():
    """A 股 5 类状态齐全；XD/XR/N 可正常交易，只有停牌不可成交。"""
    assert {"交易", "停牌", "N", "XR", "XD"} <= set(schema.TRADE_STATUS_DOCS)


def test_default_panel_columns_are_known_schema_columns():
    known = {std for _, std, _ in schema.SCHEMA} | {n for n, _ in schema.DERIVED_SCHEMA}
    assert set(DEFAULT_PANEL_COLUMNS) <= known
