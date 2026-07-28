"""官方指数每日权重导出与运行时必须使用同一口径。"""

from datetime import date

import pandas as pd
import polars as pl
import pytest

from hqopt.data.benchmark import IndexBenchmarkWeights
from scripts import export_index_weight as exporter


def _official(path):
    rows = []
    for index in exporter.IDX2KEY.values():
        for d, wa, wb in [
            (date(2024, 6, 28), 0.6, 0.4),
            (date(2024, 7, 31), 0.5, 0.5),
        ]:
            rows.extend([
                {"index": index, "date": d, "code": "A", "weight": wa},
                {"index": index, "date": d, "code": "B", "weight": wb},
            ])
    pl.DataFrame(rows).write_parquet(path)


def _panel():
    rows = []
    for d, a_price in [
        (date(2024, 6, 28), 10.0),
        (date(2024, 7, 15), 20.0),
        (date(2024, 7, 31), 11.0),
    ]:
        for code, price in [("A", a_price), ("B", 10.0)]:
            rows.append({
                "date": d,
                "code": code,
                "adj_close": price,
                "free_mv": 80.0,
                "total_mv": 100.0,
                "is_hs300": 1,
                "is_zz500": 1,
                "is_zz1000": 1,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_daily_export_matches_runtime_exactly(tmp_path, monkeypatch):
    official = tmp_path / "official.parquet"
    _official(official)
    panel = _panel()
    monkeypatch.setattr(exporter, "load_panel", lambda *args, **kwargs: panel)

    daily = exporter.build_daily_weights(
        official, date(2024, 6, 28), date(2024, 7, 31)
    )
    exported = (
        daily.filter(
            (pl.col("index") == "zz1000")
            & (pl.col("date") == date(2024, 7, 15))
        )
        .select(["code", "weight"])
        .to_pandas()
        .set_index("code")["weight"]
        .sort_index()
    )

    runtime = IndexBenchmarkWeights(
        index="zz1000",
        panel=panel,
        source="official_drift",
        official_path=official,
    )
    runtime.precompute(date(2024, 6, 28), date(2024, 7, 31), panel=panel)
    expected = runtime.get_weights(date(2024, 7, 15)).sort_index()

    pd.testing.assert_series_equal(
        exported, expected, check_names=False, rtol=1e-12, atol=1e-12
    )
    assert exported.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    row = daily.filter(
        (pl.col("index") == "zz1000")
        & (pl.col("date") == date(2024, 7, 15))
    ).row(0, named=True)
    assert row["anchor_date"] == date(2024, 6, 28)
    assert row["snapshot_age_days"] == 17
    assert row["effective_date"] == date(2024, 7, 31)
    assert row["method"] == "official_drift"
    assert row["fallback_reason"] is None

    final_row = daily.filter(
        (pl.col("index") == "zz1000")
        & (pl.col("date") == date(2024, 7, 31))
    ).row(0, named=True)
    assert final_row["effective_date"] is None


def test_daily_export_handles_fallback_for_only_one_index(tmp_path, monkeypatch):
    official = tmp_path / "official.parquet"
    _official(official)
    panel = _panel().with_columns(
        pl.when(
            (pl.col("date") == date(2024, 7, 15)) & (pl.col("code") == "B")
        )
        .then(0)
        .otherwise(pl.col("is_zz500"))
        .alias("is_zz500")
    )
    monkeypatch.setattr(exporter, "load_panel", lambda *args, **kwargs: panel)

    daily = exporter.build_daily_weights(
        official, date(2024, 6, 28), date(2024, 7, 31)
    )

    zz500 = daily.filter(
        (pl.col("index") == "zz500")
        & (pl.col("date") == date(2024, 7, 15))
    )
    assert zz500["method"].unique().to_list() == ["reconstruct"]
    assert zz500["fallback_reason"].unique().to_list() == ["roster_changed"]
    zz1000 = daily.filter(
        (pl.col("index") == "zz1000")
        & (pl.col("date") == date(2024, 7, 15))
    )
    assert zz1000["method"].unique().to_list() == ["official_drift"]
    assert zz1000["fallback_reason"].null_count() == zz1000.height
