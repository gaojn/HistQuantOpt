"""IndexBenchmarkWeights 官方权重源 + 市值重构兜底测试。"""
from datetime import date

import polars as pl
import pytest

from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights


def _make_official(path, index="zz1000"):
    """两个月度快照：2024-06-28 (A:.6,B:.4)、2024-07-31 (A:.5,B:.5)。"""
    pl.DataFrame({
        "index": [index] * 4,
        "date": [date(2024, 6, 28), date(2024, 6, 28),
                 date(2024, 7, 31), date(2024, 7, 31)],
        "code": ["A", "B", "A", "B"],
        "weight": [0.6, 0.4, 0.5, 0.5],
    }).write_parquet(path)


def test_official_asof_picks_latest_snapshot(tmp_path):
    p = tmp_path / "w.parquet"
    _make_official(p)
    bm = IndexBenchmarkWeights(index="zz1000", source="official", official_path=p)

    # 7-15 → 取 ≤该日最近快照 6-28
    w = bm.get_weights(date(2024, 7, 15), tickers=["A", "B", "C"])
    assert w["A"] == pytest.approx(0.6)
    assert w["B"] == pytest.approx(0.4)
    assert w["C"] == 0.0          # 非成分股补 0

    # 8-01 → 取 7-31 快照
    w2 = bm.get_weights(date(2024, 8, 1), tickers=["A", "B"])
    assert w2["A"] == pytest.approx(0.5)


def test_official_renormalizes(tmp_path):
    p = tmp_path / "w.parquet"
    _make_official(p)
    bm = IndexBenchmarkWeights(index="zz1000", source="official", official_path=p)
    w = bm.get_weights(date(2024, 7, 15), tickers=["A", "B", "C"])
    assert w.sum() == pytest.approx(1.0)


def test_official_before_start_falls_back_to_reconstruct(tmp_path):
    """早于官方最早快照 → 回退 free_mv 重构；无 panel 时按重构路径报错。"""
    p = tmp_path / "w.parquet"
    _make_official(p)
    bm = IndexBenchmarkWeights(index="zz1000", source="official", official_path=p)
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 1, 1), tickers=["A", "B"])


def test_official_missing_index_falls_back(tmp_path):
    """官方文件无该指数 → 回退重构。"""
    p = tmp_path / "w.parquet"
    _make_official(p, index="zz1000")
    bm = IndexBenchmarkWeights(index="hs300", source="official", official_path=p)
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 7, 15), tickers=["A", "B"])


def test_missing_official_file_falls_back(tmp_path):
    bm = IndexBenchmarkWeights(
        index="zz1000", source="official", official_path=tmp_path / "nope.parquet"
    )
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 7, 15), tickers=["A", "B"])


def test_reconstruct_source_ignores_official(tmp_path):
    """source=reconstruct 不读官方，直接走重构路径。"""
    p = tmp_path / "w.parquet"
    _make_official(p)
    bm = IndexBenchmarkWeights(index="zz1000", source="reconstruct", official_path=p)
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 7, 15), tickers=["A", "B"])


def test_invalid_source():
    with pytest.raises(ValueError):
        IndexBenchmarkWeights(index="zz1000", source="bad")
