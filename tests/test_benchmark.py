"""IndexBenchmarkWeights 官方权重源 + 市值重构兜底测试。"""
from datetime import date

import polars as pl
import pytest

from hqopt.data.benchmark import IndexBenchmarkWeights


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


# ─────────────────────────────────────────────────────────────────
# T4: _build_weight_matrix ffill 掩码修复
# ─────────────────────────────────────────────────────────────────

def _make_panel_3stocks():
    """
    构造 3 支股票、5 个交易日的小面板：
    - A, B: 全程在成分内
    - C: 前 3 天在成分，之后掉出（is_zz500=0）
    第 2 天 B 停牌（free_mv/total_mv=NaN）用于验证停牌日 ffill 正常工作
    """
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
             date(2024, 1, 5), date(2024, 1, 8)]
    rows = []
    for d in dates:
        for code, (in5, total_mv, free_mv) in {
            "A": (1, 100.0, 80.0),
            "B": (1, 200.0, 160.0),
            "C": (1 if d <= date(2024, 1, 4) else 0, 50.0, 40.0),  # C 第4天后掉出
        }.items():
            # B 在第2天停牌（市值 NaN）
            if code == "B" and d == date(2024, 1, 3):
                total_mv_r = None
                free_mv_r = None
            else:
                total_mv_r = total_mv
                free_mv_r = free_mv
            rows.append({
                "code": code, "date": d,
                "total_mv": total_mv_r, "free_mv": free_mv_r,
                "is_zz500": in5,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_ffill_respects_roster_boundary():
    """掉出指数后权重归零；停牌日权重仍在（ffill 正常）。"""
    panel = _make_panel_3stocks()
    bm = IndexBenchmarkWeights(index="zz500", panel=panel, source="reconstruct")

    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
             date(2024, 1, 5), date(2024, 1, 8)]
    bm.precompute(dates[0], dates[-1])

    # ① 掉出指数后 C 权重应为 0
    w_after = bm.get_weights(date(2024, 1, 5), tickers=["A", "B", "C"])
    assert w_after["C"] == pytest.approx(0.0, abs=1e-8), (
        f"C 掉出指数后权重应为 0，实为 {w_after['C']:.6f}"
    )
    assert w_after["A"] + w_after["B"] == pytest.approx(1.0, abs=1e-6)

    # ② 停牌日（B 第2天无市值）权重应仍存在（ffill 使用前一天市值）
    w_susp = bm.get_weights(date(2024, 1, 3), tickers=["A", "B", "C"])
    assert w_susp["B"] > 0.0, f"停牌日 B 权重应通过 ffill 保持，实为 {w_susp['B']:.6f}"
    assert w_susp["A"] + w_susp["B"] + w_susp["C"] == pytest.approx(1.0, abs=1e-6)
