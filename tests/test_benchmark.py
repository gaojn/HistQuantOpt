"""IndexBenchmarkWeights 官方权重源 + 市值重构兜底测试。"""
from datetime import date

import pandas as pd
import polars as pl
import pytest

from hqopt.data.benchmark import (
    DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    IndexBenchmarkWeights,
    drift_official_weights,
)


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
    bm = IndexBenchmarkWeights(index="zz1000", source="official_frozen", official_path=p)

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
    bm = IndexBenchmarkWeights(index="zz1000", source="official_frozen", official_path=p)
    w = bm.get_weights(date(2024, 7, 15), tickers=["A", "B", "C"])
    assert w.sum() == pytest.approx(1.0)


def test_official_before_start_falls_back_to_reconstruct(tmp_path):
    """早于官方最早快照 → 回退 free_mv 重构；无 panel 时按重构路径报错。"""
    p = tmp_path / "w.parquet"
    _make_official(p)
    bm = IndexBenchmarkWeights(index="zz1000", source="official_frozen", official_path=p)
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 1, 1), tickers=["A", "B"])


def test_official_missing_index_falls_back(tmp_path):
    """官方文件无该指数 → 回退重构。"""
    p = tmp_path / "w.parquet"
    _make_official(p, index="zz1000")
    bm = IndexBenchmarkWeights(index="hs300", source="official_frozen", official_path=p)
    with pytest.raises(RuntimeError):
        bm.get_weights(date(2024, 7, 15), tickers=["A", "B"])


def test_missing_official_file_falls_back(tmp_path):
    bm = IndexBenchmarkWeights(
        index="zz1000", source="official_frozen", official_path=tmp_path / "nope.parquet"
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


@pytest.mark.parametrize("value", [-1, True, 1.5, "30"])
def test_invalid_snapshot_age_limit(value):
    with pytest.raises(ValueError, match="max_snapshot_age_days"):
        IndexBenchmarkWeights(
            index="zz1000",
            source="official_drift",
            max_snapshot_age_days=value,
        )


# ─────────────────────────────────────────────────────────────────
# T4: _build_weight_matrix ffill 掩码修复
# ─────────────────────────────────────────────────────────────────

def _make_panel_3stocks():
    """
    构造 3 支股票、5 个交易日的小面板：
    - A, B: 全程在成分内
    - C: 前 3 天在成分，第 4 天掉出，第 5 天重新进入
    第 2 天 B 停牌（free_mv/total_mv=NaN）用于验证停牌日 ffill 正常工作
    """
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
             date(2024, 1, 5), date(2024, 1, 8)]
    rows = []
    for d in dates:
        for code, (in5, total_mv, free_mv) in {
            "A": (1, 100.0, 80.0),
            "B": (1, 200.0, 160.0),
            "C": (
                1 if d <= date(2024, 1, 4) or d == date(2024, 1, 8) else 0,
                50.0,
                40.0,
            ),
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
    """退出区间权重归零、重新进入后恢复；停牌日在册时仍可 ffill。"""
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

    # ② C 重新进入后恢复权重，证明中间退出区间不会被跨段 ffill。
    w_reentry = bm.get_weights(date(2024, 1, 8), tickers=["A", "B", "C"])
    assert w_reentry["C"] > 0.0

    # ③ 停牌日（B 第2天无市值）权重应仍存在（ffill 使用前一天市值）
    w_susp = bm.get_weights(date(2024, 1, 3), tickers=["A", "B", "C"])
    assert w_susp["B"] > 0.0, f"停牌日 B 权重应通过 ffill 保持，实为 {w_susp['B']:.6f}"
    assert w_susp["A"] + w_susp["B"] + w_susp["C"] == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────
# 官方快照 PIT 漂移
# ─────────────────────────────────────────────────────────────────


def _make_drift_panel(*, suspended: bool = False, roster_change: bool = False):
    dates = [
        date(2024, 6, 28),
        date(2024, 7, 12),
        date(2024, 7, 15),
        date(2024, 7, 31),
        date(2024, 8, 1),
    ]
    rows = []
    for d in dates:
        for code in ("A", "B", "C"):
            in_index = code in ("A", "B")
            if roster_change and d == date(2024, 7, 15):
                in_index = code in ("A", "C")
            price = 10.0
            if code == "A":
                price = {
                    date(2024, 6, 28): 10.0,
                    date(2024, 7, 12): 15.0,
                    date(2024, 7, 15): 20.0,
                    date(2024, 7, 31): 11.0,
                    date(2024, 8, 1): 12.0,
                }[d]
                if suspended and d == date(2024, 7, 15):
                    price = None
            rows.append({
                "date": d,
                "code": code,
                "adj_close": price,
                "total_mv": 100.0,
                "free_mv": 80.0,
                "is_zz1000": int(in_index),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_drift_official_weights_hand_calculation():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    anchor = pd.Series({"A": 10.0, "B": 10.0})
    target = pd.Series({"A": 20.0, "B": 10.0})

    got = drift_official_weights(weights, anchor, target)

    assert got["A"] == pytest.approx(0.75)
    assert got["B"] == pytest.approx(0.25)
    assert got.sum() == pytest.approx(1.0)


def test_official_drift_is_pit_and_exact_snapshot_wins(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel()
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 8, 1), panel=panel)

    # 7-15 只能使用 6-28 快照；7-31 的未来快照不得参与插值。
    drifted = benchmark.get_weights(date(2024, 7, 15), ["A", "B"])
    assert drifted["A"] == pytest.approx(0.75)
    assert drifted["B"] == pytest.approx(0.25)

    # 快照日退化为官方原值，不再叠加价格漂移。
    snapshot = benchmark.get_weights(date(2024, 7, 31), ["A", "B"])
    assert snapshot.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    audit = benchmark.audit_summary()
    assert audit["snapshot_as_of_by_period"]["2024-07-15"] == "2024-06-28"
    assert audit["snapshot_age_days_by_period"]["2024-07-15"] == 17
    assert audit["effective_date_by_period"]["2024-07-15"] == "2024-07-31"
    assert audit["method_by_period"]["2024-07-15"] == "official_drift"


def test_official_drift_defaults_to_30_calendar_days_and_falls_back_on_day_31(
    tmp_path,
):
    path = tmp_path / "official.parquet"
    pl.DataFrame(
        {
            "index": ["zz1000", "zz1000"],
            "date": [date(2024, 6, 1), date(2024, 6, 1)],
            "code": ["A", "B"],
            "weight": [0.6, 0.4],
        }
    ).write_parquet(path)
    rows = []
    for d in (date(2024, 6, 1), date(2024, 7, 1), date(2024, 7, 2)):
        for code in ("A", "B"):
            rows.append(
                {
                    "date": d,
                    "code": code,
                    "adj_close": 20.0 if code == "A" and d > date(2024, 6, 1) else 10.0,
                    "total_mv": 100.0,
                    "free_mv": 80.0,
                    "is_zz1000": 1,
                }
            )
    panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    benchmark = IndexBenchmarkWeights(
        index="zz1000",
        panel=panel,
        source="official_drift",
        official_path=path,
    )
    benchmark.precompute(date(2024, 6, 1), date(2024, 7, 2), panel=panel)

    day_30 = benchmark.get_weights(date(2024, 7, 1), ["A", "B"])
    day_31 = benchmark.get_weights(date(2024, 7, 2), ["A", "B"])

    assert benchmark.max_snapshot_age_days == DEFAULT_MAX_SNAPSHOT_AGE_DAYS == 30
    assert day_30.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    assert day_31.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    audit = benchmark.audit_summary()
    assert audit["stale_snapshot_period_count"] == 1
    assert audit["snapshot_age_days_by_period"]["2024-07-02"] == 31
    assert audit["fallback_reason_by_period"]["2024-07-02"] == (
        "snapshot_stale: age_days=31, limit_days=30"
    )


def test_snapshot_age_limit_can_be_explicitly_disabled(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel()
    benchmark = IndexBenchmarkWeights(
        index="zz1000",
        panel=panel,
        source="official_drift",
        official_path=path,
        max_snapshot_age_days=None,
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 8, 1), panel=panel)

    got = benchmark.get_weights(date(2024, 7, 31), ["A", "B"])

    assert got.to_dict() == pytest.approx({"A": 0.5, "B": 0.5})
    assert benchmark.audit_summary()["max_snapshot_age_days"] is None


def test_suspended_stock_uses_last_valid_price(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel(suspended=True)
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 7, 15), panel=panel)

    got = benchmark.get_weights(date(2024, 7, 15), ["A", "B"])

    # A 用 7-12 最近有效价 15；0.6*1.5 / (0.6*1.5+0.4) = 9/13。
    assert got["A"] == pytest.approx(9 / 13)
    assert got["B"] == pytest.approx(4 / 13)


def test_roster_change_warns_records_and_falls_back(tmp_path, caplog):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel(roster_change=True)
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 7, 15), panel=panel)

    got = benchmark.get_weights(date(2024, 7, 15), ["A", "B", "C"])

    assert got.to_dict() == pytest.approx({"A": 0.5, "B": 0.0, "C": 0.5})
    audit = benchmark.audit_summary()
    assert audit["fallback_reason_by_period"] == {"2024-07-15": "roster_changed"}
    assert audit["roster_change_period_count"] == 1
    assert "回退 reconstruct" in caplog.text


def test_exact_snapshot_date_still_checks_roster_consistency(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel().with_columns(
        pl.when(pl.col("date") == date(2024, 7, 31))
        .then((pl.col("code") != "B").cast(pl.Int64))
        .otherwise(pl.col("is_zz1000"))
        .alias("is_zz1000")
    )
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )

    got = benchmark.get_weights(date(2024, 7, 31), ["A", "B", "C"])

    assert got.to_dict() == pytest.approx({"A": 0.5, "B": 0.0, "C": 0.5})
    assert benchmark.audit_summary()["fallback_reason_by_period"] == {
        "2024-07-31": "roster_changed"
    }


def test_empty_target_roster_is_data_error_and_falls_back(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel().with_columns(
        pl.when(pl.col("date") == date(2024, 7, 15))
        .then(pl.lit(0))
        .otherwise(pl.col("is_zz1000"))
        .alias("is_zz1000")
    )
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )

    got = benchmark.get_weights(date(2024, 7, 15), ["A", "B", "C"])

    assert got.sum() == pytest.approx(1.0)
    assert benchmark.audit_summary()["method_by_period"]["2024-07-15"] == (
        "reconstruct"
    )
    assert benchmark.audit_summary()["fallback_reason_by_period"] == {
        "2024-07-15": "target_roster_empty"
    }


def test_missing_anchor_price_falls_back_and_is_audited(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel().with_columns(
        pl.when(
            (pl.col("date") == date(2024, 6, 28)) & (pl.col("code") == "B")
        )
        .then(None)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official_drift", official_path=path
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 7, 15), panel=panel)

    got = benchmark.get_weights(date(2024, 7, 15), ["A", "B", "C"])

    assert got.to_dict() == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})
    reason = benchmark.audit_summary()["fallback_reason_by_period"]["2024-07-15"]
    assert reason.startswith("price_coverage:")


def test_adjusted_price_avoids_false_weight_move_on_bonus_or_cash_event():
    """送股/除权/现金事件已反映在后复权价中时，权重不应因原始价跳空而误漂。"""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    adjusted_anchor = pd.Series({"A": 100.0, "B": 100.0})
    adjusted_target = pd.Series({"A": 100.0, "B": 100.0})

    got = drift_official_weights(weights, adjusted_anchor, adjusted_target)

    assert got.to_dict() == pytest.approx({"A": 0.6, "B": 0.4})


def test_official_alias_uses_drift_mode(tmp_path):
    path = tmp_path / "official.parquet"
    _make_official(path)
    panel = _make_drift_panel()
    benchmark = IndexBenchmarkWeights(
        index="zz1000", panel=panel, source="official", official_path=path
    )
    benchmark.precompute(date(2024, 6, 28), date(2024, 7, 15), panel=panel)

    assert benchmark.source == "official_drift"
    assert benchmark.get_weights(date(2024, 7, 15), ["A", "B"])["A"] == pytest.approx(0.75)


def test_reconstruct_source_records_method_for_each_requested_date():
    panel = _make_panel_3stocks()
    benchmark = IndexBenchmarkWeights(
        index="zz500", panel=panel, source="reconstruct"
    )
    target = date(2024, 1, 3)

    benchmark.get_weights(target)

    assert benchmark.audit_summary()["method_by_period"] == {
        target.isoformat(): "reconstruct"
    }
