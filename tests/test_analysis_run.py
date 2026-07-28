"""analysis/run.py（hqopt attribute 入口层）测试。

CNE6 面板 / 因子收益 / 归因器在模块边界打桩，回测引擎用真实实现——
重点验证入口层的接线：权重文件解析、实际权重重放并传给归因器、结果落地。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hqopt.analysis.run as runmod
from hqopt.analysis.run import _load_weights, run_attribution
from hqopt.backtest.execution import (
    sell_only_path_for_weights,
    write_sell_only_manifest,
)

# ── 构造最小行情面板（2 票 × 5 交易日，全程可交易）─────────────────


def _panel() -> pl.DataFrame:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4),
             date(2024, 1, 5), date(2024, 1, 8)]
    rows = []
    for d in dates:
        for code in ("A", "B"):
            price = 10.0
            rows.append({
                "code": code, "date": d,
                "adj_close": price, "adj_vwap": price, "close": price,
                "limit_up": 20.0, "limit_down": 1.0,
                "trade_status": "交易",
            })
    return pl.DataFrame(rows)


class _FakeAttributionResult:
    def __init__(self) -> None:
        self.summary = pd.DataFrame({"贡献": [0.01]}, index=["Size"])
        self.daily = pd.DataFrame(
            {"coverage_pct": [1.0, 0.5]},
            index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        )
        self.factor_daily = pd.DataFrame(
            {"Size": [0.001]}, index=pd.to_datetime(["2024-01-03"])
        )

    def __str__(self) -> str:
        return "fake-attribution-result"


class _RecordingAttributor:
    """捕获 run() 收到的参数，供断言接线正确。"""

    calls: list[dict] = []

    def __init__(self, risk_model, factor_loader) -> None:
        del risk_model, factor_loader

    def run(self, weight_df, bm_matrix, adj_close, actual_weight_df=None):
        _RecordingAttributor.calls.append({
            "weight_df": weight_df,
            "bm_matrix": bm_matrix,
            "adj_close": adj_close,
            "actual_weight_df": actual_weight_df,
        })
        return _FakeAttributionResult()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """打桩外部数据依赖，返回 (weights_path, 调用记录清空)。"""
    _RecordingAttributor.calls = []
    monkeypatch.setattr(runmod, "load_panel",
                        lambda *a, **k: _panel())
    monkeypatch.setattr(
        runmod, "CNE6RiskModel", lambda data_dir=None, query_dates=None: object()
    )
    monkeypatch.setattr(
        runmod, "FactorReturnLoader", lambda data_dir=None: object()
    )
    monkeypatch.setattr(runmod, "ReturnAttributor", _RecordingAttributor)

    weight_df = pd.DataFrame(
        {"A": [0.6, 0.5], "B": [0.4, 0.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    weights_path = tmp_path / "weights.parquet"
    weight_df.to_parquet(weights_path)
    return weights_path


# ── _load_weights：长表 / 宽表等价 ──────────────────────────────


def test_load_weights_long_and_wide_equivalent(tmp_path):
    wide = pd.DataFrame(
        {"A": [0.6, 0.5], "B": [0.4, 0.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    wide_path = tmp_path / "wide.parquet"
    wide.to_parquet(wide_path)

    long = wide.stack().rename("weight").reset_index()
    long.columns = ["date", "code", "weight"]
    long_path = tmp_path / "long.parquet"
    long.to_parquet(long_path)

    from_wide = _load_weights(wide_path)
    from_long = _load_weights(long_path)
    pd.testing.assert_frame_equal(
        from_wide.sort_index(axis=1), from_long.sort_index(axis=1),
        check_names=False,
    )
    assert isinstance(from_wide.index, pd.DatetimeIndex)


# ── run_attribution 接线 ────────────────────────────────────────


def test_run_attribution_passes_replayed_actual_weights(patched, tmp_path):
    """归因器必须收到成交账本重放出的逐日实际权重（非 None、逐日索引）。"""
    out_dir = tmp_path / "out"
    result = run_attribution(
        patched, "2024-01-02", "2024-01-08",
        index="all",                      # 非成分指数 → 等权基准，不依赖官方权重
        out_dir=out_dir,
        cost_buy=0.0, cost_sell=0.0,
    )

    assert isinstance(result, _FakeAttributionResult)
    call = _RecordingAttributor.calls[-1]
    actual = call["actual_weight_df"]
    assert actual is not None
    # T+1 成交：1/2 提交的目标从 1/3 起持有，实际权重应≈目标
    assert actual.loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(0.6, abs=1e-6)
    assert actual.loc[pd.Timestamp("2024-01-02")].sum() == 0.0  # 首日尚未成交
    # 等权基准：两票各 0.5
    bm = call["bm_matrix"]
    assert np.allclose(bm.loc[pd.Timestamp("2024-01-03")].values, [0.5, 0.5])

    # 输出文件落地
    assert (out_dir / "attribution_summary.csv").exists()
    assert (out_dir / "attribution_daily.parquet").exists()
    assert (out_dir / "attribution_factor_daily.parquet").exists()
    assert (out_dir / "attribution_execution_stats.json").exists()


def test_run_attribution_applies_sell_only_sidecar(patched):
    """归因重放必须读取 sidecar；制度性只卖股票不能按新目标反向加仓。"""
    weight_df = pd.DataFrame(
        {"A": [0.6, 0.8], "B": [0.4, 0.2]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    weight_df.to_parquet(patched)
    sell_only = pd.DataFrame(
        {"A": [False, True], "B": [False, False]},
        index=weight_df.index,
    )
    sell_only.to_parquet(sell_only_path_for_weights(patched))
    write_sell_only_manifest(patched)

    run_attribution(
        patched,
        "2024-01-02",
        "2024-01-08",
        index="all",
        cost_buy=0.0,
        cost_sell=0.0,
    )

    actual = _RecordingAttributor.calls[-1]["actual_weight_df"]
    assert actual.loc[pd.Timestamp("2024-01-05"), "A"] == pytest.approx(0.6)
    assert actual.loc[pd.Timestamp("2024-01-05"), "B"] == pytest.approx(0.2)


def test_run_attribution_filters_weights_to_date_range(patched):
    """区间过滤生效：只保留 [start, end] 内的调仓日。"""
    run_attribution(
        patched, "2024-01-04", "2024-01-08",
        index="all", cost_buy=0.0, cost_sell=0.0,
    )
    call = _RecordingAttributor.calls[-1]
    assert list(call["weight_df"].index) == [pd.Timestamp("2024-01-04")]


def test_run_attribution_empty_range_raises(patched):
    with pytest.raises(ValueError, match="无数据"):
        run_attribution(
            patched, "2025-01-01", "2025-02-01",
            index="all",
        )
