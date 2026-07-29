import json
import logging
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hqopt.backtest.run as runmod
from hqopt.backtest.execution import (
    publish_batch_bundle,
    sell_only_manifest_path_for_weights,
    sell_only_path_for_weights,
    write_sell_only_manifest,
)
from hqopt.backtest.run import (
    _align_sell_only_metadata,
    _load_sell_only_metadata,
    _save_execution_stats,
    run_backtest,
)


def test_save_execution_stats_records_expired_orders(tmp_path):
    output = _save_execution_stats(
        {
            "expired_order_count": np.int64(2),
            "expired_notional": np.float64(123.45),
            "order_states": {"A": "expired", "B": "filled"},
        },
        tmp_path / "execution_stats.json",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "expired_order_count": 2,
        "expired_notional": 123.45,
        "order_states": {"A": "expired", "B": "filled"},
    }


def test_load_sell_only_metadata_is_optional_and_boolean(tmp_path):
    weights = tmp_path / "weights.parquet"
    assert _load_sell_only_metadata(weights) is None

    pd.DataFrame(
        {"A": [0.6], "B": [0.4]},
        index=pd.to_datetime(["2024-01-02"]),
    ).to_parquet(weights)
    metadata = pd.DataFrame(
        {"A": [1], "B": [0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    metadata.to_parquet(sell_only_path_for_weights(weights))
    write_sell_only_manifest(weights)

    loaded = _load_sell_only_metadata(weights)
    assert loaded is not None
    assert loaded.dtypes.tolist() == [bool, bool]
    assert bool(loaded.loc[pd.Timestamp("2024-01-02"), "A"]) is True


def test_sell_only_metadata_missing_warns_and_uses_external_weight_mode(
    tmp_path,
    caplog,
):
    weights = tmp_path / "weights.parquet"
    weight_df = pd.DataFrame(
        {"A": [1.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    with caplog.at_level(logging.WARNING):
        aligned = _align_sell_only_metadata(None, weight_df, weights)

    assert aligned is None
    assert "未找到只卖元数据" in caplog.text
    assert "不推断 sell-only" in caplog.text


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            pd.DataFrame(
                {"A": [False]},
                index=pd.to_datetime(["2024-01-02"]),
            ),
            "日期与权重文件不一致",
        ),
        (
            pd.DataFrame(
                {"A": [False, False], "C": [False, False]},
                index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
            ),
            "股票列与权重文件不一致",
        ),
    ],
)
def test_sell_only_metadata_alignment_is_strict(tmp_path, metadata, message):
    weights = tmp_path / "weights.parquet"
    weight_df = pd.DataFrame(
        {"A": [0.6, 0.5], "B": [0.4, 0.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    with pytest.raises(ValueError, match=message):
        _align_sell_only_metadata(metadata, weight_df, weights)


def test_load_sell_only_metadata_rejects_non_binary_values(tmp_path):
    weights = tmp_path / "weights.parquet"
    pd.DataFrame(
        {"A": [1.0]},
        index=pd.to_datetime(["2024-01-02"]),
    ).to_parquet(weights)
    metadata = pd.DataFrame(
        {"A": [2]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    metadata.to_parquet(sell_only_path_for_weights(weights))
    write_sell_only_manifest(weights)

    with pytest.raises(ValueError, match="只能包含"):
        _load_sell_only_metadata(weights)


def test_load_sell_only_metadata_requires_valid_content_manifest(tmp_path):
    weights = tmp_path / "weights.parquet"
    weight_df = pd.DataFrame(
        {"A": [1.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    weight_df.to_parquet(weights)
    sidecar = sell_only_path_for_weights(weights)
    pd.DataFrame({"A": [False]}, index=weight_df.index).to_parquet(sidecar)

    with pytest.raises(ValueError, match="缺少内容绑定清单"):
        _load_sell_only_metadata(weights)

    write_sell_only_manifest(weights)
    pd.DataFrame({"A": [True]}, index=weight_df.index).to_parquet(sidecar)
    with pytest.raises(ValueError, match="哈希不匹配"):
        _load_sell_only_metadata(weights)

    assert sell_only_manifest_path_for_weights(weights).exists()


def test_run_backtest_applies_sell_only_sidecar(tmp_path, monkeypatch):
    """回测入口必须把 sidecar 传入成交账本，阻止制度性只卖股票反向加仓。"""
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    rows = [
        {
            "date": trading_date,
            "code": ticker,
            "adj_close": 10.0,
            "adj_vwap": 10.0,
            "close": 10.0,
            "limit_up": 11.0,
            "limit_down": 9.0,
            "trade_status": "交易",
        }
        for trading_date in dates
        for ticker in ("A", "B")
    ]
    monkeypatch.setattr(runmod, "load_panel", lambda *args, **kwargs: pl.DataFrame(rows))
    def reject_index_returns(*args, **kwargs):
        del args, kwargs
        raise AssertionError("equal_weight 不应读取指数收益")

    monkeypatch.setattr(runmod, "load_index_returns", reject_index_returns)

    weights = pd.DataFrame(
        {"A": [0.6, 0.8], "B": [0.4, 0.2]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )
    weight_path = tmp_path / "weights.parquet"
    sell_only = pd.DataFrame(
        {"A": [False, True], "B": [False, False]},
        index=weights.index,
    )
    publish_batch_bundle(
        weights,
        sell_only,
        {"alpha_quality": {"synthetic": True}},
        weight_path,
    )
    _, stats = run_backtest(
        weight_path,
        "2024-01-02",
        "2024-01-08",
        index="equal_weight",
        cost_buy=0.0,
        cost_sell=0.0,
        risk_free=0.0,
        out_dir=tmp_path / "report",
    )

    assert stats["final_actual_weights"]["A"] == pytest.approx(0.6)
    assert stats["final_actual_weights"]["B"] == pytest.approx(0.2)
    assert stats["final_cash_pct"] == pytest.approx(0.2)
    html = (tmp_path / "report" / "report.html").read_text(encoding="utf-8")
    assert "全市场等权基准" in html
    assert "回测业绩完全不可信" not in html
