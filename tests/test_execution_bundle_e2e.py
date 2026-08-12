"""真实 batch pipeline → 回测 → 归因的快速、无外部数据端到端回归。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hqopt.analysis.run as attribution_run
import hqopt.backtest.run as backtest_run
import hqopt.pipeline.batch_optimize as batch
from hqopt.backtest.execution import (
    SELL_ONLY_MANIFEST_VERSION,
    validate_sell_only_manifest,
)

_TRADE_DATES = [
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
]
_DATE_INDEX = pd.to_datetime(_TRADE_DATES)
_CODES = ("A", "C", "D")


def _market_panel() -> pl.DataFrame:
    """A 在失败候选日成交；C 在下一成功日可买；D 仅 T 日可买。"""
    rows = []
    for trading_date in _TRADE_DATES:
        for code in _CODES:
            at_limit_up = (
                (code == "C" and trading_date == date(2024, 1, 3))
                or (code == "D" and trading_date >= date(2024, 1, 5))
            )
            rows.append(
                {
                    "date": trading_date,
                    "code": code,
                    "adj_close": 10.0,
                    "adj_vwap": 10.0,
                    "close": 10.0,
                    "limit_up": 10.0 if at_limit_up else 11.0,
                    "limit_down": 9.0,
                    "amount": 1_000.0,
                    "float_mv": 10_000.0,
                    "free_mv": 10_000.0,
                    "total_mv": 10_000.0,
                    "free_turnover": 1.0,
                    "trade_status": "交易",
                    "industry_l1": "BankFinance",
                    "list_days": 365,
                    "is_hs300": 0,
                    "is_zz500": 0,
                    "is_zz1000": 1,
                    "is_st": 0,
                }
            )
    return pl.DataFrame(rows)


class _RiskSnapshot:
    def __init__(self, tickers: list[str]) -> None:
        self.covered_mask = np.ones(len(tickers), dtype=bool)

    def style_loading(self) -> pd.DataFrame:
        return pd.DataFrame()


class _RiskModel:
    coverage = (_TRADE_DATES[0], _TRADE_DATES[-1])

    def __init__(self, data_dir=None, query_dates=None) -> None:
        del data_dir, query_dates

    def at(self, target_date: date, tickers: list[str]) -> _RiskSnapshot:
        del target_date
        return _RiskSnapshot(tickers)


@dataclass
class _OptimizationResult:
    weights: np.ndarray
    feasible: bool
    status: str

    @property
    def is_feasible(self) -> bool:
        return self.feasible

    @property
    def n_positions(self) -> int:
        return int((self.weights > 1e-8).sum())


class _SequencedOptimizer:
    """成功旧目标 → 失败继续旧目标 → 成功目标；随后均失败。"""

    def __init__(self) -> None:
        self.calls = 0
        self.prev_weights: list[pd.Series | None] = []

    def optimize(self, alpha, snapshot, *, prev_weight=None, **kwargs):
        del alpha, kwargs
        self.prev_weights.append(
            None
            if prev_weight is None
            else pd.Series(prev_weight.copy(), index=snapshot.tickers)
        )
        call = self.calls
        self.calls += 1
        weights = np.zeros(snapshot.n_stocks)
        if call == 0:
            weights[snapshot.tickers.index("A")] = 0.5
            weights[snapshot.tickers.index("C")] = 0.5
            return _OptimizationResult(weights, True, "optimal")
        if call == 2:
            weights[snapshot.tickers.index("D")] = 1.0
            return _OptimizationResult(weights, True, "optimal")
        return _OptimizationResult(weights, False, "infeasible")


def _batch_config(tmp_path) -> dict:
    return {
        "strategy": "alpha_max",
        "index": "all",
        "backtest": {
            "start_date": "2024-01-02",
            "end_date": "2024-01-09",
            "rebalance_freq": 1,
            "initial_value": 1_000.0,
        },
        "universe": {
            "exclude_bj": False,
            "exclude_st": False,
            "top_n": None,
        },
        "optimizer": {
            "weight_upper": 1.0,
            "industry_upper": 1.0,
            "min_constituent_ratio": 0.0,
            "diversification_penalty": 0.0,
            "risk_aversion": None,
            "max_turnover": 2.0,
            "turnover_penalty": 0.0,
            "style_bound": None,
        },
        "alpha": {"source": "file", "synthetic": False},
        "execution": {"cost_buy": 0.0, "cost_sell": 0.0},
        "output": {"weights": str(tmp_path / "weights.parquet")},
    }


class _AttributionResult:
    def __init__(self) -> None:
        self.summary = pd.DataFrame(
            {
                "累计贡献": [0.0, 0.0],
                "年化贡献": [0.0, 0.0],
                "占主动收益%": [np.nan, np.nan],
                "t统计": [np.nan, np.nan],
                "年化波动": [0.0, 0.0],
            },
            index=["Country", "合计(主动收益)"],
        )
        self.daily = pd.DataFrame(
            {"coverage_pct": [1.0], "relative_active_return": [0.0]},
            index=pd.to_datetime(["2024-01-03"]),
        )
        self.factor_daily = pd.DataFrame(
            {"Country": [0.0]},
            index=pd.to_datetime(["2024-01-03"]),
        )

    def __str__(self) -> str:
        return "synthetic-attribution"


class _RecordingAttributor:
    actual_weights: pd.DataFrame | None = None

    def __init__(self, risk_model, factor_loader) -> None:
        del risk_model, factor_loader

    def run(
        self,
        weight_df,
        benchmark_weight_df,
        adj_close,
        actual_weight_df=None,
        realized_portfolio_return=None,
    ):
        del weight_df, benchmark_weight_df, adj_close, realized_portfolio_return
        type(self).actual_weights = actual_weight_df
        return _AttributionResult()


def test_real_batch_bundle_replays_identically_in_backtest_and_attribution(
    tmp_path,
    monkeypatch,
):
    """信号日先执行旧目标；成功候选收盘覆盖，失败候选继续旧目标。"""
    panel = _market_panel()
    optimizer = _SequencedOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _RiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    alpha = pd.DataFrame(
        {"A": 1.0, "C": 0.0, "D": -1.0},
        index=_DATE_INDEX,
    )
    config = _batch_config(tmp_path)
    weight_df = batch.run_batch_optimize(config, panel=panel, alpha_df=alpha)

    successful_dates = pd.to_datetime(weight_df.index)
    assert successful_dates.tolist() == [_DATE_INDEX[0], _DATE_INDEX[2]]
    assert optimizer.calls == len(_TRADE_DATES)
    assert optimizer.prev_weights[1]["A"] == pytest.approx(0.5)
    assert optimizer.prev_weights[1]["C"] == pytest.approx(0.0)
    assert optimizer.prev_weights[2]["A"] == pytest.approx(0.5)
    assert optimizer.prev_weights[2]["C"] == pytest.approx(0.5)

    weight_path = tmp_path / "weights.parquet"
    stats_path = tmp_path / "batch_execution_stats.json"
    manifest_path = validate_sell_only_manifest(weight_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SELL_ONLY_MANIFEST_VERSION
    assert manifest["batch_execution_stats_file"] == stats_path.name
    assert len(manifest["batch_execution_stats_sha256"]) == 64

    monkeypatch.setattr(
        backtest_run,
        "load_panel",
        lambda *args, **kwargs: panel,
    )
    monkeypatch.setattr(
        backtest_run,
        "load_index_returns",
        lambda *args, **kwargs: pd.Series(0.0, index=_DATE_INDEX),
    )
    monkeypatch.setattr(
        attribution_run,
        "load_panel",
        lambda *args, **kwargs: panel,
    )
    monkeypatch.setattr(
        attribution_run,
        "CNE6RiskModel",
        lambda data_dir=None, query_dates=None: object(),
    )
    monkeypatch.setattr(
        attribution_run,
        "FactorReturnLoader",
        lambda data_dir=None: object(),
    )
    monkeypatch.setattr(
        attribution_run,
        "ReturnAttributor",
        _RecordingAttributor,
    )

    backtest_result, backtest_stats = backtest_run.run_backtest(
        weight_path,
        _TRADE_DATES[0],
        _TRADE_DATES[-1],
        index="equal_weight",
        cost_buy=0.0,
        cost_sell=0.0,
        risk_free=0.0,
        initial_value=1_000.0,
    )
    attribution_out = tmp_path / "attribution"
    attribution_run.run_attribution(
        weight_path,
        _TRADE_DATES[0],
        _TRADE_DATES[-1],
        index="equal_weight",
        out_dir=attribution_out,
        cost_buy=0.0,
        cost_sell=0.0,
        initial_value=1_000.0,
    )
    attribution_report = attribution_out / "attribution_report.html"
    assert attribution_report.exists()
    assert "收益归因报告" in attribution_report.read_text(encoding="utf-8")

    batch_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    # 目标换手单独校验，其余字段保持精确匹配以锁定 schema。
    optimization = dict(batch_stats["optimization"])
    target_turnover = optimization.pop("target_turnover")
    assert optimization == {
        "candidate_period_count": 6,
        "failed_period_count": 4,
        "successful_period_count": 2,
        "post_solve_validation": {
            "absolute_tolerance": 1e-5,
            "failure_count": 0,
            "failures_by_period": {},
            "max_observed_violation": 0.0,
            "max_violation_by_period": {},
        },
    }
    # 两个成功期中只有第二期有「上期」；该期完全换仓，且上期满仓故 cash_gap=0，
    # 因此 net == gross。
    assert target_turnover["by_period"] == {
        "2024-01-04": {"gross": 2.0, "cash_gap": 0.0, "net": 2.0},
    }
    assert target_turnover["gross_mean"] == pytest.approx(2.0)
    assert target_turnover["net_mean"] == pytest.approx(2.0)
    assert target_turnover["cash_gap_mean"] == pytest.approx(0.0)
    assert "max_turnover" in target_turnover["definition"]
    assert batch_stats["alpha_quality"]["skipped_period_count"] == 0
    assert batch_stats["alpha_quality"]["zero_variance_period_count"] == 0
    assert len(batch_stats["alpha_quality"]["as_of_by_period"]) == 6
    attribution_stats = json.loads(
        (attribution_out / "attribution_execution_stats.json").read_text(
            encoding="utf-8"
        )
    )
    for stats in (backtest_stats, attribution_stats):
        assert stats["final_shares"] == pytest.approx(batch_stats["final_shares"])
        assert stats["order_states"] == batch_stats["order_states"]
        assert stats["expired_order_count"] == batch_stats["expired_order_count"]
        assert stats["expired_notional"] == pytest.approx(
            batch_stats["expired_notional"]
        )

    assert batch_stats["final_shares"] == {}
    assert batch_stats["order_states"] == {
        "A": "filled",
        "C": "filled",
        "D": "expired",
    }
    assert batch_stats["expired_order_count"] == 1
    assert batch_stats["expired_notional"] == pytest.approx(1_000.0)

    actual_weights = backtest_result.actual_weights.reindex(columns=_CODES, fill_value=0.0)
    assert actual_weights.loc[_DATE_INDEX[1], "A"] == pytest.approx(0.5)
    assert actual_weights.loc[_DATE_INDEX[1], "C"] == pytest.approx(0.0)
    assert actual_weights.loc[_DATE_INDEX[2], "C"] == pytest.approx(0.5)
    assert actual_weights.loc[_DATE_INDEX[2], "D"] == pytest.approx(0.0)
    pd.testing.assert_frame_equal(
        _RecordingAttributor.actual_weights.reindex(columns=_CODES, fill_value=0.0),
        actual_weights,
    )
