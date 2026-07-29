"""逐期优化必须使用实际成交持仓，而不是上一期目标权重。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hqopt.pipeline.batch_optimize as batch
from hqopt.backtest.execution import (
    sell_only_manifest_path_for_weights,
    sell_only_path_for_weights,
    validate_sell_only_manifest,
)


class _FakeRiskSnapshot:
    def __init__(self, tickers: list[str], *, fully_covered: bool = True) -> None:
        self.covered_mask = np.ones(len(tickers), dtype=bool)
        if not fully_covered:
            self.covered_mask[0] = False

    def style_loading(self) -> pd.DataFrame:
        return pd.DataFrame()


class _FakeRiskModel:
    coverage = (date(2024, 1, 2), date(2024, 1, 4))

    def __init__(self, data_dir=None, query_dates=None) -> None:
        del data_dir, query_dates

    def at(self, target_date: date, tickers: list[str]) -> _FakeRiskSnapshot:
        del target_date
        return _FakeRiskSnapshot(tickers)


class _PartiallyCoveredRiskModel(_FakeRiskModel):
    def at(self, target_date: date, tickers: list[str]) -> _FakeRiskSnapshot:
        del target_date
        return _FakeRiskSnapshot(tickers, fully_covered=False)


@dataclass
class _FakeResult:
    weights: np.ndarray
    status: str = "optimal"

    @property
    def is_feasible(self) -> bool:
        return True

    @property
    def n_positions(self) -> int:
        return int((self.weights > 1e-8).sum())

    def tracking_error_l2(self) -> float:
        return 0.0


class _RecordingOptimizer:
    def __init__(self) -> None:
        self.config = None
        self.prev_weights: list[np.ndarray | None] = []
        self.sell_only_masks: list[np.ndarray] = []
        self.ticker_history: list[list[str]] = []

    def optimize(self, alpha, snapshot, *, prev_weight=None, **kwargs) -> _FakeResult:
        del alpha, kwargs
        self.prev_weights.append(None if prev_weight is None else prev_weight.copy())
        self.sell_only_masks.append(snapshot.sell_only_mask.copy())
        self.ticker_history.append(list(snapshot.tickers))
        weights = np.zeros(snapshot.n_stocks)
        target_ticker = "A" if len(self.prev_weights) == 1 else "B"
        weights[snapshot.tickers.index(target_ticker)] = 1.0
        return _FakeResult(weights)


class _FakeBenchmarkWeights:
    def __init__(self, **kwargs) -> None:
        del kwargs

    def precompute(self, *args, **kwargs) -> None:
        del args, kwargs

    def get_weights(self, target_date, tickers) -> pd.Series:
        del target_date
        return pd.Series(1.0 / len(tickers), index=tickers)

    def audit_summary(self) -> dict:
        return {
            "source": "official_drift",
            "price_field": "adj_close",
            "fallback_period_count": 0,
        }


def _panel() -> pl.DataFrame:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    rows = []
    for d in dates:
        for code in ("A", "B"):
            a_blocked = code == "A" and d >= date(2024, 1, 3)
            close = 11.0 if a_blocked else 10.0
            rows.append({
                "date": d,
                "code": code,
                "adj_close": close,
                "adj_vwap": close,
                "close": close,
                "limit_up": 11.0 if a_blocked else 20.0,
                "limit_down": 1.0,
                "amount": 1000.0,
                "float_mv": 10000.0,
                "free_mv": 10000.0,
                "total_mv": 10000.0,
                "free_turnover": 1.0,
                "trade_status": "交易",
                "industry_l1": "BankFinance",
                "list_days": 365,
                "is_hs300": 0,
                "is_zz500": 0,
                "is_zz1000": 1,
                "is_st": 0,
            })
    return pl.DataFrame(rows)


def _candidate_skip_panel() -> pl.DataFrame:
    """A 在 T+1 涨停，但在下一候选调仓日恢复可买。"""
    return _panel().with_columns(
        *[
            pl.when(
                (pl.col("date") == date(2024, 1, 4))
                & (pl.col("code") == "A")
            )
            .then(10.0 if column != "limit_up" else 20.0)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("adj_close", "adj_vwap", "close", "limit_up")
        ]
    )


def _assert_skipped_candidate_executes_old_target(
    tmp_path,
    config: dict,
) -> None:
    """候选日若未发布目标，旧目标必须在恢复交易的当日成交。"""
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    weight_df = batch.run_batch_optimize(
        config,
        panel=_candidate_skip_panel(),
        alpha_df=alpha,
    )

    assert list(weight_df.index) == [date(2024, 1, 2)]
    stats = json.loads(
        (tmp_path / "batch_execution_stats.json").read_text(encoding="utf-8")
    )
    assert stats["final_shares"] == pytest.approx({"A": 10.0})
    assert stats["order_states"] == {"A": "filled"}
    assert stats["target_pending"] is False
    assert stats["expired_order_count"] == 0


def _suspension_freeze_panel() -> pl.DataFrame:
    """三期调仓面板：第二个信号日 A 停牌，次日复牌并涨价。"""
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    rows = []
    for d in dates:
        for code in ("A", "B", "C"):
            price = 20.0 if code == "A" and d >= date(2024, 1, 5) else 10.0
            rows.append({
                "date": d,
                "code": code,
                "adj_close": price,
                "adj_vwap": price,
                "close": price,
                "limit_up": price * 1.1,
                "limit_down": price * 0.9,
                "amount": 1000.0,
                "float_mv": 10000.0,
                "free_mv": 10000.0,
                "total_mv": 10000.0,
                "free_turnover": 1.0,
                "trade_status": (
                    "停牌"
                    if code == "A" and d == date(2024, 1, 4)
                    else "交易"
                ),
                "industry_l1": "BankFinance",
                "list_days": 365,
                "is_hs300": 0,
                "is_zz500": 0,
                "is_zz1000": 1,
                "is_st": 0,
            })
    return pl.DataFrame(rows)


def test_second_period_prev_weight_uses_failed_execution_state(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    config = {
        "strategy": "alpha_max",
        "index": "all",
        "backtest": {
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
            "rebalance_freq": 2,
            "initial_value": 100.0,
        },
        "universe": {"exclude_bj": False, "exclude_st": False, "top_n": None},
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
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    panel = _panel().with_columns(
        *[
            pl.when(
                (pl.col("date") == date(2024, 1, 4)) & (pl.col("code") == "A")
            )
            .then(10.0 if column != "limit_up" else 20.0)
            .otherwise(pl.col(column))
            .alias(column)
            for column in ("adj_close", "adj_vwap", "close", "limit_up")
        ]
    )
    batch.run_batch_optimize(config, panel=panel, alpha_df=alpha)

    assert optimizer.prev_weights[0] is None
    # T+1（1/3）A 涨停未成交；新调仓日（1/4）恢复交易后先补买旧目标 A，
    # 第二期优化必须看到信号日成交后的真实持仓。
    assert optimizer.prev_weights[1] is not None
    assert optimizer.prev_weights[1][0] == pytest.approx(1.0)
    assert optimizer.prev_weights[1].sum() == pytest.approx(1.0)


def test_snapshot_failure_candidate_executes_old_pending_target(
    tmp_path,
    monkeypatch,
):
    """快照失败不发布权重行，但旧目标在候选日仍应正常执行。"""
    optimizer = _RecordingOptimizer()
    original_build = batch.RealMarketAdapter.build_snapshot_from_panel

    def build_or_fail(self, panel, target_date, *args, **kwargs):
        if target_date == date(2024, 1, 4):
            raise ValueError("synthetic snapshot failure")
        return original_build(self, panel, target_date, *args, **kwargs)

    monkeypatch.setattr(
        batch.RealMarketAdapter,
        "build_snapshot_from_panel",
        build_or_fail,
    )
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    _assert_skipped_candidate_executes_old_target(
        tmp_path,
        _alpha_config(tmp_path),
    )
    assert len(optimizer.prev_weights) == 1


def test_missing_risk_snapshot_candidate_executes_old_pending_target(
    tmp_path,
    monkeypatch,
):
    """风险快照无覆盖不发布权重行，但旧目标在候选日仍应正常执行。"""

    class _MissingSecondRiskModel(_FakeRiskModel):
        def at(self, target_date: date, tickers: list[str]):
            if target_date == date(2024, 1, 4):
                return None
            return super().at(target_date, tickers)

    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _MissingSecondRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    _assert_skipped_candidate_executes_old_target(
        tmp_path,
        _alpha_config(tmp_path),
    )
    assert len(optimizer.prev_weights) == 1


def test_low_benchmark_risk_coverage_candidate_executes_old_pending_target(
    tmp_path,
    monkeypatch,
):
    """指数风险覆盖不足不发布权重行，但旧目标在候选日仍应正常执行。"""

    class _SecondTickerUncoveredRiskModel(_FakeRiskModel):
        def at(self, target_date: date, tickers: list[str]) -> _FakeRiskSnapshot:
            del target_date
            snapshot = _FakeRiskSnapshot(tickers)
            snapshot.covered_mask = np.array(
                [ticker != "B" for ticker in tickers],
                dtype=bool,
            )
            return snapshot

    class _ChangingBenchmarkWeights(_FakeBenchmarkWeights):
        def get_weights(self, target_date, tickers) -> pd.Series:
            weights = {"A": 1.0, "B": 0.0}
            if target_date == date(2024, 1, 4):
                weights = {"A": 0.0, "B": 1.0}
            return pd.Series(weights).reindex(tickers).fillna(0.0)

    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _SecondTickerUncoveredRiskModel)
    monkeypatch.setattr(batch, "IndexBenchmarkWeights", _ChangingBenchmarkWeights)
    monkeypatch.setattr(batch, "IndexEnhanceOptimizer", lambda config: optimizer)
    config = _index_config(tmp_path)
    config["backtest"]["end_date"] = "2024-01-04"
    config["backtest"]["rebalance_freq"] = 2

    _assert_skipped_candidate_executes_old_target(tmp_path, config)
    assert len(optimizer.prev_weights) == 1


def test_signal_day_suspension_keeps_shares_in_batch_feedback(tmp_path, monkeypatch):
    """T 日停牌股冻结股数；即使 T+1 复牌涨价，下一期也看到真实漂移权重。"""

    class _ThreeTargetOptimizer(_RecordingOptimizer):
        def optimize(self, alpha, snapshot, *, prev_weight=None, **kwargs) -> _FakeResult:
            del alpha, kwargs
            self.prev_weights.append(None if prev_weight is None else prev_weight.copy())
            self.sell_only_masks.append(snapshot.sell_only_mask.copy())
            self.ticker_history.append(list(snapshot.tickers))
            target_by_call = [
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "C": 0.5},
                {"A": 0.5, "C": 0.5},
            ][len(self.prev_weights) - 1]
            return _FakeResult(
                np.array([target_by_call.get(ticker, 0.0) for ticker in snapshot.tickers])
            )

    optimizer = _ThreeTargetOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)
    config = _alpha_config(tmp_path)
    config["backtest"]["end_date"] = "2024-01-08"
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0], "C": [-1.0, -1.0, -1.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-08"]),
    )

    weights = batch.run_batch_optimize(
        config,
        panel=_suspension_freeze_panel(),
        alpha_df=alpha,
    )

    second_target = weights.loc[date(2024, 1, 4)]
    assert second_target["A"] == pytest.approx(0.5, abs=1e-12)
    third_prev = pd.Series(optimizer.prev_weights[2], index=optimizer.ticker_history[2])
    # A 的 5 股未交易，复牌后市值 100；卖 B 得 50 并买 C 50，故权重为 2/3、1/3。
    assert third_prev["A"] == pytest.approx(2 / 3, abs=1e-12)
    assert third_prev["B"] == pytest.approx(0.0, abs=1e-12)
    assert third_prev["C"] == pytest.approx(1 / 3, abs=1e-12)


def test_uncovered_stock_is_passed_to_optimizer_as_sell_only(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _PartiallyCoveredRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    config = _alpha_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert optimizer.sell_only_masks[0].tolist() == [True, False]
    sell_only = pd.read_parquet(
        sell_only_path_for_weights(config["output"]["weights"])
    )
    assert bool(sell_only.loc[date(2024, 1, 2), "A"]) is True
    stats = json.loads(
        (tmp_path / "batch_execution_stats.json").read_text(encoding="utf-8")
    )
    assert stats["final_shares"].get("A", 0.0) == pytest.approx(0.0)


def test_index_optimization_skips_low_benchmark_risk_coverage(tmp_path, monkeypatch, caplog):
    """基准风险覆盖率不足时跳过该期并告警，不中断整段回测、不调用优化器。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _PartiallyCoveredRiskModel)
    monkeypatch.setattr(batch, "IndexBenchmarkWeights", _FakeBenchmarkWeights)
    monkeypatch.setattr(batch, "IndexEnhanceOptimizer", lambda config: optimizer)

    config = _index_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0], "B": [0.0]}, index=pd.to_datetime(["2024-01-02"]),
    )

    # 唯一一个调仓日被跳过 → 无任何权重记录 → 整体报"所有期均求解失败"
    with pytest.raises(RuntimeError, match="所有期均求解失败"):
        batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert optimizer.prev_weights == []  # 优化器从未被调用
    assert any("基准风险覆盖率" in message for message in caplog.messages)


def test_index_run_persists_benchmark_quality(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "IndexBenchmarkWeights", _FakeBenchmarkWeights)
    monkeypatch.setattr(batch, "IndexEnhanceOptimizer", lambda config: optimizer)
    config = _index_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0], "B": [0.0]}, index=pd.to_datetime(["2024-01-02"])
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    stats = json.loads(
        (tmp_path / "batch_execution_stats.json").read_text(encoding="utf-8")
    )
    assert stats["benchmark_quality"] == {
        "source": "official_drift",
        "price_field": "adj_close",
        "fallback_period_count": 0,
    }


def test_dust_weights_are_cleaned_without_rescaling_constraints(tmp_path, monkeypatch):
    """数值粉尘清零后保留为现金，不放大其余权重而破坏硬约束。"""

    class _DustyOptimizer(_RecordingOptimizer):
        def optimize(self, alpha, snapshot, *, prev_weight=None, **kwargs) -> _FakeResult:
            del alpha, kwargs
            self.prev_weights.append(None if prev_weight is None else prev_weight.copy())
            self.sell_only_masks.append(snapshot.sell_only_mask.copy())
            weights = np.zeros(snapshot.n_stocks)
            weights[snapshot.tickers.index("A")] = 1.0 - 3e-7
            weights[snapshot.tickers.index("B")] = 3e-7  # 数值粉尘
            return _FakeResult(weights)

    optimizer = _DustyOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    config = _alpha_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    weight_df = batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    dust = weight_df[(weight_df > 0) & (weight_df < batch._DUST_WEIGHT_TOL)]
    assert dust.count().sum() == 0, f"权重矩阵仍残留粉尘：{dust.dropna(how='all')}"
    assert weight_df.loc[date(2024, 1, 2), "A"] == pytest.approx(1.0 - 3e-7)
    assert weight_df.loc[date(2024, 1, 2)].sum() < 1.0


def test_material_aggregate_dust_is_not_removed():
    """单票虽小、累计已超容差的权重必须保留，避免累计破坏硬约束。"""
    snapshot = SimpleNamespace(
        tickers=["A", "B", "C"],
        suspended_mask=np.array([False, False, False]),
    )
    raw = np.array([1.0 - 1.2e-6, 6e-7, 6e-7])

    cleaned = batch._clean_target_weights(raw, snapshot, prev_weight=None)

    assert cleaned["B"] == pytest.approx(6e-7)
    assert cleaned["C"] == pytest.approx(6e-7)
    assert cleaned.sum() == pytest.approx(1.0)


def test_stuck_delisted_holding_does_not_block_rebalance(tmp_path, monkeypatch, caplog):
    """持仓退市（后续无行情）时，后续调仓日应告警并继续优化，而非永久跳过。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    # A 只在第一天有行情，之后从面板消失（退市）；B 全程正常。
    # 通过预置账本持仓构造"已持有的退市票"（正常成交路径造不出来：无行情的买单不会成交）。
    base_panel = _panel()
    c_rows = (
        base_panel.filter(pl.col("code") == "B")
        .with_columns(pl.lit("C").alias("code"))
    )
    panel = pl.concat([base_panel, c_rows]).filter(
        ~((pl.col("code") == "A") & (pl.col("date") > date(2024, 1, 2)))
    )
    config = _alpha_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0], "C": [-1.0, -1.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    original_ledger = batch.ExecutionLedger

    def _seeded_ledger(*args, **kwargs):
        ledger = original_ledger(*args, **kwargs)
        # 预置一笔 A 的持仓（1 股 × 10 元 = 总资产 10%），随后 A 退市无行情。
        ledger.shares["A"] = 1.0
        ledger.last_price["A"] = 10.0
        ledger.cash -= 10.0
        return ledger

    monkeypatch.setattr(batch, "ExecutionLedger", _seeded_ledger)

    weight_df = batch.run_batch_optimize(config, panel=panel, alpha_df=alpha)

    # 两个调仓日都完成了优化（未被滞留持仓阻塞）
    assert len(optimizer.prev_weights) == 2
    assert len(weight_df) == 2
    assert any("滞留资产" in message for message in caplog.messages)


def test_explicit_zero_risk_aversion_reaches_optimizer(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()

    def make_optimizer(config):
        optimizer.config = config
        return optimizer

    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", make_optimizer)
    config = _alpha_config(tmp_path)
    config["optimizer"]["risk_aversion"] = 0.0
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert optimizer.config.risk_aversion == 0.0


def test_explicit_zero_optional_constraints_are_preserved(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()

    def make_optimizer(config):
        optimizer.config = config
        return optimizer

    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", make_optimizer)
    config = _alpha_config(tmp_path)
    config["optimizer"]["max_turnover"] = 0.0
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert optimizer.config.max_turnover == 0.0


def test_invalid_strategy_is_rejected(tmp_path):
    config = _alpha_config(tmp_path)
    config["strategy"] = "index_enahnce"

    with pytest.raises(ValueError, match="strategy 须为"):
        batch.run_batch_optimize(config, panel=_panel(), alpha_df=pd.DataFrame())


def test_invalid_alpha_source_is_rejected(tmp_path):
    config = _alpha_config(tmp_path)
    config["alpha"]["source"] = "synthetci"

    with pytest.raises(ValueError, match="alpha.source 须为"):
        batch.run_batch_optimize(config, panel=_panel())


def test_nested_output_directory_is_created(tmp_path, monkeypatch):
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)
    config = _alpha_config(tmp_path)
    output = tmp_path / "new" / "nested" / "weights.parquet"
    config["output"]["weights"] = str(output)
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert output.exists()
    assert sell_only_path_for_weights(output).exists()
    assert sell_only_manifest_path_for_weights(output).exists()
    assert validate_sell_only_manifest(output) == sell_only_manifest_path_for_weights(output)
    stats_path = output.parent / "batch_execution_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["expired_order_count"] >= 0
    assert stats["expired_notional"] >= 0.0
    assert stats["optimization"]["candidate_period_count"] == 2
    assert stats["optimization"]["successful_period_count"] == 2
    assert stats["optimization"]["failed_period_count"] == 0
    assert stats["alpha_quality"]["max_staleness_days_limit"] == 15
    assert stats["alpha_quality"]["standardized"] is True


def test_batch_execution_stats_advance_last_target_to_end_date(tmp_path, monkeypatch):
    """最后一期目标也必须执行到 end_date，统计不能停在最后信号日。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)
    config = _alpha_config(tmp_path)
    config["backtest"]["end_date"] = "2024-01-08"
    config["backtest"]["rebalance_freq"] = 3
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0], "C": [-1.0, -1.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-05"]),
    )

    batch.run_batch_optimize(
        config,
        panel=_suspension_freeze_panel(),
        alpha_df=alpha,
    )

    stats = json.loads(
        (tmp_path / "batch_execution_stats.json").read_text(encoding="utf-8")
    )
    assert stats["target_pending"] is False
    assert stats["final_shares"].get("A", 0.0) == pytest.approx(0.0)
    assert stats["final_shares"]["B"] == pytest.approx(20.0)


def test_file_alpha_marked_synthetic_writes_and_clears_warning(
    tmp_path, monkeypatch, caplog
):
    """文件型合成 Alpha 也必须落地水印；真实 Alpha 成功覆盖后清除旧水印。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)
    config = _alpha_config(tmp_path)
    config["alpha"]["synthetic"] = True
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    warning = tmp_path / batch.SYNTHETIC_ALPHA_WARNING_FILE
    assert warning.exists()
    assert "回测业绩完全不可信" in warning.read_text(encoding="utf-8")
    assert any("合成 Alpha 警告" in message for message in caplog.messages)

    config["alpha"]["synthetic"] = False
    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)
    assert not warning.exists()


def _alpha_config(tmp_path) -> dict:
    return {
        "strategy": "alpha_max",
        "index": "all",
        "backtest": {
            "start_date": "2024-01-02",
            "end_date": "2024-01-04",
            "rebalance_freq": 2,
            "initial_value": 100.0,
        },
        "universe": {"exclude_bj": False, "exclude_st": False, "top_n": None},
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


def test_preloaded_alpha_cannot_bypass_metadata_validation(tmp_path):
    """预载矩阵也必须先校验 source/synthetic，不能因提前返回而绕过。"""
    config = _alpha_config(tmp_path)
    config["alpha"] = {}
    alpha = pd.DataFrame(
        {"A": [1.0], "B": [0.0]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    with pytest.raises(ValueError, match="缺少 alpha.source"):
        batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)


def test_stale_alpha_is_warned_but_still_optimized(tmp_path, monkeypatch, caplog):
    """Alpha 陈旧超过告警阈值时必须显式告警，不能静默沿用过期信号。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    # Alpha 面板止于 2023-12-01，回测区间 2024-01-02~04 → 每期都陈旧 30+ 天
    config = _alpha_config(tmp_path)
    config["alpha"]["max_staleness_days"] = None  # 显式调试模式：只告警不跳过
    alpha = pd.DataFrame(
        {"A": [1.0], "B": [0.0]}, index=pd.to_datetime(["2023-12-01"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert len(optimizer.prev_weights) == 2          # 仍然完成优化
    assert any("Alpha 陈旧" in message for message in caplog.messages)
    assert any("业绩不可直接采信" in message for message in caplog.messages)


def test_alpha_beyond_max_staleness_skips_period(tmp_path, monkeypatch, caplog):
    """配置 max_staleness_days 后，过期期数必须跳过优化而非沿用旧信号。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    config = _alpha_config(tmp_path)
    config["alpha"]["max_staleness_days"] = 1
    # 只有首个调仓日有 Alpha；第二个调仓日（01-04）陈旧 2 天 > 1 → 跳过
    alpha = pd.DataFrame(
        {"A": [1.0], "B": [0.0]}, index=pd.to_datetime(["2024-01-02"]),
    )

    weight_df = batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert len(optimizer.prev_weights) == 1
    assert len(weight_df) == 1
    assert any("跳过优化：Alpha 不可用" in message for message in caplog.messages)


def test_file_alpha_defaults_to_15_day_staleness_limit():
    policy = batch._build_alpha_policy({"source": "file"}, rebal_freq=10)

    assert policy.max_staleness_days == 15


@pytest.mark.parametrize("value", [-1, 1.5, True, "15"])
def test_invalid_alpha_staleness_limit_is_rejected(value):
    with pytest.raises(ValueError, match="必须是非负整数或 null"):
        batch._build_alpha_policy(
            {"source": "file", "max_staleness_days": value},
            rebal_freq=10,
        )


def test_zero_variance_alpha_skips_only_affected_period(
    tmp_path, monkeypatch, caplog
):
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)
    alpha = pd.DataFrame(
        {"A": [1.0, 5.0], "B": [0.0, 5.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    weight_df = batch.run_batch_optimize(
        _alpha_config(tmp_path),
        panel=_panel(),
        alpha_df=alpha,
    )

    assert len(optimizer.prev_weights) == 1
    assert len(weight_df) == 1
    assert any("Alpha 截面无区分度" in message for message in caplog.messages)
    stats = json.loads(
        (tmp_path / "batch_execution_stats.json").read_text(encoding="utf-8")
    )
    assert stats["optimization"] == {
        "candidate_period_count": 2,
        "failed_period_count": 1,
        "successful_period_count": 1,
        "post_solve_validation": {
            "absolute_tolerance": 1e-5,
            "failure_count": 0,
            "failures_by_period": {},
            "max_observed_violation": 0.0,
            "max_violation_by_period": {},
        },
    }
    assert stats["alpha_quality"]["skipped_period_count"] == 1
    assert stats["alpha_quality"]["zero_variance_period_count"] == 1
    assert stats["alpha_quality"]["as_of_by_period"] == {
        "2024-01-02": "2024-01-02"
    }


def test_alpha_standardization_can_be_disabled(tmp_path, monkeypatch):
    """standardize=false 时按原始量纲传给优化器（供已自行标准化的因子使用）。"""
    received: list[np.ndarray] = []

    class _AlphaCapturingOptimizer(_RecordingOptimizer):
        def optimize(self, alpha, snapshot, **kwargs):
            received.append(np.asarray(alpha, dtype=float).copy())
            return super().optimize(alpha, snapshot, **kwargs)

    optimizer = _AlphaCapturingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    config = _alpha_config(tmp_path)
    config["alpha"]["standardize"] = False
    alpha = pd.DataFrame(
        {"A": [30.0, 30.0], "B": [10.0, 10.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert sorted(received[0].tolist()) == [10.0, 30.0]


def test_alpha_standardized_by_default(tmp_path, monkeypatch):
    """默认对优化域做截面 z-score，使风险/成本系数的标定不随因子量纲漂移。"""
    received: list[np.ndarray] = []

    class _AlphaCapturingOptimizer(_RecordingOptimizer):
        def optimize(self, alpha, snapshot, **kwargs):
            received.append(np.asarray(alpha, dtype=float).copy())
            return super().optimize(alpha, snapshot, **kwargs)

    optimizer = _AlphaCapturingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    alpha = pd.DataFrame(
        {"A": [30.0, 30.0], "B": [10.0, 10.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(_alpha_config(tmp_path), panel=_panel(), alpha_df=alpha)

    assert received[0].mean() == pytest.approx(0.0, abs=1e-12)
    assert received[0].std(ddof=0) == pytest.approx(1.0)


def _index_config(tmp_path) -> dict:
    config = _alpha_config(tmp_path)
    config["strategy"] = "index_enhance"
    config["index"] = "zz1000"
    config["backtest"]["end_date"] = "2024-01-02"
    config["optimizer"] = {
        "weight_upper": 1.0,
        "min_constituent_ratio": 0.0,
        "industry_active_bound": 1.0,
        "style_active_bound": 1.0,
        "tracking_penalty": 1.0,
        "max_turnover": 2.0,
        "turnover_penalty": 0.0,
        "min_risk_coverage": 0.95,
    }
    return config


# ── _ExecutionWalker：账本推进语义 ────────────────────────────────


class _RecordingLedger:
    """记录账本调用序列，验证「信号日只估值、不成交」的执行语义。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []

    @staticmethod
    def _tagged_date(adj_close) -> date:
        """截面用 adj_close 编码日号（见 _walker_days），便于断言推进顺序。"""
        return date(2024, 1, int(round(float(adj_close.iloc[0]))))

    def mark_to_market(self, adj_close) -> None:
        self.calls.append(("mark", self._tagged_date(adj_close)))

    def step(self, **kwargs) -> None:
        self.calls.append(("step", self._tagged_date(kwargs["adj_close"])))


def _walker_days(dates: list[date]) -> dict[date, pl.DataFrame]:
    """每个交易日一行；adj_close 存日号，供 _RecordingLedger 反解出推进日期。"""
    return {
        d: pl.DataFrame({
            "date": [d], "code": ["A"],
            "adj_close": [float(d.day)], "adj_vwap": [float(d.day)],
            "close": [float(d.day)],
            "limit_up": [99.0], "limit_down": [1.0], "trade_status": ["交易"],
        })
        for d in dates
    }


def _make_walker(dates: list[date]):
    ledger = _RecordingLedger()
    walker = batch._ExecutionWalker(ledger, dates, _walker_days(dates))
    return ledger, walker


def _dates(*days: int) -> list[date]:
    return [date(2024, 1, d) for d in days]


def test_walker_settles_prior_days_and_executes_old_target_on_signal_day():
    """调仓日之前及信号日均按序执行，收盘后才能由新目标覆盖。"""
    dates = _dates(2, 3, 4, 5)
    ledger, walker = _make_walker(dates)

    walker.open_signal_day(date(2024, 1, 4))

    assert [kind for kind, _ in ledger.calls] == ["step", "step", "step"]
    assert ledger.calls[-1] == ("step", date(2024, 1, 4))


def test_walker_executes_signal_day_exactly_once():
    """信号日的旧目标只执行一次，不依赖求解结果二次回放。"""
    dates = _dates(2, 3)
    ledger, walker = _make_walker(dates)

    walker.open_signal_day(date(2024, 1, 2))

    assert ledger.calls == [("step", date(2024, 1, 2))]


def test_walker_does_not_rewind_across_rebalance_dates():
    """游标单调前进：同一交易日不会被重复成交。"""
    dates = _dates(2, 3, 4, 5, 8)
    ledger, walker = _make_walker(dates)

    walker.open_signal_day(date(2024, 1, 3))
    walker.open_signal_day(date(2024, 1, 5))
    walker.finish()

    settled = [d for kind, d in ledger.calls if kind == "step"]
    assert settled == sorted(settled)
    assert len(settled) == len(set(settled))          # 无重复成交


def test_walker_finish_advances_remaining_days():
    """区间末尾必须补记，否则漏掉最后一批 T+1/T+2/T+3 成交或过期。"""
    dates = _dates(2, 3, 4, 5)
    ledger, walker = _make_walker(dates)

    walker.open_signal_day(date(2024, 1, 2))
    walker.finish()

    stepped = [d for kind, d in ledger.calls if kind == "step"]
    assert stepped == _dates(2, 3, 4, 5)


def test_walker_rejects_rebalance_date_outside_trading_calendar():
    dates = _dates(2, 3)
    _, walker = _make_walker(dates)

    with pytest.raises(RuntimeError, match="不在成交交易日序列中"):
        walker.open_signal_day(date(2024, 1, 9))


def test_walker_rejects_missing_execution_slice():
    dates = _dates(2, 3, 4)
    ledger = _RecordingLedger()
    days = _walker_days(dates)
    del days[date(2024, 1, 3)]
    walker = batch._ExecutionWalker(ledger, dates, days)

    with pytest.raises(RuntimeError, match="缺少成交行情截面"):
        walker.open_signal_day(date(2024, 1, 4))


def test_walker_rejects_missing_signal_day_slice():
    dates = _dates(2, 3)
    ledger = _RecordingLedger()
    days = _walker_days(dates)
    del days[date(2024, 1, 3)]
    walker = batch._ExecutionWalker(ledger, dates, days)

    with pytest.raises(RuntimeError, match="缺少信号日行情截面"):
        walker.open_signal_day(date(2024, 1, 3))
