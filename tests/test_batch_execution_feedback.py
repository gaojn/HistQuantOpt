"""逐期优化必须使用实际成交持仓，而不是上一期目标权重。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

import hqopt.pipeline.batch_optimize as batch


class _FakeRiskSnapshot:
    def __init__(self, tickers: list[str], *, fully_covered: bool = True) -> None:
        self.covered_mask = np.ones(len(tickers), dtype=bool)
        if not fully_covered:
            self.covered_mask[0] = False

    def style_loading(self) -> pd.DataFrame:
        return pd.DataFrame()


class _FakeRiskModel:
    coverage = (date(2024, 1, 2), date(2024, 1, 4))

    def __init__(self, data_dir=None) -> None:
        del data_dir

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


class _RecordingOptimizer:
    def __init__(self) -> None:
        self.config = None
        self.prev_weights: list[np.ndarray | None] = []
        self.sell_only_masks: list[np.ndarray] = []

    def optimize(self, alpha, snapshot, *, prev_weight=None, **kwargs) -> _FakeResult:
        del alpha, kwargs
        self.prev_weights.append(None if prev_weight is None else prev_weight.copy())
        self.sell_only_masks.append(snapshot.sell_only_mask.copy())
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
        "alpha": {"source": "file"},
        "execution": {"cost_buy": 0.0, "cost_sell": 0.0},
        "output": {"weights": str(tmp_path / "weights.parquet")},
    }
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
    )

    batch.run_batch_optimize(config, panel=_panel(), alpha_df=alpha)

    assert optimizer.prev_weights[0] is None
    # 首期目标 A=100%，但两个执行日 A 都涨停，实际仍为全现金；第二期必须看到全零股票权重。
    assert optimizer.prev_weights[1] is not None
    assert optimizer.prev_weights[1].sum() == 0.0


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


def test_dust_weights_are_cleaned_before_submission(tmp_path, monkeypatch):
    """求解器数值粉尘（<1e-6）必须在提交目标前清零，避免账本买入微量仓位。"""

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
    assert weight_df.loc[date(2024, 1, 2), "A"] == pytest.approx(1.0, abs=1e-9)


def test_stuck_delisted_holding_does_not_block_rebalance(tmp_path, monkeypatch, caplog):
    """持仓退市（后续无行情）时，后续调仓日应告警并继续优化，而非永久跳过。"""
    optimizer = _RecordingOptimizer()
    monkeypatch.setattr(batch, "CNE6RiskModel", _FakeRiskModel)
    monkeypatch.setattr(batch, "AlphaMaxOptimizer", lambda config: optimizer)

    # A 只在第一天有行情，之后从面板消失（退市）；B 全程正常。
    # 通过预置账本持仓构造"已持有的退市票"（正常成交路径造不出来：无行情的买单不会成交）。
    panel = _panel().filter(
        ~((pl.col("code") == "A") & (pl.col("date") > date(2024, 1, 2)))
    )
    config = _alpha_config(tmp_path)
    alpha = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]},
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
        "alpha": {"source": "file"},
        "execution": {"cost_buy": 0.0, "cost_sell": 0.0},
        "output": {"weights": str(tmp_path / "weights.parquet")},
    }


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
