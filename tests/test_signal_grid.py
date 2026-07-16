"""合成信号网格标定器测试（research/signal_grid.py）。

signal_grid 本身是「用未来收益反推标定曲线」的研究工具（模块顶部已有醒目
⚠️ 前视警告，不用于实盘），测试目标只是保证其内部数学（IC/ICIR/自相关/
换手计算）在已知构造下给出预期结果，而非验证其"未来信息"用法本身。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hqopt.research.signal_grid import SignalGridRunner, _pivot_adj, run_grid


def _make_price_panel(n_dates: int = 120, n_codes: int = 80, seed: int = 0) -> pl.DataFrame:
    """构造几何随机游走价格面板（date × code），字段含 date/code/adj_close。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    codes = [f"{i:06d}.SH" for i in range(n_codes)]

    rets = rng.normal(0, 0.02, size=(n_dates, n_codes))
    price = 10.0 * np.exp(np.cumsum(rets, axis=0))

    rows = []
    for i, dt in enumerate(dates):
        for j, code in enumerate(codes):
            rows.append((dt.date(), code, float(price[i, j])))
    return pl.DataFrame(
        rows, schema=["date", "code", "adj_close"], orient="row"
    )


@pytest.fixture
def panel():
    return _make_price_panel()


def test_pivot_adj_shape_and_values():
    panel = _make_price_panel(n_dates=5, n_codes=3, seed=1)
    wide = _pivot_adj(panel, "adj_close")
    assert wide.shape == (5, 3)
    assert set(wide.columns) == {"000000.SH", "000001.SH", "000002.SH"}
    assert list(wide.index) == sorted(wide.index)


def test_runner_init_computes_fwd_ret_and_dates(panel):
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    # 末尾 fwd_days 天没有未来收益，gen_dates 应短于全部日期
    n_all_dates = panel.select("date").unique().height
    assert len(runner.gen_dates) == n_all_dates - 5
    # rebal_dates 是 gen_dates 按 rebal_freq 抽样，长度应为 ceil(len/freq)
    assert len(runner.rebal_dates) == len(range(0, len(runner.gen_dates), 5))
    assert runner.periods_per_year == pytest.approx(252.0 / 5)


def test_gen_alpha_perfect_ic_when_rho_one(panel):
    """ic_mean=1.0, ic_std≈0 时 rho 被 clip 到 0.95（gen_alpha 内部上限），
    合成因子应与未来收益强同向（但因 clip+残差噪声不会到 1.0）。"""
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    alpha = runner.gen_alpha(ic_mean=1.0, ic_std=1e-6, decay=0.0, seed=42)
    result = runner.evaluate(alpha)
    assert result["ic"] > 0.9


def test_gen_alpha_near_zero_ic_when_rho_zero(panel):
    """ic_mean=0, ic_std≈0 时 rho≈0，合成因子应与未来收益基本无关。"""
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    alpha = runner.gen_alpha(ic_mean=0.0, ic_std=1e-6, decay=0.0, seed=42)
    result = runner.evaluate(alpha)
    assert abs(result["ic"]) < 0.15


def test_decay_increases_autocorrelation(panel):
    """decay 越高，相邻期因子排名相关性（autocorr）应越高、换手应越低。"""
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    low_decay = runner.evaluate(
        runner.gen_alpha(ic_mean=0.08, ic_std=0.1, decay=0.0, seed=1)
    )
    high_decay = runner.evaluate(
        runner.gen_alpha(ic_mean=0.08, ic_std=0.1, decay=0.95, seed=1)
    )
    assert high_decay["autocorr"] > low_decay["autocorr"]
    assert high_decay["turnover"] < low_decay["turnover"]


def test_run_point_averages_across_seeds_and_reports_inputs(panel):
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    res = runner.run_point(ic_mean=0.1, ic_std=0.1, decay=0.8, seeds=(1, 2, 3))
    assert res["n_seeds"] == 3
    assert res["in_ic"] == pytest.approx(0.1)
    assert res["in_icir"] == pytest.approx(1.0)
    assert res["in_decay"] == pytest.approx(0.8)
    for key in ("ic", "ic_std", "icir", "ir_theory", "ls_ir", "ls_ann", "autocorr", "turnover"):
        assert key in res


def test_run_grid_shape_and_ic_monotonicity(panel):
    """跑一个精简网格：更高 ic_mean 组的实测 IC 应整体更高（弱单调）。"""
    runner = SignalGridRunner(panel, fwd_days=5, rebal_freq=5, min_names=50)
    df = run_grid(
        runner,
        ic_list=[0.02, 0.15],
        icir_list=[0.8],
        decay_list=[0.0],
        seeds=(1, 2),
        verbose=False,
    )
    assert len(df) == 3  # 2 (ic_icir sweep) + 1 (decay sweep)
    ic_icir_rows = df[df["sweep"] == "ic_icir"].sort_values("in_ic")
    assert ic_icir_rows["ic"].iloc[0] < ic_icir_rows["ic"].iloc[1]
