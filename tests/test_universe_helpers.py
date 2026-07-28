"""候选池辅助函数测试：冲击成本向量、合成 Alpha、北交所/ST 过滤。"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from hqopt.data.generator import MarketDataGenerator
from hqopt.pipeline.universe import (
    build_cost_vector,
    build_synthetic_alpha,
    filter_universe,
)

START = date(2024, 1, 1)


def _price_panel(
    series: dict[str, list[float]],
    amounts: dict[str, float] | None = None,
    n_days: int | None = None,
) -> pl.DataFrame:
    """由每只股票的收盘价序列构造最简面板（date/code/adj_close/amount）。"""
    codes = list(series)
    n_days = n_days or len(next(iter(series.values())))
    rows = []
    for i in range(n_days):
        day = START + timedelta(days=i)
        for code in codes:
            rows.append({
                "date": day,
                "code": code,
                "adj_close": float(series[code][i]),
                "amount": float((amounts or {}).get(code, 1_000_000.0)),
            })
    return pl.DataFrame(rows)


# ── build_cost_vector ────────────────────────────────────────────


def test_cost_vector_falls_back_to_equal_weight_on_short_window():
    """可用交易日不足 5 天时无法估波动，退回等权（全 1）。"""
    panel = _price_panel({"A": [10.0] * 3, "B": [10.0] * 3})

    cost = build_cost_vector(["A", "B"], panel, START + timedelta(days=2))

    np.testing.assert_allclose(cost, [1.0, 1.0])


def test_cost_vector_median_normalized_to_one():
    rng = np.random.default_rng(0)
    series = {
        code: list(100 * np.cumprod(1 + rng.normal(0, 0.02, 20)))
        for code in ("A", "B", "C")
    }
    panel = _price_panel(series)

    cost = build_cost_vector(["A", "B", "C"], panel, START + timedelta(days=19))

    assert np.median(cost) == pytest.approx(1.0)


def test_cost_vector_higher_for_volatile_and_illiquid_names():
    """c_i = σ_i / sqrt(ADV_i)：波动大或成交额小的股票冲击成本更高。"""
    rng = np.random.default_rng(1)
    calm = list(100 * np.cumprod(1 + rng.normal(0, 0.005, 20)))
    wild = list(100 * np.cumprod(1 + rng.normal(0, 0.05, 20)))
    panel = _price_panel(
        {"CALM_LIQUID": calm, "WILD_LIQUID": wild, "CALM_ILLIQUID": calm},
        amounts={
            "CALM_LIQUID": 1e7,
            "WILD_LIQUID": 1e7,
            "CALM_ILLIQUID": 1e4,
        },
    )
    tickers = ["CALM_LIQUID", "WILD_LIQUID", "CALM_ILLIQUID"]

    cost = build_cost_vector(tickers, panel, START + timedelta(days=19))
    by_name = dict(zip(tickers, cost, strict=True))

    assert by_name["WILD_LIQUID"] > by_name["CALM_LIQUID"]      # 同流动性，波动更大 → 更贵
    assert by_name["CALM_ILLIQUID"] > by_name["CALM_LIQUID"]    # 同波动，流动性更差 → 更贵


def test_cost_vector_ignores_data_after_target_date():
    """只能用调仓日及之前的数据——目标日之后的行情不得影响成本估计（防前视）。"""
    rng = np.random.default_rng(2)
    base = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 20)))
    tickers = ["A", "B"]
    target = START + timedelta(days=19)

    quiet_future = _price_panel({"A": base, "B": base[::-1]})
    # 在目标日之后追加剧烈波动的行情
    shocked = pl.concat([
        quiet_future,
        _price_panel(
            {"A": [100.0, 200.0, 50.0], "B": [100.0, 30.0, 300.0]},
            n_days=3,
        ).with_columns(pl.col("date") + pl.duration(days=25)),
    ])

    np.testing.assert_allclose(
        build_cost_vector(tickers, quiet_future, target),
        build_cost_vector(tickers, shocked, target),
    )


def test_cost_vector_fills_missing_tickers_with_one():
    panel = _price_panel({"A": [10.0 + i for i in range(10)]})

    cost = build_cost_vector(["A", "MISSING"], panel, START + timedelta(days=9))

    assert cost[1] == pytest.approx(1.0)


def test_cost_vector_clipped_to_sane_range():
    """极端离群值必须被裁剪到 [0.1, 10]，避免单票惩罚失控。"""
    rng = np.random.default_rng(3)
    flat = [100.0 + 1e-9 * i for i in range(20)]        # 近似零波动
    wild = list(100 * np.cumprod(1 + rng.normal(0, 0.3, 20)))
    panel = _price_panel(
        {"FLAT": flat, "WILD": wild, "MID": list(100 * np.cumprod(1 + rng.normal(0, 0.02, 20)))},
        amounts={"FLAT": 1e9, "WILD": 1e3, "MID": 1e6},
    )

    cost = build_cost_vector(["FLAT", "WILD", "MID"], panel, START + timedelta(days=19))

    assert cost.min() >= 0.1
    assert cost.max() <= 10.0


def test_cost_vector_length_matches_tickers():
    panel = _price_panel({"A": [10.0] * 10, "B": [11.0] * 10})

    assert len(build_cost_vector(["A", "B"], panel, START + timedelta(days=9))) == 2


# ── build_synthetic_alpha ────────────────────────────────────────


def _wide_panel(n_stocks: int = 60, n_days: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"T{i:03d}" for i in range(n_stocks)]
    rows = []
    prices = dict.fromkeys(codes, 100.0)
    for i in range(n_days):
        day = START + timedelta(days=i)
        for c in codes:
            prices[c] *= 1 + rng.normal(0, 0.02)
            rows.append({
                "date": day, "code": c,
                "adj_close": prices[c], "amount": 1e6,
            })
    return pl.DataFrame(rows)


def test_synthetic_alpha_shape_and_index():
    alpha = build_synthetic_alpha(_wide_panel(), fwd_days=5, seed=7)

    assert alpha.index.name == "date"
    assert not alpha.empty
    assert alpha.shape[1] <= 60


def test_synthetic_alpha_is_deterministic_per_seed():
    panel = _wide_panel()
    a = build_synthetic_alpha(panel, fwd_days=5, seed=42)
    b = build_synthetic_alpha(panel, fwd_days=5, seed=42)
    c = build_synthetic_alpha(panel, fwd_days=5, seed=43)

    assert a.equals(b)
    assert not a.equals(c)


def test_synthetic_alpha_is_cross_sectionally_standardized():
    alpha = build_synthetic_alpha(_wide_panel(), fwd_days=5, seed=1)
    row = alpha.iloc[0].dropna()

    assert row.mean() == pytest.approx(0.0, abs=1e-8)
    assert row.std() == pytest.approx(1.0, rel=1e-6)   # 实现用 pandas 默认 ddof=1


def test_synthetic_alpha_correlates_with_future_return_by_construction():
    """合成因子按未来收益构造 → 与未来收益正相关。这正是它含前视、
    只能用于流程标定、绝不可用于真实业绩评估的原因。"""
    panel = _wide_panel(seed=5)
    alpha = build_synthetic_alpha(panel, fwd_days=5, ic_mean=0.5, ic_std=0.01, decay=0.0, seed=9)

    adj = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    fwd = adj.shift(-5) / adj - 1

    day = alpha.index[0]
    common = alpha.columns.intersection(fwd.columns)
    ic = alpha.loc[day, common].corr(fwd.loc[day, common])
    assert ic > 0.2


def test_synthetic_alpha_decay_creates_autocorrelation():
    """decay>0 时相邻期因子自相关显著高于 decay=0。"""
    panel = _wide_panel(seed=11)
    persistent = build_synthetic_alpha(panel, fwd_days=5, decay=0.95, seed=3)
    memoryless = build_synthetic_alpha(panel, fwd_days=5, decay=0.0, seed=3)

    def mean_autocorr(df):
        pairs = [
            df.iloc[i].corr(df.iloc[i + 1])
            for i in range(min(5, len(df) - 1))
        ]
        return float(np.nanmean(pairs))

    assert mean_autocorr(persistent) > mean_autocorr(memoryless)


# ── filter_universe 过滤分支 ─────────────────────────────────────


def _st_panel(tickers: list[str], target: date, st_codes: set[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [target] * len(tickers),
        "code": tickers,
        "is_st": [1 if t in st_codes else 0 for t in tickers],
    })


@pytest.fixture
def snap30():
    return MarketDataGenerator(n_stocks=30, seed=42).generate()


def _rename_tickers(snapshot, renamed: list[str]):
    """把快照里所有按 ticker 索引的 Series 换成新代码（用于构造 .BJ 场景）。"""
    def relabel(series):
        return None if series is None else series.set_axis(renamed)

    return replace(
        snapshot,
        tickers=renamed,
        industry=relabel(snapshot.industry),
        adv=relabel(snapshot.adv),
        status=relabel(snapshot.status),
        prev_weight=relabel(snapshot.prev_weight),
        market_cap=relabel(snapshot.market_cap),
        is_constituent=relabel(snapshot.is_constituent),
        sell_only=relabel(snapshot.sell_only),
    )


def test_filter_universe_excludes_bse_tickers(snap30):
    """北交所（.BJ）必须被剔除。"""
    target = date(2024, 1, 2)
    tickers = list(snap30.tickers)
    renamed = [f"{t.split('.')[0]}.BJ" for t in tickers[:5]] + tickers[5:]
    snap = _rename_tickers(snap30, renamed)
    panel = _st_panel(renamed, target, set())

    kept = filter_universe(snap, panel, target, exclude_bj=True, exclude_st=False)
    assert not any(t.endswith(".BJ") for t in kept.tickers)

    kept_all = filter_universe(snap, panel, target, exclude_bj=False, exclude_st=False)
    assert sum(t.endswith(".BJ") for t in kept_all.tickers) == 5


def test_filter_universe_excludes_st_by_panel_flag(snap30):
    """ST 判定走面板当日 is_st 字段（point-in-time），不用静态名单。"""
    target = date(2024, 1, 2)
    st_codes = set(snap30.tickers[:4])
    panel = _st_panel(list(snap30.tickers), target, st_codes)

    kept = filter_universe(snap30, panel, target, exclude_bj=False, exclude_st=True)

    assert st_codes.isdisjoint(kept.tickers)
    assert len(kept.tickers) == len(snap30.tickers) - 4
