"""收益归因单测（手算合成数据，验证分解算式而非依赖真实 ClickHouse 数据）。

构造 2 只票（A/B）、2 个因子（Size/Country）、2 个调仓期的最小场景，
使个股收益恰好满足 Barra 恒等式 r_i = X_i·f + u_i，令残差严格为 0——
这样残差校验能证明「分解算式本身正确」，与真实数据里因覆盖缺口产生
的非零残差（见 attribution.py 顶部说明）区分开。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import hqopt.analysis.run as attribution_run
from hqopt.analysis.attribution import ReturnAttributor
from hqopt.backtest.engine import RealisticBacktester
from hqopt.data.benchmark import (
    benchmark_returns_from_rebalance_weights,
    equal_weight_benchmark_weights,
)
from hqopt.risk.attribution_data import FactorReturnLoader
from hqopt.risk.cne6_risk import CNE6RiskModel

FACTORS = ["Size", "Country"]

# 固定的因子收益 / 特质收益（每日不变，便于手算）
F_SIZE = 0.01     # Size 因子日收益
F_COUNTRY = 0.0   # Country 因子日收益（设 0，隔离验证 Country 恒为 0 的性质）
U_A = 0.002       # A 特质日收益
U_B = -0.001      # B 特质日收益

# 个股暴露：A 高 Size 暴露，B 低（对称），Country 恒为 1（全市场共同因子）
EXPOSURE = {"A": {"Size": 1.0, "Country": 1.0}, "B": {"Size": -1.0, "Country": 1.0}}

REBAL_DATES = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")]
# 5 个交易日：01-03/04/05 属期 1（T0 权重持有），01-08/09 属期 2（T1 权重持有）
TRADING_DATES = [
    pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05"),
    pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09"),
]
BASE_DATE = pd.Timestamp("2024-01-02")   # adj_close 基准日（供首日 pct_change 用）


def _write_risk_panels(data_dir: Path) -> None:
    """构造最小 exposure_panel / factor_cov_panel（两个调仓日暴露相同，简化手算）。"""
    exp_rows = []
    for rdate in REBAL_DATES:
        for code, exp in EXPOSURE.items():
            exp_rows.append({
                "rebal_date": rdate.date(), "code": code,
                "Size": exp["Size"], "Country": exp["Country"], "spec_var": 0.01,
            })
    pl.DataFrame(exp_rows).write_parquet(data_dir / "exposure_panel.parquet")

    cov_rows = []
    for rdate in REBAL_DATES:
        cov_rows.append({"rebal_date": rdate.date(), "factor": "Size", "Size": 0.01, "Country": 0.0})
        cov_rows.append({"rebal_date": rdate.date(), "factor": "Country", "Size": 0.0, "Country": 0.01})
    pl.DataFrame(cov_rows).write_parquet(data_dir / "factor_cov_panel.parquet")


def _write_factor_return(data_dir: Path) -> None:
    rows = []
    for d in TRADING_DATES:
        rows.append({"trade_date": d.date(), "factor_name": "Size", "ret": F_SIZE})
        rows.append({"trade_date": d.date(), "factor_name": "Country", "ret": F_COUNTRY})
    pl.DataFrame(rows).write_parquet(data_dir / "factor_return.parquet")


def _write_specific_return(data_dir: Path) -> None:
    rows = []
    for d in TRADING_DATES:
        rows.append({"trade_date": d.date(), "code": "A", "u": U_A})
        rows.append({"trade_date": d.date(), "code": "B", "u": U_B})
    pl.DataFrame(rows).write_parquet(data_dir / "specific_return.parquet")


def _adj_close() -> pd.DataFrame:
    """构造满足 r_i = X_i·f + u_i 恒等式的价格序列（常数日收益 → cumprod）。"""
    r_a = EXPOSURE["A"]["Size"] * F_SIZE + EXPOSURE["A"]["Country"] * F_COUNTRY + U_A
    r_b = EXPOSURE["B"]["Size"] * F_SIZE + EXPOSURE["B"]["Country"] * F_COUNTRY + U_B
    dates = [BASE_DATE, *TRADING_DATES]
    price_a = [100.0]
    price_b = [100.0]
    for _ in TRADING_DATES:
        price_a.append(price_a[-1] * (1 + r_a))
        price_b.append(price_b[-1] * (1 + r_b))
    return pd.DataFrame({"A": price_a, "B": price_b}, index=pd.DatetimeIndex(dates))


@pytest.fixture
def attributor(tmp_path) -> ReturnAttributor:
    _write_risk_panels(tmp_path)
    _write_factor_return(tmp_path)
    _write_specific_return(tmp_path)
    risk_model = CNE6RiskModel(data_dir=tmp_path)
    factor_loader = FactorReturnLoader(data_dir=tmp_path)
    return ReturnAttributor(risk_model, factor_loader)


@pytest.fixture
def weight_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [0.6, 0.7], "B": [0.4, 0.3]}, index=REBAL_DATES,
    )


@pytest.fixture
def benchmark_weight_df() -> pd.DataFrame:
    # 单一快照（早于两期调仓日），asof 对两期均生效：基准恒 50/50
    return pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[REBAL_DATES[0]])


def test_residual_zero_when_fully_covered(attributor, weight_df, benchmark_weight_df):
    """暴露/收益完全对齐（满足 Barra 恒等式）时，残差应严格为 0。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    assert result.daily["residual"].abs().max() < 1e-10
    assert result.summary.loc["残差", "累计贡献"] == pytest.approx(0.0, abs=1e-10)


def test_equal_weight_benchmark_return_matches_shared_backtest_contract(
    attributor,
    weight_df,
):
    prices = _adj_close()
    benchmark_weights = equal_weight_benchmark_weights(prices)
    expected = benchmark_returns_from_rebalance_weights(
        benchmark_weights,
        prices,
        list(weight_df.index),
    )

    result = attributor.run(weight_df, benchmark_weights, prices)

    pd.testing.assert_series_equal(
        result.daily["benchmark_return"],
        expected.reindex(result.daily.index),
        check_names=False,
    )


def test_style_contribution_matches_hand_calc(attributor, weight_df, benchmark_weight_df):
    """期 1（T0 权重持有 3 天）：主动 Size 暴露 0.2，风格贡献每日应为 0.2*0.01=0.002。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    period1_days = result.daily.index[:3]   # 01-03/04/05
    period2_days = result.daily.index[3:]   # 01-08/09

    assert result.daily.loc[period1_days, "style_total"].values == pytest.approx(0.002, abs=1e-9)
    # 期 2：主动 Size 暴露 0.4（0.7-0.5 vs 0.3-0.5），贡献 0.4*0.01=0.004
    assert result.daily.loc[period2_days, "style_total"].values == pytest.approx(0.004, abs=1e-9)


def test_specific_contribution_matches_hand_calc(attributor, weight_df, benchmark_weight_df):
    """期 1 特质贡献 = 0.1*0.002 + (-0.1)*(-0.001) = 0.0003；期 2 = 0.2*0.002+(-0.2)*(-0.001)=0.0006。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    period1_days = result.daily.index[:3]
    period2_days = result.daily.index[3:]

    assert result.daily.loc[period1_days, "specific"].values == pytest.approx(0.0003, abs=1e-9)
    assert result.daily.loc[period2_days, "specific"].values == pytest.approx(0.0006, abs=1e-9)


def test_country_contribution_is_zero(attributor, weight_df, benchmark_weight_df):
    """组合、基准均满仓（权重和=1）时，Country 主动暴露恒为 0（与因子收益无关）。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    assert result.daily["country"].abs().max() < 1e-10


def test_style_direction_is_positive(attributor, weight_df, benchmark_weight_df):
    """组合超配 Size 暴露高的 A、低配 B，Size 因子收益为正 → Size 贡献方向应为正。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    assert result.summary.loc["Size", "累计贡献"] > 0
    assert result.summary.loc["Size", "t统计"] > 0


def test_linked_contributions_sum_to_exact_geometric_excess(
    attributor, weight_df, benchmark_weight_df
):
    """链接贡献之和严格等于组合/基准净值比，而非主动日差的近似复利。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    items_sum = result.summary.drop(index="合计(主动收益)")["累计贡献"].sum()
    total = result.summary.loc["合计(主动收益)", "累计贡献"]
    assert items_sum == pytest.approx(total, abs=1e-9)

    portfolio_growth = float((1.0 + result.daily["portfolio_return"]).prod())
    benchmark_growth = float((1.0 + result.daily["benchmark_return"]).prod())
    exact_geometric_excess = portfolio_growth / benchmark_growth - 1.0
    legacy_approximation = float((1.0 + result.daily["active_return"]).prod() - 1.0)
    assert total == pytest.approx(exact_geometric_excess, abs=1e-9)
    assert float((1.0 + result.daily["relative_active_return"]).prod() - 1.0) \
        == pytest.approx(exact_geometric_excess, abs=1e-9)
    assert exact_geometric_excess != pytest.approx(legacy_approximation, abs=1e-9)


def test_coverage_gap_leaks_into_residual(attributor, weight_df, benchmark_weight_df):
    """引入模型未覆盖的第三只票 C（持仓有权重，但不在暴露/特质收益数据中），
    其收益应完整计入"真实主动收益"却在归因项里查不到 → 残差非零，
    coverage_pct < 1，证明覆盖缺口确实会漏进残差（而非静默吞掉）。"""
    wdf = weight_df.copy()
    wdf["C"] = [0.0, 0.1]   # 仅第二期持有 C，且 C 不在风险模型/因子收益数据里
    wdf["A"] = [0.6, 0.6]
    wdf["B"] = [0.4, 0.3]

    ac = _adj_close().copy()
    ac["C"] = [100.0] * len(ac)
    ac.loc[ac.index[-2:], "C"] = [110.0, 121.0]   # C 在期 2 两天各涨 10%

    result = attributor.run(wdf, benchmark_weight_df, ac)
    period2_days = result.daily.index[3:]
    assert (result.daily.loc[period2_days, "coverage_pct"] < 1.0).all()
    assert result.daily.loc[period2_days, "residual"].abs().min() > 1e-6


def test_risk_snapshot_preserves_missing_name_coverage(tmp_path):
    """风险模型填数仅用于数值稳定，不能把未覆盖股票伪装成零暴露已覆盖。"""
    _write_risk_panels(tmp_path)
    risk_model = CNE6RiskModel(data_dir=tmp_path)

    snapshot = risk_model.at(REBAL_DATES[0].date(), ["A", "MISSING"])

    assert snapshot is not None
    assert snapshot.covered_mask.tolist() == [True, False]
    assert snapshot.X[1].tolist() == pytest.approx([0.0, 0.0])
    assert snapshot.delta[1] == pytest.approx(0.01)


def test_actual_execution_weights_override_targets(attributor, weight_df, benchmark_weight_df):
    """目标未成交时，归因必须使用实际现金状态，而不是假设目标已持有。"""
    actual = pd.DataFrame(0.0, index=[BASE_DATE, *TRADING_DATES], columns=["A", "B"])
    actual.loc[TRADING_DATES[3]:, ["A", "B"]] = [0.7, 0.3]

    result = attributor.run(
        weight_df,
        benchmark_weight_df,
        _adj_close(),
        actual_weight_df=actual,
    )

    # 期 1 实际全现金：主动权重为 A/B 各 -0.5，特质贡献为 -0.0005；
    # 若错误使用目标 60/40，则会得到 +0.0003。
    assert result.daily.iloc[:3]["specific"].values == pytest.approx(-0.0005, abs=1e-9)


def test_attribution_coverage_uses_original_mask(tmp_path):
    """单个因子缺失但其余暴露非零时，也必须按未覆盖处理。"""
    _write_risk_panels(tmp_path)
    exposure_path = tmp_path / "exposure_panel.parquet"
    exposure = pl.read_parquet(exposure_path).with_columns(
        pl.when(pl.col("code") == "B")
        .then(None)
        .otherwise(pl.col("Size"))
        .alias("Size")
    )
    exposure.write_parquet(exposure_path)
    _write_factor_return(tmp_path)
    _write_specific_return(tmp_path)
    attributor = ReturnAttributor(
        CNE6RiskModel(data_dir=tmp_path),
        FactorReturnLoader(data_dir=tmp_path),
    )
    portfolio = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]}, index=REBAL_DATES)
    benchmark = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[REBAL_DATES[0]])

    result = attributor.run(portfolio, benchmark, _adj_close())

    # 主动 L1 权重 A/B 各 1；仅 A 完整覆盖，因此覆盖率为 50%。
    assert result.daily["coverage_pct"].values == pytest.approx(0.5, abs=1e-9)


def test_run_attribution_replays_blocked_execution(tmp_path, monkeypatch):
    """归因主路径应重放成交；首期全部停牌时必须识别为现金，而非目标持仓。"""
    _write_risk_panels(tmp_path)
    _write_factor_return(tmp_path)
    _write_specific_return(tmp_path)
    weights = pd.DataFrame(
        {"A": [0.6, 0.7], "B": [0.4, 0.3]}, index=REBAL_DATES
    )
    weight_path = tmp_path / "weights.parquet"
    weights.to_parquet(weight_path)

    prices = _adj_close()
    rows = []
    for d in prices.index:
        for code in ("A", "B"):
            price = float(prices.loc[d, code])
            rows.append({
                "date": d.date(),
                "code": code,
                "adj_close": price,
                "adj_vwap": price,
                "close": price,
                "limit_up": price * 2,
                "limit_down": price / 2,
                "trade_status": "停牌" if d in TRADING_DATES[:3] else "交易",
            })
    panel = pl.DataFrame(rows)
    monkeypatch.setattr(attribution_run, "load_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr(
        attribution_run,
        "FactorReturnLoader",
        lambda data_dir=None: FactorReturnLoader(data_dir=tmp_path),
    )

    result = attribution_run.run_attribution(
        weight_path,
        BASE_DATE.date(),
        TRADING_DATES[-1].date(),
        index="equal_weight",
        cne6_data_dir=tmp_path,
        cost_buy=0.0,
        cost_sell=0.0,
    )

    assert result.daily.iloc[:3]["specific"].values == pytest.approx(-0.0005, abs=1e-9)


# ═══════════════════════════════════════════════════════════════
#  归因组合收益必须与真实净值一致（防同日权重前视）
# ═══════════════════════════════════════════════════════════════


def _drifting_actual_weights(adj_close: pd.DataFrame, w0: dict) -> pd.DataFrame:
    """买入持有下逐日**收盘后**的实际权重——与回测账本的写入时点一致。

    回测引擎在当日 mark-to-market 之后才记录 actual_weight（backtest/engine.py），
    因此每一行都已含当日涨跌，随价格逐日漂移。这正是能照出同日自我引用的形态；
    旧测试用的分段常数权重不漂移，照不出来。
    """
    px = adj_close.loc[[BASE_DATE, *TRADING_DATES]]
    shares = pd.Series(w0) / px.loc[BASE_DATE]
    value = px.mul(shares, axis=1)
    return value.div(value.sum(axis=1), axis=0)


def test_portfolio_return_reproduces_buy_and_hold_nav(
    attributor, weight_df, benchmark_weight_df
):
    """归因的组合日收益连乘必须等于真实买入持有净值。

    这是防前视的核心断言：``actual_weight_df`` 的第 d 行是 d 日**收盘后**权重，
    必须配 d+1 日收益。若错配成同日，会凭空多出 Σ w_i·r_i²（截面收益方差，
    恒为正），实测可达 +11%/年，且主要堆进「特质(选股)」项。

    注意残差自检（test_residual_zero_when_fully_covered）挡不住这个 bug：
    explained 与 active_return 用同一个有偏权重，偏差两边同步出现、相减抵消，
    残差恒为 0 而两边都错。故必须直接对齐外部真值——净值。
    """
    adj_close = _adj_close()
    w0 = {"A": 0.6, "B": 0.4}
    actual = _drifting_actual_weights(adj_close, w0)

    result = attributor.run(
        weight_df, benchmark_weight_df, adj_close, actual_weight_df=actual,
    )

    # 真值：期初 w0 买入后持有不动，区间末净值（区间 = 首个调仓日之后的全部交易日）
    px = adj_close.loc[[BASE_DATE, *TRADING_DATES]]
    shares = pd.Series(w0) / px.loc[BASE_DATE]
    nav = px.mul(shares, axis=1).sum(axis=1)
    true_total = nav.iloc[-1] / nav.iloc[0] - 1.0

    attributed = (1.0 + result.daily["portfolio_return"]).prod() - 1.0
    assert attributed == pytest.approx(true_total, abs=1e-12), (
        f"归因组合收益 {attributed:+.6%} 与真实净值 {true_total:+.6%} 不符；"
        "差额恒为正通常意味着用了同日权重（前视）"
    )


def test_execution_effect_reconciles_delayed_vwap_and_fee_nav(
    attributor, benchmark_weight_df
):
    """延期到 T+3 且 VWAP≠收盘、费用非零时，归因组合收益仍须等于回测 NAV。"""
    adj_close = _adj_close()
    weight_df = pd.DataFrame(
        {"A": [0.6], "B": [0.4]},
        index=[BASE_DATE],
    )
    adj_vwap = adj_close.copy()
    execution_day = TRADING_DATES[2]
    adj_vwap.loc[execution_day] = adj_close.loc[execution_day] * 0.95
    close_raw = adj_close.copy()
    limit_up = adj_close * 2.0
    limit_down = adj_close * 0.5
    trade_status = pd.DataFrame(
        "交易",
        index=adj_close.index,
        columns=adj_close.columns,
    )
    trade_status.loc[TRADING_DATES[:2]] = "停牌"

    backtest_result, _ = RealisticBacktester(
        cost_buy=0.01,
        cost_sell=0.02,
        risk_free=0.0,
    ).run(
        weight_df=weight_df,
        adj_close=adj_close,
        adj_vwap=adj_vwap,
        close_raw=close_raw,
        limit_up_df=limit_up,
        limit_down_df=limit_down,
        trade_status_df=trade_status,
        benchmark_ret=pd.Series(0.0, index=adj_close.index),
        initial_value=100.0,
    )
    result = attributor.run(
        weight_df,
        benchmark_weight_df,
        adj_close,
        actual_weight_df=backtest_result.actual_weights,
        realized_portfolio_return=backtest_result.daily_ret,
    )

    assert backtest_result.actual_weights.loc[TRADING_DATES[1]].sum() == 0.0
    assert backtest_result.actual_weights.loc[execution_day].sum() > 0.9
    assert abs(result.daily.loc[execution_day, "execution_effect"]) > 1e-6

    attributed = float((1.0 + result.daily["portfolio_return"]).prod() - 1.0)
    true_total = float(
        backtest_result.nav.loc[result.daily.index[-1]]
        / backtest_result.nav.loc[BASE_DATE]
        - 1.0
    )
    assert attributed == pytest.approx(true_total, abs=1e-12)

    decomposed = (
        result.daily["explained"]
        + result.daily["residual"]
        + result.daily["execution_effect"]
    )
    assert decomposed.values == pytest.approx(
        result.daily["active_return"].values,
        abs=1e-12,
    )
    linked_items = result.summary.drop(index="合计(主动收益)")["累计贡献"].sum()
    assert linked_items == pytest.approx(
        result.summary.loc["合计(主动收益)", "累计贡献"],
        abs=1e-12,
    )


def test_same_day_weight_would_overstate_return(attributor, weight_df, benchmark_weight_df):
    """反向锁：同日权重口径确实高估，证明上一条断言不是恒真的空断言。"""
    adj_close = _adj_close()
    w0 = {"A": 0.6, "B": 0.4}
    actual = _drifting_actual_weights(adj_close, w0)
    daily_ret = adj_close.pct_change(fill_method=None)

    same_day, lagged = 1.0, 1.0
    for d in TRADING_DATES:
        r = daily_ret.loc[d]
        same_day *= 1.0 + float((actual.loc[d] * r).sum())
        prev = actual.index[actual.index.get_loc(d) - 1]
        lagged *= 1.0 + float((actual.loc[prev] * r).sum())

    assert same_day > lagged, "同日权重必须高估，否则这组测试数据照不出该 bug"
