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

from portfolio_optimizer.analysis.attribution import ReturnAttributor
from portfolio_optimizer.risk.attribution_data import FactorReturnLoader
from portfolio_optimizer.risk.cne6_risk import CNE6RiskModel

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


def test_linked_contributions_sum_to_total(attributor, weight_df, benchmark_weight_df):
    """Carino 链接后，各归因项（不含"合计"行本身）累计贡献之和 = 总主动收益累计贡献。"""
    result = attributor.run(weight_df, benchmark_weight_df, _adj_close())
    items_sum = result.summary.drop(index="合计(主动收益)")["累计贡献"].sum()
    total = result.summary.loc["合计(主动收益)", "累计贡献"]
    assert items_sum == pytest.approx(total, abs=1e-9)

    # 累计贡献应等于真实几何累计主动收益 Π(1+a_t)-1
    geo_total = float((1 + result.daily["active_return"]).prod() - 1)
    assert total == pytest.approx(geo_total, abs=1e-9)


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
