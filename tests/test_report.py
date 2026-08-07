from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import hqopt.backtest.report as report_module
from hqopt.backtest.engine import BacktestResult, PerformanceMetrics
from hqopt.backtest.report import (
    _build_holdings_table,
    _turnover_summary,
    _yearly_turnover_table,
    generate_html_report,
)


def _metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        annual_return=0.10,
        annual_vol=0.20,
        sharpe=0.5,
        max_drawdown=-0.10,
        calmar=1.0,
        win_rate_monthly=0.6,
        avg_monthly_excess=0.01,
        annual_excess_return=0.03,
        info_ratio=0.8,
        tracking_error=0.05,
        excess_max_drawdown=-0.04,
        excess_calmar=0.75,
    )


def _result() -> BacktestResult:
    idx = pd.bdate_range("2020-01-02", periods=504)
    daily_ret = pd.Series(0.0002, index=idx)
    bm_ret = pd.Series(0.0001, index=idx)
    nav = (1 + daily_ret).cumprod()
    bm_nav = (1 + bm_ret).cumprod()
    excess_nav = nav / bm_nav
    turnover = pd.Series(
        [0.10, 0.20, 0.15, 0.05],
        index=pd.to_datetime(["2020-03-31", "2020-09-30", "2021-03-31", "2021-09-30"]),
        name="turnover",
    )
    return BacktestResult(
        nav=nav,
        bm_nav=bm_nav,
        excess_nav=excess_nav,
        daily_ret=daily_ret,
        bm_ret=bm_ret,
        excess_ret=daily_ret - bm_ret,
        turnover=turnover,
        portfolio_metrics=_metrics(),
        benchmark_metrics=_metrics(),
    )


def test_turnover_summary_counts_rebalance_periods_not_fill_days():
    """一次调仓的分次成交不得各计一期。

    成交可跨 T+1/T+2/T+3，逐成交日序列因此比调仓期多行。若报告口径误用它，
    ``mean()`` 分母偏大→平均换手被系统性低估，``rebalances`` 也会多计。
    缺这条语义测试正是该缺陷此前得以存活的原因。
    """
    result = _result()
    # 4 次调仓，其中两次各有一笔 T+2 续成交 → 逐成交日 6 行，总量与按期一致。
    result.turnover = pd.Series(
        [0.08, 0.02, 0.20, 0.12, 0.03, 0.05],
        index=pd.to_datetime([
            "2020-03-31", "2020-04-01",
            "2020-09-30",
            "2021-03-31", "2021-04-01",
            "2021-09-30",
        ]),
        name="turnover",
    )
    result.turnover_by_rebalance = pd.Series(
        [0.10, 0.20, 0.15, 0.05],
        index=pd.to_datetime(["2020-03-31", "2020-09-30", "2021-03-31", "2021-09-30"]),
        name="turnover",
    )

    summary = _turnover_summary(result)
    assert summary["rebalances"] == 4.0                      # 不是 6
    assert abs(summary["avg_turnover"] - 0.125) < 1e-9       # 0.50/4，不是 0.50/6
    assert abs(summary["annualized_turnover"] - 0.25) < 1e-9  # 总量不受聚合影响

    yearly = _yearly_turnover_table(result)
    assert yearly.loc[2020, "rebalance_count"] == 2          # 不是 3
    assert abs(yearly.loc[2020, "avg_turnover"] - 0.15) < 1e-9


def test_turnover_summary_and_yearly_turnover():
    result = _result()
    summary = _turnover_summary(result)
    assert abs(summary["annualized_turnover"] - 0.25) < 1e-9
    assert abs(summary["avg_turnover"] - 0.125) < 1e-9
    assert abs(summary["rebalances_per_year"] - 2.0) < 1e-9

    yearly = _yearly_turnover_table(result)
    assert list(yearly.index) == [2020, 2021]
    assert yearly.loc[2020, "rebalance_count"] == 2
    assert abs(yearly.loc[2020, "turnover"] - 0.30) < 1e-9
    assert abs(yearly.loc[2021, "avg_turnover"] - 0.10) < 1e-9


def test_generate_html_report_includes_turnover_summary(tmp_path: Path):
    result = _result()
    out = tmp_path / "report.html"
    generate_html_report(result, output_path=out, title="测试报告")

    html = out.read_text(encoding="utf-8")
    assert "年化双边换手率" in html
    assert "平均单期双边换手" in html
    assert "调仓次数" in html
    assert "双边换手" in html
    assert "所有 Calmar 均为对应区间" in html
    assert "CAGR/MDD" in html
    assert '"marker":{"color":"#c0392b"' in html

    yearly = pd.read_parquet(tmp_path / "report_data" / "yearly.parquet")
    assert {"rebalance_count", "turnover", "avg_turnover"}.issubset(yearly.columns)


def test_generate_html_report_renders_escaped_warning_banner(tmp_path: Path):
    out = tmp_path / "report.html"

    generate_html_report(
        _result(),
        output_path=out,
        warning_banner="含未来信息，回测业绩完全不可信\n<script>alert(1)</script>",
    )

    html = out.read_text(encoding="utf-8")
    assert 'class="warning-banner"' in html
    assert "含未来信息，回测业绩完全不可信" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def _fake_panel(target_date) -> pl.DataFrame:
    """模拟 load_panel 返回的截面：三只持仓 + 一只当日缺行情（退市滞留仓）。"""
    return pl.DataFrame({
        "code":         ["SZ000001", "SZ000002", "SZ000003"],
        "date":         [target_date] * 3,
        "name":         ["平安银行", "万科A", "国农科技"],
        "industry_l1":  ["银行", "房地产", "农林牧渔"],
        "total_mv":     [3_000_000.0, 1_500_000.0, 80_000.0],  # 万元
    })


def _result_with_holdings() -> BacktestResult:
    result = _result()
    idx = pd.to_datetime(["2020-03-31", "2020-09-30"])
    result.actual_weights = pd.DataFrame(
        {
            "SZ000001": [0.30, 0.40],
            "SZ000002": [0.30, 0.35],
            "SZ000003": [0.20, 0.20],
            "SZ000004": [0.10, 0.0],   # 末期权重≈0，应被过滤
        },
        index=idx,
    )
    return result


def test_build_holdings_table_empty_without_actual_weights():
    html, df = _build_holdings_table(_result())
    assert html == ""
    assert df is None


def test_build_holdings_table_joins_name_industry_market_cap(monkeypatch):
    monkeypatch.setattr(
        report_module, "load_panel", lambda t1, t2, columns, cache_dir=None: _fake_panel(t1)
    )
    result = _result_with_holdings()

    html, df = _build_holdings_table(result)

    assert "2020-09-30" in html
    assert "平安银行" in html and "银行" in html
    assert "150.0" in html          # 万科A 总市值 1,500,000万 = 150.0亿
    assert "SZ000004" not in html   # 权重≈0 被过滤

    assert df is not None
    assert set(df["code"]) == {"SZ000001", "SZ000002", "SZ000003"}
    row = df.set_index("code").loc["SZ000002"]
    assert row["weight"] == pytest.approx(0.35)
    assert row["mv_yi"] == pytest.approx(150.0)


def test_generate_html_report_includes_holdings_section(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        report_module, "load_panel", lambda t1, t2, columns, cache_dir=None: _fake_panel(t1)
    )
    result = _result_with_holdings()
    out = tmp_path / "report.html"

    generate_html_report(result, output_path=out, title="测试报告")

    html = out.read_text(encoding="utf-8")
    assert "最后一期持仓明细（2020-09-30）" in html
    assert "万科A" in html

    holdings = pd.read_parquet(tmp_path / "report_data" / "holdings.parquet")
    assert set(holdings["code"]) == {"SZ000001", "SZ000002", "SZ000003"}
