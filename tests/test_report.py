from pathlib import Path

import pandas as pd

from hqopt.backtest.engine import BacktestResult, PerformanceMetrics
from hqopt.backtest.report import (
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
    assert "不做年化" in html
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
