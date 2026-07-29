"""独立收益归因 HTML 报告测试。"""

from __future__ import annotations

import pandas as pd

from hqopt.analysis.attribution import AttributionResult
from hqopt.analysis.report import generate_attribution_html


def _result() -> AttributionResult:
    dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
    daily = pd.DataFrame(
        {
            "relative_active_return": [0.001, -0.002, 0.003],
            "coverage_pct": [0.95, 0.75, 0.90],
        },
        index=dates,
    )
    summary = pd.DataFrame(
        {
            "累计贡献": [0.010, -0.002, 0.003, 0.001, -0.001, 0.011],
            "年化贡献": [0.100, -0.020, 0.030, 0.010, -0.010, 0.110],
            "占主动收益%": [90.9, -18.2, 27.3, 9.1, -9.1, 100.0],
            "t统计": [2.1, -0.5, 1.2, 0.3, -0.2, 1.8],
            "年化波动": [0.05, 0.02, 0.03, 0.01, 0.01, 0.06],
        },
        index=[
            "Size",
            "行业合计",
            "特质(选股)",
            "残差",
            "执行影响(含费用)",
            "合计(主动收益)",
        ],
    )
    return AttributionResult(
        factor_daily=pd.DataFrame({"Size": [0.001, 0.0, 0.002]}, index=dates),
        daily=daily,
        summary=summary,
    )


def test_generate_attribution_html_is_offline_and_complete(tmp_path):
    output = tmp_path / "attribution_report.html"

    path = generate_attribution_html(
        _result(),
        output,
        benchmark="zz1000",
        model_name="barra_cne6_S",
    )

    html = path.read_text(encoding="utf-8")
    assert path == output
    assert "收益归因报告" in html
    assert "Carino 链接累计贡献" in html
    assert "特质(选股)" in html
    assert "执行影响(含费用)" in html
    assert "风险模型覆盖率低于 80% 的交易日共 1 日" in html
    assert "plotly.js" in html
    assert "attribution-contribution-chart" in html
    assert "attribution-active-return-chart" in html
    assert "attribution-coverage-chart" in html
    assert '"marker":{"color":"#c0392b"}' in html


def test_generate_attribution_html_escapes_metadata(tmp_path):
    output = tmp_path / "attribution_report.html"

    generate_attribution_html(
        _result(),
        output,
        benchmark="<script>",
        model_name="<model>",
        title="<title>",
    )

    html = output.read_text(encoding="utf-8")
    assert "基准：<script>" not in html
    assert "<h1><title>" not in html
    assert "&lt;SCRIPT&gt;" in html
    assert "&lt;model&gt;" in html
    assert "&lt;title&gt;" in html


def test_generate_attribution_html_labels_equal_weight_in_chinese(tmp_path):
    output = tmp_path / "attribution_report.html"

    generate_attribution_html(
        _result(),
        output,
        benchmark="equal_weight",
        model_name="barra_cne6_S",
    )

    html = output.read_text(encoding="utf-8")
    assert "基准：全市场等权" in html
    assert "EQUAL_WEIGHT" not in html
