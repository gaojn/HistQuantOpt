"""收益归因 HTML 报告生成器。"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hqopt.analysis.attribution import AttributionResult

_TOTAL_ROW = "合计(主动收益)"
_LOW_COVERAGE = 0.80
_POSITIVE = "#c0392b"
_NEGATIVE = "#27ae60"
_NEUTRAL = "#3498db"


def _summary_value(
    result: AttributionResult,
    row: str,
    column: str,
) -> float:
    if row not in result.summary.index or column not in result.summary.columns:
        return 0.0
    value = float(result.summary.loc[row, column])
    return value if np.isfinite(value) else 0.0


def _signed_class(value: float) -> str:
    return "positive" if value >= 0.0 else "negative"


def _metric_cards(result: AttributionResult) -> str:
    total = _summary_value(result, _TOTAL_ROW, "累计贡献")
    annual = _summary_value(result, _TOTAL_ROW, "年化贡献")
    specific = _summary_value(result, "特质(选股)", "累计贡献")
    residual = _summary_value(result, "残差", "累计贡献")
    execution = _summary_value(result, "执行影响(含费用)", "累计贡献")
    coverage = pd.to_numeric(
        result.daily.get("coverage_pct", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    coverage_mean = float(coverage.mean()) if len(coverage) else float("nan")

    cards = (
        ("累计主动收益", total, f"{total * 100:+.2f}%", "Carino 精确链接"),
        ("年化主动贡献", annual, f"{annual * 100:+.2f}%", "按归因交易日年化"),
        ("特质（选股）", specific, f"{specific * 100:+.2f}%", "累计链接贡献"),
        ("模型残差", residual, f"{residual * 100:+.2f}%", "累计链接贡献"),
        ("执行影响", execution, f"{execution * 100:+.2f}%", "含费用与成交偏差"),
    )
    html = '<div class="metric-grid">'
    for title, value, display, sub in cards:
        html += (
            '<div class="metric-card">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value {_signed_class(value)}">{display}</div>'
            f'<div class="metric-sub">{sub}</div>'
            "</div>"
        )
    coverage_display = (
        f"{coverage_mean * 100:.1f}%" if np.isfinite(coverage_mean) else "—"
    )
    html += (
        '<div class="metric-card">'
        '<div class="metric-title">平均风险模型覆盖率</div>'
        f'<div class="metric-value">{coverage_display}</div>'
        f'<div class="metric-sub">低于 80%：{int((coverage < _LOW_COVERAGE).sum())} 日</div>'
        "</div></div>"
    )
    return html


def _contribution_chart(result: AttributionResult) -> str:
    summary = result.summary.drop(index=_TOTAL_ROW, errors="ignore")
    values = pd.to_numeric(summary["累计贡献"], errors="coerce").fillna(0.0) * 100
    colors = [_POSITIVE if value >= 0.0 else _NEGATIVE for value in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[str(value) for value in values.index],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>累计贡献=%{x:.3f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_color="#7f8c8d", line_width=1)
    fig.update_layout(
        height=max(430, 27 * len(values) + 120),
        template="plotly_white",
        xaxis_title="累计贡献（%）",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=30, t=25, b=45),
        showlegend=False,
    )
    return fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        div_id="attribution-contribution-chart",
    )


def _active_return_chart(result: AttributionResult) -> str:
    daily = result.daily.sort_index()
    relative = pd.to_numeric(
        daily["relative_active_return"], errors="coerce"
    ).fillna(0.0)
    cumulative = (1.0 + relative).cumprod() - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.10,
        subplot_titles=("累计主动收益", "逐日相对主动收益"),
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative * 100,
            mode="lines",
            name="累计主动收益",
            line=dict(color=_NEUTRAL, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=relative.index,
            y=relative * 100,
            name="逐日主动收益",
            marker_color=_POSITIVE,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_color="#95a5a6", line_width=1, row=2, col=1)
    fig.update_layout(
        height=610,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=55, r=30, t=55, b=40),
    )
    fig.update_yaxes(title_text="累计（%）", row=1, col=1)
    fig.update_yaxes(title_text="单日（%）", row=2, col=1)
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="attribution-active-return-chart",
    )


def _coverage_chart(result: AttributionResult) -> str:
    coverage = pd.to_numeric(
        result.daily["coverage_pct"], errors="coerce"
    ).sort_index()
    fig = go.Figure(
        go.Scatter(
            x=coverage.index,
            y=coverage * 100,
            mode="lines",
            line=dict(color="#8e44ad", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(142,68,173,0.10)",
            hovertemplate="%{x|%Y-%m-%d}<br>覆盖率=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=_LOW_COVERAGE * 100,
        line_dash="dash",
        line_color="#d97706",
        annotation_text="80% 提示线",
    )
    fig.update_layout(
        height=310,
        template="plotly_white",
        yaxis_title="覆盖率（%）",
        yaxis_range=[0, 102],
        margin=dict(l=55, r=30, t=30, b=40),
        showlegend=False,
    )
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="attribution-coverage-chart",
    )


def _summary_table(result: AttributionResult) -> str:
    columns = (
        "累计贡献",
        "年化贡献",
        "占主动收益%",
        "t统计",
        "年化波动",
    )
    html = (
        '<div class="table-wrap"><table class="stats-table"><thead><tr>'
        "<th>归因项</th>"
        + "".join(f"<th>{column}</th>" for column in columns)
        + "</tr></thead><tbody>"
    )
    for name, row in result.summary.iterrows():
        row_class = ' class="total-row"' if name == _TOTAL_ROW else ""
        cumulative = float(row.get("累计贡献", np.nan))
        annual = float(row.get("年化贡献", np.nan))
        active_pct = float(row.get("占主动收益%", np.nan))
        t_stat = float(row.get("t统计", np.nan))
        annual_vol = float(row.get("年化波动", np.nan))

        def pct(value: float, *, already_percent: bool = False) -> str:
            if not np.isfinite(value):
                return "—"
            scaled = value if already_percent else value * 100
            return f"{scaled:+.2f}%"

        def number(value: float) -> str:
            return f"{value:+.2f}" if np.isfinite(value) else "—"

        color_class = _signed_class(cumulative) if np.isfinite(cumulative) else ""
        html += (
            f"<tr{row_class}>"
            f"<td>{escape(str(name))}</td>"
            f'<td class="{color_class}">{pct(cumulative)}</td>'
            f"<td>{pct(annual)}</td>"
            f"<td>{pct(active_pct, already_percent=True)}</td>"
            f"<td>{number(t_stat)}</td>"
            f"<td>{pct(annual_vol)}</td>"
            "</tr>"
        )
    return html + "</tbody></table></div>"


_CSS = """
<style>
  body {
    margin: 0; padding: 24px; background: #f5f6f8; color: #2c3e50;
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  .container { max-width: 1300px; margin: 0 auto; }
  h1 { margin: 0 0 5px; font-size: 28px; font-weight: 650; }
  .subtitle { color: #7f8c8d; margin-bottom: 20px; font-size: 13px; }
  .warning {
    background: #fff3cd; border-left: 4px solid #d97706; border-radius: 6px;
    color: #7c2d12; padding: 12px 14px; margin-bottom: 18px; font-size: 13px;
  }
  .section {
    background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .section h2 {
    margin: 0 0 16px; font-size: 17px; color: #34495e;
    border-left: 3px solid #3498db; padding-left: 10px;
  }
  .metric-grid {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;
  }
  .metric-card {
    background: #fafbfc; border-radius: 6px; padding: 14px;
    border-left: 3px solid #95a5a6;
  }
  .metric-title { font-size: 12px; color: #7f8c8d; margin-bottom: 6px; }
  .metric-value { font-size: 23px; font-weight: 650; }
  .metric-sub { font-size: 11px; color: #95a5a6; margin-top: 4px; }
  .positive { color: #c0392b; }
  .negative { color: #27ae60; }
  .table-wrap { overflow-x: auto; }
  .stats-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .stats-table th, .stats-table td {
    padding: 8px 12px; text-align: right; border-bottom: 1px solid #ecf0f1;
    white-space: nowrap;
  }
  .stats-table th {
    background: #f8f9fa; color: #34495e; border-bottom: 2px solid #bdc3c7;
  }
  .stats-table th:first-child, .stats-table td:first-child { text-align: left; }
  .stats-table .total-row { background: #fdf6e3; font-weight: 650; }
  .note { color: #667085; font-size: 12px; line-height: 1.7; }
  @media (max-width: 800px) {
    body { padding: 12px; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
</style>
"""


def generate_attribution_html(
    result: AttributionResult,
    output_path: str | Path = "output/attribution_report.html",
    *,
    benchmark: str,
    model_name: str,
    title: str = "收益归因报告",
) -> Path:
    """生成离线可查看的独立收益归因 HTML 报告。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily = result.daily.sort_index()
    start, end = daily.index.min(), daily.index.max()
    coverage = pd.to_numeric(daily["coverage_pct"], errors="coerce")
    low_coverage_days = int((coverage < _LOW_COVERAGE).sum())
    warning_html = ""
    if low_coverage_days:
        warning_html = (
            '<div class="warning">'
            f"风险模型覆盖率低于 80% 的交易日共 {low_coverage_days} 日；"
            "对应期间的归因结果，尤其残差占比，需要谨慎解释。"
            "</div>"
        )

    safe_title = escape(title)
    benchmark_label = (
        "全市场等权" if benchmark == "equal_weight" else benchmark.upper()
    )
    subtitle = (
        f"区间：{start:%Y-%m-%d} ~ {end:%Y-%m-%d}　|　"
        f"基准：{escape(benchmark_label)}　|　模型：{escape(model_name)}　|　"
        f"归因交易日：{len(daily)}　|　"
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}"
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  {_CSS}
</head>
<body>
  <main class="container">
    <h1>{safe_title}</h1>
    <div class="subtitle">{subtitle}</div>
    {warning_html}

    <section class="section">
      <h2>归因概览</h2>
      {_metric_cards(result)}
    </section>

    <section class="section">
      <h2>Carino 链接累计贡献</h2>
      {_contribution_chart(result)}
    </section>

    <section class="section">
      <h2>主动收益路径</h2>
      {_active_return_chart(result)}
    </section>

    <section class="section">
      <h2>风险模型覆盖率</h2>
      {_coverage_chart(result)}
    </section>

    <section class="section">
      <h2>完整归因汇总</h2>
      {_summary_table(result)}
    </section>

    <section class="section note">
      <h2>口径说明</h2>
      累计贡献使用 Carino 多期链接，合计严格对应组合相对基准的几何主动收益。
      风格、行业、Country、特质和残差解释收盘持有收益；执行影响单列费用、
      VWAP 成交与收盘持有收益之间的差异。红色表示正贡献，绿色表示负贡献。
    </section>
  </main>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


__all__ = ["generate_attribution_html"]
