"""
回测结果 HTML 报告生成器（Plotly 交互式）。

包含图表：
  1. 净值曲线（组合 vs 基准 vs 超额）
  2. 回撤曲线
  3. 超额净值与超额回撤
  4. 月度超额收益热图
  5. 月度超额收益表格（越大越红，越小越绿）
  6. 调仓换手率

统计表：
  - 总体绩效（年化收益/波动/Sharpe/最大回撤/Calmar/IR/TE/超额回撤/胜率）
  - 年度分解（按年统计，含超额TE和超额回撤）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_optimizer.backtest.engine import BacktestResult


# ─────────────────────────────────────────────────────────────────
# 工具：年度绩效计算
# ─────────────────────────────────────────────────────────────────

def _annual_metrics(ret: pd.Series, bm: pd.Series) -> dict:
    n_days  = len(ret)
    n_years = n_days / 252 if n_days > 0 else 1
    total   = (1 + ret).prod() - 1
    ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = ret.std() * np.sqrt(252) if len(ret) > 1 else 0
    sharpe  = (ret.mean() / (ret.std() + 1e-12)) * np.sqrt(252) if len(ret) > 1 else 0
    nav     = (1 + ret).cumprod()
    mdd     = float((nav / nav.cummax() - 1).min()) if len(nav) > 0 else 0
    calmar  = ann_ret / (abs(mdd) + 1e-12)
    bm_tot  = (1 + bm).prod() - 1
    exc     = ret - bm
    exc_vol = exc.std() * np.sqrt(252) if len(exc) > 1 else 0
    ir      = (exc.mean() * 252) / (exc_vol + 1e-12)

    # 超额净值回撤（几何）
    port_nav     = (1 + ret).cumprod()
    bm_nav_loc   = (1 + bm).cumprod()
    exc_nav_loc  = port_nav / bm_nav_loc.replace(0, np.nan)
    exc_nav_loc  = exc_nav_loc.ffill().fillna(1.0)
    exc_dd_series = exc_nav_loc / exc_nav_loc.cummax() - 1
    exc_max_dd   = float(exc_dd_series.min()) if len(exc_dd_series) > 0 else 0.0

    return {
        "annual_return": ann_ret,
        "annual_vol":    ann_vol,
        "sharpe":        sharpe,
        "max_dd":        mdd,
        "calmar":        calmar,
        "excess_return": (1 + total) / (1 + bm_tot) - 1 if (1 + bm_tot) > 1e-8 else 0.0,
        "info_ratio":    ir,
        "excess_vol":    exc_vol,
        "excess_max_dd": exc_max_dd,
    }


# ─────────────────────────────────────────────────────────────────
# 图表生成
# ─────────────────────────────────────────────────────────────────

def _date_to_num(idx: pd.DatetimeIndex) -> list[str]:
    """将日期索引转为 YYYYMMDD 数字字符串，用于横坐标显示。"""
    return [d.strftime("%Y%m%d") for d in idx]


def _make_nav_chart(result: BacktestResult) -> str:
    """净值 + 回撤双轴图。"""
    nav      = result.nav
    bm_nav   = result.bm_nav
    exc_nav  = result.excess_nav
    drawdown = nav / nav.cummax() - 1
    x_dates  = _date_to_num(nav.index)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("净值曲线", "组合回撤"),
    )

    fig.add_trace(go.Scatter(
        x=x_dates, y=nav.values, name="组合",
        line=dict(color="#e74c3c", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_dates, y=bm_nav.values, name="基准",
        line=dict(color="#95a5a6", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_dates, y=exc_nav.values, name="超额净值",
        line=dict(color="#3498db", width=1.5, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_dates, y=drawdown.values * 100,
        name="回撤(%)", fill="tozeroy",
        line=dict(color="#c0392b", width=1),
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.01),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_xaxes(type="category", tickangle=-45, nticks=20)
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="nav-chart")


def _make_excess_chart(result: BacktestResult) -> str:
    """超额净值 + 超额回撤双轴图。"""
    exc_nav = result.excess_nav
    exc_dd  = exc_nav / exc_nav.cummax() - 1
    x_dates = _date_to_num(exc_nav.index)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("超额净值（几何法：组合净值 / 基准净值）", "超额回撤"),
    )

    fig.add_trace(go.Scatter(
        x=x_dates, y=exc_nav.values, name="超额净值",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.08)",
    ), row=1, col=1)

    fig.add_hline(
        y=1.0, line_dash="dot", line_color="#95a5a6", line_width=1,
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=x_dates, y=exc_dd.values * 100,
        name="超额回撤(%)", fill="tozeroy",
        line=dict(color="#3498db", width=1),
        fillcolor="rgba(52,152,219,0.15)",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=550,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.01),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_xaxes(type="category", tickangle=-45, nticks=20)
    fig.update_yaxes(title_text="超额净值", row=1, col=1)
    fig.update_yaxes(title_text="超额回撤(%)", row=2, col=1)
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="excess-chart")




def _make_turnover_chart(result: BacktestResult) -> str:
    """换手率柱状图。"""
    to = result.turnover.dropna()
    if len(to) == 0:
        return ""
    x_dates = _date_to_num(to.index)
    fig = go.Figure(data=go.Bar(
        x=x_dates, y=to.values * 100,
        marker_color="#8e44ad",
    ))
    fig.add_hline(
        y=to.mean() * 100, line_dash="dot", line_color="#7f8c8d",
        annotation_text=f"均值: {to.mean()*100:.1f}%",
    )
    fig.update_layout(
        title="调仓换手率（双边）",
        height=300,
        template="plotly_white",
        yaxis_title="换手率 (%)",
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_xaxes(type="category", tickangle=-45, nticks=20)
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="turnover-chart")


# ─────────────────────────────────────────────────────────────────
# 统计表
# ─────────────────────────────────────────────────────────────────

def _build_overall_card(result: BacktestResult) -> str:
    pm = result.portfolio_metrics
    bm = result.benchmark_metrics

    cards = [
        ("年化收益",    f"{pm.annual_return*100:+.2f}%",       f"基准 {bm.annual_return*100:+.2f}%"),
        ("年化波动",    f"{pm.annual_vol*100:.2f}%",            f"基准 {bm.annual_vol*100:.2f}%"),
        ("Sharpe",     f"{pm.sharpe:.3f}",                     f"基准 {bm.sharpe:.3f}"),
        ("最大回撤",    f"{pm.max_drawdown*100:.2f}%",          f"基准 {bm.max_drawdown*100:.2f}%"),
        ("Calmar",     f"{pm.calmar:.3f}",                     f"基准 {bm.calmar:.3f}"),
        ("年化超额",    f"{pm.annual_excess_return*100:+.2f}%", "—"),
        ("跟踪误差TE",  f"{pm.tracking_error*100:.2f}%",        "—"),
        ("信息比率IR",  f"{pm.info_ratio:.3f}",                 "—"),
        ("超额最大回撤", f"{pm.excess_max_drawdown*100:.2f}%",  "—"),
        ("超额Calmar",  f"{pm.excess_calmar:.3f}",              "—"),
        ("月度胜率",    f"{pm.win_rate_monthly*100:.1f}%",      "—"),
        ("月均超额",    f"{pm.avg_monthly_excess*100:+.3f}%",   "—"),
    ]

    html = '<div class="metric-grid">'
    for title, value, sub in cards:
        sign_class = ""
        if value.startswith("+"):  sign_class = "positive"
        elif value.startswith("-"): sign_class = "negative"
        html += f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value {sign_class}">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """
    html += "</div>"
    return html


def _build_yearly_table(result: BacktestResult) -> str:
    """
    年度分解表：
      - 各年行：累计收益（口径一致），超额=几何方法
      - 全期行：年化收益（标注 *），超额=年化超额
    """
    rows = []
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        bm_grp = result.bm_ret.reindex(grp.index).fillna(0)
        m = _annual_metrics(grp, bm_grp)
        port_total = (1 + grp).prod() - 1
        bm_total   = (1 + bm_grp).prod() - 1
        excess_geo = (1 + port_total) / (1 + bm_total) - 1 if (1 + bm_total) > 1e-8 else 0.0
        rows.append({
            "年份":      f"{year}<span style='color:#95a5a6;font-size:10px'> ({len(grp)}天)</span>",
            "组合收益":  port_total,
            "基准收益":  bm_total,
            "超额收益":  excess_geo,
            "波动率":    m["annual_vol"],
            "最大回撤":  m["max_dd"],
            "Sharpe":   m["sharpe"],
            "Calmar":   m["calmar"],
            "信息比率":  m["info_ratio"],
            "超额TE":    m["excess_vol"],
            "超额回撤":  m["excess_max_dd"],
        })

    # 总体行（年化口径）
    rows.append({
        "年份":      "<b>全期(年化*)</b>",
        "组合收益":  result.portfolio_metrics.annual_return,
        "基准收益":  result.benchmark_metrics.annual_return,
        "超额收益":  result.portfolio_metrics.annual_excess_return,
        "波动率":    result.portfolio_metrics.annual_vol,
        "最大回撤":  result.portfolio_metrics.max_drawdown,
        "Sharpe":   result.portfolio_metrics.sharpe,
        "Calmar":   result.portfolio_metrics.calmar,
        "信息比率":  result.portfolio_metrics.info_ratio,
        "超额TE":    result.portfolio_metrics.tracking_error,
        "超额回撤":  result.portfolio_metrics.excess_max_drawdown,
    })

    html = """
    <table class="stats-table">
        <thead><tr>
            <th>年份</th><th>组合收益</th><th>基准收益</th><th>超额收益</th>
            <th>波动率</th><th>最大回撤</th><th>Sharpe</th><th>Calmar</th>
            <th>IR</th><th>超额TE</th><th>超额回撤</th>
        </tr></thead><tbody>
    """

    for i, r in enumerate(rows):
        cls = "total-row" if i == len(rows) - 1 else ""

        def fmt_pct(v, sign=False):
            s = "+" if sign and v >= 0 else ""
            return f"{s}{v*100:.2f}%"

        def color(v):
            return "positive" if v > 0 else ("negative" if v < 0 else "")

        html += f"""
        <tr class="{cls}">
            <td>{r['年份']}</td>
            <td class="{color(r['组合收益'])}">{fmt_pct(r['组合收益'], True)}</td>
            <td class="{color(r['基准收益'])}">{fmt_pct(r['基准收益'], True)}</td>
            <td class="{color(r['超额收益'])}">{fmt_pct(r['超额收益'], True)}</td>
            <td>{fmt_pct(r['波动率'])}</td>
            <td class="negative">{fmt_pct(r['最大回撤'])}</td>
            <td>{r['Sharpe']:.3f}</td>
            <td>{r['Calmar']:.3f}</td>
            <td>{r['信息比率']:.3f}</td>
            <td>{fmt_pct(r['超额TE'])}</td>
            <td class="negative">{fmt_pct(r['超额回撤'])}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html


def _build_monthly_excess_table(result: BacktestResult) -> str:
    """
    月度超额收益表格：越大标记红色，越小标记绿色。
    行=年份，列=月份，最后加行胜率统计。
    """
    monthly_port = (1 + result.daily_ret).resample("ME").prod() - 1
    monthly_bm   = (1 + result.bm_ret).resample("ME").prod() - 1
    excess       = (monthly_port - monthly_bm) * 100  # 百分比

    years  = sorted(excess.index.year.unique())
    month_names = ["1月","2月","3月","4月","5月","6月",
                   "7月","8月","9月","10月","11月","12月"]

    # 构建网格
    grid: dict[int, dict[int, float | None]] = {}
    for y in years:
        grid[y] = {}
        for m in range(1, 13):
            sel = excess[(excess.index.year == y) & (excess.index.month == m)]
            grid[y][m] = float(sel.iloc[0]) if len(sel) > 0 else None

    # 全局 min/max 用于颜色归一化
    all_vals = [v for row in grid.values() for v in row.values() if v is not None]
    v_max = max(abs(max(all_vals)), abs(min(all_vals))) if all_vals else 1.0

    def cell_style(v: float | None) -> str:
        if v is None:
            return "background:#f8f9fa; color:#bdc3c7;"
        intensity = min(abs(v) / (v_max + 1e-8), 1.0)
        if v > 0:
            # 越大越红：正值 → 红色渐变
            r = int(192 + 63 * intensity)
            g = int(57 + (57 - 57) * intensity)
            b = int(43 + (43 - 43) * intensity)
            alpha = 0.15 + 0.65 * intensity
            bg = f"rgba({r},{g},{b},{alpha:.2f})"
            txt = "#7b0000" if intensity > 0.5 else "#333"
        else:
            # 越小越绿：负值 → 绿色渐变
            r = int(39 + (39 - 39) * intensity)
            g = int(174 - 50 * intensity)
            b = int(96 - 30 * intensity)
            alpha = 0.15 + 0.65 * intensity
            bg = f"rgba({r},{g},{b},{alpha:.2f})"
            txt = "#004d00" if intensity > 0.5 else "#333"
        return f"background:{bg}; color:{txt};"

    # 月度胜率（按月）
    month_win: list[str] = []
    for m in range(1, 13):
        vals = [grid[y][m] for y in years if grid[y][m] is not None]
        if vals:
            win = sum(1 for v in vals if v > 0) / len(vals)
            month_win.append(f"{win*100:.0f}%")
        else:
            month_win.append("—")

    # 年度胜率（按年）
    year_win: list[str] = []
    for y in years:
        vals = [grid[y][m] for m in range(1, 13) if grid[y][m] is not None]
        if vals:
            win = sum(1 for v in vals if v > 0) / len(vals)
            year_win.append(f"{win*100:.0f}%")
        else:
            year_win.append("—")

    # 年度超额合计
    year_sum: list[str] = []
    for y in years:
        vals = [grid[y][m] for m in range(1, 13) if grid[y][m] is not None]
        year_sum.append(f"{sum(vals):+.2f}%" if vals else "—")

    html = '<table class="monthly-table"><thead><tr>'
    html += "<th>年份</th>"
    for mn in month_names:
        html += f"<th>{mn}</th>"
    html += "<th>胜率</th><th>合计</th></tr></thead><tbody>"

    for idx, y in enumerate(years):
        html += f"<tr><td style='font-weight:600'>{y}</td>"
        for m in range(1, 13):
            v = grid[y][m]
            style = cell_style(v)
            text  = f"{v:+.2f}%" if v is not None else "—"
            html += f"<td style='{style} text-align:center; font-size:11px; padding:5px;'>{text}</td>"
        html += f"<td style='text-align:center; font-weight:500'>{year_win[idx]}</td>"
        html += f"<td style='text-align:center; font-weight:500'>{year_sum[idx]}</td></tr>"

    # 月度胜率行
    html += "<tr style='border-top:2px solid #bdc3c7; background:#f8f9fa; font-weight:600'>"
    html += "<td>月胜率</td>"
    for mw in month_win:
        html += f"<td style='text-align:center; font-size:11px'>{mw}</td>"
    html += "<td colspan='2'></td></tr>"

    html += "</tbody></table>"

    pm = result.portfolio_metrics
    win_rate_html = (
        f"<div style='margin-top:10px; font-size:13px; color:#34495e;'>"
        f"  超额月度胜率（全期）: "
        f"  <span style='font-size:18px; font-weight:700; color:#27ae60'>"
        f"    {pm.win_rate_monthly*100:.1f}%"
        f"  </span>"
        f"  &nbsp;&nbsp;月均超额: "
        f"  <span style='font-size:18px; font-weight:700;"
        f"    color:{'#e74c3c' if pm.avg_monthly_excess >= 0 else '#27ae60'}'>"
        f"    {pm.avg_monthly_excess*100:+.3f}%"
        f"  </span>"
        f"</div>"
    )

    return win_rate_html + html


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

_CSS = """
<style>
  body {
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    margin: 0; padding: 24px; background: #f5f6f8; color: #2c3e50;
  }
  .container { max-width: 1300px; margin: 0 auto; }
  h1 { margin: 0 0 4px 0; font-weight: 600; color: #2c3e50; }
  .subtitle { color: #7f8c8d; margin-bottom: 24px; font-size: 13px; }
  .section {
    background: #fff; border-radius: 8px; padding: 20px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .section h2 {
    margin: 0 0 16px 0; font-size: 16px; color: #34495e;
    border-left: 3px solid #3498db; padding-left: 10px;
  }
  .metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  }
  .metric-card {
    background: #fafbfc; border-radius: 6px; padding: 14px;
    border-left: 3px solid #95a5a6;
  }
  .metric-title { font-size: 12px; color: #7f8c8d; margin-bottom: 6px; }
  .metric-value { font-size: 22px; font-weight: 600; line-height: 1.2; }
  .metric-sub { font-size: 11px; color: #95a5a6; margin-top: 4px; }
  .positive { color: #27ae60; }
  .negative { color: #c0392b; }
  .stats-table {
    width: 100%; border-collapse: collapse; font-size: 13px;
  }
  .stats-table th, .stats-table td {
    padding: 8px 12px; text-align: right; border-bottom: 1px solid #ecf0f1;
  }
  .stats-table th {
    background: #f8f9fa; color: #34495e; font-weight: 600;
    border-bottom: 2px solid #bdc3c7;
  }
  .stats-table th:first-child, .stats-table td:first-child { text-align: left; }
  .stats-table .total-row {
    background: #fdf6e3; font-weight: 600;
    border-top: 2px solid #f39c12;
  }
  .stats-table tr:hover { background: #f8f9fa; }
  .total-row:hover { background: #fdf2d4 !important; }
  .monthly-table {
    width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px;
  }
  .monthly-table th {
    background: #f8f9fa; color: #34495e; font-weight: 600;
    padding: 7px 4px; text-align: center; border-bottom: 2px solid #bdc3c7;
    font-size: 11px;
  }
  .monthly-table td:first-child { text-align: left; padding-left: 8px; }
  .monthly-table tr:hover td { filter: brightness(0.95); }
</style>
"""


def generate_html_report(
    result: BacktestResult,
    output_path: Path | str = "output/backtest_report.html",
    title: str = "量化多头组合回测报告",
    subtitle: str | None = None,
) -> Path:
    """
    生成回测 HTML 报告。

    Parameters
    ----------
    result : BacktestResult
        回测结果对象
    output_path : Path | str
        HTML 输出路径
    title : str
        报告标题
    subtitle : str | None
        副标题，默认显示回测时间区间

    Returns
    -------
    Path: 输出的 HTML 文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if subtitle is None:
        d0, d1 = result.nav.index[0], result.nav.index[-1]
        subtitle = (
            f"回测区间: {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}  |  "
            f"交易日数: {len(result.nav)}  |  "
            f"再平衡次数: {len(result.turnover)}  |  "
            f"生成时间: {datetime.now():%Y-%m-%d %H:%M}"
        )

    overall_card    = _build_overall_card(result)
    yearly_table    = _build_yearly_table(result)
    nav_chart       = _make_nav_chart(result)
    excess_chart    = _make_excess_chart(result)
    monthly_table   = _build_monthly_excess_table(result)
    turnover_chart  = _make_turnover_chart(result)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  {_CSS}
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>

    <div class="section">
      <h2>总体绩效</h2>
      {overall_card}
    </div>

    <div class="section">
      <h2>年度绩效分解</h2>
      {yearly_table}
    </div>

    <div class="section">
      <h2>净值与回撤</h2>
      {nav_chart}
    </div>

    <div class="section">
      <h2>超额净值与超额回撤</h2>
      {excess_chart}
    </div>

    <div class="section">
      <h2>月度超额明细（越大越红，越小越绿）</h2>
      {monthly_table}
    </div>

    <div class="section">
      <h2>调仓换手</h2>
      {turnover_chart}
    </div>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
