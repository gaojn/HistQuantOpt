"""
回测结果 HTML 报告生成器（Plotly 交互式）。

包含图表：
  1. 净值与回撤整合图（组合/基准/超额净值 + 组合/超额回撤，按年份背景色带）
  2. 月度超额收益表格（越大越红，越小越绿）
  3. 调仓换手率

统计表：
  - 总体绩效（年化收益/波动/Sharpe/最大回撤/Calmar/IR/跟踪误差/超额回撤/胜率）
  - 年度分解（按年统计，含超额跟踪误差和超额回撤）
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hqopt.backtest.engine import BacktestResult
from hqopt.io.data_panel import load_panel
from hqopt.pipeline.universe import load_alpha_panel

# 换手一节的主色（柱状图与指标卡片左边框共用，保持同组视觉一致）
_TURNOVER_COLOR = "#c0392b"

# ─────────────────────────────────────────────────────────────────
# 工具：年度绩效计算
# ─────────────────────────────────────────────────────────────────

def _annual_metrics(ret: pd.Series, bm: pd.Series, risk_free: float = 0.02) -> dict:
    """单个分组（通常是一个自然年）的绩效指标。

    口径与 ``engine.calc_metrics`` 对齐，两处漂移会让同一张报告里的同名指标
    不可比（见 tests/test_metric_consistency.py）：

    - **信息比率**分子用几何年化超额 ``(1+R_p)/(1+R_b)-1`` 后年化，
      不用算术的 ``日均超额×252``。IR 的分母 TE 本身是年化量，分子必须同为年化。
    - **Calmar** 采用行业标准口径：年化收益(CAGR) / 最大回撤，年度行与全期行
      统一使用年化收益作分子，跨不同长度区间可比。不足一年的分组年化后数值
      会明显放大（短样本本就对年化敏感），属预期现象，不额外改写公式；
      解读年度 Calmar 时结合样本天数判断即可。
    """
    n_days  = len(ret)
    n_years = n_days / 252 if n_days > 0 else 1
    total   = (1 + ret).prod() - 1
    ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = ret.std() * np.sqrt(252) if len(ret) > 1 else 0
    # 与 calc_metrics 保持一致：扣除日化无风险利率后再年化
    rf_daily = (1 + risk_free) ** (1 / 252) - 1
    sharpe  = ((ret.mean() - rf_daily) / (ret.std() + 1e-12)) * np.sqrt(252) if len(ret) > 1 else 0
    nav     = (1 + ret).cumprod()
    mdd     = float((nav / nav.cummax() - 1).min()) if len(nav) > 0 else 0
    # 行业标准口径：年化收益(CAGR) / 最大回撤
    calmar  = ann_ret / (abs(mdd) + 1e-12)
    bm_tot  = (1 + bm).prod() - 1
    exc     = ret - bm
    exc_vol = exc.std() * np.sqrt(252) if len(exc) > 1 else 0
    # 几何超额后年化，与 calc_metrics.info_ratio 同口径
    total_exc_geo = (1 + total) / max(1 + bm_tot, 1e-8) - 1
    ann_exc = (1 + total_exc_geo) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ir      = ann_exc / (exc_vol + 1e-12)

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
        "annual_excess_return": ann_exc,
        "info_ratio":    ir,
        "excess_vol":    exc_vol,
        "excess_max_dd": exc_max_dd,
    }


def _turnover_summary(result: BacktestResult) -> dict[str, float]:
    """汇总换手指标：年化双边换手、平均单期换手、年均调仓次数。

    一律按**调仓期**统计。若误用逐成交日的 ``result.turnover``，一次调仓在
    T+1/T+2/T+3 的分次成交会各占一行，使 ``mean()`` 分母偏大、平均换手被
    系统性低估（实测可低估约 1.7 倍），``rebalances`` 也会多计。
    """
    to = result.rebalance_turnover
    n_years = len(result.daily_ret) / 252 if len(result.daily_ret) > 0 else 1.0
    if len(to) == 0 or n_years <= 0:
        return {
            "annualized_turnover": 0.0,
            "avg_turnover": 0.0,
            "rebalances_per_year": 0.0,
            "rebalances": 0.0,
        }
    return {
        "annualized_turnover": float(to.sum() / n_years),
        "avg_turnover": float(to.mean()),
        "rebalances_per_year": float(len(to) / n_years),
        "rebalances": float(len(to)),
    }


def _yearly_turnover_table(result: BacktestResult) -> pd.DataFrame:
    """按自然年汇总双边换手率与调仓次数（按调仓期计，理由见 _turnover_summary）。"""
    to = result.rebalance_turnover
    if len(to) == 0:
        return pd.DataFrame(columns=["rebalance_count", "turnover", "avg_turnover"])

    yearly = pd.DataFrame({"turnover": to})
    yearly["year"] = yearly.index.year
    out = yearly.groupby("year").agg(
        rebalance_count=("turnover", "size"),
        turnover=("turnover", "sum"),
        avg_turnover=("turnover", "mean"),
    )
    out.index.name = "year"
    return out


# ─────────────────────────────────────────────────────────────────
# 图表生成
# ─────────────────────────────────────────────────────────────────

def _make_nav_chart(result: BacktestResult) -> str:
    """
    净值 + 回撤整合图。
      上图：组合净值(蓝) / 基准净值(黑) / 超额净值(红)
      下图：组合回撤 与 超额回撤
    """
    nav      = result.nav
    bm_nav   = result.bm_nav
    exc_nav  = result.excess_nav
    drawdown = nav / nav.cummax() - 1
    exc_dd   = exc_nav / exc_nav.cummax() - 1
    x        = nav.index   # 真实日期轴

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("净值曲线", "回撤（组合 vs 超额）"),
    )

    # ── 上图：净值 ──
    fig.add_trace(go.Scatter(
        x=x, y=nav.values, name="组合净值",
        line=dict(color="#3498db", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=bm_nav.values, name="基准净值",
        line=dict(color="#2c3e50", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=exc_nav.values, name="超额净值",
        line=dict(color="#e60000", width=2),
    ), row=1, col=1)

    fig.add_hline(
        y=1.0, line_dash="dot", line_color="#95a5a6", line_width=1,
        row=1, col=1,
    )

    # ── 下图：回撤（组合填充 + 超额线） ──
    fig.add_trace(go.Scatter(
        x=x, y=drawdown.values * 100,
        name="组合回撤", fill="tozeroy",
        line=dict(color="#3498db", width=1),
        fillcolor="rgba(52,152,219,0.15)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=exc_dd.values * 100,
        name="超额回撤",
        line=dict(color="#e60000", width=1.3),
    ), row=2, col=1)

    # 年份背景色带（数据下方）
    _add_year_bands(fig, nav.index, rows=(1, 2))

    fig.update_layout(
        height=640,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0.01),
        margin=dict(l=50, r=30, t=85, b=40),
    )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
    # 第一个图内嵌 plotly.js（离线可用，后续图复用全局 Plotly）
    return fig.to_html(include_plotlyjs=True, full_html=False, div_id="nav-chart")


def _add_year_bands(fig, idx: pd.DatetimeIndex,
                    rows: tuple[int, ...] = (1, 2)) -> None:
    """按年份给图表背景加交替色带（奇数年浅灰 / 偶数年浅蓝），并在顶部标注年份。"""
    xmin, xmax = idx.min(), idx.max()
    for y in sorted(idx.year.unique()):
        x0 = max(pd.Timestamp(y, 1, 1), xmin)
        x1 = min(pd.Timestamp(y + 1, 1, 1), xmax)
        # 柔和冷色交替（奇数年 slate 灰 / 偶数年天蓝），低饱和不抢线条
        color = "rgba(148,163,184,0.10)" if y % 2 == 1 else "rgba(96,165,250,0.07)"
        for r in rows:
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=color, line_width=0,
                layer="below", row=r, col=1,
            )
        # 不再加顶部年份文字（底部 x 轴已显示年份，避免与子图标题重合）




def _make_turnover_chart(result: BacktestResult) -> str:
    """换手率柱状图（每根柱=一个调仓期，不把分次成交拆成多根）。"""
    to = result.rebalance_turnover
    if len(to) == 0:
        return ""
    fig = go.Figure(data=go.Bar(
        x=to.index, y=to.values * 100,
        marker_color=_TURNOVER_COLOR,
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
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="turnover-chart")


def _build_turnover_card(result: BacktestResult) -> str:
    s = _turnover_summary(result)

    def card(title: str, value: str, sub: str) -> str:
        return (
            f'<div class="metric-card" style="border-left-color:{_TURNOVER_COLOR}">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            '</div>'
        )

    return (
        '<div class="metric-grid" style="grid-template-columns:repeat(3,1fr)">'
        + card("年化双边换手率", f"{s['annualized_turnover']*100:.1f}%", "按全期平均年度累计")
        + card("平均单期双边换手", f"{s['avg_turnover']*100:.1f}%", "按实际调仓期平均")
        + card("年均调仓次数", f"{s['rebalances_per_year']:.1f}", f"全期共 {int(s['rebalances'])} 次")
        + '</div>'
    )


# ─────────────────────────────────────────────────────────────────
# 统计表
# ─────────────────────────────────────────────────────────────────

def _build_overall_card(result: BacktestResult) -> str:
    pm = result.portfolio_metrics
    bm = result.benchmark_metrics

    def card(title: str, value: str, sub: str = "—", accent: str = "#95a5a6",
             colored: bool = False) -> str:
        # 默认黑色；仅 colored=True 的重要指标按正红负绿着色
        sc = ""
        if colored:
            sc = "negative" if value.strip().startswith("-") else "positive"
        return (
            f'<div class="metric-card" style="border-left-color:{accent}">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value {sc}">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>'
        )

    def group(label: str, cols: int, *cards_html: str) -> str:
        return (
            f'<div class="metric-group-label">{label}</div>'
            f'<div class="metric-grid" style="grid-template-columns:repeat({cols},1fr)">'
            + "".join(cards_html)
            + "</div>"
        )

    # ── 第一行：超额指标（与下一行逐列对应，方便对比） ──
    g_excess = group(
        "超额指标", 5,
        card("年化超额",    f"{pm.annual_excess_return*100:.2f}%", "组合 vs 基准",    "#3498db", colored=True),
        card("跟踪误差",    f"{pm.tracking_error*100:.2f}%",       "年化超额波动",     "#3498db"),
        card("信息比率IR",  f"{pm.info_ratio:.2f}",                "超额/跟踪误差",    "#3498db", colored=True),
        card("超额Calmar",  f"{pm.excess_calmar:.2f}",             "年化超额/超额回撤", "#3498db"),
        card("超额最大回撤", f"{pm.excess_max_drawdown*100:.2f}%",  "超额净值最大下行",  "#c0392b", colored=True),
    )

    # ── 第二行：组合绩效（列与上一行对应：收益/波动/风险调整/Calmar/回撤） ──
    g_port = group(
        "组合绩效", 5,
        card("年化收益",  f"{pm.annual_return*100:.2f}%",  f"基准 {bm.annual_return*100:.2f}%",  "#e74c3c"),
        card("年化波动",  f"{pm.annual_vol*100:.2f}%",      f"基准 {bm.annual_vol*100:.2f}%",      "#95a5a6"),
        card("Sharpe",   f"{pm.sharpe:.2f}",               f"基准 {bm.sharpe:.2f}",               "#f39c12"),
        card("Calmar",   f"{pm.calmar:.2f}",               f"年化收益/回撤；基准 {bm.calmar:.2f}", "#f39c12"),
        card("最大回撤",  f"{pm.max_drawdown*100:.2f}%",    f"基准 {bm.max_drawdown*100:.2f}%",    "#c0392b", colored=True),
    )

    return g_excess + g_port


def _build_yearly_table(result: BacktestResult) -> str:
    """
    年度分解表：
      - 各年行：累计收益（口径一致），超额=几何方法
      - 全期行：年化收益（标注 *），超额=年化超额
    """
    rows = []
    yearly_turnover = _yearly_turnover_table(result)
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        bm_grp = result.bm_ret.reindex(grp.index).fillna(0)
        m = _annual_metrics(grp, bm_grp, risk_free=result.risk_free)
        port_total = (1 + grp).prod() - 1
        bm_total   = (1 + bm_grp).prod() - 1
        excess_geo = (1 + port_total) / (1 + bm_total) - 1 if (1 + bm_total) > 1e-8 else 0.0
        yto = yearly_turnover.loc[year] if year in yearly_turnover.index else None
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
            "调仓次数":  int(yto["rebalance_count"]) if yto is not None else 0,
            "双边换手":  float(yto["turnover"]) if yto is not None else 0.0,
            "超额TE":    m["excess_vol"],
            "超额回撤":  m["excess_max_dd"],
        })

    # 总体行：收益/波动等展示年化值；Calmar 分子仍是全期累计收益。
    tsum = _turnover_summary(result)
    rows.append({
        "年份":      "<b>全期(收益年化*)</b>",
        "组合收益":  result.portfolio_metrics.annual_return,
        "基准收益":  result.benchmark_metrics.annual_return,
        "超额收益":  result.portfolio_metrics.annual_excess_return,
        "波动率":    result.portfolio_metrics.annual_vol,
        "最大回撤":  result.portfolio_metrics.max_drawdown,
        "Sharpe":   result.portfolio_metrics.sharpe,
        "Calmar":   result.portfolio_metrics.calmar,
        "信息比率":  result.portfolio_metrics.info_ratio,
        "调仓次数":  f"{tsum['rebalances_per_year']:.1f}",
        "双边换手":  tsum["annualized_turnover"],
        "超额TE":    result.portfolio_metrics.tracking_error,
        "超额回撤":  result.portfolio_metrics.excess_max_drawdown,
    })

    html = """
    <p style="font-size:11px; color:#7f8c8d; margin:4px 0 4px 0">
    年度行收益为当年累计；全期行收益列为年化（标 *）。所有 Calmar 均为对应区间
    年化收益 / 同期最大回撤（行业标准 CAGR/MDD 口径）。Sharpe / IR / 波动率 / 跟踪误差均为年化。
    </p>
    <table class="stats-table">
        <thead><tr>
            <th>年份</th><th>组合收益</th><th>基准收益</th><th>超额收益</th>
            <th>波动率</th><th>Sharpe</th><th>Calmar</th>
            <th>IR</th><th>调仓次数</th><th>双边换手</th><th>跟踪误差</th><th>最大回撤</th><th>超额回撤</th>
        </tr></thead><tbody>
    """

    for i, r in enumerate(rows):
        cls = "total-row" if i == len(rows) - 1 else ""

        def fmt_pct(v):
            return f"{v*100:.2f}%"

        def color(v):
            # 正红负绿；仅用于需着色的列
            return "positive" if v > 0 else ("negative" if v < 0 else "")

        html += f"""
        <tr class="{cls}">
            <td>{r['年份']}</td>
            <td>{fmt_pct(r['组合收益'])}</td>
            <td>{fmt_pct(r['基准收益'])}</td>
            <td class="{color(r['超额收益'])}">{fmt_pct(r['超额收益'])}</td>
            <td>{fmt_pct(r['波动率'])}</td>
            <td>{r['Sharpe']:.2f}</td>
            <td>{r['Calmar']:.2f}</td>
            <td class="{color(r['信息比率'])}">{r['信息比率']:.2f}</td>
            <td>{r['调仓次数']}</td>
            <td>{fmt_pct(r['双边换手'])}</td>
            <td>{fmt_pct(r['超额TE'])}</td>
            <td class="{color(r['最大回撤'])}">{fmt_pct(r['最大回撤'])}</td>
            <td class="{color(r['超额回撤'])}">{fmt_pct(r['超额回撤'])}</td>
        </tr>
        """
    html += "</tbody></table>"
    return html


def _load_last_alpha_slice(
    alpha_path: Path | str | None, target_date, codes: pd.Index
) -> pd.Series:
    """取 ``target_date`` 当期 alpha 值，取 ≤ target_date 最近一次信号（与优化器
    ``get_alpha_for_date`` 的 as-of 口径一致，不引入前视）。展示原始值，不做
    优化器内部可能的截面标准化。``alpha_path`` 为空或该期无信号时全部为 NaN。
    """
    empty = pd.Series(np.nan, index=codes)
    if alpha_path is None:
        return empty
    alpha_df = load_alpha_panel(alpha_path)
    avail = alpha_df.index[alpha_df.index <= pd.Timestamp(target_date)]
    if len(avail) == 0:
        return empty
    return alpha_df.loc[avail[-1]].reindex(codes)


def _build_holdings_table(
    result: BacktestResult,
    cache_dir: Path | str | None = None,
    alpha_path: Path | str | None = None,
) -> tuple[str, pd.DataFrame | None]:
    """最后一期优化器目标持仓：代码/名称/市值/行业/权重/Alpha值。

    取 ``target_weights``（优化器原始目标权重，未经撮合/涨跌停/停牌调整，区别于
    执行后的 ``actual_weights``）最后一行对应调仓日的截面，与本地行情缓存的
    name/industry_l1/total_mv 按 code 关联；``alpha_path`` 提供时同步关联当期
    alpha 原始值。行情缓存当日缺行的股票（如退市滞留仓）显示"—"而不报错。
    ``target_weights`` 缺失时返回空表。
    """
    weights = result.target_weights
    if weights is None or weights.empty:
        return "", None

    last_date = weights.index[-1]
    holdings = weights.iloc[-1]
    holdings = holdings[holdings > 1e-6].sort_values(ascending=False)
    if holdings.empty:
        return "", None

    target_date = last_date.date() if hasattr(last_date, "date") else last_date
    try:
        panel = load_panel(
            target_date, target_date,
            columns=["code", "date", "name", "industry_l1", "total_mv"],
            cache_dir=cache_dir,
        ).to_pandas().set_index("code")
    except FileNotFoundError:
        # 行情缓存缺失（如 CI/新环境）时降级：只展示代码与权重，
        # 名称/行业/市值显示"—"，不让整份报告失败。
        panel = pd.DataFrame(columns=["name", "industry_l1", "total_mv"])

    df = pd.DataFrame({"weight": holdings})
    df["name"] = panel["name"].reindex(df.index).fillna("—")
    df["industry"] = panel["industry_l1"].reindex(df.index).fillna("—")
    df["mv_yi"] = panel["total_mv"].reindex(df.index) / 1e4   # 万元 → 亿元
    df["alpha"] = _load_last_alpha_slice(alpha_path, target_date, df.index)

    rows = []
    for rank, (code, r) in enumerate(df.iterrows(), start=1):
        mv_str = f"{r['mv_yi']:.1f}" if pd.notna(r["mv_yi"]) else "—"
        alpha_str = f"{r['alpha']:.4f}" if pd.notna(r["alpha"]) else "—"
        rows.append(f"""
        <tr>
            <td>{rank}</td>
            <td>{escape(str(code))}</td>
            <td>{escape(str(r['name']))}</td>
            <td>{mv_str}</td>
            <td>{escape(str(r['industry']))}</td>
            <td>{r['weight']*100:.2f}%</td>
            <td>{alpha_str}</td>
        </tr>
        """)

    html = f"""
    <p style="font-size:11px; color:#7f8c8d; margin:4px 0 4px 0">
    截止 {target_date:%Y-%m-%d} 的优化器目标持仓（现金与权重≈0 的仓位不显示，
    未经撮合/涨跌停/停牌调整，与实际成交持仓可能有差异）。名称/行业/市值取自
    当日行情缓存截面；Alpha 为该期信号原始值（≤当日最近一次，未做截面标准化）。
    </p>
    <table class="stats-table">
        <thead><tr>
            <th>排名</th><th>代码</th><th>名称</th><th>总市值(亿元)</th><th>行业</th><th>权重</th><th>Alpha值</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """
    return html, df.reset_index().rename(columns={"index": "code"})


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
        # 过滤绑定到变量而非重复下标取值，使「已排除 None」对类型检查可见。
        vals = [v for v in (grid[y][m] for y in years) if v is not None]
        if vals:
            win = sum(1 for v in vals if v > 0) / len(vals)
            month_win.append(f"{win*100:.0f}%")
        else:
            month_win.append("—")

    # 年度胜率（按年）
    year_win: list[str] = []
    for y in years:
        vals = [v for v in (grid[y][m] for m in range(1, 13)) if v is not None]
        if vals:
            win = sum(1 for v in vals if v > 0) / len(vals)
            year_win.append(f"{win*100:.0f}%")
        else:
            year_win.append("—")

    # 年度超额合计
    year_sum: list[str] = []
    for y in years:
        vals = [v for v in (grid[y][m] for m in range(1, 13)) if v is not None]
        year_sum.append(f"{sum(vals):.2f}%" if vals else "—")

    # 注：月度超额为算术差（月组合收益 − 月基准收益）；年度/全期超额采用几何口径
    html = '<p style="font-size:11px; color:#7f8c8d; margin:4px 0 4px 0">月度超额为算术差；年度/全期超额为几何口径。</p>'
    html += '<table class="monthly-table"><thead><tr>'
    html += "<th>年份</th>"
    for mn in month_names:
        html += f"<th>{mn}</th>"
    html += "<th>胜率</th><th>合计</th></tr></thead><tbody>"

    for idx, y in enumerate(years):
        html += f"<tr><td style='font-weight:600'>{y}</td>"
        for m in range(1, 13):
            v = grid[y][m]
            style = cell_style(v)
            text  = f"{v:.2f}%" if v is not None else "—"
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
        f"    color:{'#c0392b' if pm.avg_monthly_excess >= 0 else '#27ae60'}'>"
        f"    {pm.avg_monthly_excess*100:.3f}%"
        f"  </span>"
        f"</div>"
    )

    return win_rate_html + html


# ─────────────────────────────────────────────────────────────────
# 报告数据落地（parquet）
# ─────────────────────────────────────────────────────────────────

def _save_report_data(
    result: BacktestResult, output_path: Path, holdings_df: pd.DataFrame | None = None
) -> Path:
    """
    把报告里展示的数据保存为 parquet，存到 <报告名>_data/ 目录。
      - timeseries.parquet : 净值/回撤/日收益（净值与回撤图）
      - turnover.parquet   : 调仓换手率
      - metrics.parquet    : 总体绩效（组合 vs 基准）
      - yearly.parquet     : 年度分解表
      - monthly_excess.parquet : 月度超额（组合/基准/超额）
      - holdings.parquet   : 最后一期优化器目标持仓（代码/名称/行业/权重/市值/alpha），无持仓数据时不写
    """
    base = output_path.parent / f"{output_path.stem}_data"
    base.mkdir(parents=True, exist_ok=True)

    nav, bm_nav, exc_nav = result.nav, result.bm_nav, result.excess_nav
    ts = pd.DataFrame({
        "nav":             nav,
        "bm_nav":          bm_nav,
        "excess_nav":      exc_nav,
        "port_drawdown":   nav / nav.cummax() - 1,
        "excess_drawdown": exc_nav / exc_nav.cummax() - 1,
        "port_ret":        result.daily_ret,
        "bm_ret":          result.bm_ret,
    })
    ts.index.name = "date"
    ts.to_parquet(base / "timeseries.parquet")

    # 与卡片/图表同口径：按调仓期。逐成交日的原始序列在顶层 turnover.parquet。
    result.rebalance_turnover.rename("turnover").to_frame().to_parquet(
        base / "turnover.parquet"
    )

    pm, bm = result.portfolio_metrics, result.benchmark_metrics
    fields = ["annual_return", "annual_vol", "sharpe", "calmar", "max_drawdown",
              "annual_excess_return", "tracking_error", "info_ratio",
              "excess_calmar", "excess_max_drawdown",
              "win_rate_monthly", "avg_monthly_excess"]
    metrics = pd.DataFrame({
        "portfolio": {f: getattr(pm, f, np.nan) for f in fields},
        "benchmark": {f: getattr(bm, f, np.nan) for f in fields},
    })
    metrics.index.name = "metric"
    metrics.to_parquet(base / "metrics.parquet")

    # 年度分解
    yearly_turnover = _yearly_turnover_table(result)
    yrows = []
    for year, grp in result.daily_ret.groupby(result.daily_ret.index.year):
        bm_grp = result.bm_ret.reindex(grp.index).fillna(0)
        m = _annual_metrics(grp, bm_grp, risk_free=result.risk_free)
        port_total = (1 + grp).prod() - 1
        bm_total   = (1 + bm_grp).prod() - 1
        yto = yearly_turnover.loc[year] if year in yearly_turnover.index else None
        yrows.append({
            "year":      int(year),
            "n_days":    len(grp),
            "rebalance_count": int(yto["rebalance_count"]) if yto is not None else 0,
            "turnover":  float(yto["turnover"]) if yto is not None else 0.0,
            "avg_turnover": float(yto["avg_turnover"]) if yto is not None else 0.0,
            "port_ret":  port_total,
            "bm_ret":    bm_total,
            "excess_ret": (1 + port_total) / (1 + bm_total) - 1 if (1 + bm_total) > 1e-8 else 0.0,
            "vol":       m["annual_vol"],
            "max_dd":    m["max_dd"],
            "sharpe":    m["sharpe"],
            "calmar":    m["calmar"],
            "info_ratio": m["info_ratio"],
            "excess_te": m["excess_vol"],
            "excess_dd": m["excess_max_dd"],
        })
    pd.DataFrame(yrows).set_index("year").to_parquet(base / "yearly.parquet")

    # 月度超额
    mp = (1 + result.daily_ret).resample("ME").prod() - 1
    mb = (1 + result.bm_ret).resample("ME").prod() - 1
    monthly = pd.DataFrame({"port_ret": mp, "bm_ret": mb, "excess_ret": mp - mb})
    monthly.index.name = "month"
    monthly.to_parquet(base / "monthly_excess.parquet")

    if holdings_df is not None and not holdings_df.empty:
        holdings_df.to_parquet(base / "holdings.parquet")

    return base


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
  .warning-banner {
    background: #fff3cd; border: 2px solid #d97706; color: #7c2d12;
    border-radius: 8px; padding: 14px 16px; margin: 0 0 20px 0;
    font-size: 16px; font-weight: 700; line-height: 1.6;
  }
  .section {
    background: #fff; border-radius: 8px; padding: 20px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .section h2 {
    margin: 0 0 16px 0; font-size: 16px; color: #34495e;
    border-left: 3px solid #3498db; padding-left: 10px;
  }
  .metric-grid {
    display: grid; gap: 12px; margin-bottom: 16px;
  }
  .metric-group-label {
    font-size: 11px; font-weight: 600; color: #95a5a6;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 16px 0 8px 2px;
  }
  .metric-group-label:first-child { margin-top: 0; }
  .metric-card {
    background: #fafbfc; border-radius: 6px; padding: 14px;
    border-left: 3px solid #95a5a6;
  }
  .metric-title { font-size: 12px; color: #7f8c8d; margin-bottom: 6px; }
  .metric-value { font-size: 22px; font-weight: 600; line-height: 1.2; }
  .metric-sub { font-size: 11px; color: #95a5a6; margin-top: 4px; }
  .positive { color: #c0392b; }   /* 正数红色（A股惯例） */
  .negative { color: #27ae60; }   /* 负数绿色 */
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
    warning_banner: str | None = None,
    cache_dir: Path | str | None = None,
    alpha_path: Path | str | None = None,
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
    warning_banner : str | None
        需要置顶展示的风险水印；例如含未来信息的合成 Alpha 警告
    cache_dir : Path | str | None
        最后一期持仓明细表用到的行情 parquet 缓存目录，None 用默认路径。
        与 ``run_backtest`` 的 ``cache_dir`` 应保持一致。
    alpha_path : Path | str | None
        最后一期持仓明细表 Alpha 列用到的 alpha parquet 路径（与优化时的
        ``alpha.path`` 一致）。None 时该列全部显示"—"。

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
            f"再平衡次数: {len(result.rebalance_turnover)}  |  "
            f"生成时间: {datetime.now():%Y-%m-%d %H:%M}"
        )

    overall_card    = _build_overall_card(result)
    yearly_table    = _build_yearly_table(result)
    nav_chart       = _make_nav_chart(result)
    monthly_table   = _build_monthly_excess_table(result)
    turnover_chart  = _make_turnover_chart(result)
    turnover_card   = _build_turnover_card(result)
    holdings_html, holdings_df = _build_holdings_table(
        result, cache_dir=cache_dir, alpha_path=alpha_path
    )
    holdings_date = (
        result.target_weights.index[-1]
        if result.target_weights is not None and not result.target_weights.empty
        else None
    )
    holdings_section = ""
    if holdings_html:
        holdings_section = f"""
    <div class="section">
      <h2>最后一期优化器目标持仓（{holdings_date:%Y-%m-%d}）</h2>
      {holdings_html}
    </div>
"""
    warning_html = ""
    if warning_banner:
        safe_warning = escape(warning_banner).replace("\n", "<br>")
        warning_html = f'<div class="warning-banner">⚠️ {safe_warning}</div>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  {_CSS}
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>
    {warning_html}

    <div class="section">
      <h2>总体绩效</h2>
      {overall_card}
    </div>

    <div class="section">
      <h2>年度绩效分解</h2>
      {yearly_table}
    </div>

    <div class="section">
      <h2>净值与回撤（含超额）</h2>
      {nav_chart}
    </div>

    <div class="section">
      <h2>月度超额明细（越大越红，越小越绿）</h2>
      {monthly_table}
    </div>

    <div class="section">
      <h2>调仓换手</h2>
      {turnover_card}
      {turnover_chart}
    </div>
    {holdings_section}
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")

    # 同步把报告展示用到的数据落地为 parquet
    _save_report_data(result, output_path, holdings_df=holdings_df)

    return output_path
