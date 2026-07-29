"""收益归因：把每期主动收益拆成风格 / 行业 / Country / 个股特质贡献。

用于回答"超额是真选股能力，还是优化器偷偷吃了风格 beta"——把超额收益
用 CNE6 因子模型分解，而非只看一条净值曲线。

方法
----
每个调仓期 ``(T_i, T_{i+1}]`` 使用信号日 T_i 的风险暴露快照；若提供成交账本
生成的逐日实际权重，则主动权重随实际成交、价格漂移逐日更新，否则沿用目标权重。
这样停牌、涨跌停和延期订单不会被误当成已成交。

持有期内逐日用当日已实现的因子收益 f(t) 与个股特质收益 u(t) 计算贡献。若传入
成交账本的真实组合日收益，还会把 VWAP→收盘时点差、费用和滑点统一记为执行影响：

    真实主动收益(t)
      = 风格 + 行业 + Country + 特质 + 模型残差 + 执行影响

其中 X_active_k = Σ_i w_active_i · X_ik（该持有期内固定不变的主动暴露），
f_k(t)/u_i(t) 来自 ClickHouse test_barra_cne6_gao 对应 S/L 模型（与暴露 X 同源，见
:mod:`hqopt.risk.attribution_data`），保证暴露与收益出自
同一套模型，残差自检才有意义。

多期链接采用相对收益口径。先把每日算术贡献除以 ``1 + 基准收益(t)``，
使各项之和等于每日相对主动收益
``g_t = (1 + 组合收益(t)) / (1 + 基准收益(t)) - 1``；再用
Carino (1999) 平滑系数链接。由此保证 Σ 各归因项链接后的累计贡献严格等于
``Π(1+组合收益) / Π(1+基准收益) - 1``，不再把 ``Π(1+Rp-Rb)-1``
近似冒充真实几何超额收益（"合计"行即为该恒等式的自检）。

残差自检：每日 ``持有主动收益 - (风格+行业+Country+特质)`` 理论上应 ≈ 0
（在暴露/收益对齐正确、且持仓票均在风险模型估计域内时，两者恒等）。

已知局限——残差不会严格为 0：
    1. **覆盖缺口**：test_barra_cne6_gao 的暴露/特质收益只覆盖其估计域
       （``univ_flag==1``，剔除次新/极小市值/长期停牌等）。组合或基准
       里在估计域外的持仓，其真实收益仍计入"主动收益"，但因子/特质
       两项均为 0（暴露和特质收益都取不到），全部漏进残差。
       ``daily["coverage_pct"]`` 记录了每个持有期"被模型覆盖的主动
       权重占比"，可用于判断该期归因是否可信；但覆盖率与残差占比
       不是线性关系——未覆盖的通常正是次新/极小市值这类高波动票，
       权重占比很小也可能贡献不成比例的收益/残差（实测覆盖率
       中位数~97%，残差仍占主动收益~27%），不能只看覆盖率百分比
       就假设残差很小。
    2. **暴露按持有期冻结**：本模块暴露 X 只在信号日 T_i 取值一次、
       整个持有期 ``(T_i, T_{i+1}]`` 内保持不变（与优化器约束假设一致），
       而 f(t)/u(t) 出自模型逐日的截面回归（用当日暴露）。调仓越不频繁，
       此近似误差可能越大。
    实测（见 tests/test_attribution.py）：全覆盖的合成数据残差严格为 0，
    证明分解算式本身正确；残差非零是数据覆盖问题，不是代码逻辑问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hqopt.risk.attribution_data import FactorReturnLoader
from hqopt.risk.cne6_risk import STYLE_FACTORS, CNE6RiskModel

_TRADING_DAYS = 252
_COUNTRY = "Country"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _ensure_datetime_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    if series.index.has_duplicates:
        raise ValueError("真实组合日收益日期不能重复")
    return series.sort_index()


def _align_union(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.union(b.index)
    return a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)


def _carino_k(r: float) -> float:
    """Carino(1999) 对数平滑系数：ln(1+r)/r，r→0 时取极限 1。"""
    if r <= -1.0:
        raise ValueError(f"Carino 收益必须大于 -100%，当前为 {r:.6f}")
    if abs(r) < 1e-10:
        return 1.0
    return float(np.log1p(r) / r)


def _carino_summary(
    items: dict[str, pd.Series],
    portfolio_return: pd.Series,
    benchmark_return: pd.Series,
) -> pd.DataFrame:
    """按组合/基准净值比做精确 Carino 链接，并计算简单 t 统计。"""
    if not portfolio_return.index.equals(benchmark_return.index):
        raise ValueError("组合与基准收益日期必须完全对齐")
    n = len(portfolio_return)
    if n == 0:
        raise ValueError("无可链接的收益日期")

    benchmark_gross = 1.0 + benchmark_return
    if (benchmark_gross <= 0.0).any():
        raise ValueError("基准单日收益必须大于 -100%")
    relative_active = (1.0 + portfolio_return) / benchmark_gross - 1.0
    r_geo = float((1.0 + relative_active).prod() - 1.0)
    k = _carino_k(r_geo)
    k_t = relative_active.apply(_carino_k)

    rows = []
    for name, series in items.items():
        relative_series = series / benchmark_gross
        linked = float((relative_series * k_t / k).sum())
        annualized = linked * (_TRADING_DAYS / n)
        pct = linked / r_geo * 100 if abs(r_geo) > 1e-12 else float("nan")
        std = float(relative_series.std(ddof=1))
        t_stat = (
            float(relative_series.mean() / (std / np.sqrt(n)))
            if std > 1e-12
            else float("nan")
        )
        ann_vol = std * np.sqrt(_TRADING_DAYS)
        rows.append({
            "归因项": name,
            "累计贡献": linked,
            "年化贡献": annualized,
            "占主动收益%": pct,
            "t统计": t_stat,
            "年化波动": ann_vol,
        })
    return pd.DataFrame(rows).set_index("归因项")


@dataclass
class AttributionResult:
    """归因结果。"""
    factor_daily: pd.DataFrame   # index=date, columns=S 51/L 47因子, 逐日贡献 X_active_k·f_k(t)
    daily: pd.DataFrame          # index=date，含持有收益、真实收益、模型残差和执行影响
    summary: pd.DataFrame        # 风格+行业+Country+特质+残差+执行影响+合计

    def __str__(self) -> str:
        return self.summary.to_string(float_format=lambda v: f"{v:+.4f}")


class ReturnAttributor:
    """CNE6 因子收益归因器。

    Parameters
    ----------
    risk_model : CNE6RiskModel
        因子暴露来源（须与 factor_loader 同源，即均来自 test_barra_cne6_gao 的同一S/L模型）。
    factor_loader : FactorReturnLoader
        因子收益 / 个股特质收益来源。
    """

    def __init__(self, risk_model: CNE6RiskModel, factor_loader: FactorReturnLoader) -> None:
        self.risk_model = risk_model
        self.factor_loader = factor_loader

    def run(
        self,
        weight_df: pd.DataFrame,
        benchmark_weight_df: pd.DataFrame,
        adj_close: pd.DataFrame,
        actual_weight_df: pd.DataFrame | None = None,
        realized_portfolio_return: pd.Series | None = None,
    ) -> AttributionResult:
        """
        Parameters
        ----------
        weight_df : 组合调仓权重，index=调仓（信号）日, columns=ticker
        benchmark_weight_df : 基准权重，同形状；调仓日可与 weight_df 不同，
            按 weight_df 的调仓日 asof（取 ≤ 该日的最近基准快照）对齐。
            alpha_max 无自然基准时可传等权全市场权重矩阵。
        adj_close : 复权收盘价，index=交易日, columns=ticker。用于计算不含
            成本/执行摩擦的收盘到收盘持有收益，供 Barra 残差自检。
        actual_weight_df : 逐日实际股票权重，通常来自真实回测成交账本。
            None 时兼容旧用法，在每个持有期内沿用目标权重。
        realized_portfolio_return : 成交账本产生的真实组合日收益，包含 VWAP 成交、
            费用和滑点。提供时以它作为组合收益真值，并将其与收盘持有收益之差
            计入 ``execution_effect``；None 时执行影响为 0。

        Returns
        -------
        AttributionResult
        """
        weight_df = _ensure_datetime_index(weight_df)
        benchmark_weight_df = _ensure_datetime_index(benchmark_weight_df)
        adj_close = _ensure_datetime_index(adj_close)
        if actual_weight_df is not None:
            actual_weight_df = _ensure_datetime_index(actual_weight_df)
        if realized_portfolio_return is not None:
            realized_portfolio_return = _ensure_datetime_series(
                realized_portfolio_return
            )

        rebal_dates = list(weight_df.index)
        if not rebal_dates:
            raise ValueError("weight_df 为空")
        bm_dates = list(benchmark_weight_df.index)

        # 持有期为 (T_i, T_{i+1}]：T_i 当天仍按上一期权重计入收益，
        # T_i 决定的新权重从 T_i+1 起才实际持有（T+1 执行）。
        trading_dates = [d for d in adj_close.index if d > rebal_dates[0]]
        daily_ret = adj_close.pct_change(fill_method=None)

        # d 日收益必须配 d-1 日**收盘**持仓：``actual_weight_df`` 的每一行由
        # 回测引擎在当日 mark-to-market 之后写入（backtest/engine.py），已经
        # 含当日涨跌。若直接用同日权重乘同日收益，等价于「当天涨得多的票自动
        # 加大权重」，凭空多出 Σ w_i·r_i²（截面收益方差，恒为正）。实测该偏差
        # 可达 +11%/年，且会按暴露结构分摊、主要堆进「特质(选股)」——恰是本
        # 模块要回答的问题，结论会被系统性带偏成「选股能力很强」。
        lagged_actual_weight = (
            actual_weight_df.shift(1) if actual_weight_df is not None else None
        )

        factor_names: list[str] | None = None
        factor_rows: dict = {}
        specific_rows: dict = {}
        holding_portfolio_rows: dict = {}
        benchmark_rows: dict = {}
        coverage_rows: dict = {}
        skipped = 0

        for i, t_i in enumerate(rebal_dates):
            t_next = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
            period_dates = [
                d for d in trading_dates
                if d > t_i and (t_next is None or d <= t_next)
            ]
            if not period_dates:
                continue

            target_weight = weight_df.loc[t_i].dropna()
            target_weight = target_weight[target_weight != 0]

            bm_asof = [d for d in bm_dates if d <= t_i]
            if not bm_asof:
                skipped += len(period_dates)
                continue
            w_b = benchmark_weight_df.loc[bm_asof[-1]].dropna()
            w_b = w_b[w_b != 0]

            if lagged_actual_weight is None:
                # 兼容旧用法：持有期内沿用目标权重。目标权重在期内是常数，
                # 不随日漂移，因此不存在同日自我引用问题，无需滞后。
                period_weight = pd.DataFrame(
                    [target_weight] * len(period_dates), index=period_dates
                ).fillna(0.0)
            else:
                period_weight = (
                    lagged_actual_weight.reindex(period_dates).ffill().fillna(0.0)
                )
            held_columns = period_weight.columns[
                (period_weight.abs() > 1e-12).any(axis=0)
            ]
            tickers = list(pd.Index(held_columns).union(w_b.index))

            risk_snap = self.risk_model.at(t_i.date(), tickers)
            if risk_snap is None:
                skipped += len(period_dates)
                continue
            if factor_names is None:
                factor_names = risk_snap.factor_names

            for d in period_dates:
                w_p = period_weight.loc[d].reindex(tickers).fillna(0.0)
                w_b_aligned = w_b.reindex(tickers).fillna(0.0)
                w_active = w_p - w_b_aligned
                w_arr = w_active.values
                x_active = w_arr @ risk_snap.X

                f_t = self.factor_loader.factor_return(d.date(), factor_names)
                factor_rows[d] = x_active * f_t.values

                u_t = self.factor_loader.specific_return(d.date(), tickers)
                specific_rows[d] = float(w_arr @ u_t.values)

                r_t = daily_ret.loc[d].reindex(tickers).fillna(0.0)
                holding_portfolio_rows[d] = float(w_p.values @ r_t.values)
                benchmark_rows[d] = float(w_b_aligned.values @ r_t.values)
                total_active_l1 = float(np.abs(w_arr).sum())
                coverage_rows[d] = (
                    float(np.abs(w_arr[risk_snap.covered_mask]).sum()) / total_active_l1
                    if total_active_l1 > 1e-12 else float("nan")
                )

        if not factor_rows:
            raise ValueError("无可归因的调仓期（检查风险面板/因子收益覆盖范围是否与回测区间重叠）")

        factor_daily = pd.DataFrame(factor_rows, index=factor_names).T.sort_index()
        specific_daily = pd.Series(specific_rows).sort_index()
        holding_portfolio_return = pd.Series(holding_portfolio_rows).sort_index()
        benchmark_return = pd.Series(benchmark_rows).sort_index()
        coverage = pd.Series(coverage_rows).sort_index()
        if realized_portfolio_return is None:
            portfolio_return = holding_portfolio_return.copy()
        else:
            missing_dates = holding_portfolio_return.index.difference(
                realized_portfolio_return.index
            )
            if len(missing_dates):
                sample = ", ".join(str(d.date()) for d in missing_dates[:3])
                raise ValueError(f"真实组合日收益缺少归因日期：{sample}")
            portfolio_return = (
                realized_portfolio_return.reindex(holding_portfolio_return.index)
                .astype(float)
            )
            if not np.isfinite(portfolio_return.to_numpy()).all():
                raise ValueError("真实组合日收益包含 NaN 或无穷值")

        style_cols = [c for c in factor_daily.columns if c in STYLE_FACTORS]
        industry_cols = [
            c for c in factor_daily.columns if c not in STYLE_FACTORS and c != _COUNTRY
        ]

        daily = pd.DataFrame({
            "style_total": factor_daily[style_cols].sum(axis=1),
            "industry_total": factor_daily[industry_cols].sum(axis=1),
            "country": factor_daily[_COUNTRY] if _COUNTRY in factor_daily.columns else 0.0,
            "specific": specific_daily,
            "holding_portfolio_return": holding_portfolio_return,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "coverage_pct": coverage,
        })
        daily["model_active_return"] = (
            daily["holding_portfolio_return"] - daily["benchmark_return"]
        )
        daily["execution_effect"] = (
            daily["portfolio_return"] - daily["holding_portfolio_return"]
        )
        daily["active_return"] = (
            daily["portfolio_return"] - daily["benchmark_return"]
        )
        daily["explained"] = (
            daily["style_total"] + daily["industry_total"] + daily["country"] + daily["specific"]
        )
        daily["residual"] = daily["model_active_return"] - daily["explained"]
        daily["relative_active_return"] = (
            (1.0 + daily["portfolio_return"])
            / (1.0 + daily["benchmark_return"])
            - 1.0
        )

        items: dict[str, pd.Series] = {f: factor_daily[f] for f in style_cols}
        items["行业合计"] = daily["industry_total"]
        items["Country"] = daily["country"]
        items["特质(选股)"] = daily["specific"]
        items["残差"] = daily["residual"]
        items["执行影响(含费用)"] = daily["execution_effect"]
        items["合计(主动收益)"] = daily["active_return"]

        summary = _carino_summary(
            items,
            daily["portfolio_return"],
            daily["benchmark_return"],
        )

        return AttributionResult(factor_daily=factor_daily, daily=daily, summary=summary)


__all__ = ["ReturnAttributor", "AttributionResult"]
