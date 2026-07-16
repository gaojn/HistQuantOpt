"""
回测引擎（真实执行）+ 绩效指标。

RealisticBacktester：T+1 VWAP 成交 + 涨跌停 + 停牌处理。
执行规则：
  调仓信号日 T 收盘后生成目标权重，T+1 日以 adj_vwap 成交。
  涨停（close ≥ limit_up × 99.9%）：无法买入，资金留现金
  跌停（close ≤ limit_down × 100.1%）：无法卖出，进延期队列重试
  停牌（trade_status == '停牌'）：买卖均不能执行
  成本：买入 0.1%（1‰），卖出 0.2%（2‰），非对称
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hqopt.backtest.execution import ExecutionLedger


@dataclass
class PerformanceMetrics:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate_monthly: float        # 月度胜率（相对基准）
    avg_monthly_excess: float      # 月均超额收益
    annual_excess_return: float    # 年化超额收益
    info_ratio: float              # 信息比率
    tracking_error: float = 0.0    # 跟踪误差（超额收益年化波动）
    excess_max_drawdown: float = 0.0   # 超额净值最大回撤
    excess_calmar: float = 0.0     # 年化超额 / |超额最大回撤|

    def __str__(self) -> str:
        return (
            f"年化收益       : {self.annual_return*100:+.2f}%\n"
            f"年化波动       : {self.annual_vol*100:.2f}%\n"
            f"Sharpe         : {self.sharpe:.3f}\n"
            f"最大回撤       : {self.max_drawdown*100:.2f}%\n"
            f"Calmar         : {self.calmar:.3f}\n"
            f"年化超额       : {self.annual_excess_return*100:+.2f}%\n"
            f"跟踪误差(TE)   : {self.tracking_error*100:.2f}%\n"
            f"信息比率(IR)   : {self.info_ratio:.3f}\n"
            f"超额最大回撤   : {self.excess_max_drawdown*100:.2f}%\n"
            f"超额Calmar     : {self.excess_calmar:.3f}\n"
            f"月度胜率       : {self.win_rate_monthly*100:.1f}%\n"
            f"月均超额       : {self.avg_monthly_excess*100:+.3f}%"
        )


@dataclass
class BacktestResult:
    nav: pd.Series               # 组合净值（从1开始）
    bm_nav: pd.Series            # 基准净值
    excess_nav: pd.Series        # 超额净值（nav / bm_nav）
    daily_ret: pd.Series         # 组合日收益
    bm_ret: pd.Series            # 基准日收益
    excess_ret: pd.Series        # 超额日收益
    turnover: pd.Series          # 调仓日双边换手率
    portfolio_metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    risk_free: float = 0.02      # 年化无风险利率（Sharpe 用，与 calc_metrics 保持一致）
    actual_weights: pd.DataFrame | None = None  # 每日收盘实际股票权重（现金不在列内）

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  组合绩效",
            "=" * 50,
            str(self.portfolio_metrics),
            "",
            "=" * 50,
            "  基准绩效",
            "=" * 50,
            str(self.benchmark_metrics),
            "",
            f"平均调仓换手  : {self.turnover.mean()*100:.1f}%",
        ]
        return "\n".join(lines)


def calc_metrics(ret: pd.Series, bm: pd.Series, risk_free: float = 0.02) -> PerformanceMetrics:
    """根据组合日收益 ret 与基准日收益 bm 计算绩效指标。"""
    n_days  = len(ret)
    n_years = n_days / 252 if n_days > 0 else 1

    total_ret = (1 + ret).prod() - 1
    ann_ret   = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol   = ret.std() * np.sqrt(252)

    rf_daily  = (1 + risk_free) ** (1 / 252) - 1
    sharpe    = (ret.mean() - rf_daily) / (ret.std() + 1e-12) * np.sqrt(252)

    nav       = (1 + ret).cumprod()
    drawdown  = nav / nav.cummax() - 1
    max_dd    = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    calmar    = ann_ret / (abs(max_dd) + 1e-12)

    exc       = ret - bm
    # 超额收益用几何口径：(组合累计+1)/(基准累计+1)-1 后年化
    bm_total  = (1 + bm).prod() - 1
    total_exc_geo = (1 + total_ret) / (max(1 + bm_total, 1e-8)) - 1
    ann_exc   = (1 + total_exc_geo) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    exc_vol   = exc.std() * np.sqrt(252)   # 跟踪误差 TE（按日差异，不改口径）
    ir        = ann_exc / (exc_vol + 1e-12)

    # 超额净值（几何）& 超额回撤
    port_nav     = (1 + ret).cumprod()
    bm_nav_local = (1 + bm).cumprod()
    exc_nav_loc  = port_nav / bm_nav_local.replace(0, np.nan)
    exc_nav_loc  = exc_nav_loc.ffill().fillna(1.0)
    exc_dd_series = exc_nav_loc / exc_nav_loc.cummax() - 1
    exc_max_dd   = float(exc_dd_series.min()) if len(exc_dd_series) > 0 else 0.0
    exc_calmar   = ann_exc / (abs(exc_max_dd) + 1e-12)

    monthly_port = (1 + ret).resample("ME").prod() - 1
    monthly_bm   = (1 + bm).resample("ME").prod() - 1
    monthly_exc  = monthly_port - monthly_bm
    win_rate     = (monthly_exc > 0).mean()
    avg_exc_m    = monthly_exc.mean()

    return PerformanceMetrics(
        annual_return=ann_ret,
        annual_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate_monthly=float(win_rate),
        avg_monthly_excess=float(avg_exc_m),
        annual_excess_return=ann_exc,
        info_ratio=ir,
        tracking_error=float(exc_vol),
        excess_max_drawdown=exc_max_dd,
        excess_calmar=float(exc_calmar),
    )


class RealisticBacktester:
    """
    T+1 VWAP 成交的真实回测引擎。

    Parameters
    ----------
    cost_buy  : float  买入费率，默认 0.1%（1‰）
    cost_sell : float  卖出费率，默认 0.2%（2‰）
    risk_free : float  年化无风险利率（用于 Sharpe）
    """

    def __init__(
        self,
        cost_buy: float = 0.001,
        cost_sell: float = 0.002,
        risk_free: float = 0.02,
    ) -> None:
        self.cost_buy  = cost_buy
        self.cost_sell = cost_sell
        self.risk_free = risk_free

    def run(
        self,
        weight_df: pd.DataFrame,
        adj_close: pd.DataFrame,
        adj_vwap: pd.DataFrame,
        close_raw: pd.DataFrame,
        limit_up_df: pd.DataFrame,
        limit_down_df: pd.DataFrame,
        trade_status_df: pd.DataFrame,
        benchmark_ret: pd.Series | None = None,
        initial_value: float = 1e8,
    ) -> tuple[BacktestResult, dict]:
        """
        执行回测。

        Parameters
        ----------
        weight_df       : 调仓权重矩阵，index=调仓日，columns=ticker
        adj_close       : 复权收盘价（用于 NAV 估值）
        adj_vwap        : 复权 VWAP（用于 T+1 成交价）
        close_raw       : 原始收盘价（用于涨跌停判断）
        limit_up_df     : 涨停价（原始）
        limit_down_df   : 跌停价（原始）
        trade_status_df : 交易状态（'交易'/'停牌'/...）
        benchmark_ret   : 基准日收益率（None=等权全股）
        initial_value   : 初始资金（元）

        Returns
        -------
        (BacktestResult, execution_stats dict)
        """
        # 统一 weight_df.index 为 pd.Timestamp，防止与 adj_close.index 类型不一致
        # 导致 `date in set(rebal_dates)` 永远不匹配
        if not isinstance(weight_df.index, pd.DatetimeIndex):
            weight_df = weight_df.copy()
            weight_df.index = pd.to_datetime(weight_df.index)

        rebal_dates = sorted(weight_df.index)
        missing_rebal_dates = pd.DatetimeIndex(rebal_dates).difference(
            pd.DatetimeIndex(adj_close.index)
        )
        if not missing_rebal_dates.empty:
            preview = ", ".join(
                date.strftime("%Y-%m-%d") for date in missing_rebal_dates[:5]
            )
            suffix = (
                f"（共 {len(missing_rebal_dates)} 日）"
                if len(missing_rebal_dates) > 5
                else ""
            )
            raise ValueError(f"调仓日不在行情交易日历中: {preview}{suffix}")
        first_rebal = rebal_dates[0]

        all_dates = adj_close.index[adj_close.index >= first_rebal]

        # ── 预处理：对齐到 all_dates ───────────────────────────────
        def align(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(index=all_dates)

        ac_raw = align(adj_close)        # 当日真实收盘价（退市/缺行为 NaN）
        # 估值用「最后有效价」：退市或数据缺失时，持仓按上一有效价计入 NAV，
        # 避免持仓凭空归零再于复牌/补数时跳回，造成净值假摔。
        # 注意：仅估值 ffill，成交价 av 不 ffill（退市后不可假装能成交）。
        ac   = ac_raw.ffill()
        av   = align(adj_vwap)
        cr   = align(close_raw)
        lu   = align(limit_up_df)
        ld   = align(limit_down_df)
        ts   = align(trade_status_df)

        # 成交账本同时被逐期优化使用，保证目标生成和回测采用完全一致的执行语义。
        ledger = ExecutionLedger(
            initial_value=initial_value,
            cost_buy=self.cost_buy,
            cost_sell=self.cost_sell,
        )

        # ── 输出 ─────────────────────────────────────────────────
        port_values  = pd.Series(0.0, index=all_dates)
        turnover_rec: dict = {}
        cash_ratio_rec: list[float] = []
        actual_weight_rows: dict = {}

        rebal_date_set = set(rebal_dates)

        for date in all_dates:
            day_result = ledger.step(
                adj_close=ac_raw.loc[date],
                adj_vwap=av.loc[date],
                close_raw=cr.loc[date],
                limit_up=lu.loc[date],
                limit_down=ld.loc[date],
                trade_status=ts.loc[date],
            )
            if day_result.turnover > 0:
                turnover_rec[date] = day_result.turnover

            # ── 2. 计算当日 NAV（最近有效 adj_close 估值）────────
            nav_val = ledger.nav
            port_values[date] = nav_val
            actual_weight_rows[date] = ledger.actual_weights()
            if nav_val > 1e-8:
                cash_ratio_rec.append(ledger.cash / nav_val)

            # ── 3. T 日收盘后提交目标，最早于下一交易日执行 ─────
            if date in rebal_date_set:
                ledger.submit_target(weight_df.loc[date].fillna(0.0))

        # ── 组合指标 ─────────────────────────────────────────────
        nav      = port_values / initial_value
        port_ret = nav.pct_change().fillna(0.0)

        if benchmark_ret is None:
            bm_ret = (
                adj_close.pct_change(fill_method=None)
                .fillna(0.0).mean(axis=1)
                .reindex(all_dates).fillna(0.0)
            )
        else:
            bm_ret = benchmark_ret.reindex(all_dates).fillna(0.0)

        bm_nav  = (1 + bm_ret).cumprod()
        exc_ret = port_ret - bm_ret
        exc_nav = nav / bm_nav

        turnover_s = pd.Series(turnover_rec, name="turnover")
        actual_weights = pd.DataFrame(actual_weight_rows).T.fillna(0.0)
        actual_weights.index.name = "date"

        # 退市/长停滞留告警：回测末日仍持有、但当日无真实价（靠 ffill 陈旧价估值）
        # 的持仓 —— 这些票实际已无法成交（退市或长期停牌），其估值不可全信。
        last_date    = all_dates[-1]
        ac_raw_last  = ac_raw.loc[last_date]
        ac_last      = ac.loc[last_date]
        stuck_cnt    = 0
        stuck_val    = 0.0
        for t, sh in ledger.shares.items():
            if sh <= 1e-10:
                continue
            if pd.isna(ac_raw_last.get(t)) and pd.notna(ac_last.get(t)):
                stuck_cnt += 1
                stuck_val += sh * float(ac_last.get(t))
        final_nav = float(port_values[last_date])
        stale_value_pct = stuck_val / final_nav if final_nav > 1e-8 else 0.0

        pm = calc_metrics(port_ret, bm_ret, self.risk_free)
        bm = calc_metrics(bm_ret, pd.Series(0.0, index=bm_ret.index), self.risk_free)

        result = BacktestResult(
            nav=nav,
            bm_nav=bm_nav,
            excess_nav=exc_nav,
            daily_ret=port_ret,
            bm_ret=bm_ret,
            excess_ret=exc_ret,
            turnover=turnover_s,
            portfolio_metrics=pm,
            benchmark_metrics=bm,
            risk_free=self.risk_free,
            actual_weights=actual_weights,
        )

        exec_stats = {
            "buy_fail_count":     ledger.buy_fail_count,
            "sell_defer_count":   ledger.sell_defer_count,
            "avg_cash_pct":       float(np.mean(cash_ratio_rec)) if cash_ratio_rec else 0.0,
            "final_cash_pct":     float(ledger.cash / final_nav) if final_nav > 1e-8 else 0.0,
            "final_actual_weights": ledger.actual_weights().to_dict(),
            "target_pending":     ledger.pending_target is not None,
            "delisted_stuck_count": stuck_cnt,        # 末日靠陈旧价估值的滞留持仓数
            "stale_value_pct":    float(stale_value_pct),  # 其价值占末日 NAV 比例
        }

        return result, exec_stats
