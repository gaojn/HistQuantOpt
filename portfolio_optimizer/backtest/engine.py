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
    ann_exc   = exc.mean() * 252
    exc_vol   = exc.std() * np.sqrt(252)   # 跟踪误差 TE
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
        first_rebal = rebal_dates[0]

        all_dates = adj_close.index[adj_close.index >= first_rebal]

        # ── 预处理：对齐到 all_dates ───────────────────────────────
        def align(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(index=all_dates)

        ac   = align(adj_close)
        av   = align(adj_vwap)
        cr   = align(close_raw)
        lu   = align(limit_up_df)
        ld   = align(limit_down_df)
        ts   = align(trade_status_df)

        # 涨跌停和停牌布尔矩阵（ticker 轴对齐到 adj_close 的 columns）
        common = adj_close.columns
        is_lup = (cr.reindex(columns=common) >= lu.reindex(columns=common) * 0.999).fillna(False)
        is_ldn = (cr.reindex(columns=common) <= ld.reindex(columns=common) * 1.001).fillna(False)
        is_sus = (ts.reindex(columns=common) == "停牌").fillna(False)

        # ── 状态 ─────────────────────────────────────────────────
        shares: dict[str, float]       = {}   # ticker → 持仓份额（分数股）
        cash: float                    = initial_value
        pending_sells: dict[str, float] = {}  # ticker → 待卖份额

        # ── 输出 ─────────────────────────────────────────────────
        port_values  = pd.Series(0.0, index=all_dates)
        turnover_rec: dict = {}
        cash_ratio_rec: list[float] = []

        # 执行统计
        buy_fail_cnt  = 0
        sell_defer_cnt = 0

        pending_rebal: pd.Timestamp | None = None   # T 日信号，T+1 执行

        for date in all_dates:
            # 取当日各矩阵的行（Series，index=ticker）
            ac_row = ac.loc[date]
            av_row = av.loc[date]
            lup_row = is_lup.loc[date]
            ldn_row = is_ldn.loc[date]
            sus_row = is_sus.loc[date]

            def exec_p(ticker: str) -> float:
                """当日 adj_vwap，若无效返回 0。"""
                v = av_row.get(ticker)
                return float(v) if v is not None and pd.notna(v) and v > 0 else 0.0

            def cant_buy(ticker: str) -> bool:
                return bool(sus_row.get(ticker, False)) or bool(lup_row.get(ticker, False))

            def cant_sell(ticker: str) -> bool:
                return bool(sus_row.get(ticker, False)) or bool(ldn_row.get(ticker, False))

            # ── 1. 执行待卖单（跌停/停牌后延续尝试）────────────────
            for ticker in list(pending_sells):
                sh = pending_sells[ticker]
                if sh < 1e-10:
                    del pending_sells[ticker]
                    continue
                if cant_sell(ticker):
                    continue   # 继续延迟
                p = exec_p(ticker)
                if p <= 0:
                    continue
                actual_sh = shares.get(ticker, 0.0)
                sh_sold   = min(sh, actual_sh)
                if sh_sold > 1e-10:
                    shares[ticker] = actual_sh - sh_sold
                    cash += sh_sold * p * (1.0 - self.cost_sell)
                del pending_sells[ticker]

            # ── 2. 执行 T 日调仓信号（T+1 成交）─────────────────
            if pending_rebal is not None and pending_rebal in weight_df.index:
                tgt_w = weight_df.loc[pending_rebal].fillna(0.0)

                # 当前总市值（按 adj_vwap 估值）
                total_val = cash
                for t, sh in shares.items():
                    p = exec_p(t)
                    if p > 0:
                        total_val += sh * p

                # 目标价值
                tgt_vals = (tgt_w * total_val).to_dict()

                # 当前价值
                cur_vals: dict[str, float] = {}
                for t, sh in shares.items():
                    p = exec_p(t)
                    if p > 0:
                        cur_vals[t] = sh * p

                # 新调仓清空旧的待卖（以新目标为准）
                pending_sells.clear()

                sell_orders: dict[str, float] = {}
                buy_orders:  dict[str, float] = {}
                all_tickers = set(cur_vals) | set(tgt_vals)

                for ticker in all_tickers:
                    delta = tgt_vals.get(ticker, 0.0) - cur_vals.get(ticker, 0.0)
                    if delta < -1.0:
                        sell_orders[ticker] = -delta
                    elif delta > 1.0:
                        buy_orders[ticker] = delta

                # 先执行卖出，释放现金
                sell_total = 0.0
                for ticker, sell_val in sell_orders.items():
                    # 先判可否卖出：不可卖（跌停/停牌）则进延期队列
                    if cant_sell(ticker):
                        p = exec_p(ticker)
                        if p > 0:
                            pending_sells[ticker] = pending_sells.get(ticker, 0.0) + sell_val / p
                            sell_defer_cnt += 1
                        continue
                    p = exec_p(ticker)
                    if p <= 0:
                        continue
                    actual_sh = shares.get(ticker, 0.0)
                    sh_to_sell = min(sell_val / p, actual_sh)
                    if sh_to_sell > 1e-10:
                        shares[ticker] = actual_sh - sh_to_sell
                        proceeds = sh_to_sell * p * (1.0 - self.cost_sell)
                        cash += proceeds
                        sell_total += sh_to_sell * p

                # 再执行买入（按需比例缩放，防超支）
                buy_demand = sum(
                    v for t, v in buy_orders.items() if not cant_buy(t) and exec_p(t) > 0
                )
                scale = min(1.0, cash / (buy_demand * (1.0 + self.cost_buy) + 1e-8)) \
                        if buy_demand > 0 else 0.0

                buy_total = 0.0
                for ticker, buy_val in buy_orders.items():
                    p = exec_p(ticker)
                    if p <= 0:
                        continue
                    if cant_buy(ticker):
                        buy_fail_cnt += 1
                        continue
                    actual_buy = buy_val * scale
                    if actual_buy < 1.0:
                        continue
                    sh_bought = actual_buy / p
                    cost = sh_bought * p * self.cost_buy
                    shares[ticker] = shares.get(ticker, 0.0) + sh_bought
                    cash -= sh_bought * p + cost
                    buy_total += sh_bought * p

                cash = max(cash, 0.0)

                # 记录换手率（双边 / 总资产）
                turnover_rec[date] = (sell_total + buy_total) / (total_val + 1e-12)
                pending_rebal = None

            # ── 3. 计算当日 NAV（adj_close 估值）───────────────
            nav_val = cash
            for t, sh in shares.items():
                p = ac_row.get(t)
                if p is not None and pd.notna(p) and p > 0 and sh > 1e-10:
                    nav_val += sh * p
            port_values[date] = nav_val
            if nav_val > 1e-8:
                cash_ratio_rec.append(cash / nav_val)

            # ── 4. 更新调仓信号 ─────────────────────────────────
            if date in set(rebal_dates):
                pending_rebal = date

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
        )

        exec_stats = {
            "buy_fail_count":   buy_fail_cnt,
            "sell_defer_count": sell_defer_cnt,
            "avg_cash_pct":     float(np.mean(cash_ratio_rec)) if cash_ratio_rec else 0.0,
        }

        return result, exec_stats
