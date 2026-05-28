"""
向量化回测引擎。

逻辑：
  - 调仓日按目标权重建仓，扣除单边交易成本
  - 调仓间隔内权重随价格自然漂移
  - 每日组合收益 = Σ w_i * r_i（漂移后权重）
  - 基准收益 = HS300 / ZZ500 等权日收益（或外部传入）

输入：
  weight_df  : pd.DataFrame  (调仓日期 × ticker)
  adj_close  : pd.DataFrame  (全部交易日 × ticker)
  benchmark  : pd.Series     (全部交易日，日收益率，可选)

输出：
  BacktestResult  包含净值、超额收益、绩效指标等
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


class Backtester:
    """
    向量化回测引擎。

    Parameters
    ----------
    cost_one_way : float
        单边交易成本（佣金+冲击），默认 0.15%
    risk_free : float
        无风险年化利率，用于 Sharpe 计算，默认 0.02
    """

    def __init__(
        self,
        cost_one_way: float = 0.0015,
        risk_free: float = 0.02,
    ) -> None:
        self.cost_one_way = cost_one_way
        self.risk_free = risk_free

    def run(
        self,
        weight_df: pd.DataFrame,
        adj_close: pd.DataFrame,
        benchmark_ret: pd.Series | None = None,
    ) -> BacktestResult:
        """
        执行回测。

        Parameters
        ----------
        weight_df : pd.DataFrame
            调仓权重矩阵，index=调仓日，columns=ticker
        adj_close : pd.DataFrame
            复权收盘价，index=所有交易日，columns=ticker
        benchmark_ret : pd.Series | None
            基准日收益率序列；None 时用 adj_close 全股等权收益

        Returns
        -------
        BacktestResult
        """
        # ── 对齐 ticker & 日期 ─────────────────────────────────────
        tickers = weight_df.columns.tolist()
        adj = adj_close.reindex(columns=tickers).sort_index()

        # 日收益率（复权）
        daily_ret_all = adj.pct_change(fill_method=None).fillna(0.0)

        # 回测区间：weight_df 首个调仓日的次日起
        rebal_dates = sorted(weight_df.index)
        all_dates   = adj.index[adj.index >= rebal_dates[0]]

        # ── 逐日组合收益（含自然漂移）────────────────────────────
        port_ret   = pd.Series(0.0, index=all_dates)
        turnover_s = pd.Series(dtype=float)

        # 初始化权重为第一个调仓日的权重
        rebal_idx = 0
        w_current = weight_df.loc[rebal_dates[0]].reindex(tickers).fillna(0.0).values.copy()
        rebal_set  = set(rebal_dates)

        turnover_records: dict = {}

        for dt in all_dates:
            # 当日收益向量
            r = daily_ret_all.loc[dt].reindex(tickers).fillna(0.0).values

            # 如果是调仓日（且不是第一个），先扣交易成本
            if dt in rebal_set and dt != rebal_dates[0]:
                w_target  = weight_df.loc[dt].reindex(tickers).fillna(0.0).values
                # 漂移后权重归一化
                w_drifted = w_current / (w_current.sum() + 1e-12)
                bilateral_turnover = float(np.abs(w_target - w_drifted).sum())
                cost = bilateral_turnover * self.cost_one_way
                # 当日收益先用漂移权重，再扣成本，然后切换到目标权重
                day_ret = float(w_drifted @ r) - cost
                port_ret[dt] = day_ret
                turnover_records[dt] = bilateral_turnover
                # 更新为目标权重
                w_current = w_target.copy()
            else:
                # 普通持有日：用当前权重计算收益
                w_norm    = w_current / (w_current.sum() + 1e-12)
                day_ret   = float(w_norm @ r)
                port_ret[dt] = day_ret
                # 自然漂移更新权重
                w_current = w_current * (1.0 + r)

        turnover_s = pd.Series(turnover_records, name="turnover")

        # ── 基准收益 ───────────────────────────────────────────────
        if benchmark_ret is None:
            # 等权全股基准
            benchmark_ret = daily_ret_all.mean(axis=1)
        bm_ret = benchmark_ret.reindex(all_dates).fillna(0.0)

        # ── 净值 ───────────────────────────────────────────────────
        nav     = (1 + port_ret).cumprod()
        bm_nav  = (1 + bm_ret).cumprod()
        exc_ret = port_ret - bm_ret
        exc_nav = nav / bm_nav

        # ── 绩效指标 ───────────────────────────────────────────────
        pm = self._calc_metrics(port_ret, bm_ret)
        bm = self._calc_metrics(bm_ret, pd.Series(0.0, index=bm_ret.index))

        return BacktestResult(
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

    def _calc_metrics(
        self,
        ret: pd.Series,
        bm: pd.Series,
    ) -> PerformanceMetrics:
        n_days   = len(ret)
        n_years  = n_days / 252

        # 年化收益
        total_ret = (1 + ret).prod() - 1
        ann_ret   = (1 + total_ret) ** (1 / n_years) - 1

        # 年化波动
        ann_vol   = ret.std() * np.sqrt(252)

        # Sharpe
        rf_daily  = (1 + self.risk_free) ** (1 / 252) - 1
        sharpe    = (ret.mean() - rf_daily) / (ret.std() + 1e-12) * np.sqrt(252)

        # 最大回撤
        nav       = (1 + ret).cumprod()
        drawdown  = nav / nav.cummax() - 1
        max_dd    = float(drawdown.min())

        # Calmar
        calmar    = ann_ret / (abs(max_dd) + 1e-12)

        # 超额收益相关
        exc       = ret - bm
        ann_exc   = exc.mean() * 252
        exc_vol   = exc.std() * np.sqrt(252)   # 跟踪误差 TE
        ir        = ann_exc / (exc_vol + 1e-12)

        # 超额净值（几何）& 超额回撤
        port_nav     = (1 + ret).cumprod()
        bm_nav_local = (1 + bm).cumprod()
        # 防止 bm_nav=0 引起除零
        exc_nav_loc  = port_nav / bm_nav_local.replace(0, np.nan)
        exc_nav_loc  = exc_nav_loc.ffill().fillna(1.0)
        exc_dd_series = exc_nav_loc / exc_nav_loc.cummax() - 1
        exc_max_dd   = float(exc_dd_series.min()) if len(exc_dd_series) > 0 else 0.0
        exc_calmar   = ann_exc / (abs(exc_max_dd) + 1e-12)

        # 月度胜率
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
