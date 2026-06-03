"""
候选池过滤与合成 Alpha 生成。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from portfolio_optimizer.data.generator import MarketSnapshot


def build_cost_vector(
    tickers: list[str],
    panel: pl.DataFrame,
    target_date: date,
    lookback: int = 20,
) -> np.ndarray:
    """
    计算个股冲击成本代理向量，归一化使中位数 = 1。

    公式：c_i = σ_i / sqrt(ADV_i)

    来源：Almgren-Chriss 冲击模型，单笔冲击成本 ≈ σ × sqrt(Q/ADV)。
    在下单量占组合比例固定时，c_i ∝ σ_i / sqrt(ADV_i)：
    波动大或流动性差的股票冲击成本更高。

    Parameters
    ----------
    tickers     : 目标股票列表
    panel       : 行情面板（需含 adj_close、amount、date、code）
    target_date : 调仓日（取该日及之前 lookback 个交易日）
    lookback    : 滚动窗口（交易日数），默认 20

    Returns
    -------
    np.ndarray, shape (N,)
        归一化成本权重，中位数=1，缺失/异常值填 1.0（等权）
    """
    ts = pd.Timestamp(target_date)

    # 取目标日之前（含）lookback 个交易日
    avail_dates = sorted(
        panel.filter(pl.col("date") <= target_date)
        .select("date").unique()["date"].to_list()
    )
    window_dates = avail_dates[-lookback:]
    if len(window_dates) < 5:
        return np.ones(len(tickers))

    sub = (
        panel.filter(
            (pl.col("date").is_in(window_dates)) &
            (pl.col("code").is_in(tickers))
        )
        .select(["date", "code", "adj_close", "amount"])
        .to_pandas()
        .pivot(index="date", columns="code", values=["adj_close", "amount"])
        .sort_index()
    )

    adj   = sub["adj_close"].reindex(columns=tickers)
    amt   = sub["amount"].reindex(columns=tickers)

    # 年化波动率（20日滚动标准差 × √252）
    daily_ret = adj.pct_change(fill_method=None)
    vol = daily_ret.std(ddof=1) * np.sqrt(252)          # pd.Series, index=ticker

    # ADV：窗口内日均成交额（千元）
    adv = amt.mean()                                     # pd.Series, index=ticker

    # c_i = σ_i / sqrt(ADV_i)，对零/NaN 做保护
    adv_safe = adv.clip(lower=1.0)
    c_raw = vol / np.sqrt(adv_safe)

    # 归一化：除以中位数，使中位数股票成本权重 = 1
    median = c_raw.median()
    if median > 1e-12:
        c_raw = c_raw / median

    # 缺失/异常 → 填 1.0（等权，不额外惩罚）
    c_raw = c_raw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    c_raw = c_raw.clip(lower=0.1, upper=10.0)          # 防极端值

    return c_raw.reindex(tickers).fillna(1.0).values.astype(float)


def filter_universe(
    snapshot: MarketSnapshot,
    panel: pl.DataFrame,
    target_date: date,
    exclude_bj: bool = True,
    exclude_st: bool = True,
    top_n: int | None = None,
) -> MarketSnapshot:
    """
    候选池过滤：剔除北交所、ST，可选按市值截取 TOP_N。

    Parameters
    ----------
    snapshot   : 原始市场快照
    panel      : 行情面板（含 is_st 字段）
    target_date: 调仓日
    exclude_bj : 是否剔除北交所（.BJ 结尾）
    exclude_st : 是否剔除 ST 股票
    top_n      : 按自由流通市值取前 N 只，None = 不限制
    """
    tickers = snapshot.tickers
    keep = list(tickers)

    if exclude_bj:
        keep = [t for t in keep if not t.endswith(".BJ")]

    if exclude_st:
        st_df = (
            panel.filter(pl.col("date") == target_date)
            .select(["code", "is_st"])
            .to_pandas()
            .set_index("code")
        )
        st_set = set(st_df[st_df["is_st"] == 1].index)
        keep = [t for t in keep if t not in st_set]

    if top_n is not None and len(keep) > top_n:
        cap = snapshot.market_cap.reindex(keep).fillna(0.0)
        keep = cap.nlargest(top_n).index.tolist()

    return replace(
        snapshot,
        tickers=keep,
        industry=snapshot.industry.reindex(keep),
        adv=snapshot.adv.reindex(keep),
        status=snapshot.status.reindex(keep),
        prev_weight=snapshot.prev_weight.reindex(keep).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(keep),
        is_constituent=(
            snapshot.is_constituent.reindex(keep)
            if snapshot.is_constituent is not None else None
        ),
    )


def build_synthetic_alpha(
    panel: pl.DataFrame,
    fwd_days: int = 5,
    ic_mean: float = 0.08,
    ic_std: float = 0.10,
    decay: float = 0.80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成 Alpha 矩阵（仅用于验证流程，不代表真实可交易收益）。

    Returns
    -------
    pd.DataFrame  index=date, columns=ticker
    """
    rng = np.random.default_rng(seed)
    adj = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    fwd_ret = adj.shift(-fwd_days) / adj - 1
    dates = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    rows: dict = {}
    f_prev: pd.Series | None = None
    for dt in dates:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50:
            continue
        mu, sig = r.mean(), r.std()
        if sig < 1e-8:
            continue
        z_r = (r - mu) / sig
        rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = pd.Series(
            rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps,
            index=r.index,
        )
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        if f_prev is None or decay == 0.0:
            f = new_sig
        else:
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    decay * f_prev[common]
                    + np.sqrt(max(1 - decay**2, 0)) * new_sig[common]
                )
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f
        rows[dt] = f

    alpha_df = pd.DataFrame(rows).T
    alpha_df.index.name = "date"
    return alpha_df


def get_alpha_for_date(
    alpha_df: pd.DataFrame,
    target_date: date,
    tickers: list[str],
) -> np.ndarray:
    """取最近可用日期的 Alpha，对齐到 tickers。"""
    ts = pd.Timestamp(target_date)
    avail = alpha_df.index[alpha_df.index <= ts]
    if len(avail) == 0:
        return np.zeros(len(tickers))
    return alpha_df.loc[avail[-1]].reindex(tickers).fillna(0.0).values.astype(float)
