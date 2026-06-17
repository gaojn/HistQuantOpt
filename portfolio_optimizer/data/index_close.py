"""
指数日收盘价加载器。

从本地 CSV（默认 data/指数收盘价信息.csv）读取官方宽基指数收盘价，
供回测/报告作为基准（benchmark）使用。

CSV 格式（宽表）：
    date(YYYYMMDD), 沪深300, 中证500, 中证800, 中证1000, 中证全指, 中证红利, 万得全A

用法::

    from portfolio_optimizer.data.index_close import load_index_close, load_index_returns
    close = load_index_close("zz1000", "2020-01-01", "2026-05-31")   # 收盘价 Series
    ret   = load_index_returns("zz1000", "2020-01-01", "2026-05-31") # 日收益 Series
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/指数收盘价信息.csv")

# 指数 key（与项目其它处一致的英文 key / 别名）→ CSV 列名
_INDEX_COL: dict[str, str] = {
    "hs300": "沪深300",
    "zz500": "中证500",
    "zz800": "中证800",
    "zz1000": "中证1000",
    "csiall": "中证全指",      # 000985 中证全指
    "zzall": "中证全指",
    "winda": "万得全A",
    "wanda": "万得全A",
    "dividend": "中证红利",
}


def _resolve_column(index: str) -> str:
    """把指数 key（hs300/zz1000/…）或中文列名解析成 CSV 列名。"""
    if index in _INDEX_COL:
        return _INDEX_COL[index]
    return index  # 允许直接传中文列名，如 "中证全指"


def load_index_close(
    index: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    path: Path | str = DEFAULT_PATH,
) -> pd.Series:
    """
    读取某指数的日收盘价。

    Parameters
    ----------
    index : str
        指数 key（hs300/zz500/zz800/zz1000/csiall/winda）或 CSV 中文列名。
    start, end : 可选
        日期区间（含端点）。
    path : 默认 data/指数收盘价信息.csv

    Returns
    -------
    pd.Series  index=DatetimeIndex(date), name=列名，已剔除缺失值并排序。
    """
    col = _resolve_column(index)
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(
            f"指数列 '{col}' 不在 {path}，可用列：{[c for c in df.columns if c != 'date']}"
        )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    s = df.set_index("date")[col].astype("float64").dropna().sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    if s.empty:
        raise ValueError(f"指数 '{col}' 在 {start}~{end} 区间无数据")
    return s.rename(col)


def load_index_returns(
    index: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    path: Path | str = DEFAULT_PATH,
) -> pd.Series:
    """
    读取某指数的日收益率（pct_change）。

    为保证区间内首日有收益，会多取一天前置收盘价计算收益后再裁剪到 [start, end]。
    """
    # 多取前置数据，避免 start 当日收益为 NaN
    raw = load_index_close(index, start=None, end=end, path=path)
    ret = raw.pct_change().dropna()
    if start is not None:
        ret = ret[ret.index >= pd.Timestamp(start)]
    return ret.rename(_resolve_column(index))


def available_indices(path: Path | str = DEFAULT_PATH) -> list[str]:
    """返回 CSV 中实际有数据（非全空）的指数列名。"""
    df = pd.read_csv(path)
    return [c for c in df.columns if c != "date" and df[c].notna().any()]
