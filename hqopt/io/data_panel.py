"""A 股日频行情面板（Polars LazyFrame 端到端）。

直接读取 ``data/cache/ashare_daily_<year>.parquet``，按需选列、切片日期。
涨跌停价 ``limit_up`` / ``limit_down`` 使用缓存真实字段，不在此模块推算。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# 逻辑列名 -> 缓存物理列名（旧缓存可能仍用 avg_price）
_CACHE_COLUMN_ALIASES: dict[str, str] = {
    "avg_price": "vwap",
}

# 行情面板默认列：覆盖因子计算、universe 过滤、T+1 成交、涨跌停判定
DEFAULT_PANEL_COLUMNS: tuple[str, ...] = (
    "code",
    "date",
    "adj_close",
    "adj_factor",
    "vwap",
    "adj_vwap",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "limit_up",
    "limit_down",
    "volume",
    "amount",
    "turnover",
    "float_mv",
    "industry_l1",
    "is_st",
    "is_hs300",
    "is_zz500",
    "is_zz1000",
    "trade_status",
    "list_date",
    "delist_date",
)


def _years_in_range(t1: date, t2: date) -> list[int]:
    if t1 > t2:
        raise ValueError(f"t1 ({t1}) 不能晚于 t2 ({t2})")
    return list(range(t1.year, t2.year + 1))


def _cache_files(years: Sequence[int], cache_dir: Path) -> list[Path]:
    paths: list[Path] = []
    missing: list[int] = []
    for y in years:
        p = cache_dir / f"ashare_daily_{y}.parquet"
        if p.exists():
            paths.append(p)
        else:
            missing.append(y)
    if missing:
        missing_str = ", ".join(str(y) for y in missing)
        raise FileNotFoundError(
            f"缺少本地缓存: {missing_str} 年。"
            f"请确认 {cache_dir} 目录下存在对应年份的 parquet 文件。"
        )
    return paths


def _cache_schema_names(files: Sequence[Path]) -> set[str]:
    return set(pl.scan_parquet(files[0]).collect_schema().names())


def _physical_columns(logical: Sequence[str], cache_names: set[str]) -> tuple[list[str], dict[str, str]]:
    """将逻辑列名映射为 Parquet 物理列；返回 (select 列表, 加载后 rename)。"""
    select: list[str] = []
    rename: dict[str, str] = {}
    for col in logical:
        if col in cache_names:
            if col not in select:
                select.append(col)
            continue
        alias = _CACHE_COLUMN_ALIASES.get(col)
        if alias and alias in cache_names:
            if alias not in select:
                select.append(alias)
            rename[alias] = col
            continue
        if col == "adj_vwap" and "adj_vwap" not in cache_names:
            continue
        raise KeyError(
            f"缓存缺少列 {col!r}（已尝试别名 {_CACHE_COLUMN_ALIASES.get(col)!r}）。"
            f"可用列: {sorted(cache_names)}"
        )
    return select, rename


def load_panel(
    t1: date,
    t2: date,
    columns: Sequence[str] | None = None,
    cache_dir: Path | str | None = None,
    add_adj_vwap: bool = True,
) -> pl.DataFrame:
    """加载行情面板。

    Args:
        t1: 起始日期（闭区间）
        t2: 结束日期（闭区间）
        columns: 要返回的列；None 时返回 DEFAULT_PANEL_COLUMNS。code/date 总是包含。
        cache_dir: 缓存目录，默认 data/cache。
        add_adj_vwap: 缓存无 adj_vwap 时，是否用 vwap * adj_factor 派生。

    Returns:
        polars.DataFrame，按 (date, code) 升序排序。
    """
    cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    logical = _resolve_columns(columns, add_adj_vwap=add_adj_vwap)
    files = _cache_files(_years_in_range(t1, t2), cache_path)
    cache_names = _cache_schema_names(files)
    physical, rename = _physical_columns(logical, cache_names)

    lf = pl.scan_parquet(files).select(physical)
    if rename:
        lf = lf.rename(rename)
    lf = lf.with_columns(pl.col("date").cast(pl.Date))
    lf = lf.filter((pl.col("date") >= t1) & (pl.col("date") <= t2))

    collected = lf.collect()
    if add_adj_vwap and "adj_vwap" in logical and "adj_vwap" not in collected.columns:
        vwap_col = "vwap" if "vwap" in collected.columns else "avg_price"
        if vwap_col in collected.columns and "adj_factor" in collected.columns:
            collected = collected.with_columns(
                (pl.col(vwap_col) * pl.col("adj_factor")).alias("adj_vwap")
            )

    return collected.sort(["date", "code"])


def _resolve_columns(
    columns: Sequence[str] | None,
    *,
    add_adj_vwap: bool,
) -> list[str]:
    if columns is None:
        selected = list(DEFAULT_PANEL_COLUMNS)
    else:
        selected = list(dict.fromkeys(["code", "date", *columns]))

    if add_adj_vwap and "adj_vwap" not in selected:
        selected.append("adj_vwap")
        for needed in ("vwap", "adj_factor"):
            if needed not in selected:
                selected.append(needed)

    return selected
