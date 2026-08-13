"""权重 → T+1 VWAP 真实回测 → HTML 报告。

供 CLI（`hqopt backtest` / `hqopt run`）与脚本复用。权重文件支持：
  - 长表 parquet：列 [date, code, weight]
  - 宽表 parquet：index=date, columns=ticker
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from hqopt.backtest.engine import RealisticBacktester
from hqopt.backtest.execution import (
    align_sell_only_matrix,
    bundle_io_lock,
    ensure_bundle_not_in_progress,
    normalize_sell_only_matrix,
    sell_only_manifest_path_for_weights,
    sell_only_path_for_weights,
    synthetic_alpha_warning_path_for_weights,
    validate_sell_only_manifest,
)
from hqopt.backtest.report import generate_html_report
from hqopt.constants import (
    SYNTHETIC_ALPHA_WARNING_FILE,
    SYNTHETIC_ALPHA_WARNING_TEXT,
)
from hqopt.data.benchmark import (
    benchmark_returns_from_rebalance_weights,
    equal_weight_benchmark_weights,
)
from hqopt.data.index_close import load_index_returns
from hqopt.io.data_panel import load_panel

logger = logging.getLogger(__name__)

_EQUAL_WEIGHT_BENCHMARK = "equal_weight"


def _save_execution_stats(stats: dict[str, Any], path: str | Path) -> Path:
    """把执行统计以可审计 JSON 落盘。"""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item(),
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _load_weights(path: str | Path) -> pd.DataFrame:
    """加载权重，统一输出宽表（index=DatetimeIndex，columns=ticker）。"""
    df = pd.read_parquet(path)
    if {"date", "code", "weight"}.issubset(df.columns):       # 长表
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="code", values="weight").sort_index()
    else:                                                      # 宽表
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    df.index.name = "date"
    return df


def _load_sell_only_metadata(weight_path: str | Path) -> pd.DataFrame | None:
    """读取优化阶段输出的只卖不买矩阵；外部权重无旁路文件时返回 None。"""
    ensure_bundle_not_in_progress(weight_path)
    path = sell_only_path_for_weights(weight_path)
    if not path.exists():
        manifest = sell_only_manifest_path_for_weights(weight_path)
        if manifest.exists():
            validate_sell_only_manifest(weight_path)
        return None
    validate_sell_only_manifest(weight_path)
    frame = pd.read_parquet(path)
    return normalize_sell_only_matrix(
        frame,
        context=f"sell-only 元数据 {path}：",
    )


def _load_execution_bundle(
    weight_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """在同一共享锁内连续读取权重及其 sell-only 元数据。"""
    with bundle_io_lock(weight_path, exclusive=False):
        sell_only_metadata = _load_sell_only_metadata(weight_path)
        weight_df = _load_weights(weight_path)
    return weight_df, sell_only_metadata


def _align_sell_only_metadata(
    metadata: pd.DataFrame | None,
    weight_df: pd.DataFrame,
    weight_path: str | Path,
) -> pd.DataFrame | None:
    """严格绑定权重与 sell-only 元数据，避免错位时静默放开交易限制。"""
    metadata_path = sell_only_path_for_weights(weight_path)
    if metadata is None:
        logger.warning(
            "  未找到只卖元数据：%s；按外部权重模式执行，不推断 sell-only 限制",
            metadata_path,
        )
        return None

    logger.info("  已加载只卖元数据：%s", metadata_path)
    return align_sell_only_matrix(metadata, weight_df)


def _to_wide(panel: pl.DataFrame, col: str) -> pd.DataFrame:
    """从 polars 面板 pivot 指定列为宽表，index=DatetimeIndex。"""
    wide = (
        panel.select(["date", "code", col]).to_pandas()
        .pivot(index="date", columns="code", values=col)
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide


def _parse_date(s: str | date | pd.Timestamp) -> date:
    if isinstance(s, date):
        return s
    return pd.Timestamp(s).date()


def _detect_warning_banner(weight_path: str | Path) -> str | None:
    """权重 bundle 带合成 Alpha 警告文件时，报告必须携带水印。

    优化阶段（batch.publish）会在合成 Alpha 运行的权重旁落按 stem 隔离的
    SYNTHETIC_ALPHA_WARNING.txt；回测端据此自动接上 HTML 置顶横幅，
    避免"含前视信号的漂亮报告"被无提示地截图外传。目录级旧水印文件也
    保守地一并触发（宁可误报，不可漏报）。
    """
    weights = Path(weight_path).resolve()
    candidates = [
        synthetic_alpha_warning_path_for_weights(weights),
        weights.parent / SYNTHETIC_ALPHA_WARNING_FILE,   # 旧版目录级布局
    ]
    for warning_file in candidates:
        if warning_file.is_file():
            text = warning_file.read_text(encoding="utf-8").strip()
            return text or SYNTHETIC_ALPHA_WARNING_TEXT
    return None


def run_backtest(
    weight_path: str | Path,
    start_date: str | date,
    end_date: str | date,
    index: str = "zz1000",
    cost_buy: float = 0.001,
    cost_sell: float = 0.002,
    risk_free: float = 0.02,
    initial_value: float = 1e8,
    out_dir: str | Path | None = None,
    title: str | None = None,
    cache_dir: str | Path | None = None,
    index_close_path: str | Path | None = None,
    alpha_path: str | Path | None = None,
    warning_banner: str | None = None,
) -> tuple:
    """
    加载权重文件，执行 T+1 VWAP 真实回测，生成 HTML 报告。

    Parameters
    ----------
    weight_path : 权重 parquet 路径（长表或宽表）
    start_date / end_date : 回测区间（权重早于 start 的记录会被裁剪）
    index       : 基准 key（equal_weight / hs300 / zz500 / zz1000 / csiall ...）
    cost_buy / cost_sell : 买/卖费率（非对称），默认 1‰ / 2‰
    risk_free   : 无风险年化利率（Sharpe 用）
    initial_value : 初始资金（元）
    out_dir     : 报告输出目录，None 则不落地文件
    title       : 报告标题，None 自动生成
    cache_dir   : 行情 parquet 缓存目录，None 用默认
    index_close_path : 指数收盘价 CSV 路径，None 用默认
    alpha_path  : 报告"最后一期目标持仓"表 Alpha 列用的 alpha parquet 路径，
                  None 时该列显示"—"
    warning_banner : HTML 报告置顶风险水印文本。None（默认）时自动探测：
                  权重 bundle 带 SYNTHETIC_ALPHA_WARNING.txt 即取其内容。
                  注意空字符串 "" 会显式关闭探测与水印——仅当确认权重
                  绝无前视信号时才可传入

    Returns
    -------
    (BacktestResult, exec_stats dict)
    """
    t1 = _parse_date(start_date)
    t2 = _parse_date(end_date)

    # ── 1. 权重 ──────────────────────────────────────────────
    logger.info(f"\n[1] 加载权重：{weight_path}")
    all_weight_df, sell_only_metadata = _load_execution_bundle(weight_path)
    weight_df = all_weight_df[
        (all_weight_df.index >= pd.Timestamp(t1))
        & (all_weight_df.index <= pd.Timestamp(t2))
    ]
    if weight_df.empty:
        raise ValueError(f"权重文件在 {t1}~{t2} 无数据，请检查日期区间")
    sell_only_df = _align_sell_only_metadata(
        sell_only_metadata,
        all_weight_df,
        weight_path,
    )
    if sell_only_df is not None:
        sell_only_df = sell_only_df.loc[weight_df.index]
    actual_start = weight_df.index.min().date()
    actual_end = weight_df.index.max().date()
    n_stocks = (weight_df > 1e-6).any(axis=0).sum()
    logger.info(f"  调仓日={len(weight_df)}  股票池={n_stocks}  区间={actual_start}~{actual_end}")

    # ── 2. 行情面板（多取年初数据，确保首期 ADV/VWAP）──────────
    data_start = date(t1.year, 1, 1)
    logger.info(f"\n[2] 加载行情面板（{data_start} ~ {t2}）...")
    panel = load_panel(
        data_start, t2,
        columns=["code", "date", "adj_close", "adj_vwap", "close",
                 "limit_up", "limit_down", "trade_status"],
        cache_dir=cache_dir,
    )
    logger.info(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    # ── 3. 宽表 ──────────────────────────────────────────────
    logger.info("\n[3] 构建回测宽表...")
    adj_close_w = _to_wide(panel, "adj_close")
    adj_vwap_w = _to_wide(panel, "adj_vwap")
    close_raw_w = _to_wide(panel, "close")
    limit_up_w = _to_wide(panel, "limit_up")
    limit_down_w = _to_wide(panel, "limit_down")
    trade_status_w = _to_wide(panel, "trade_status")

    # ── 4. 基准收益 ──────────────────────────────────────────
    if index == _EQUAL_WEIGHT_BENCHMARK:
        logger.info("\n[4] 构建基准收益（全市场等权）...")
        benchmark_weights = equal_weight_benchmark_weights(adj_close_w)
        bm_ret = benchmark_returns_from_rebalance_weights(
            benchmark_weights,
            adj_close_w,
            list(weight_df.index),
        )
    else:
        logger.info(f"\n[4] 加载基准收益（{index.upper()}）...")
        index_close_kwargs = {} if index_close_path is None else {"path": index_close_path}
        bm_ret = (
            load_index_returns(index, start=str(t1), end=str(t2), **index_close_kwargs)
            .reindex(adj_close_w.index[adj_close_w.index >= pd.Timestamp(t1)])
            .fillna(0.0)
        )

    # ── 5. 回测 ──────────────────────────────────────────────
    logger.info("\n[5] 执行回测（T+1 VWAP，涨跌停/停牌，非对称成本）...")
    bt = RealisticBacktester(cost_buy=cost_buy, cost_sell=cost_sell, risk_free=risk_free)
    result, exec_stats = bt.run(
        weight_df=weight_df, adj_close=adj_close_w, adj_vwap=adj_vwap_w,
        close_raw=close_raw_w, limit_up_df=limit_up_w, limit_down_df=limit_down_w,
        trade_status_df=trade_status_w, benchmark_ret=bm_ret, initial_value=initial_value,
        sell_only_df=sell_only_df,
    )

    logger.info(f"\n{result.summary()}")
    logger.info(
        f"\n  执行统计：买入阻断={exec_stats['buy_fail_count']}次  "
        f"卖出阻断={exec_stats['sell_defer_count']}次  "
        f"过期订单={exec_stats['expired_order_count']}笔  "
        f"过期未成交金额={exec_stats['expired_notional']:,.2f}元  "
        f"平均现金占比={exec_stats['avg_cash_pct']*100:.1f}%"
    )
    if exec_stats.get("delisted_stuck_count", 0) > 0:
        logger.info(
            f"  ⚠️ 退市/长停滞留持仓={exec_stats['delisted_stuck_count']}只  "
            f"陈旧价估值占末日NAV={exec_stats['stale_value_pct']*100:.2f}%"
            f"（无法成交，净值含此部分不可全信）"
        )

    # ── 6. 输出 ──────────────────────────────────────────────
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        benchmark_label = (
            "全市场等权" if index == _EQUAL_WEIGHT_BENCHMARK else index.upper()
        )
        report_title = title or (
            f"回测报告  {benchmark_label}基准  {actual_start}~{actual_end}"
        )
        if warning_banner is None:
            warning_banner = _detect_warning_banner(weight_path)
        if warning_banner:
            logger.warning("  ⚠️ 报告将携带置顶风险水印（合成/前视 Alpha）")
        report_path = generate_html_report(
            result, output_path=out_path / "report.html", title=report_title,
            cache_dir=cache_dir, alpha_path=alpha_path,
            warning_banner=warning_banner,
        )
        logger.info(f"\n  HTML 报告：{report_path}")
        nav_df = pd.DataFrame({
            "nav": result.nav, "bm_nav": result.bm_nav, "excess_nav": result.excess_nav,
            "port_ret": result.daily_ret, "bm_ret": result.bm_ret, "excess_ret": result.excess_ret,
        })
        nav_df.to_parquet(out_path / "nav.parquet")
        result.turnover.to_frame("turnover").to_parquet(out_path / "turnover.parquet")
        if result.actual_weights is not None:
            result.actual_weights.to_parquet(out_path / "actual_weights.parquet")
        execution_stats_path = _save_execution_stats(
            exec_stats,
            out_path / "execution_stats.json",
        )
        logger.info(f"  净值数据：{out_path / 'nav.parquet'}")
        logger.info(f"  执行统计：{execution_stats_path}")

    return result, exec_stats
