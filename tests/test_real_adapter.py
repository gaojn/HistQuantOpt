"""
_compute_status 单元测试。

测试四个场景：
1. trade_status="XD" 且 close≈limit_up → LIMIT_UP
2. trade_status="XR" 且 close≈limit_down → LIMIT_DOWN
3. trade_status="停牌"（即使 close≈limit_up） → SUSPENDED
4. trade_status="交易" 且价格正常 → NORMAL
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hqopt.data.generator import TradingStatus
from hqopt.data.real_adapter import RealMarketAdapter


def _make_today(trade_status: str, close: float, limit_up: float, limit_down: float) -> pd.DataFrame:
    """构造最小 today DataFrame（单票）。"""
    return pd.DataFrame(
        {
            "trade_status": [trade_status],
            "close":        [close],
            "limit_up":     [limit_up],
            "limit_down":   [limit_down],
            "list_days":    [365],   # 已上市一年，避免次新判定
            "is_st":        [False],
        },
        index=["000001.SZ"],
    )


@pytest.fixture()
def adapter() -> RealMarketAdapter:
    return RealMarketAdapter()


# ── 场景1：XD 日收盘=涨停价 → LIMIT_UP ──────────────────────────────
def test_xd_at_limit_up(adapter: RealMarketAdapter) -> None:
    today = _make_today("XD", close=12.0, limit_up=12.0, limit_down=9.8)
    result = adapter._compute_status(today)
    assert result.iloc[0] == TradingStatus.LIMIT_UP, (
        f"XD日涨停应为LIMIT_UP，实为{result.iloc[0]}"
    )


# ── 场景2：XR 日收盘=跌停价 → LIMIT_DOWN ─────────────────────────────
def test_xr_at_limit_down(adapter: RealMarketAdapter) -> None:
    today = _make_today("XR", close=9.8, limit_up=12.0, limit_down=9.8)
    result = adapter._compute_status(today)
    assert result.iloc[0] == TradingStatus.LIMIT_DOWN, (
        f"XR日跌停应为LIMIT_DOWN，实为{result.iloc[0]}"
    )


# ── 场景3：停牌（即使价格=涨停价）→ SUSPENDED ──────────────────────────
def test_suspended_overrides_limit_up(adapter: RealMarketAdapter) -> None:
    today = _make_today("停牌", close=12.0, limit_up=12.0, limit_down=9.8)
    result = adapter._compute_status(today)
    assert result.iloc[0] == TradingStatus.SUSPENDED, (
        f"停牌应为SUSPENDED，实为{result.iloc[0]}"
    )


def test_suspended_overrides_st_and_new_listing(adapter: RealMarketAdapter) -> None:
    """停牌优先级最高，避免 ST/次新覆盖后丢失持仓冻结语义。"""
    today = _make_today("停牌", close=10.0, limit_up=11.0, limit_down=9.0)
    today.loc[:, "is_st"] = True
    today.loc[:, "list_days"] = 10

    result = adapter._compute_status(today)

    assert result.iloc[0] == TradingStatus.SUSPENDED


# ── 场景4：正常交易日，价格未触界 → NORMAL ────────────────────────────
def test_normal_trading(adapter: RealMarketAdapter) -> None:
    today = _make_today("交易", close=10.5, limit_up=12.0, limit_down=9.8)
    result = adapter._compute_status(today)
    assert result.iloc[0] == TradingStatus.NORMAL, (
        f"正常交易应为NORMAL，实为{result.iloc[0]}"
    )


# ─────────────────────────────────────────────────────────────────
# T8：_compute_adv 新实现与旧逻辑一致性测试
# ─────────────────────────────────────────────────────────────────

def _old_compute_adv(panel: "pl.DataFrame", target_date, tickers, adv_window=20):
    """旧逻辑内联（基准对照）：逐票 groupby.apply。"""
    df = (
        panel
        .filter(pl.col("date") <= target_date)
        .filter(pl.col("code").is_in(tickers))
        .select(["code", "date", "amount", "trade_status"])
        .sort(["code", "date"])
        .to_pandas()
    )
    df.loc[df["trade_status"] == "停牌", "amount"] = np.nan
    adv = (
        df.groupby("code")["amount"]
        .apply(lambda s: s.dropna().tail(adv_window).mean() * 1000)
        .reindex(tickers)
        .fillna(1e5)
    )
    adv.name = "adv"
    return adv


def _make_adv_panel():
    """
    构造含 NaN / 停牌空洞的小面板（2 只股票，10 个交易日）。
    - AA：第3、7天停牌（amount 有值但 trade_status=停牌）
    - BB：数据不足 window（只有 3 个非停牌日），验证 min_periods=1
    """
    dates = [date(2024, 1, d) for d in range(2, 12)]  # 10 天
    rows = []
    for i, d in enumerate(dates):
        # AA：周期性停牌
        susp_aa = (i == 2 or i == 6)
        rows.append({
            "code": "AA", "date": d,
            "amount": 100.0 + i * 5,
            "trade_status": "停牌" if susp_aa else "交易",
        })
        # BB：只有 3 天有数据（其余停牌）
        susp_bb = (i >= 3)
        rows.append({
            "code": "BB", "date": d,
            "amount": 200.0 + i * 3,
            "trade_status": "停牌" if susp_bb else "交易",
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_adv_new_matches_old_logic():
    """新 polars 预计算 ADV 与旧 groupby.apply 逻辑在多日期上数值一致。"""
    panel = _make_adv_panel()
    tickers = ["AA", "BB"]
    adv_window = 5

    adapter_new = RealMarketAdapter(adv_window=adv_window)

    test_dates = [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 11)]
    for td in test_dates:
        new_adv = adapter_new._compute_adv(panel, td, tickers)
        old_adv = _old_compute_adv(panel, td, tickers, adv_window)
        for code in tickers:
            assert new_adv[code] == pytest.approx(old_adv[code], rel=1e-6, abs=1.0), (
                f"[{td}] {code}: new={new_adv[code]:.2f} old={old_adv[code]:.2f}"
            )
