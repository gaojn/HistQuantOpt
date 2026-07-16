"""候选池过滤与 carry 逻辑测试。"""
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hqopt.data.generator import MarketDataGenerator
from hqopt.pipeline.universe import filter_universe


@pytest.fixture
def snap30():
    """30 只股票的合成快照（seed=42）。"""
    return MarketDataGenerator(n_stocks=30, seed=42).generate()


def _make_panel(snap, target_date: date) -> pl.DataFrame:
    """生成最简 polars 面板（is_st 全为 0）。"""
    return pl.DataFrame({
        "date": [target_date] * len(snap.tickers),
        "code": snap.tickers,
        "is_st": [0] * len(snap.tickers),
    })


TARGET = date(2024, 1, 2)


def test_no_prev_holdings_unchanged(snap30):
    """prev_holdings=None 时行为与旧版一致：sell_only 为 None，tickers 不变。"""
    panel = _make_panel(snap30, TARGET)
    result = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=None,
    )
    assert result.sell_only is None
    # sell_only_mask 全 False
    assert not result.sell_only_mask.any()
    assert len(result.tickers) == 20


def test_carry_dropped_holding_included(snap30):
    """掉出 top_n 的持仓票被携带回来，标记 sell_only=True。"""
    panel = _make_panel(snap30, TARGET)

    # top_n=20 → 排名 21-30 的票会掉出
    sorted_by_cap = snap30.market_cap.sort_values(ascending=False)
    drop_ticker = sorted_by_cap.index[20]  # 第 21 名（rank 20，0-indexed）

    prev_holdings = pd.Series({drop_ticker: 0.05})

    result = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=prev_holdings,
    )

    # carry 票必须在结果域内
    assert drop_ticker in result.tickers
    # 必须标记为 sell_only
    assert result.sell_only is not None
    assert bool(result.sell_only[drop_ticker]) is True
    # 总票数 = 20 + 1（carry）
    assert len(result.tickers) == 21

    # sell_only_mask 只有 carry 票为 True
    so_mask = result.sell_only_mask
    assert so_mask.sum() == 1
    carry_pos = result.tickers.index(drop_ticker)
    assert so_mask[carry_pos] is np.True_


def test_no_holding_drop_not_carried(snap30):
    """掉出 top_n 但无持仓的票不被携带。"""
    panel = _make_panel(snap30, TARGET)
    sorted_by_cap = snap30.market_cap.sort_values(ascending=False)
    drop_ticker = sorted_by_cap.index[20]

    # 该票持仓为 0（不满足 > 1e-8）
    prev_holdings = pd.Series({drop_ticker: 0.0})

    result = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=prev_holdings,
    )

    assert drop_ticker not in result.tickers
    assert len(result.tickers) == 20


def test_ticker_not_in_snapshot_not_carried(snap30):
    """不在 snapshot.tickers（真退市 / 无行情）的票不被携带。"""
    panel = _make_panel(snap30, TARGET)
    prev_holdings = pd.Series({"GHOST.SH": 0.05})

    result = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=prev_holdings,
    )
    assert "GHOST.SH" not in result.tickers
    assert len(result.tickers) == 20


def test_filter_then_optimize_carry_stock(snap30):
    """
    集成测试（Step 3e）：filter_universe + 优化器组合。

    上期持仓包含一只本期掉出 top_n 的票；
    验证该票在优化器中：
      - 仍在 snapshot.tickers（sell_only=True）
      - 优化后权重 ≤ 上期权重（不加仓）
      - 卖出量计入换手（total_turnover > 0 且不含被错误冻结）
    """
    from dataclasses import replace

    import numpy as np

    from hqopt.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer

    panel = _make_panel(snap30, TARGET)
    sorted_by_cap = snap30.market_cap.sort_values(ascending=False)
    drop_ticker = sorted_by_cap.index[20]   # 第 21 名，会被 top_n=20 排除

    # 上期持仓：drop_ticker 有 5% 仓位，其余若干正常票均匀持仓
    keep_tickers = sorted_by_cap.index[:20].tolist()
    prev_w_dict = {t: 0.05 for t in keep_tickers[:5]}   # 前5只各5%
    prev_w_dict[drop_ticker] = 0.05                       # 掉池票也有5%
    # 归一
    total = sum(prev_w_dict.values())
    prev_w_dict = {t: v/total for t, v in prev_w_dict.items()}

    prev_holdings = pd.Series(prev_w_dict)

    result_snap = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=prev_holdings,
    )

    # carry 票必须在 snapshot
    assert drop_ticker in result_snap.tickers
    assert result_snap.sell_only is not None
    assert bool(result_snap.sell_only[drop_ticker])

    # 对齐 prev_weight 数组
    ps = prev_holdings.reindex(result_snap.tickers).fillna(0.0).values

    # 极高 alpha 给 drop_ticker，验证 sell_only 仍阻止加仓
    alpha = np.zeros(len(result_snap.tickers))
    drop_pos = result_snap.tickers.index(drop_ticker)
    alpha[drop_pos] = 100.0

    cfg = AlphaMaxConfig(weight_upper=0.15, industry_upper=0.5, max_turnover=1.0)
    opt_result = AlphaMaxOptimizer(cfg).optimize(alpha, result_snap, prev_weight=ps)

    assert opt_result.is_feasible
    # drop_ticker 不得加仓
    assert opt_result.weights[drop_pos] <= ps[drop_pos] + 1e-5
    # 总换手 > 0（至少有卖出的动作）
    turnover = float(np.abs(opt_result.weights - ps).sum())
    assert turnover > 1e-4


def test_multiple_carry_stocks(snap30):
    """多只掉池持仓票均被携带并标记 sell_only。"""
    panel = _make_panel(snap30, TARGET)
    sorted_by_cap = snap30.market_cap.sort_values(ascending=False)
    drop_tickers = sorted_by_cap.index[20:23].tolist()  # 3 只

    prev_holdings = pd.Series({t: 0.05 for t in drop_tickers})

    result = filter_universe(
        snap30, panel, TARGET,
        exclude_bj=False, exclude_st=False,
        top_n=20,
        prev_holdings=prev_holdings,
    )

    assert len(result.tickers) == 23
    assert result.sell_only is not None
    for t in drop_tickers:
        assert bool(result.sell_only[t]) is True
    assert result.sell_only_mask.sum() == 3
