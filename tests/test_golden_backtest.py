"""回测引擎端到端 golden 回归。

锁定「权重 + 行情 → 净值 / 绩效指标 / 执行统计」这条链路的数值口径。
CHANGELOG 中绝大多数 ⚠️ 破坏性变更都落在这一层，本测试是它们的自动化哨兵：
口径一旦变化此处必然失败，从而强制显式确认并登记需重跑范围。

设计约束
--------
1. **零外部数据**：行情面板由本文件解析式生成，不读 ``data/``、不连 ClickHouse，
   CI 上可独立运行。
2. **零随机数**：价格由整数取模递推得到，再除以 100 得到精确两位小数。
   ``np.random.default_rng`` 的 bit-stream 不保证跨 NumPy 版本一致，用 RNG 会让
   golden 在升级依赖时假失败；纯整数构造在任何平台与版本上逐位相同。
3. **零超越函数**：不使用 sin/exp，避免不同 libm 的 1 ulp 差异在涨跌停这类
   离散分支判定上被放大成完全不同的成交路径。
4. **覆盖真实分支**：面板注入停牌、涨停、跌停、退市与 sell-only，使 golden 锚住
   完整执行语义，而非仅理想成交路径。涨跌停距离特意做成三档（封板 / 准封板 /
   远离），准封板一档让 ``LIMIT_TOL`` 的口径变化也能被这条基线捕捉——只有
   「封板」和「远离」两档时，改动该容差不会翻转任何判定，参数会静默失去保护。

基线数字不是策略基准
--------------------
合成面板不含任何 alpha，组合是高换手的确定性轮换，因此基线里组合年化为负、
跑输等权基准属于预期结果。本测试锚定的是「同一输入必得同一输出」的口径稳定性，
与策略优劣无关；负收益路径反而额外覆盖了负 Sharpe、负 IR、超额回撤的计算分支。

基线维护
--------
口径**有意**变更时，用以下命令重生成基线，并在 CHANGELOG 记录影响与需重跑范围：

    HQOPT_UPDATE_GOLDEN=1 pytest tests/test_golden_backtest.py

未设该环境变量时基线只读，绝不会被测试自动改写。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import hqopt.backtest.run as runmod
from hqopt.backtest.engine import RealisticBacktester
from hqopt.backtest.execution import publish_batch_bundle
from hqopt.backtest.run import run_backtest
from hqopt.constants import LIMIT_TOL

# 用于校验合成面板覆盖了封板容差敏感区：把容差放宽到这个量级时，准封板样本的
# 判定必须翻转。取值只需远大于 LIMIT_TOL 且小于「远离封板」档的 10%。
WIDENED_TOL = 5e-2

GOLDEN_PATH = Path(__file__).parent / "golden" / "backtest_baseline.json"

# 相对容差。回测引擎是纯 numpy/pandas 算术、不经过任何凸优化求解器，
# 因此可以死锁到接近机器精度；留 1e-9 只为吸收求和顺序等实现细节差异。
RTOL = 1e-9

N_STOCKS = 40
N_DAYS = 120
REBALANCE_EVERY = 21          # 约每月一次调仓
N_HOLDINGS = 12               # 每期等权持有数
FIRST_TRADING_DAY = date(2024, 1, 2)

# 退市股：末段直接从面板消失（真实退市没有行情行），用于锚住陈旧价估值分支。
# 退市日必须晚于「末次调仓(105) + T+1 成交(106)」，否则这些票根本建不了仓，
# 只会变成 expired 订单，走不到「持仓中途退市」这条路径。
DELISTED_STOCKS = (37, 38, 39)
DELIST_FROM_DAY = 112


# ──────────────────────────────────────────────────────────────
# 确定性行情构造：全部走整数取模，最后一步才除以 100
# ──────────────────────────────────────────────────────────────
def _close_cents(stock: int, day: int) -> int:
    """收盘价（分）。波动幅度约 1~3%，量级 8~30 元，贴近 A 股常见价位。"""
    wave = ((stock * 37 + day * 17) % 61) - 30
    trend = (day * (3 + stock % 5)) % 23
    return 1000 + stock * 50 + wave * 2 + trend


def _vwap_cents(stock: int, day: int) -> int:
    """VWAP（分）。与收盘价系统性错开，避免 VWAP=close 掩盖时点差口径问题。"""
    offset = ((stock * 11 + day * 5) % 9) - 4
    return _close_cents(stock, day) + offset


def _is_suspended(stock: int, day: int) -> bool:
    return (stock * 7 + day * 3) % 97 == 0


def _is_limit_up(stock: int, day: int) -> bool:
    return (stock * 5 + day * 11) % 89 == 0


def _is_limit_down(stock: int, day: int) -> bool:
    return (stock * 3 + day * 19) % 101 == 0


def _is_near_limit_up(stock: int, day: int) -> bool:
    """准涨停：close/limit_up ≈ 0.98，落在封板容差的敏感区间内。

    只有"已封板"和"远离封板"两档时，`LIMIT_TOL` 改动不会翻转任何判定，
    该参数就完全不被 golden 锚定（实测：1e-3 → 5e-2 时基线纹丝不动）。
    这一档把比率放在 (1-5e-2, 1-1e-3) 之间，使容差口径变化必然改变成交。
    """
    return (stock * 23 + day * 7) % 73 == 0


def _is_near_limit_down(stock: int, day: int) -> bool:
    """准跌停：close/limit_down ≈ 1.02，作用同 :func:`_is_near_limit_up`。"""
    return (stock * 29 + day * 13) % 79 == 0


def _is_delisted(stock: int, day: int) -> bool:
    return stock in DELISTED_STOCKS and day >= DELIST_FROM_DAY


def _trading_days() -> list[date]:
    return [
        day.date()
        for day in pd.bdate_range(FIRST_TRADING_DAY, periods=N_DAYS)
    ]


def build_panel() -> pl.DataFrame:
    """构造合成行情面板，字段与 ``load_panel`` 的回测取数口径一致。"""
    days = _trading_days()
    rows = []
    for day_index, trading_day in enumerate(days):
        for stock in range(N_STOCKS):
            if _is_delisted(stock, day_index):
                continue          # 退市后无行情行

            close_cents = _close_cents(stock, day_index)
            suspended = _is_suspended(stock, day_index)

            # 三档涨跌停距离：封板（close==limit，触发不可买/不可卖）、
            # 准封板（比率 0.98/1.02，落在容差敏感区，锚定 LIMIT_TOL 口径）、
            # 远离（±10%，正常成交）。
            if _is_limit_up(stock, day_index):
                limit_up_cents = close_cents
            elif _is_near_limit_up(stock, day_index):
                limit_up_cents = close_cents * 100 // 98
            else:
                limit_up_cents = close_cents * 110 // 100

            if _is_limit_down(stock, day_index):
                limit_down_cents = close_cents
            elif _is_near_limit_down(stock, day_index):
                limit_down_cents = close_cents * 100 // 102
            else:
                limit_down_cents = close_cents * 90 // 100

            rows.append(
                {
                    "date": trading_day,
                    "code": f"SZ{stock:06d}",
                    "adj_close": close_cents / 100.0,
                    "adj_vwap": _vwap_cents(stock, day_index) / 100.0,
                    "close": close_cents / 100.0,
                    "limit_up": limit_up_cents / 100.0,
                    "limit_down": limit_down_cents / 100.0,
                    "trade_status": "停牌" if suspended else "交易",
                }
            )
    return pl.DataFrame(rows)


def build_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造调仓权重与 sell-only 矩阵（等权持仓，成分按确定性规则轮换）。"""
    days = _trading_days()
    rebalance_indices = list(range(0, N_DAYS - 5, REBALANCE_EVERY))
    codes = [f"SZ{stock:06d}" for stock in range(N_STOCKS)]

    weight_rows: list[dict[str, float]] = []
    sell_only_rows: list[dict[str, bool]] = []
    for period in range(len(rebalance_indices)):
        # 持仓成分随期数轮换，制造真实换手（含买入与卖出两侧）。
        holdings = {
            (stock * 13 + period * 29) % N_STOCKS for stock in range(N_HOLDINGS)
        }
        # 末期显式并入退市股，走到「持仓中途退市 → 无法成交、靠陈旧价估值滞留」
        # 分支。与轮换规则解耦，避免改动选股规则时静默丢失该分支覆盖。
        if period == len(rebalance_indices) - 1:
            holdings |= set(DELISTED_STOCKS)
        holdings = sorted(holdings)
        weight = 1.0 / len(holdings)

        weight_row = dict.fromkeys(codes, 0.0)
        sell_only_row = dict.fromkeys(codes, False)
        for stock in holdings:
            weight_row[f"SZ{stock:06d}"] = weight

        # 自第 2 期起，把两只目标票标记 sell-only：目标权重为正但禁止加仓，
        # 对应「掉出候选池仍持仓」的制度性只卖场景。
        if period >= 1:
            for stock in holdings[:2]:
                sell_only_row[f"SZ{stock:06d}"] = True

        weight_rows.append(weight_row)
        sell_only_rows.append(sell_only_row)

    index = pd.to_datetime([days[i] for i in rebalance_indices])
    weights = pd.DataFrame(weight_rows, index=index)
    sell_only = pd.DataFrame(sell_only_rows, index=index)
    return weights, sell_only


def run_golden_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """在合成面板上跑完整 ``run_backtest``，返回 (BacktestResult, exec_stats)。"""
    panel = build_panel()
    monkeypatch.setattr(runmod, "load_panel", lambda *args, **kwargs: panel)

    def _reject_index_returns(*args, **kwargs):
        del args, kwargs
        raise AssertionError("equal_weight 基准不应读取外部指数收益")

    monkeypatch.setattr(runmod, "load_index_returns", _reject_index_returns)

    weights, sell_only = build_bundle()
    weight_path = tmp_path / "weights.parquet"
    publish_batch_bundle(
        weights,
        sell_only,
        {"alpha_quality": {"synthetic": True}},
        weight_path,
    )

    days = _trading_days()
    return run_backtest(
        weight_path,
        str(days[0]),
        str(days[-1]),
        index="equal_weight",
        cost_buy=0.001,
        cost_sell=0.002,
        risk_free=0.02,
        initial_value=1e8,
        out_dir=None,
    )


# ──────────────────────────────────────────────────────────────
# 基线读写
# ──────────────────────────────────────────────────────────────
def _series_digest(series: pd.Series) -> str:
    """整条序列的哈希：捕捉首末值相同、中间轨迹漂移的口径变更。"""
    payload = ",".join(f"{value:.12e}" for value in series.to_numpy())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect(result, stats: dict) -> dict:
    metrics = result.portfolio_metrics
    return {
        "metrics": {
            "annual_return": metrics.annual_return,
            "annual_vol": metrics.annual_vol,
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
            "calmar": metrics.calmar,
            "annual_excess_return": metrics.annual_excess_return,
            "tracking_error": metrics.tracking_error,
            "info_ratio": metrics.info_ratio,
            "excess_max_drawdown": metrics.excess_max_drawdown,
            "excess_calmar": metrics.excess_calmar,
            "win_rate_monthly": metrics.win_rate_monthly,
            "avg_monthly_excess": metrics.avg_monthly_excess,
        },
        "benchmark_metrics": {
            "annual_return": result.benchmark_metrics.annual_return,
            "annual_vol": result.benchmark_metrics.annual_vol,
            "max_drawdown": result.benchmark_metrics.max_drawdown,
        },
        "execution": {
            "buy_fail_count": int(stats["buy_fail_count"]),
            "sell_defer_count": int(stats["sell_defer_count"]),
            "expired_order_count": int(stats["expired_order_count"]),
            "expired_notional": float(stats["expired_notional"]),
            "avg_cash_pct": float(stats["avg_cash_pct"]),
            "final_cash_pct": float(stats["final_cash_pct"]),
            "delisted_stuck_count": int(stats["delisted_stuck_count"]),
            "stale_value_pct": float(stats["stale_value_pct"]),
        },
        "series": {
            "n_days": int(len(result.nav)),
            # 成交日数 ≠ 调仓期数：一次调仓可在 T+1/T+2/T+3 分次成交。
            "n_fill_days": int(len(result.turnover)),
            "n_rebalance_periods": int(len(result.rebalance_turnover)),
            "turnover_by_rebalance_mean": float(result.rebalance_turnover.mean()),
            "turnover_by_rebalance_sum": float(result.rebalance_turnover.sum()),
            "nav_first": float(result.nav.iloc[0]),
            "nav_last": float(result.nav.iloc[-1]),
            "bm_nav_last": float(result.bm_nav.iloc[-1]),
            "excess_nav_last": float(result.excess_nav.iloc[-1]),
            "turnover_fill_day_mean": float(result.turnover.mean()),
            "nav_digest": _series_digest(result.nav),
            "excess_nav_digest": _series_digest(result.excess_nav),
        },
    }


def _load_baseline() -> dict:
    if not GOLDEN_PATH.is_file():
        pytest.fail(
            f"缺少 golden 基线 {GOLDEN_PATH}。"
            "首次生成：HQOPT_UPDATE_GOLDEN=1 pytest tests/test_golden_backtest.py"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


_DRIFT_HINT = (
    "回测口径与 golden 基线不一致。\n"
    "  · 若为**非预期**变化：这是回归，请定位改动而非改基线。\n"
    "  · 若为**有意**的口径变更：更新基线并在 CHANGELOG 记录影响与需重跑范围——\n"
    "      HQOPT_UPDATE_GOLDEN=1 pytest tests/test_golden_backtest.py"
)


@pytest.fixture(scope="module")
def golden_actual(tmp_path_factory) -> dict:
    """跑一次回测供本模块所有断言复用，避免重复执行。"""
    monkeypatch = pytest.MonkeyPatch()
    try:
        tmp_path = tmp_path_factory.mktemp("golden")
        result, stats = run_golden_backtest(tmp_path, monkeypatch)
    finally:
        monkeypatch.undo()

    actual = _collect(result, stats)
    if os.environ.get("HQOPT_UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return actual


def test_golden_performance_metrics(golden_actual):
    """12 项组合绩效指标锁定到基线。"""
    expected = _load_baseline()["metrics"]
    for name, value in expected.items():
        assert golden_actual["metrics"][name] == pytest.approx(value, rel=RTOL), (
            f"绩效指标 {name} 漂移：{golden_actual['metrics'][name]!r} != {value!r}\n{_DRIFT_HINT}"
        )


def test_golden_benchmark_metrics(golden_actual):
    """基准口径锁定——等权基准的构造方式变化会在此暴露。"""
    expected = _load_baseline()["benchmark_metrics"]
    for name, value in expected.items():
        assert golden_actual["benchmark_metrics"][name] == pytest.approx(value, rel=RTOL), (
            f"基准指标 {name} 漂移：{golden_actual['benchmark_metrics'][name]!r} != {value!r}\n"
            f"{_DRIFT_HINT}"
        )


def test_golden_execution_stats(golden_actual):
    """成交执行统计锁定：阻断、顺延、过期与现金占比。"""
    expected = _load_baseline()["execution"]
    for name, value in expected.items():
        actual = golden_actual["execution"][name]
        if isinstance(value, int):
            assert actual == value, (
                f"执行统计 {name} 漂移：{actual!r} != {value!r}\n{_DRIFT_HINT}"
            )
        else:
            assert actual == pytest.approx(value, rel=RTOL), (
                f"执行统计 {name} 漂移：{actual!r} != {value!r}\n{_DRIFT_HINT}"
            )


def test_golden_nav_series(golden_actual):
    """净值与超额净值整条序列锁定（含哈希，捕捉中间轨迹漂移）。"""
    expected = _load_baseline()["series"]
    for name in ("n_days", "n_fill_days", "n_rebalance_periods"):
        assert golden_actual["series"][name] == expected[name], (
            f"序列长度 {name} 变化：{golden_actual['series'][name]} != {expected[name]}\n{_DRIFT_HINT}"
        )
    for name in (
        "nav_first", "nav_last", "bm_nav_last", "excess_nav_last",
        "turnover_fill_day_mean", "turnover_by_rebalance_mean",
        "turnover_by_rebalance_sum",
    ):
        assert golden_actual["series"][name] == pytest.approx(expected[name], rel=RTOL), (
            f"序列关键点 {name} 漂移：{golden_actual['series'][name]!r} != {expected[name]!r}\n"
            f"{_DRIFT_HINT}"
        )
    for name in ("nav_digest", "excess_nav_digest"):
        assert golden_actual["series"][name] == expected[name], (
            f"{name} 不一致：首末值可能相同但中间轨迹已漂移。\n{_DRIFT_HINT}"
        )


def test_turnover_by_rebalance_is_per_period_and_conserves_total(golden_actual):
    """每期换手序列必须与调仓日一一对应，且总量与逐成交日守恒。

    这是「换手率按调仓期而非成交日统计」的语义锚点：
    - 期数 == 权重文件行数（某期被完全阻断时记 0，不得从分母消失）
    - 总量守恒（聚合只改分组，不改总换手）
    - 场景本身必须真的存在跨日成交，否则这条测试是空转
    """
    series = golden_actual["series"]
    n_rebalance_days = len(build_bundle()[0])

    assert series["n_rebalance_periods"] == n_rebalance_days, (
        f"每期换手行数 {series['n_rebalance_periods']} 与调仓日数 "
        f"{n_rebalance_days} 不一致"
    )
    assert series["n_fill_days"] > series["n_rebalance_periods"], (
        "合成场景没有跨日续成交，本测试无法证伪按成交日统计的错误口径"
    )
    # 逐成交日均值 × 成交日数 == 每期总量：证明两者只是分组不同
    fill_total = series["turnover_fill_day_mean"] * series["n_fill_days"]
    assert fill_total == pytest.approx(series["turnover_by_rebalance_sum"], rel=1e-9), (
        "聚合改变了总换手，说明归属逻辑有误"
    )
    # 错误口径会低估平均换手，此处确认差异真实存在（防回归到 mean() 写法）
    assert (
        series["turnover_by_rebalance_mean"] > series["turnover_fill_day_mean"]
    ), "每期均值未高于逐成交日均值，聚合可能未生效"


def test_engine_default_costs_stay_asymmetric():
    """默认费率单独锚定：golden 显式传费率以隔离配置，默认值不在其射程内。

    买卖非对称（卖侧含印花税，故 > 买侧）是 A 股回测真实性的硬要求，
    默认值漂移不应悄悄发生。
    """
    engine = RealisticBacktester()
    assert engine.cost_buy == pytest.approx(0.001)
    assert engine.cost_sell == pytest.approx(0.002)
    assert engine.cost_sell > engine.cost_buy


def test_golden_panel_covers_limit_tolerance_band():
    """守护封板容差敏感区：必须存在 close/limit 落在容差带内的样本。

    若所有股票都「已封板」或「远离封板」，改动 ``LIMIT_TOL`` 不会翻转任何判定，
    这个口径就静默失去 golden 保护——本测试确保该带始终有样本。
    """
    panel = build_panel().to_pandas()
    up_ratio = panel["close"] / panel["limit_up"]
    down_ratio = panel["close"] / panel["limit_down"]

    # 当前容差下判定为「未封板」，容差放宽到 WIDENED_TOL 即翻转为「封板」。
    in_up_band = ((up_ratio > 1 - WIDENED_TOL) & (up_ratio < 1 - LIMIT_TOL)).sum()
    in_down_band = ((down_ratio < 1 + WIDENED_TOL) & (down_ratio > 1 + LIMIT_TOL)).sum()

    assert in_up_band > 0, "面板缺少准涨停样本，LIMIT_TOL 口径未被锚定"
    assert in_down_band > 0, "面板缺少准跌停样本，LIMIT_TOL 口径未被锚定"


def test_golden_scenario_exercises_all_execution_paths(golden_actual):
    """守护 golden 输入本身：若合成面板退化成理想成交，基线就失去意义。

    这些分支正是回测真实性的核心，必须确认它们真的被走到。
    """
    execution = golden_actual["execution"]
    assert execution["buy_fail_count"] > 0, "合成面板未触发涨停/停牌买入阻断"
    assert execution["sell_defer_count"] > 0, "合成面板未触发跌停/停牌卖出顺延"
    assert execution["delisted_stuck_count"] > 0, "合成面板未触发退市滞留持仓"
    assert golden_actual["series"]["turnover_by_rebalance_mean"] > 0, "合成权重未产生换手"
