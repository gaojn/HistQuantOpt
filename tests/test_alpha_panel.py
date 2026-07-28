"""Alpha 面板加载、日期解析与截面取值的单测。

覆盖研究员接入外部 Alpha 的完整入口路径：
    parquet（长表/宽表，多种日期表示）→ load_alpha_panel
                                     → get_alpha_for_date（陈旧度 + 截面标准化）
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hqopt.pipeline.universe import (
    AlphaSlice,
    AlphaZeroVarianceError,
    _parse_dates_flexible,
    get_alpha_for_date,
    load_alpha_panel,
)

TICKERS = ["A", "B", "C"]


def _panel(index, values=None) -> pd.DataFrame:
    """构造宽表 Alpha 面板（index=date, columns=ticker）。"""
    if values is None:
        values = [[1.0, 2.0, 3.0]] * len(index)
    return pd.DataFrame(values, index=pd.to_datetime(index), columns=TICKERS)


# ── 日期解析 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        pd.to_datetime(["2024-01-02", "2024-01-03"]),           # datetime64
        [date(2024, 1, 2), date(2024, 1, 3)],                   # datetime.date
        ["2024-01-02", "2024-01-03"],                           # ISO 字符串
        ["2024/01/02", "2024/01/03"],                           # 斜杠分隔
        [20240102, 20240103],                                   # YYYYMMDD 整数
        ["20240102", "20240103"],                               # YYYYMMDD 字符串
    ],
)
def test_parse_dates_flexible_handles_all_supported_forms(raw):
    parsed = _parse_dates_flexible(raw)
    assert list(parsed) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_parse_dates_flexible_yyyymmdd_int_not_read_as_epoch():
    """20240102 必须解析为 2024-01-02，而非纳秒时间戳（1970 年）。"""
    parsed = _parse_dates_flexible([20240102])
    assert parsed[0].year == 2024


def test_parse_dates_flexible_mixed_string_formats():
    parsed = _parse_dates_flexible(["2024-01-02", "2024/01/03"])
    assert list(parsed) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


# ── 面板加载 ─────────────────────────────────────────────────────


def test_load_alpha_panel_wide_format(tmp_path):
    path = tmp_path / "wide.parquet"
    _panel(["2024-01-03", "2024-01-02"]).to_parquet(path)

    loaded = load_alpha_panel(path)

    assert list(loaded.columns) == TICKERS
    assert loaded.index.name == "date"
    assert loaded.index.is_monotonic_increasing        # 加载后必须按日期升序
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_alpha_panel_long_format_pivots(tmp_path):
    path = tmp_path / "long.parquet"
    pd.DataFrame({
        "date": [20240102, 20240102, 20240103, 20240103],
        "code": ["A", "B", "A", "B"],
        "alpha": [1.0, 2.0, 3.0, 4.0],
    }).to_parquet(path)

    loaded = load_alpha_panel(path)

    assert list(loaded.columns) == ["A", "B"]
    assert loaded.loc[pd.Timestamp("2024-01-03"), "B"] == pytest.approx(4.0)


def test_load_alpha_panel_long_format_accepts_date_objects(tmp_path):
    path = tmp_path / "long_date.parquet"
    pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 2)],
        "code": ["A", "B"],
        "alpha": [1.0, 2.0],
    }).to_parquet(path)

    loaded = load_alpha_panel(path)

    assert loaded.index[0] == pd.Timestamp("2024-01-02")


# ── 截面取值：陈旧度 ─────────────────────────────────────────────


def test_alpha_asof_uses_latest_available_and_reports_zero_staleness():
    panel = _panel(["2024-01-02", "2024-01-03"])

    got = get_alpha_for_date(panel, date(2024, 1, 3), TICKERS)

    assert isinstance(got, AlphaSlice)
    assert got.as_of == date(2024, 1, 3)
    assert got.staleness_days == 0
    assert got.n_valid == 3


def test_alpha_staleness_days_measured_in_calendar_days():
    panel = _panel(["2024-01-02"])

    got = get_alpha_for_date(panel, date(2024, 1, 12), TICKERS)

    assert got.as_of == date(2024, 1, 2)
    assert got.staleness_days == 10


def test_alpha_beyond_max_staleness_returns_none():
    """过期信号必须显式不可用，而不是静默沿用最后一期。"""
    panel = _panel(["2024-01-02"])

    assert get_alpha_for_date(
        panel, date(2024, 1, 12), TICKERS, max_staleness_days=5
    ) is None
    # 边界：恰好等于阈值仍可用
    assert get_alpha_for_date(
        panel, date(2024, 1, 12), TICKERS, max_staleness_days=10
    ) is not None


def test_alpha_entirely_after_target_raises():
    """面板整体晚于回测区间是配置错误，不能静默返回全零 alpha。"""
    panel = _panel(["2026-01-02"])

    with pytest.raises(ValueError, match="无 2024-01-02 及之前的数据"):
        get_alpha_for_date(panel, date(2024, 1, 2), TICKERS)


def test_alpha_zero_coverage_on_universe_returns_none():
    """优化域内一只都没覆盖 → 不可用（否则 alpha 恒 0，优化退化为纯风险最小化）。"""
    panel = _panel(["2024-01-02"])

    assert get_alpha_for_date(panel, date(2024, 1, 2), ["X", "Y"]) is None


def test_alpha_partial_coverage_is_usable_and_counted():
    panel = _panel(["2024-01-02"])

    got = get_alpha_for_date(panel, date(2024, 1, 2), ["A", "B", "X"])

    assert got.n_valid == 2
    assert len(got.values) == 3


# ── 截面取值：标准化 ─────────────────────────────────────────────


def test_alpha_standardized_to_zero_mean_unit_std():
    panel = _panel(["2024-01-02"], values=[[1.0, 2.0, 3.0]])

    got = get_alpha_for_date(panel, date(2024, 1, 2), TICKERS)

    assert got.standardized is True
    assert got.values.mean() == pytest.approx(0.0, abs=1e-12)
    assert got.values.std(ddof=0) == pytest.approx(1.0)
    # 保序
    assert list(np.argsort(got.values)) == [0, 1, 2]


def test_alpha_standardization_is_scale_invariant():
    """同一因子排序、不同量纲，标准化后必须得到完全相同的 alpha 向量。

    这是 risk_aversion / turnover_penalty / diversification_penalty 默认标定
    能够成立的前提：未标准化时 α 乘 100 倍就能把分散组合压成单票全仓。
    """
    base = np.array([[-1.0, 0.0, 2.0]])
    results = []
    for scale, shift in [(1.0, 0.0), (100.0, 0.0), (0.01, 5.0), (7.0, -3.0)]:
        panel = _panel(["2024-01-02"], values=base * scale + shift)
        results.append(get_alpha_for_date(panel, date(2024, 1, 2), TICKERS).values)

    for other in results[1:]:
        np.testing.assert_allclose(results[0], other, atol=1e-12)


def test_alpha_standardize_disabled_keeps_raw_scale():
    panel = _panel(["2024-01-02"], values=[[10.0, 20.0, 30.0]])

    got = get_alpha_for_date(panel, date(2024, 1, 2), TICKERS, standardize=False)

    assert got.standardized is False
    np.testing.assert_allclose(got.values, [10.0, 20.0, 30.0])


def test_alpha_flat_cross_section_raises():
    """截面无差异时必须显式拒绝，不能静默退化成纯风险最小化。"""
    panel = _panel(["2024-01-02"], values=[[5.0, 5.0, 5.0]])

    with pytest.raises(AlphaZeroVarianceError, match="Alpha 截面无区分度"):
        get_alpha_for_date(panel, date(2024, 1, 2), TICKERS)


def test_alpha_all_zero_raises_even_when_standardization_disabled():
    panel = _panel(["2024-01-02"], values=[[0.0, 0.0, 0.0]])

    with pytest.raises(AlphaZeroVarianceError, match="Alpha 截面无区分度"):
        get_alpha_for_date(
            panel,
            date(2024, 1, 2),
            TICKERS,
            standardize=False,
        )


def test_alpha_missing_filled_with_cross_section_neutral_zero():
    """缺失票填 0；标准化后 0 恰为截面均值，语义上是"中性"而非"最差"。"""
    panel = pd.DataFrame(
        [[1.0, 3.0, np.nan]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=TICKERS,
    )

    got = get_alpha_for_date(panel, date(2024, 1, 2), TICKERS)

    assert got.n_valid == 2
    assert got.values[2] == pytest.approx(0.0)
    # 有效值仍严格保序
    assert got.values[0] < got.values[1]


def test_alpha_infinities_treated_as_missing():
    panel = pd.DataFrame(
        [[1.0, 2.0, np.inf]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=TICKERS,
    )

    got = get_alpha_for_date(panel, date(2024, 1, 2), TICKERS)

    assert got.n_valid == 2
    assert np.isfinite(got.values).all()


def test_alpha_values_aligned_to_requested_ticker_order():
    panel = _panel(["2024-01-02"], values=[[1.0, 2.0, 3.0]])

    got = get_alpha_for_date(panel, date(2024, 1, 2), ["C", "A", "B"])

    # 原顺序 A<B<C，请求顺序 C,A,B → 值应随之重排
    assert got.values[0] > got.values[2] > got.values[1]
