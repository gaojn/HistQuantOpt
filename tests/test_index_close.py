"""指数收盘价加载器测试（自带临时 CSV，不依赖本地数据文件）。"""
import pandas as pd
import pytest

from portfolio_optimizer.data.index_close import (
    load_index_close,
    load_index_returns,
    available_indices,
)

CSV = """date,沪深300,中证500,中证1000,中证红利,万得全A
20200102,4000.0,5000.0,6000.0,,5500.0
20200103,4040.0,5050.0,6060.0,,5555.0
20200106,4000.0,5100.0,6000.0,,5600.0
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "idx.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


def test_load_close_by_key(csv_path):
    s = load_index_close("hs300", path=csv_path)
    assert isinstance(s, pd.Series)
    assert s.index[0] == pd.Timestamp("2020-01-02")
    assert s.iloc[1] == 4040.0


def test_load_close_by_chinese_name(csv_path):
    s = load_index_close("中证1000", path=csv_path)
    assert s.iloc[0] == 6000.0


def test_date_range_filter(csv_path):
    s = load_index_close("zz500", start="2020-01-03", end="2020-01-06", path=csv_path)
    assert s.index[0] == pd.Timestamp("2020-01-03")
    assert len(s) == 2


def test_returns_first_day_not_nan(csv_path):
    # start 当日收益应基于前一日收盘，非 NaN
    r = load_index_returns("hs300", start="2020-01-03", path=csv_path)
    assert r.index[0] == pd.Timestamp("2020-01-03")
    assert abs(r.iloc[0] - 0.01) < 1e-9


def test_available_excludes_all_nan(csv_path):
    cols = available_indices(path=csv_path)
    assert "中证红利" not in cols          # 整列空 → 排除
    assert "沪深300" in cols


def test_missing_column_raises(csv_path):
    with pytest.raises(KeyError):
        load_index_close("不存在指数", path=csv_path)
