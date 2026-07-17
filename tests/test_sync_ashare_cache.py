from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from scripts import sync_ashare_cache as sync


def test_price_select_matches_verified_wind_schema() -> None:
    mappings = {
        "S_INFO_WINDCODE": "code",
        "TRADE_DT": "date",
        "S_INFO_NAME": "name",
        "S_DQ_PRECLOSE": "pre_close",
        "S_DQ_OPEN": "open",
        "S_DQ_HIGH": "high",
        "S_DQ_LOW": "low",
        "S_DQ_CLOSE": "close",
        "S_DQ_LIMIT": "limit_up",
        "S_DQ_STOPPING": "limit_down",
        "S_DQ_PCTCHANGE": "pct_change",
        "S_DQ_VOLUME": "volume",
        "S_DQ_AMOUNT": "amount",
        "S_DQ_ADJPRECLOSE": "adj_pre_close",
        "S_DQ_ADJOPEN": "adj_open",
        "S_DQ_ADJHIGH": "adj_high",
        "S_DQ_ADJLOW": "adj_low",
        "S_DQ_ADJCLOSE": "adj_close",
        "S_DQ_AVGPRICE": "vwap",
        "S_DQ_ADJFACTOR": "adj_factor",
        "S_DQ_TRADESTATUS": "trade_status",
        "S_VAL_MV": "total_mv",
        "S_DQ_MV": "float_mv",
        "S_DQ_TURN": "turnover",
        "S_DQ_FREETURNOVER": "free_turnover",
        "TOT_SHR_TODAY": "total_shares",
        "FLOAT_A_SHR_TODAY": "float_shares",
        "FREE_SHARES_TODAY": "free_shares",
        "CITICS_IND_NAME_L1": "industry_l1",
        "CITICS_IND_NAME_L2": "industry_l2",
        "CITICS_IND_NAME_L3": "industry_l3",
        "S_INFO_LISTDATE": "list_date",
        "S_INFO_DELISTDATE": "delist_date",
        "IN_HS300": "is_hs300",
        "IN_ZZ500": "is_zz500",
        "IN_ZZ1000": "is_zz1000",
        "IS_ST": "is_st",
    }
    for physical, alias in mappings.items():
        assert re.search(rf"\b{physical}\s+AS\s+{alias}\b", sync.PRICE_SELECT)
    assert "FROM wind_db.VW_ASHARE_STOCK_DAILY" in sync.PRICE_SELECT
    assert "WHERE TRADE_DT >= '{y}-01-01' AND TRADE_DT <= '{y}-12-31'" in sync.PRICE_SELECT


def _source_frame() -> pl.DataFrame:
    derived = {"adj_vwap", "free_mv", "list_days"}
    string_columns = {
        "code",
        "name",
        "trade_status",
        "industry_l1",
        "industry_l2",
        "industry_l3",
        "list_date",
        "delist_date",
    }
    flag_columns = {"is_hs300", "is_zz500", "is_zz1000", "is_st"}
    values: dict[str, list[object]] = {}
    for column in sync.PRICE_COLUMNS:
        if column in derived:
            continue
        if column in string_columns:
            values[column] = [column]
        elif column in flag_columns:
            values[column] = [0]
        else:
            values[column] = [1.0]
    values.update(
        {
            "code": ["000001.SZ"],
            "date": [date(2026, 7, 16)],
            "name": ["平安银行"],
            "close": [10.77],
            "vwap": [10.7951],
            "adj_factor": [85.329579],
            "free_shares": [816048.1215],
            "trade_status": ["交易"],
            "list_date": ["20260715"],
            "delist_date": ["20991231"],
            "is_hs300": [1],
        }
    )
    return pl.DataFrame(values).with_columns(
        pl.col("is_hs300", "is_zz500", "is_zz1000", "is_st").cast(pl.UInt8)
    )


def test_sync_prices_preserves_cache_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries: list[str] = []

    def fake_query(sql: str) -> pl.DataFrame:
        queries.append(sql)
        return _source_frame()

    monkeypatch.setattr(sync, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sync, "query_df", fake_query)

    sync.sync_prices([2026])

    assert "2026-01-01" in queries[0]
    result = pl.read_parquet(tmp_path / "ashare_daily_2026.parquet")
    assert result.columns == sync.PRICE_COLUMNS
    assert result.schema["date"] == pl.Datetime("ms")
    row = result.row(0, named=True)
    assert row["adj_vwap"] == pytest.approx(10.7951 * 85.329579)
    assert row["free_mv"] == pytest.approx(10.77 * 816048.1215)
    assert row["list_days"] == 1
