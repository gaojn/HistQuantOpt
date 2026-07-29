from __future__ import annotations

import pytest

from hqopt.data import clickhouse_db


@pytest.fixture(autouse=True)
def _clean_clickhouse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PORT",
        "CLICKHOUSE_DB",
        "CLICKHOUSE_USER",
        clickhouse_db.PWD_ENV,
        clickhouse_db.PWD_ENV_UNIFIED,
    ):
        monkeypatch.delenv(name, raising=False)


def test_cfg_accepts_unified_wind_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(clickhouse_db.PWD_ENV_UNIFIED, "unified-secret")

    cfg = clickhouse_db._cfg()

    assert cfg["pwd"] == "unified-secret"
    assert cfg["db"] == "the_quant"
    assert cfg["user"] == "dw_player"


def test_cfg_legacy_password_keeps_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(clickhouse_db.PWD_ENV, "legacy-secret")
    monkeypatch.setenv(clickhouse_db.PWD_ENV_UNIFIED, "unified-secret")

    assert clickhouse_db._cfg()["pwd"] == "legacy-secret"


def test_cfg_missing_password_names_unified_variable() -> None:
    with pytest.raises(RuntimeError, match=clickhouse_db.PWD_ENV_UNIFIED):
        clickhouse_db._cfg()


def _make_parquet_bytes() -> bytes:
    import io

    import polars as pl

    buf = io.BytesIO()
    pl.DataFrame({"v": [1]}).write_parquet(buf)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def test_query_df_retries_transient_incomplete_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IncompleteRead 是瞬时网络错误，必须重试而非让整批导出作废。"""
    import http.client

    monkeypatch.setenv(clickhouse_db.PWD_ENV_UNIFIED, "secret")
    monkeypatch.setattr(clickhouse_db.time, "sleep", lambda _s: None)
    payload = _make_parquet_bytes()
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.IncompleteRead(b"", 2)
        return _FakeResponse(payload)

    monkeypatch.setattr(clickhouse_db.urllib.request, "urlopen", fake_urlopen)
    df = clickhouse_db.query_df("SELECT 1 AS v")
    assert calls["n"] == 3
    assert df.height == 1


def test_query_df_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import http.client

    monkeypatch.setenv(clickhouse_db.PWD_ENV_UNIFIED, "secret")
    monkeypatch.setattr(clickhouse_db.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        raise http.client.IncompleteRead(b"", 2)

    monkeypatch.setattr(clickhouse_db.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="响应中断"):
        clickhouse_db.query_df("SELECT 1 AS v", max_attempts=3)
    assert calls["n"] == 3


def test_query_df_does_not_retry_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 4xx/5xx 是确定性拒绝，重试只会放大无效负载。"""
    import io as _io
    import urllib.error

    monkeypatch.setenv(clickhouse_db.PWD_ENV_UNIFIED, "secret")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "http://x", 400, "Bad Request", {}, _io.BytesIO(b"boom")
        )

    monkeypatch.setattr(clickhouse_db.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        clickhouse_db.query_df("SELECT 1 AS v")
    assert calls["n"] == 1
