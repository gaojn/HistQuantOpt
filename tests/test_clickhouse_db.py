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
