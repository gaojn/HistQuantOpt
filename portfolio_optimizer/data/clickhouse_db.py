"""ClickHouse ``the_quant`` 只读连接层。

HTTP 接口 + ``FORMAT Parquet`` → polars，零额外依赖（标准库 urllib）。
连接参数走环境变量，**密码绝不入代码/git**：

| 变量 | 默认 | 说明 |
|---|---|---|
| ``CLICKHOUSE_HOST`` | ``124.222.224.45`` | 主机 |
| ``CLICKHOUSE_PORT`` | ``18123`` | HTTP 端口 |
| ``CLICKHOUSE_DB``   | ``the_quant`` | 库名 |
| ``CLICKHOUSE_USER`` | ``dw_player`` | 只读账号 |
| ``CLICKHOUSE_PASSWORD`` | —（**必填**） | 只读密码，仅环境变量 |

用法::

    from portfolio_optimizer.data.clickhouse_db import query_df
    df = query_df("SELECT * FROM ashare_eod_prices WHERE trade_dt='2024-06-03'")
"""

from __future__ import annotations

import base64
import io
import os
import re
import urllib.error
import urllib.request

import polars as pl

DEFAULTS = {
    "host": "124.222.224.45",
    "port": "18123",
    "db": "the_quant",
    "user": "dw_player",
}
PWD_ENV = "CLICKHOUSE_PASSWORD"
_TIMEOUT = 600  # 单次查询上限（秒）；按年拉行情足够

_FORMAT_RE = re.compile(r"\bFORMAT\s+\w+\s*$", re.IGNORECASE)


def _cfg() -> dict:
    pwd = os.environ.get(PWD_ENV)
    if not pwd:
        raise RuntimeError(
            f"请设置环境变量 {PWD_ENV}（ClickHouse 只读密码，绝不入代码/git）。\n"
            f"  export {PWD_ENV}='...'\n"
            f"  其余可选: CLICKHOUSE_HOST/PORT/DB/USER（默认 the_quant/dw_player）"
        )
    return {
        "host": os.environ.get("CLICKHOUSE_HOST", DEFAULTS["host"]),
        "port": os.environ.get("CLICKHOUSE_PORT", DEFAULTS["port"]),
        "db": os.environ.get("CLICKHOUSE_DB", DEFAULTS["db"]),
        "user": os.environ.get("CLICKHOUSE_USER", DEFAULTS["user"]),
        "pwd": pwd,
    }


def query_df(sql: str, *, timeout: int = _TIMEOUT) -> pl.DataFrame:
    """执行只读查询，返回 polars DataFrame（经 ``FORMAT Parquet``）。

    Args:
        sql: SQL 语句；未显式带 ``FORMAT`` 子句时自动追加 ``FORMAT Parquet``。
        timeout: 超时秒数。

    Returns:
        polars.DataFrame；空结果返回空 DataFrame。

    Note: 仅供受控的内部 sync 脚本使用（SQL 非用户输入），无注入面。
    """
    cfg = _cfg()
    q = sql.strip().rstrip(";")
    if not _FORMAT_RE.search(q):
        q += "\nFORMAT Parquet"
    url = f"http://{cfg['host']}:{cfg['port']}/?database={cfg['db']}"
    req = urllib.request.Request(url, data=q.encode("utf-8"), method="POST")
    token = base64.b64encode(f"{cfg['user']}:{cfg['pwd']}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read()  # noqa: S310 (固定内网HTTP)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"ClickHouse 查询失败 HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ClickHouse 连接失败: {e.reason}") from e
    if not raw:
        return pl.DataFrame()
    return pl.read_parquet(io.BytesIO(raw))


def ping() -> str:
    """连通性检查，返回服务端版本字符串。"""
    df = query_df("SELECT version() AS v")
    return df.item(0, "v") if df.height else "?"


__all__ = ["query_df", "ping", "PWD_ENV", "DEFAULTS"]
