"""ClickHouse ``the_quant`` 只读连接层。

HTTP 接口 + ``FORMAT Parquet`` → polars，零额外依赖（标准库 urllib）。
连接参数走环境变量，**密码绝不入代码/git**：

| 变量 | 默认 | 说明 |
|---|---|---|
| ``CLICKHOUSE_HOST`` | ``124.222.224.45`` | 主机 |
| ``CLICKHOUSE_PORT`` | ``18123`` | HTTP 端口 |
| ``CLICKHOUSE_DB``   | ``the_quant`` | 库名（SQL 里写 ``库名.表名`` 可跨库，无需切换） |
| ``CLICKHOUSE_USER`` | ``dw_player`` | 只读账号 |
| ``CLICKHOUSE_WIND_PASSWORD`` | —（**必填**） | 只读密码，仅环境变量；2026-07-10 起服务端凭证已合并，单一密码覆盖全部库（wind_db/the_quant/test_barra_cne6_gao/...） |
| ``CLICKHOUSE_PASSWORD`` | — | 旧变量名，仍兼容（优先级高于上者） |

用法::

    from hqopt.data.clickhouse_db import query_df
    df = query_df("SELECT * FROM ashare_eod_prices WHERE trade_dt='2024-06-03'")
"""

from __future__ import annotations

import base64
import http.client
import io
import os
import re
import socket
import time
import urllib.error
import urllib.request

import polars as pl

DEFAULTS = {
    "host": "124.222.224.45",
    "port": "18123",
    "db": "the_quant",
    "user": "dw_player",
}
PWD_ENV = "CLICKHOUSE_PASSWORD"          # 旧变量名，优先读取（向后兼容）
PWD_ENV_UNIFIED = "CLICKHOUSE_WIND_PASSWORD"  # 合并后的单一凭证（覆盖全部库）
_TIMEOUT = 600  # 单次查询上限（秒）；按年拉行情足够

_FORMAT_RE = re.compile(r"\bFORMAT\s+\w+\s*$", re.IGNORECASE)


def _cfg() -> dict:
    pwd = os.environ.get(PWD_ENV) or os.environ.get(PWD_ENV_UNIFIED)
    if not pwd:
        raise RuntimeError(
            f"请设置环境变量 {PWD_ENV_UNIFIED}（ClickHouse 只读密码，绝不入代码/git；"
            f"单一凭证覆盖全部库，旧变量 {PWD_ENV} 仍兼容）。\n"
            f"  export {PWD_ENV_UNIFIED}='...'\n"
            f"  其余可选: CLICKHOUSE_HOST/PORT/DB/USER（默认 the_quant/dw_player）"
        )
    return {
        "host": os.environ.get("CLICKHOUSE_HOST", DEFAULTS["host"]),
        "port": os.environ.get("CLICKHOUSE_PORT", DEFAULTS["port"]),
        "db": os.environ.get("CLICKHOUSE_DB", DEFAULTS["db"]),
        "user": os.environ.get("CLICKHOUSE_USER", DEFAULTS["user"]),
        "pwd": pwd,
    }


# 大结果集的 chunked 响应可能在传输中途被截断（实测 2026-07-29：CNE6 暴露
# 分季导出在第 31/50 块抛 IncompleteRead，30 块已完成的工作全部作废）。
# 这类异常是瞬时网络问题、重试即可恢复，且不属于 HTTPError/URLError，
# 故单独列出并做指数退避重试；HTTP 4xx/5xx 是服务端确定性拒绝，不重试。
_TRANSIENT_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionResetError,
    socket.timeout,
)
_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 2.0
# 服务端内存超限：退避 4/16/64 秒，等待并发查询释放内存
_MEM_BACKOFF_BASE = 4.0
_MEMORY_LIMIT_RE = re.compile(r"Code:\s*241|MEMORY_LIMIT_EXCEEDED")


def query_df(
    sql: str, *, timeout: int = _TIMEOUT, max_attempts: int = _MAX_ATTEMPTS
) -> pl.DataFrame:
    """执行只读查询，返回 polars DataFrame（经 ``FORMAT Parquet``）。

    Args:
        sql: SQL 语句；未显式带 ``FORMAT`` 子句时自动追加 ``FORMAT Parquet``。
        timeout: 超时秒数。
        max_attempts: 瞬时网络错误的最大尝试次数（含首次），退避 2/4/8 秒。

    Returns:
        polars.DataFrame；空结果返回空 DataFrame。

    Note: 仅供受控的内部 sync 脚本使用（SQL 非用户输入），无注入面。
    """
    cfg = _cfg()
    q = sql.strip().rstrip(";")
    if not _FORMAT_RE.search(q):
        q += "\nFORMAT Parquet"
    url = f"http://{cfg['host']}:{cfg['port']}/?database={cfg['db']}"
    token = base64.b64encode(f"{cfg['user']}:{cfg['pwd']}".encode()).decode()
    raw = b""
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=q.encode("utf-8"), method="POST")
        req.add_header("Authorization", f"Basic {token}")
        try:
            raw = urllib.request.urlopen(req, timeout=timeout).read()  # noqa: S310 (固定内网HTTP)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:800]
            # 服务端内存超限（Code 241）取决于**并发负载**，本质是瞬时错误：
            # 实测同一查询在他人占用 14 GiB 时失败、负载回落后成功。故与其他
            # 4xx/5xx（确定性拒绝）区别对待，退避重试；退避比网络类更长，
            # 给并发查询留出释放内存的时间。
            if _MEMORY_LIMIT_RE.search(body) and attempt < max_attempts:
                time.sleep(_MEM_BACKOFF_BASE ** attempt)
                continue
            raise RuntimeError(f"ClickHouse 查询失败 HTTP {e.code}: {body}") from e
        except _TRANSIENT_ERRORS as e:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"ClickHouse 响应中断，重试 {max_attempts} 次仍失败: {e!r}"
                ) from e
            time.sleep(_BACKOFF_BASE ** attempt)
        except urllib.error.URLError as e:
            # URLError 包装的底层瞬时错误（如 socket.timeout）同样值得重试
            if isinstance(e.reason, _TRANSIENT_ERRORS) and attempt < max_attempts:
                time.sleep(_BACKOFF_BASE ** attempt)
                continue
            raise RuntimeError(f"ClickHouse 连接失败: {e.reason}") from e
    if not raw:
        return pl.DataFrame()
    return pl.read_parquet(io.BytesIO(raw))


def ping() -> str:
    """连通性检查，返回服务端版本字符串。"""
    df = query_df("SELECT version() AS v")
    return df.item(0, "v") if df.height else "?"


__all__ = ["query_df", "ping", "PWD_ENV", "PWD_ENV_UNIFIED", "DEFAULTS"]
