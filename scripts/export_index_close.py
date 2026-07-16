"""从 wind_db.AINDEXEODPRICES 导出宽基指数日收盘价，生成回测基准 CSV。

输出与 data/指数收盘价信息.csv 同格式（宽表，date 为 YYYYMMDD）：
    date, 沪深300, 中证500, 中证800, 中证1000, 中证全指, 中证红利

供 hqopt.data.index_close 作为回测基准读取。

数据源：ClickHouse wind_db（与项目主库 the_quant 同实例、不同库/密码）。
复用 clickhouse_db.query_df，连接前注入 wind_db 库名与密码：
    密码走环境变量 CLICKHOUSE_WIND_PASSWORD（绝不入代码/git）。

运行：
    CLICKHOUSE_WIND_PASSWORD=... python scripts/export_index_close.py
    # 可选：--start 2010-01-01 --end 2026-06-30 --out data/指数收盘价信息.csv

注：原 CSV 含「万得全A」（Wind 自编指数，wind_db 无），已用「中证全指」替代其角色
（index_close.py 中 winda/wanda 别名已重定向到中证全指）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

# CSV 中文列名 → Wind 指数代码（s_info_windcode）。
# 中证全指 / 中证红利用中证指数公司口径 .CSI（与官方一致）。
NAME2CODE: dict[str, str] = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证800": "000906.SH",
    "中证1000": "000852.SH",
    "中证全指": "000985.CSI",
    "中证红利": "000922.CSI",
}
COLUMN_ORDER = list(NAME2CODE.keys())

DEFAULT_OUT = Path("data/指数收盘价信息.csv")


def fetch_wide(start: str, end: str) -> pd.DataFrame:
    """拉取并 pivot 成宽表（index=YYYYMMDD int，columns=中文指数名）。"""
    # 注入 wind_db 连接（库名 + 专用密码），复用 the_quant 的 HTTP 查询层
    pwd = os.environ.get("CLICKHOUSE_WIND_PASSWORD")
    if not pwd:
        raise RuntimeError(
            "请设置环境变量 CLICKHOUSE_WIND_PASSWORD（wind_db 只读密码，绝不入代码/git）。"
        )
    os.environ["CLICKHOUSE_PASSWORD"] = pwd
    os.environ["CLICKHOUSE_DB"] = "wind_db"
    from hqopt.data.clickhouse_db import query_df

    codes = "','".join(NAME2CODE.values())
    sql = f"""
        SELECT trade_dt, s_info_windcode, s_dq_close
        FROM AINDEXEODPRICES
        WHERE s_info_windcode IN ('{codes}')
          AND trade_dt BETWEEN '{start}' AND '{end}'
    """
    df = query_df(sql).to_pandas()
    if df.empty:
        raise RuntimeError("查询为空，请检查日期区间与指数代码")

    code2name = {v: k for k, v in NAME2CODE.items()}
    df["name"] = df["s_info_windcode"].map(code2name)
    df["date"] = pd.to_datetime(df["trade_dt"]).dt.strftime("%Y%m%d").astype(int)
    df["s_dq_close"] = df["s_dq_close"].astype(float)

    wide = (
        df.pivot(index="date", columns="name", values="s_dq_close")
        .reindex(columns=COLUMN_ORDER)
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide


def main() -> None:
    p = argparse.ArgumentParser(description="导出宽基指数日收盘价 → 回测基准 CSV")
    p.add_argument("--start", default="2010-01-01", help="起始日（含），默认 2010-01-01")
    p.add_argument("--end", default="2026-12-31", help="结束日（含），默认 2026-12-31")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出 CSV，默认 {DEFAULT_OUT}")
    args = p.parse_args()

    print(f"[1/2] 拉取 {len(NAME2CODE)} 个指数收盘价（{args.start} ~ {args.end}）...")
    wide = fetch_wide(args.start, args.end)
    cov = wide.notna().sum()
    print(f"      {len(wide)} 个交易日  {wide['date'].min()}~{wide['date'].max()}")
    for c in COLUMN_ORDER:
        print(f"        {c}: {int(cov[c])} 个非空")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        bak = out.with_suffix(out.suffix + ".bak")
        shutil.copy(out, bak)
        print(f"[2/2] 原文件已备份 → {bak}")
    wide.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"      已写入 {out}")


if __name__ == "__main__":
    main()
