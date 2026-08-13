"""随机选股信号生成器——轮动回测（hqopt rotate）的基线/压测信号。

每个交易日从全市场「当日正常交易」的股票里随机选 0 ~ K 只（只数也随机，
均匀取自 {0, 1, ..., K}），输出 [date, code] 长表。date=T（信号日），
配合 `hqopt rotate` 即 T+1 开盘买入、T+H 收盘卖出。

用途：随机信号没有任何选股能力，回测结果应围绕市场基准波动，可用来
检验轮动回测引擎的执行成本、涨跌停/停牌处理是否引入系统性偏差。

示例：
    python scripts/gen_random_picks.py \
        --start 2020-01-01 --end 2026-05-31 \
        --k-max 5 --seed 42 \
        --out output/rotate_random/picks_random.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hqopt.io.data_panel import load_panel

logger = logging.getLogger(__name__)


def generate_random_picks(
    start: str,
    end: str,
    k_max: int = 5,
    seed: int = 42,
    cache_dir: str | Path | None = None,
):
    """逐交易日随机选 0~k_max 只当日正常交易的股票，返回 pandas 长表。"""
    panel = load_panel(
        pd.Timestamp(start).date(), pd.Timestamp(end).date(),
        columns=["code", "date", "trade_status"],
        cache_dir=cache_dir,
    )
    tradable = (
        panel.filter(pl.col("trade_status") == "交易")
        .select(["date", "code"])
        .to_pandas()
    )

    rng = np.random.default_rng(seed)
    frames = []
    for day, group in tradable.groupby("date"):
        k = int(rng.integers(0, k_max + 1))
        if k == 0:
            continue
        codes = rng.choice(group["code"].to_numpy(), size=k, replace=False)
        frames.append({"date": day, "codes": codes})

    picks = pd.DataFrame(
        [
            {"date": row["date"], "code": code}
            for row in frames
            for code in row["codes"]
        ],
        columns=["date", "code"],
    ).sort_values(["date", "code"]).reset_index(drop=True)
    return picks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="每日全市场随机选 0~K 只股票")
    ap.add_argument("--start", required=True, help="起始日 如 2020-01-01")
    ap.add_argument("--end", required=True, help="截止日 如 2026-05-31")
    ap.add_argument("--k-max", type=int, default=5, help="每日最多选股只数（默认 5）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42，可复现）")
    ap.add_argument("--out", required=True, help="输出 parquet 路径")
    ap.add_argument("--cache-dir", default=None, help="行情缓存目录，默认用仓库内置")
    args = ap.parse_args()

    picks = generate_random_picks(
        args.start, args.end, k_max=args.k_max, seed=args.seed,
        cache_dir=args.cache_dir,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(out, index=False)

    n_days = picks["date"].nunique()
    logger.info(
        "已生成随机选股：%s\n  信号日=%d  总行数=%d  日均选股=%.2f  "
        "区间=%s~%s  seed=%d",
        out, n_days, len(picks), len(picks) / n_days,
        picks["date"].min().date(), picks["date"].max().date(), args.seed,
    )


if __name__ == "__main__":
    main()
