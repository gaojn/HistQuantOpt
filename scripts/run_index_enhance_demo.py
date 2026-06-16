"""
指数增强回测 Demo（默认中证1000，CNE6 风险模型，真实执行）。

完整闭环：加载行情 → CNE6 风格约束下优化权重 → 真实回测（T+1 VWAP/涨跌停/成本）
→ 生成 HTML 报告 + 落地数据 parquet。基准用官方指数收盘价（data/指数收盘价信息.csv）。

⚠️ 示例 alpha 为 VWAP5 合成因子（含未来信息），仅用于跑通流程，不代表真实可交易表现。

运行：
    python scripts/run_index_enhance_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from pathlib import Path

import pandas as pd

from portfolio_optimizer.io.data_panel import load_panel
from portfolio_optimizer.data.index_close import load_index_returns
from portfolio_optimizer.backtest.realistic_engine import RealisticBacktester
from portfolio_optimizer.backtest.report import generate_html_report
from portfolio_optimizer.pipeline.batch_optimize import run_batch_optimize, load_config

CONFIG      = "configs/index_enhance_demo.yaml"
INDEX       = "zz1000"      # 基准指数（官方收盘价列）
INDEX_NAME  = "中证1000"
ALPHA_PATH  = Path("alphas/alpha_vwap5_ic0.12_icir1.2_decay0.80.parquet")
OUT_DIR     = Path("output/index_enhance_demo")


def alpha_long_to_wide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    wide = df.pivot(index="date", columns="code", values="alpha").sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CONFIG)
    start_date = date.fromisoformat(cfg["backtest"]["start_date"])
    end_date   = date.fromisoformat(cfg["backtest"]["end_date"])

    print(f"\n{'='*70}")
    print(f"  {INDEX_NAME} 指数增强回测  |  {ALPHA_PATH.stem}")
    print(f"  区间: {start_date} ~ {end_date}  换仓: {cfg['backtest']['rebalance_freq']} 日")
    print(f"{'='*70}")

    print(f"\n[1] 加载行情数据...")
    panel = load_panel(
        date(start_date.year, 1, 1), end_date,
        columns=[
            "code", "date", "adj_close", "adj_vwap", "close",
            "limit_up", "limit_down", "amount",
            "float_mv", "free_mv", "total_mv",
            "free_turnover", "trade_status",
            "industry_l1", "list_days",
            "is_hs300", "is_zz500", "is_zz1000", "is_st",
        ],
    )
    print(f"  交易日={panel['date'].n_unique()}  股票={panel['code'].n_unique()}")

    print("\n[2] 预计算回测用宽表...")

    def to_wide(col: str) -> pd.DataFrame:
        d = (
            panel.select(["date", "code", col]).to_pandas()
            .pivot(index="date", columns="code", values=col).sort_index()
        )
        d.index = pd.to_datetime(d.index)
        return d

    adj_close_w    = to_wide("adj_close")
    adj_vwap_w     = to_wide("adj_vwap")
    close_raw_w    = to_wide("close")
    limit_up_w     = to_wide("limit_up")
    limit_down_w   = to_wide("limit_down")
    trade_status_w = to_wide("trade_status")

    print(f"\n[3] 加载 {INDEX_NAME} 官方指数收盘价作为基准...")
    bm_ret = (
        load_index_returns(INDEX, start_date, end_date)
        .reindex(adj_close_w.index).fillna(0.0)
    )
    bm_ret.name = INDEX.upper()

    print(f"\n[4] 优化权重（CNE6 风险模型）...")
    cfg["output"]["weights"] = str(OUT_DIR / "weights.parquet")
    weight_df = run_batch_optimize(cfg, panel=panel, alpha_df=alpha_long_to_wide(ALPHA_PATH))
    weight_df.index = pd.to_datetime(weight_df.index)

    print("\n[5] 真实执行回测（T+1 VWAP，涨跌停处理，买1‰卖2‰）...")
    ex = cfg.get("execution", {})
    bt = RealisticBacktester(
        cost_buy=float(ex.get("cost_buy", 0.001)),
        cost_sell=float(ex.get("cost_sell", 0.002)),
        risk_free=float(ex.get("risk_free", 0.02)),
    )
    result, _ = bt.run(
        weight_df=weight_df, adj_close=adj_close_w, adj_vwap=adj_vwap_w,
        close_raw=close_raw_w, limit_up_df=limit_up_w, limit_down_df=limit_down_w,
        trade_status_df=trade_status_w, benchmark_ret=bm_ret, initial_value=1e8,
    )
    print(f"\n{result.summary()}")

    report_path = generate_html_report(
        result, output_path=OUT_DIR / "report.html",
        title=f"{INDEX_NAME} 指数增强（{ALPHA_PATH.stem}，CNE6 风险模型，真实执行）",
    )
    print(f"\n  HTML 报告已生成: {report_path}")


if __name__ == "__main__":
    main()
