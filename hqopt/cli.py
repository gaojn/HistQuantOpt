"""hqopt —— A股组合优化与回测统一命令行入口。

子命令：
    hqopt run <config>                          一站式：优化 → 回测 → HTML 报告
    hqopt optimize <config>                     逐期优化，输出权重矩阵
    hqopt backtest --weights ... --index ...    权重 → 回测 → HTML 报告
    hqopt attribute --weights ... --index ...   权重 → 收益归因（风格/行业/特质）
    hqopt data sync|cne6|factor-return|index-close|index-weight   数据准备（导出脚本透传）

安装后即可使用：`pip install -e .`，然后 `hqopt run configs/index_enhance_default.yaml`。
"""

from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# data 子命令 → scripts/ 下的导出脚本（数据准备为开发期运维，依赖源码仓库）
_DATA_SCRIPTS = {
    "sync": "sync_ashare_cache.py",          # A股行情缓存（the_quant）
    "cne6": "export_cne6_panels.py",         # CNE6 风险面板（the_quant）
    "factor-return": "export_factor_attribution.py",  # 因子/特质收益（cne6_risk，收益归因用）
    "index-close": "export_index_close.py",  # 指数日收盘价（wind_db）
    "index-weight": "export_index_weight.py",# 指数官方成分权重（wind_db）
}


def _override_alpha_file(cfg: dict, alpha_file: str | None) -> None:
    """CLI 显式替换 Alpha 时按真实外部信号处理；合成文件应改 YAML 声明。"""
    if alpha_file:
        cfg["alpha"]["source"] = "file"
        cfg["alpha"]["path"] = alpha_file
        cfg["alpha"]["synthetic"] = False


def cmd_optimize(args: argparse.Namespace) -> None:
    from hqopt.pipeline.batch_optimize import load_config, run_batch_optimize
    cfg = load_config(args.config)
    if args.output:
        cfg["output"]["weights"] = args.output
    _override_alpha_file(cfg, args.alpha_file)
    if args.risk_aversion is not None:
        cfg["optimizer"]["risk_aversion"] = args.risk_aversion
    run_batch_optimize(cfg)


def cmd_backtest(args: argparse.Namespace) -> None:
    from hqopt.backtest.run import run_backtest
    # risk_free: None 时 run_backtest 使用默认值 0.02
    rf = args.risk_free if hasattr(args, "risk_free") and args.risk_free is not None else None
    run_backtest(
        args.weights, args.start, args.end, index=args.index,
        cost_buy=args.cost_buy, cost_sell=args.cost_sell,
        initial_value=args.initial_value, out_dir=args.out_dir,
        **({} if rf is None else {"risk_free": rf}),
    )


def cmd_run(args: argparse.Namespace) -> None:
    """一站式：优化 → 回测 → 报告。基准默认 = 指增取 config.index、多头取中证全指。"""
    from hqopt.pipeline.batch_optimize import load_config, run_batch_optimize
    from hqopt.backtest.run import run_backtest

    cfg = load_config(args.config)
    _override_alpha_file(cfg, args.alpha_file)

    strategy = cfg["strategy"]
    bench = (
        args.benchmark
        or cfg.get("backtest", {}).get("benchmark")
        or (cfg["index"] if strategy == "index_enhance" else "csiall")
    )

    # 1. 优化（权重落地到 cfg.output.weights）
    run_batch_optimize(cfg)
    wpath = cfg["output"]["weights"]

    # 2. 回测 + 报告（输出到权重同目录）
    bt, ex = cfg["backtest"], cfg.get("execution", {})
    run_backtest(
        wpath, bt["start_date"], bt["end_date"], index=bench,
        cost_buy=float(ex.get("cost_buy", 0.001)),
        cost_sell=float(ex.get("cost_sell", 0.002)),
        risk_free=float(ex.get("risk_free", 0.02)),
        initial_value=float(bt["initial_value"]),
        out_dir=Path(wpath).parent,
    )


def cmd_attribute(args: argparse.Namespace) -> None:
    from hqopt.analysis.run import run_attribution
    run_attribution(
        args.weights, args.start, args.end, index=args.index,
        out_dir=args.out_dir, cne6_data_dir=args.cne6_data_dir,
        cost_buy=args.cost_buy, cost_sell=args.cost_sell,
        initial_value=args.initial_value,
    )


def cmd_data(what: str, extra: list[str]) -> None:
    """透传到 scripts/ 下的导出脚本（其余参数原样转交）。"""
    script = ROOT / "scripts" / _DATA_SCRIPTS[what]
    if not script.exists():
        raise SystemExit(f"找不到 {script}（数据脚本需在源码仓库内运行）")
    sys.argv = [str(script), *extra]
    runpy.run_path(str(script), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hqopt", description="A股组合优化与回测 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="一站式：优化→回测→报告")
    pr.add_argument("config", help="YAML 配置路径")
    pr.add_argument("--benchmark", default=None, help="回测基准指数 key（默认按策略推断）")
    pr.add_argument("--alpha-file", default=None, help="覆盖配置里的 alpha 路径")
    pr.set_defaults(func=cmd_run)

    po = sub.add_parser("optimize", help="逐期优化，输出权重")
    po.add_argument("config", help="YAML 配置路径")
    po.add_argument("--output", default=None, help="权重输出 parquet（覆盖配置）")
    po.add_argument("--alpha-file", default=None, help="覆盖配置里的 alpha 路径")
    po.add_argument("--risk-aversion", type=float, default=None, help="覆盖 risk_aversion")
    po.set_defaults(func=cmd_optimize)

    pb = sub.add_parser("backtest", help="权重→回测→报告")
    pb.add_argument("--weights", required=True, help="权重 parquet（长表/宽表）")
    pb.add_argument("--start", required=True, help="回测起始日 如 2020-01-01")
    pb.add_argument("--end", required=True, help="回测截止日 如 2026-05-31")
    pb.add_argument("--index", default="zz1000", help="基准指数 key")
    pb.add_argument("--out-dir", default=None, help="报告输出目录")
    pb.add_argument("--cost-buy", type=float, default=0.001)
    pb.add_argument("--cost-sell", type=float, default=0.002)
    pb.add_argument("--initial-value", type=float, default=1e8)
    pb.add_argument("--risk-free", type=float, default=None,
                    help="年化无风险利率（Sharpe 用），不传时默认 0.02")
    pb.set_defaults(func=cmd_backtest)

    pa = sub.add_parser("attribute", help="权重→收益归因（风格/行业/Country/特质分解）")
    pa.add_argument("--weights", required=True, help="权重 parquet（长表/宽表）")
    pa.add_argument("--start", required=True, help="归因起始日 如 2020-01-01")
    pa.add_argument("--end", required=True, help="归因截止日 如 2026-05-31")
    pa.add_argument("--index", default="zz1000",
                     help="基准：hs300/zz500/zz1000 用官方成分权重，其余（如 all/csiall）用全市场等权")
    pa.add_argument("--out-dir", default=None, help="归因结果输出目录")
    pa.add_argument("--cne6-data-dir", default=None, help="CNE6 风险面板目录，默认短周期 S")
    pa.add_argument("--cost-buy", type=float, default=0.001)
    pa.add_argument("--cost-sell", type=float, default=0.002)
    pa.add_argument("--initial-value", type=float, default=1e8)
    pa.set_defaults(func=cmd_attribute)

    pdt = sub.add_parser("data", help="数据准备（透传到导出脚本）")
    pdt.add_argument("what", choices=list(_DATA_SCRIPTS), help="要导出的数据")
    pdt.set_defaults(func=None)   # data 走透传分支

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if args.cmd == "data":
        cmd_data(args.what, extra)
        return
    if extra:
        parser.error(f"未知参数: {' '.join(extra)}")
    args.func(args)


if __name__ == "__main__":
    main()
