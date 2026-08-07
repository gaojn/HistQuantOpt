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
    "cne6": "export_cne6_panels.py",         # CNE6S/L 风险面板（test_barra_cne6_gao）
    "factor-return": "export_factor_attribution.py",  # S/L因子/特质收益（同源归因）
    "index-close": "export_index_close.py",  # 指数日收盘价（wind_db）
    "index-weight": "export_index_weight.py",# 指数官方成分权重（wind_db）
}


def _override_alpha_file(cfg: dict, alpha_file: str | None) -> None:
    """CLI 只替换 Alpha 文件路径，不推断或改写其可信度声明。"""
    if alpha_file:
        alpha_cfg = cfg["alpha"]
        # source=synthetic 本身就是可信度声明；改成文件来源前必须将其固化，
        # 否则一个 --alpha-file 就能把含前视信号静默降级成“真实”文件。
        if alpha_cfg.get("source") == "synthetic":
            alpha_cfg["synthetic"] = True
        alpha_cfg["source"] = "file"
        alpha_cfg["path"] = str(Path(alpha_file).expanduser().resolve())


def _start_run_manifest(
    args: argparse.Namespace,
    cfg: dict,
    *,
    mode: str,
):
    """强制校验数据锁，并冻结本次运行的代码/配置/输入身份。"""
    from hqopt.io.run_manifest import (
        RunManifestRecorder,
        verify_data_bundle_lock,
    )

    evidence = verify_data_bundle_lock(cfg)
    logger.info(
        "数据锁校验通过：profile=%s  lock=%s",
        evidence.profile,
        evidence.lock_path,
    )
    return RunManifestRecorder.start(
        mode=mode,
        config=cfg,
        config_path=args.config,
        output_dir=Path(cfg["output"]["weights"]).parent,
        command=[str(value) for value in sys.argv],
        data_lock=evidence,
        project_root=Path(cfg["data"]["root"]),
    )


def _finish_run_manifest(
    recorder,
    cfg: dict,
    *,
    include_backtest: bool,
    error: Exception | None = None,
) -> None:
    from hqopt.io.run_manifest import (
        expected_run_artifacts,
        load_run_quality_checks,
    )

    weights = cfg["output"]["weights"]
    if error is not None:
        unique, _ = recorder.finalize(
            status="failed",
            artifacts=[],
            error=f"{type(error).__name__}: {error}",
        )
        logger.error("失败运行清单已保存：%s", unique)
        return

    artifacts = expected_run_artifacts(
        weights, include_backtest=include_backtest
    )
    quality_checks = load_run_quality_checks(weights)
    unique, _ = recorder.finalize(
        status="complete",
        artifacts=artifacts,
        quality_checks=quality_checks,
    )
    logger.info("运行清单已保存：%s", unique)


def cmd_optimize(args: argparse.Namespace) -> None:
    from hqopt.pipeline.batch_optimize import load_config, run_batch_optimize
    cfg = load_config(args.config)
    if args.output:
        cfg["output"]["weights"] = str(
            Path(args.output).expanduser().resolve()
        )
    _override_alpha_file(cfg, args.alpha_file)
    if args.risk_aversion is not None:
        cfg["optimizer"]["risk_aversion"] = args.risk_aversion
    recorder = _start_run_manifest(args, cfg, mode="optimize")
    try:
        run_batch_optimize(cfg)
        _finish_run_manifest(recorder, cfg, include_backtest=False)
    except Exception as exc:
        _finish_run_manifest(
            recorder, cfg, include_backtest=False, error=exc
        )
        raise


def _default_backtest_out_dir(weight_path: str | Path) -> Path:
    """标准权重写回策略目录；外部权重统一写到项目 output/backtest。"""
    resolved = Path(weight_path).expanduser().resolve()
    output_root = (ROOT / "output").resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError:
        return output_root / "backtest"
    return resolved.parent


def _start_standalone_manifest(
    args: argparse.Namespace,
    *,
    mode: str,
    output_dir: str | Path,
    effective_config: dict,
):
    """为无 YAML 配置的独立命令冻结参数、代码状态和权重输入。"""
    from hqopt.io.run_manifest import (
        RunManifestRecorder,
        execution_bundle_inputs,
    )

    return RunManifestRecorder.start(
        mode=mode,
        config=effective_config,
        config_path=None,
        output_dir=output_dir,
        command=[str(value) for value in sys.argv],
        data_lock=None,
        input_files=execution_bundle_inputs(args.weights),
    )


def _finish_standalone_manifest(
    recorder,
    *,
    mode: str,
    output_dir: str | Path,
    error: Exception | None = None,
) -> None:
    """完成独立命令清单；失败也保留不可覆盖的审计记录。"""
    from hqopt.io.run_manifest import expected_standalone_artifacts

    if error is not None:
        unique, _ = recorder.finalize(
            status="failed",
            artifacts=[],
            quality_checks={"data_lock": "not_verified"},
            error=f"{type(error).__name__}: {error}",
        )
        logger.error("失败运行清单已保存：%s", unique)
        return

    artifacts = expected_standalone_artifacts(output_dir, mode=mode)
    unique, _ = recorder.finalize(
        status="complete",
        artifacts=artifacts,
        quality_checks={
            "data_lock": "not_verified",
            "input_contract": "validated_or_explicit_external_weights",
        },
    )
    logger.info("运行清单已保存：%s", unique)


def cmd_backtest(args: argparse.Namespace) -> None:
    from hqopt.backtest.run import run_backtest
    # risk_free: None 时 run_backtest 使用默认值 0.02
    rf = args.risk_free if hasattr(args, "risk_free") and args.risk_free is not None else None
    out_dir = args.out_dir or _default_backtest_out_dir(args.weights)
    effective_config = {
        "weights": str(Path(args.weights).expanduser().resolve()),
        "start_date": args.start,
        "end_date": args.end,
        "index": args.index,
        "cost_buy": args.cost_buy,
        "cost_sell": args.cost_sell,
        "risk_free": 0.02 if rf is None else rf,
        "initial_value": args.initial_value,
        "output_dir": str(Path(out_dir).expanduser().resolve()),
        "alpha_path": args.alpha_path,
    }
    recorder = _start_standalone_manifest(
        args,
        mode="backtest",
        output_dir=out_dir,
        effective_config=effective_config,
    )
    try:
        run_backtest(
            args.weights, args.start, args.end, index=args.index,
            cost_buy=args.cost_buy, cost_sell=args.cost_sell,
            initial_value=args.initial_value, out_dir=out_dir,
            alpha_path=args.alpha_path,
            **({} if rf is None else {"risk_free": rf}),
        )
        _finish_standalone_manifest(
            recorder,
            mode="backtest",
            output_dir=out_dir,
        )
    except Exception as exc:
        _finish_standalone_manifest(
            recorder,
            mode="backtest",
            output_dir=out_dir,
            error=exc,
        )
        raise


def cmd_run(args: argparse.Namespace) -> None:
    """一站式：优化 → 回测 → 报告。基准默认 = 指增取 config.index、多头取全市场等权。"""
    from hqopt.backtest.run import run_backtest
    from hqopt.pipeline.batch_optimize import load_config, run_batch_optimize

    cfg = load_config(args.config)
    _override_alpha_file(cfg, args.alpha_file)
    data_root = Path(cfg["data"]["root"])

    strategy = cfg["strategy"]
    bench = (
        args.benchmark
        or cfg.get("backtest", {}).get("benchmark")
        or (cfg["index"] if strategy == "index_enhance" else "equal_weight")
    )

    recorder = _start_run_manifest(args, cfg, mode="run")

    try:
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
            cache_dir=data_root / "data" / "cache",
            index_close_path=data_root / "data" / "指数收盘价信息.csv",
            alpha_path=cfg.get("alpha", {}).get("path"),
        )
        _finish_run_manifest(recorder, cfg, include_backtest=True)
    except Exception as exc:
        _finish_run_manifest(
            recorder, cfg, include_backtest=True, error=exc
        )
        raise


def _default_attribution_out_dir(weight_path: str | Path) -> Path:
    """标准权重留在所属策略目录；外部权重统一落到项目 output/attribution。"""
    resolved = Path(weight_path).expanduser().resolve()
    output_root = (ROOT / "output").resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError:
        return output_root / "attribution"
    return resolved.parent / "attribution"


def cmd_attribute(args: argparse.Namespace) -> None:
    from hqopt.analysis.run import run_attribution
    out_dir = args.out_dir or _default_attribution_out_dir(args.weights)
    effective_config = {
        "weights": str(Path(args.weights).expanduser().resolve()),
        "start_date": args.start,
        "end_date": args.end,
        "index": args.index,
        "benchmark_weight_source": args.benchmark_weight_source,
        "benchmark_max_snapshot_age_days": (
            args.benchmark_max_snapshot_age_days
        ),
        "cne6_data_dir": (
            str(Path(args.cne6_data_dir).expanduser().resolve())
            if args.cne6_data_dir is not None
            else None
        ),
        "cost_buy": args.cost_buy,
        "cost_sell": args.cost_sell,
        "initial_value": args.initial_value,
        "output_dir": str(Path(out_dir).expanduser().resolve()),
    }
    recorder = _start_standalone_manifest(
        args,
        mode="attribute",
        output_dir=out_dir,
        effective_config=effective_config,
    )
    try:
        run_attribution(
            args.weights, args.start, args.end, index=args.index,
            benchmark_weight_source=args.benchmark_weight_source,
            benchmark_max_snapshot_age_days=args.benchmark_max_snapshot_age_days,
            out_dir=out_dir, cne6_data_dir=args.cne6_data_dir,
            cost_buy=args.cost_buy, cost_sell=args.cost_sell,
            initial_value=args.initial_value,
        )
        _finish_standalone_manifest(
            recorder,
            mode="attribute",
            output_dir=out_dir,
        )
    except Exception as exc:
        _finish_standalone_manifest(
            recorder,
            mode="attribute",
            output_dir=out_dir,
            error=exc,
        )
        raise


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
    pb.add_argument(
        "--index",
        default="zz1000",
        help="基准 key；equal_weight=全市场等权，其余为指数 key",
    )
    pb.add_argument(
        "--out-dir",
        default=None,
        help="报告输出目录；默认策略权重目录或 output/backtest",
    )
    pb.add_argument("--cost-buy", type=float, default=0.001)
    pb.add_argument("--cost-sell", type=float, default=0.002)
    pb.add_argument("--initial-value", type=float, default=1e8)
    pb.add_argument("--risk-free", type=float, default=None,
                    help="年化无风险利率（Sharpe 用），不传时默认 0.02")
    pb.add_argument("--alpha-path", default=None,
                    help="报告最后一期目标持仓表 Alpha 列用的 alpha parquet 路径，不传则该列显示—")
    pb.set_defaults(func=cmd_backtest)

    pa = sub.add_parser("attribute", help="权重→收益归因（风格/行业/Country/特质分解）")
    pa.add_argument("--weights", required=True, help="权重 parquet（长表/宽表）")
    pa.add_argument("--start", required=True, help="归因起始日 如 2020-01-01")
    pa.add_argument("--end", required=True, help="归因截止日 如 2026-05-31")
    pa.add_argument("--index", default="zz1000",
                     help="基准：equal_weight=全市场等权；hs300/zz500/zz1000 用官方成分权重")
    pa.add_argument(
        "--benchmark-weight-source",
        choices=["official_drift", "official_frozen", "reconstruct", "official"],
        default="official_drift",
        help="成分指数权重口径（默认 official_drift）",
    )
    pa.add_argument(
        "--benchmark-max-snapshot-age-days",
        type=int,
        default=30,
        help="official_drift 快照最大陈旧自然日数（默认 30）",
    )
    pa.add_argument(
        "--out-dir",
        default=None,
        help="归因结果输出目录；默认 output/<策略目录>/attribution",
    )
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
