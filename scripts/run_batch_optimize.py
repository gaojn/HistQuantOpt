"""批量组合优化 CLI。

用法：
    python scripts/run_batch_optimize.py configs/index_enhance_demo.yaml
    python scripts/run_batch_optimize.py configs/alpha_max_demo.yaml \\
        --alpha-file output/my_alpha.parquet \\
        --output output/my_weights.parquet \\
        --risk-aversion 20

--alpha-file 指向 parquet 宽表（index=date, columns=ticker，与权重矩阵同一约定），
传入后会覆盖配置文件里的合成 Alpha（alpha.source 改为 "file"）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_optimizer.pipeline.batch_optimize import load_config, run_batch_optimize


def main() -> None:
    parser = argparse.ArgumentParser(description="批量组合优化 CLI")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--alpha-file", help="外部 Alpha parquet（index=date, columns=ticker）")
    parser.add_argument("--output", help="覆盖输出权重矩阵路径")
    parser.add_argument("--cne6-dir", help="覆盖 CNE6 风险面板目录（如 data/barra_cne6_L）")
    parser.add_argument("--risk-aversion", type=float, help="覆盖风险厌恶系数 λ")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.alpha_file:
        cfg["alpha"]["source"] = "file"
        cfg["alpha"]["path"] = args.alpha_file
    if args.output:
        cfg["output"]["weights"] = args.output
    if args.cne6_dir:
        cfg["optimizer"]["cne6_data_dir"] = args.cne6_dir
    if args.risk_aversion is not None:
        cfg["optimizer"]["risk_aversion"] = args.risk_aversion

    run_batch_optimize(cfg)


if __name__ == "__main__":
    main()
