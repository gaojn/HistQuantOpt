"""
批量优化入口（统一调用 pipeline）。

用法：
    python examples/run_optimize.py configs/zz500_enhance.yaml
    python examples/run_optimize.py configs/hs300_alpha_max.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer.pipeline.batch_optimize import run_batch_optimize

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python examples/run_optimize.py <config.yaml>")
        sys.exit(1)
    run_batch_optimize(sys.argv[1])
