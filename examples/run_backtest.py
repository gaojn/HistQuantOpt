"""
真实回测入口（统一调用 pipeline）。

用法：
    python examples/run_backtest.py configs/zz500_enhance.yaml
    python examples/run_backtest.py configs/hs300_alpha_max.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer.pipeline.backtest_pipeline import run_backtest

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python examples/run_backtest.py <config.yaml>")
        sys.exit(1)
    run_backtest(sys.argv[1])
