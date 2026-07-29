"""公开导入路径的 smoke tests。"""

import subprocess
import sys


def test_bare_package_import_is_lazy():
    code = """
import sys
import hqopt

assert "cvxpy" not in sys.modules
assert "hqopt.optimizer.alpha_max" not in sys.modules
assert "AlphaMaxOptimizer" in dir(hqopt)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_package_imports_backtest_api():
    import hqopt
    from hqopt import Backtester, BacktestResult, RealisticBacktester
    from hqopt.backtest.engine import RealisticBacktester as EngineBacktester

    assert hqopt.RealisticBacktester is EngineBacktester
    assert RealisticBacktester is EngineBacktester
    assert Backtester is EngineBacktester
    assert BacktestResult.__name__ == "BacktestResult"
