"""公开导入路径的 smoke tests。"""


def test_package_imports_backtest_api():
    import hqopt
    from hqopt import Backtester, BacktestResult, RealisticBacktester
    from hqopt.backtest.engine import RealisticBacktester as EngineBacktester

    assert hqopt.RealisticBacktester is EngineBacktester
    assert RealisticBacktester is EngineBacktester
    assert Backtester is EngineBacktester
    assert BacktestResult.__name__ == "BacktestResult"
