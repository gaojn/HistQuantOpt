"""公开导入路径的 smoke tests。"""


def test_package_imports_backtest_api():
    import portfolio_optimizer
    from portfolio_optimizer import Backtester, BacktestResult, RealisticBacktester
    from portfolio_optimizer.backtest.engine import RealisticBacktester as EngineBacktester

    assert portfolio_optimizer.RealisticBacktester is EngineBacktester
    assert RealisticBacktester is EngineBacktester
    assert Backtester is EngineBacktester
    assert BacktestResult.__name__ == "BacktestResult"
