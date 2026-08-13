"""
A股量化多头组合优化框架。

主流程：
    1. CNE6RiskModel           —— 加载 CNE6S(20风格)/L(16风格)因子风险模型
    2. AlphaMaxOptimizer       —— QP 优化器（max w'α - γ‖w‖²）
    3. RealisticBacktester     —— T+1 VWAP 真实执行回测引擎

数据层：
    RealMarketAdapter          —— parquet → MarketSnapshot
    IndexBenchmarkWeights      —— 分级靠档指数权重

详见 docs/操作指南.md。
"""

import logging
from importlib import import_module
from typing import Any

_EXPORTS = {
    # 数据
    "MarketSnapshot": ("hqopt.data.generator", "MarketSnapshot"),
    "TradingStatus": ("hqopt.data.generator", "TradingStatus"),
    "RealMarketAdapter": ("hqopt.data.real_adapter", "RealMarketAdapter"),
    "IndexBenchmarkWeights": ("hqopt.data.benchmark", "IndexBenchmarkWeights"),
    # 风险
    "CNE6RiskModel": ("hqopt.risk", "CNE6RiskModel"),
    "FactorReturnLoader": ("hqopt.risk", "FactorReturnLoader"),
    # 归因
    "AttributionResult": ("hqopt.analysis", "AttributionResult"),
    "ReturnAttributor": ("hqopt.analysis", "ReturnAttributor"),
    # 优化
    "AlphaMaxConfig": ("hqopt.optimizer.alpha_max", "AlphaMaxConfig"),
    "AlphaMaxOptimizer": ("hqopt.optimizer.alpha_max", "AlphaMaxOptimizer"),
    "AlphaMaxResult": ("hqopt.optimizer.alpha_max", "AlphaMaxResult"),
    "IndexEnhanceConfig": ("hqopt.optimizer.index_enhance", "IndexEnhanceConfig"),
    "IndexEnhanceOptimizer": ("hqopt.optimizer.index_enhance", "IndexEnhanceOptimizer"),
    "IndexEnhanceResult": ("hqopt.optimizer.index_enhance", "IndexEnhanceResult"),
    "TopNEqualConfig": ("hqopt.optimizer.topn_equal", "TopNEqualConfig"),
    "TopNEqualOptimizer": ("hqopt.optimizer.topn_equal", "TopNEqualOptimizer"),
    "TopNEqualResult": ("hqopt.optimizer.topn_equal", "TopNEqualResult"),
    # 回测
    "RealisticBacktester": ("hqopt.backtest.engine", "RealisticBacktester"),
    # Backtester 是旧公开 API 名称；真实执行回测已统一融合到 engine.py。
    "Backtester": ("hqopt.backtest.engine", "RealisticBacktester"),
    "BacktestResult": ("hqopt.backtest.engine", "BacktestResult"),
}

# 库惯例：默认挂 NullHandler，不强加日志配置；由调用方（脚本/应用）
# 通过 logging.basicConfig 决定输出与级别。
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """首次访问公开符号时再加载其实现模块，并缓存结果。"""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
