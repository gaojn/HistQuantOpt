"""
A股量化多头组合优化框架。

主流程：
    1. CNE6RiskModel           —— 加载 CNE6 因子风险模型（16 风格因子暴露 + 协方差）
    2. AlphaMaxOptimizer       —— QP 优化器（max w'α - γ‖w‖²）
    3. RealisticBacktester     —— T+1 VWAP 真实执行回测引擎

数据层：
    RealMarketAdapter          —— parquet → MarketSnapshot
    IndexBenchmarkWeights      —— 分级靠档指数权重

详见 docs/操作指南.md。
"""

import logging

from hqopt.data.generator import MarketSnapshot, TradingStatus
from hqopt.data.real_adapter import RealMarketAdapter
from hqopt.data.benchmark import IndexBenchmarkWeights
from hqopt.risk import CNE6RiskModel, FactorReturnLoader
from hqopt.analysis import AttributionResult, ReturnAttributor
from hqopt.optimizer.alpha_max import (
    AlphaMaxConfig,
    AlphaMaxOptimizer,
    AlphaMaxResult,
)
from hqopt.optimizer.index_enhance import (
    IndexEnhanceConfig,
    IndexEnhanceOptimizer,
    IndexEnhanceResult,
)
from hqopt.backtest.engine import BacktestResult, RealisticBacktester

# Backtester 是旧公开 API 名称；真实执行回测已统一融合到 engine.py。
Backtester = RealisticBacktester

# 库惯例：默认挂 NullHandler，不强加日志配置；由调用方（脚本/应用）
# 通过 logging.basicConfig 决定输出与级别。
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # 数据
    "MarketSnapshot", "TradingStatus",
    "RealMarketAdapter", "IndexBenchmarkWeights",
    # 风险
    "CNE6RiskModel", "FactorReturnLoader",
    # 归因
    "AttributionResult", "ReturnAttributor",
    # 优化
    "AlphaMaxConfig", "AlphaMaxOptimizer", "AlphaMaxResult",
    "IndexEnhanceConfig", "IndexEnhanceOptimizer", "IndexEnhanceResult",
    # 回测
    "RealisticBacktester", "Backtester", "BacktestResult",
]
