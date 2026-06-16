"""
A股量化多头组合优化框架。

主流程：
    1. CNE6RiskModel           —— 加载 CNE6 因子风险模型（16 风格因子暴露 + 协方差）
    2. AlphaMaxOptimizer       —— QP 优化器（max w'α - γ‖w‖²）
    3. Backtester              —— 向量化回测引擎

数据层：
    RealMarketAdapter          —— parquet → MarketSnapshot
    IndexBenchmarkWeights      —— 分级靠档指数权重

详见 docs/manual.md。
"""

from portfolio_optimizer.data.generator import MarketSnapshot, TradingStatus
from portfolio_optimizer.data.real_adapter import RealMarketAdapter
from portfolio_optimizer.data.benchmark import IndexBenchmarkWeights
from portfolio_optimizer.factors.alpha_factors import AlphaFactors
from portfolio_optimizer.risk import CNE6RiskModel
from portfolio_optimizer.optimizer.alpha_max import (
    AlphaMaxConfig,
    AlphaMaxOptimizer,
    AlphaMaxResult,
)
from portfolio_optimizer.optimizer.index_enhance import (
    IndexEnhanceConfig,
    IndexEnhanceOptimizer,
    IndexEnhanceResult,
)
from portfolio_optimizer.backtest.engine import Backtester, BacktestResult

__all__ = [
    # 数据
    "MarketSnapshot", "TradingStatus",
    "RealMarketAdapter", "IndexBenchmarkWeights",
    # 因子 / 风险
    "AlphaFactors", "CNE6RiskModel",
    # 优化
    "AlphaMaxConfig", "AlphaMaxOptimizer", "AlphaMaxResult",
    "IndexEnhanceConfig", "IndexEnhanceOptimizer", "IndexEnhanceResult",
    # 回测
    "Backtester", "BacktestResult",
]
