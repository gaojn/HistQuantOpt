"""投研分析层：收益归因等不参与优化/回测主链路的事后分析工具。"""

from portfolio_optimizer.analysis.attribution import AttributionResult, ReturnAttributor

__all__ = ["AttributionResult", "ReturnAttributor"]
