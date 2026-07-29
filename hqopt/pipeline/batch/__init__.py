"""``batch_optimize`` 的分阶段实现。

拆分边界见 docs/design.md §6.1。要点：**阶段一（输入准备）留在
``hqopt.pipeline.batch_optimize``**，因为测试对 ``CNE6RiskModel`` /
``AlphaMaxOptimizer`` / ``IndexEnhanceOptimizer`` / ``IndexBenchmarkWeights`` /
``ExecutionLedger`` 的 monkeypatch 打在那个模块的命名空间上；把构造这些对象的
代码移到本包会让 patch 静默失效（失败模式是假绿，不是报错）。

本包内各模块的依赖全部经 ``_BatchInputs`` 实例传入，不引用可 patch 的模块级全局。
"""
