# HistQuantOpt 文档索引

A 股量化组合优化框架，支持**量化多头**和**指数增强**两条策略流水线。

---

## 快速入门

### 流水线 B：ZZ500 指数增强（主流程）

```bash
# 批量优化 144 期（约 135s）
python examples/demo_batch_index_enhance.py

# 回测 + 生成 HTML 报告
python examples/demo_index_enhance_backtest.py

# 打开报告
open output/zz500_enhance_report.html
```

### 流水线 A：量化多头

```bash
python examples/demo_synthetic_alpha.py    # 生成合成 Alpha
python examples/demo_batch_alpha_max.py    # 96 期批量优化
python examples/demo_backtest.py           # 回测 + HTML 报告
open output/backtest_report.html
```

### CNE6 风险模型 + config 驱动批量优化（CLI）

```bash
python scripts/run_batch_optimize.py configs/zz500_enhance_cne6_horizon.yaml \
    --alpha-file output/my_alpha.parquet   # 可选：替换为自定义 Alpha
```

详见 [manual.md §12](manual.md#12-cne6-因子风险模型--config-驱动批量优化)。

### VWAP5 合成 Alpha 网格 + 批量回测

```bash
python examples/build_alphas_vwap5.py        # 生成28个不同IC/ICIR/换手的因子 -> alphas/
python examples/run_alpha_grid_pipeline.py   # 全网格批量优化+真实回测 -> output/vwap5_grid/
```

详见 [manual.md §13](manual.md#13-vwap5-合成-alpha-网格--批量回测)。

---

## 文档说明

| 文档 | 内容 |
|---|---|
| [manual.md](manual.md) | **操作手册**：项目结构、完整 API、参数配置、TE 控制、常见问题 |
| [optimization_method.md](optimization_method.md) | 量化多头优化方法详解：目标函数、7 类约束、数学推导 |
| [index_enhance_method.md](index_enhance_method.md) | 指数增强方法详解：分级靠档权重、相对约束体系 |
| [design.md](design.md) | 原始设计文档（历史参考） |

---

## 核心参数速查

### IndexEnhanceConfig（指数增强）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `weight_upper` | 0.02（ZZ500） | 单票绝对权重上限 |
| `min_constituent_ratio` | 0.80 | 目标指数成分股权重下限 |
| `industry_active_bound` | 0.07 | 行业相对基准偏离 ±7% |
| `style_active_bound` | 0.50 | 风格主动暴露 ±0.5σ |
| `tracking_penalty` | 10.0 | γ，越大越贴近基准 |
| `max_turnover` | 0.40 | 单次双边换手上限 |
| `weight_diff_l2_bound` | None | ‖w−w_bm‖₂ 硬约束（TE 精确控制） |

### TE 控制速查

| 目标 TE | 推荐配置 |
|---|---|
| ~8% | `tracking_penalty=10`（默认） |
| ~7.6% | `tracking_penalty=20` |
| ~6.7% | `tracking_penalty=40` 或 `weight_diff_l2_bound=0.088` |
| <6% | `tracking_penalty=60+` 或 `weight_diff_l2_bound=0.075` |

### AlphaMaxConfig（量化多头）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `weight_upper` | 0.02 | 单票权重上限 |
| `industry_upper` | 0.20 | 行业权重上限 |
| `diversification_penalty` | 0.05 | γ，L2 分散惩罚 |
| `style_bound` | 1.0 | 风格绝对暴露上限（σ） |
| `max_turnover` | None | 双边换手上限（None=不约束） |

---

## 实测绩效（ZZ500 增强，2023-06 ~ 2026-05，合成 Alpha）

| 指标 | 值 |
|---|---|
| 年化超额 | +53.12% |
| 跟踪误差 TE | 8.18% |
| 信息比率 IR | 6.497 |
| 超额最大回撤 | -5.14% |
| 月度胜率 | 88.9% |

> 合成 Alpha 含前瞻信息，真实因子超额通常 10~20%/年。
