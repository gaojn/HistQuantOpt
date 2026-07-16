# HistQuantOpt

A股多因子组合优化框架：基于 Barra CNE6 风险模型，输出「Alpha → 优化 → 真实回测（T+1 VWAP / 涨跌停 / 停牌 / 非对称费率）→ HTML 报告」的完整闭环，支持**量化选股**（`alpha_max`）与**指数增强**（`index_enhance`）两条策略流水线。

## 安装

```bash
pip install -e .          # 装依赖 + 注册 hqopt 命令
```

## 快速开始

数据准备完成后（见 [docs/操作指南.md § 1](docs/操作指南.md#1-环境与数据准备)）：

```bash
hqopt run configs/alpha_max_default.yaml        # 量化选股（对标中证全指）
hqopt run configs/index_enhance_default.yaml    # 指数增强（中证1000）
```

产物在 `output/<策略>_default/`：`report.html`（交互报告）、`weights.parquet`、`nav.parquet`。

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/操作指南.md](docs/操作指南.md)** | **新成员从这里开始**：数据准备（含 ClickHouse 拉数）、CLI 用法、配置参数与默认值、输出口径、调参 FAQ |
| [docs/method.md](docs/method.md) | 两策略数学模型、约束体系、团队默认参数、Barra 风格约束建议、收益归因方法 |
| [docs/design.md](docs/design.md) | 架构设计：因子体系 / 风险模型 / 模块结构 / 数据流 |

## 测试

```bash
python3 -m pytest tests/ -q
```
