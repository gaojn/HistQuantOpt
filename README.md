# HistQuantOpt

A股多因子组合优化框架：基于 Barra CNE6 风险模型，输出「Alpha → 优化 → 真实回测（T+1 VWAP / 涨跌停 / 停牌 / 非对称费率）→ HTML 报告」的完整闭环，支持**量化选股**（`alpha_max`）与**指数增强**（`index_enhance`）两条策略流水线。

## 安装

支持 Python 3.12 与 3.14.6，CI 以 3.14.6 为主。

```bash
pip install -e .          # 装依赖 + 注册 hqopt 命令
```

## 快速开始

数据准备完成后（见 [docs/操作指南.md § 1](docs/操作指南.md#1-环境与数据准备)）：

```bash
python scripts/verify_data_bundle.py --profile default \
  --lock data_bundle.default.lock.json
hqopt run configs/alpha_max_default.yaml        # 量化选股（全市场等权基准）
hqopt run configs/index_enhance_default.yaml    # 指数增强（中证1000）
```

产物在 `output/<策略>_default/`：`report.html`（交互报告）、`weights.parquet`、`nav.parquet`。
默认 YAML 的 `data.root: ".."` 相对配置文件定位项目数据根目录；因此 wheel 安装后也会
读取外部 `data/`、`alphas/`、数据清单和分档锁，不依赖 site-packages 内包含数据。

默认配置使用含未来信息的合成 Alpha，只用于流程验证；终端和产物目录会显示
`SYNTHETIC ALPHA` 提醒，HTML 报告不加水印。生产研究换成真实 Alpha 时，除通过
`--alpha-file` 覆盖路径外，还必须在 YAML 中显式设置 `alpha.source: file` 和
`alpha.synthetic: false`；CLI 及 Python API 的预载 `alpha_df` 都不会绕过或改写这两项。

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/操作指南.md](docs/操作指南.md)** | **新成员从这里开始**：数据准备（含 ClickHouse 拉数）、CLI 用法、配置参数与默认值、输出口径、调参 FAQ |
| [docs/method.md](docs/method.md) | 两策略数学模型、约束体系、团队默认参数、Barra 风格约束建议、收益归因方法 |
| [docs/design.md](docs/design.md) | 架构设计：因子体系 / 风险模型 / 模块结构 / 数据流 |
| [CHANGELOG.md](CHANGELOG.md) | **影响回测口径的变更**与需重跑的范围；升级后先看这里 |

## 测试

```bash
python3.14 -m pytest tests/ -q   # 主版本：3.14.6
python3.12 -m pytest tests/ -q   # 兼容版本
```

CI 在 3.14.6 / 3.12 两个版本上依次跑 lint、类型检查与带覆盖率门禁的测试：

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest -q --cov=hqopt --cov=scripts --cov-fail-under=82
```

三道门禁的口径：

| 门禁 | 范围 | 现状 |
|---|---|---|
| `ruff check .` | 全仓 | 零告警。规则集 E/F/I/B/UP/SIM/C4，E501 与另两条的豁免理由见 `[tool.ruff.lint]` |
| `mypy` | `hqopt` + `scripts`（范围写在 `[tool.mypy]`，无需传参） | 零错误，且不依赖任何 `# type: ignore` 兜底 |
| 覆盖率 | `hqopt` + `scripts` | 门禁 82%，当前约 84% |

覆盖率统计包含 `scripts/`——数据导出脚本同属交付面，其覆盖缺口应当可见。因此
阈值低于只统计 `hqopt` 时的 85%（彼时约 90%）：这是把现状暴露出来，不是放松要求。
`scripts/` 目前是薄弱层（`build_alphas_vwap5.py`、`export_index_close.py` 尚无测试），
补测试后应同步上调阈值。

`[tool.mypy]` 只对 polars 的 `str-bytes-safe` 做了全局豁免（其 `Series.min()/max()`
stub 返回类型含 `bytes`，而脚本格式化的是日期列，属误报）；`tests/` 不在检查范围内，
因为测试替身与 duck typing 会产生大量无价值告警。

### Golden 回归基线

`tests/test_golden_backtest.py` 在合成行情上跑完整 `run_backtest`，把绩效指标、
执行统计与整条净值序列（含 SHA-256）锁到 `tests/golden/backtest_baseline.json`。
CHANGELOG 里绝大多数 ⚠️ 破坏性变更都落在这条链路上，此测试是它们的哨兵——
口径一旦变化必然失败，强制显式确认而非静默漂移。

数据由测试自身解析式构造（零随机数、零超越函数），不读 `data/`、不连 ClickHouse，
并注入停牌、涨停、跌停、退市与 sell-only，确保锚住的是完整执行语义。

口径**有意**变更时重生成基线，并同步在 CHANGELOG 记录影响与需重跑范围：

```bash
HQOPT_UPDATE_GOLDEN=1 pytest tests/test_golden_backtest.py
```

未设该环境变量时基线只读，测试绝不会自动改写它。基线里组合收益为负是预期的——
合成面板不含 alpha，该测试锚定数值口径而非策略表现。
