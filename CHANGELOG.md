# 变更记录

本文件只记录**影响回测结果或产物口径**的变更，以及需要重跑历史结果的范围。
纯重构、测试补充、文档更新不在此列（见 git log）。

格式：`⚠️ 破坏性` = 同一份权重/配置在新旧版本下会得到不同数字。

---

## 未发布（工作区，2026-07-29）

### ⚠️ 破坏性：CNE6S 面板目录改名 `data/barra_cne6` → `data/barra_cne6_S`

`hqopt/risk/cne6_risk.py`、`hqopt/risk/attribution_data.py`、`data_manifest.yaml`、
`scripts/export_cne6_panels.py`、`scripts/export_factor_attribution.py`

S 与 L 两套模型的目录名此前不对称（S 无后缀），容易误配成跨模型混用。现统一为
`barra_cne6_S` / `barra_cne6_L`。同时对默认 S/L 目录加**因子合同校验**（fail-closed）：
因子数须为 S=51 / L=47、风格齐全、L 不得含 4 个快因子、协方差行列完整且无 NaN/Inf、
暴露面板字段齐全，任一不满足直接抛 `ValueError` 而非静默降级。

- **兼容**：旧目录名 `barra_cne6` 仍识别为 S 变体（仅用于变体推断，不再是默认路径）。
- **导出区间**：`export_cne6_panels.py` 由 2020-01-01~2026-05-31 扩至
  2014-01-01~2026-06-30，并改为按季度分块查询。
- **影响**：数据路径与校验，不改变既有面板的数值。旧目录需重命名或重新导出，
  否则 `CNE6RiskModel` 按默认路径找不到面板。
- **需重跑**：仅需重新导出面板；已产出的回测结果数值不受此项影响。

### ⚠️ 破坏性：CNE6 风险模型切换到 test_barra_cne6_gao

`scripts/export_cne6_panels.py`、`scripts/export_factor_attribution.py`、
`hqopt/risk/cne6_risk.py`、`data_manifest.yaml`

CNE6 风险面板及因子/特质收益统一改从 `test_barra_cne6_gao` 导出。短周期S采用
20个风格 + Country + 30个中信一级行业（51因子）；长周期L采用16个风格 +
Country + 30个行业（47因子）。源库协方差中的空行业因子会同时从行列过滤，
空值/空字符串/“未知”行业也不生成本地行业暴露。

- 收益归因分别导出并加载S/L的 `factor_return_*`、`specific_return_*`，禁止跨模型混用。
- `data_manifest.yaml` 分开校验S/L因子合同，并新增 `attribution_long` profile。
- **影响**：共同风格因子的暴露、协方差和特质风险也可能变化，不是仅增加4个S因子。
- **需重跑**：全部使用CNE6的历史优化、回测、收益归因和报告；旧数据锁文件失效。

### ⚠️ 破坏性：指数增强官方基准权重由月度冻结改为 PIT 每日漂移

`data/benchmark.py`、`scripts/export_index_weight.py`

默认来源改为 `official_drift`：T 日取不晚于 T 的最近官方快照 S，并按后复权
收盘价比漂移到 T 日；快照日使用官方原值，停牌只向前取最近有效价，不使用未来
快照。快照默认最多陈旧 30 个自然日；超限、成分名单变化、价格不完整或官方未覆盖
时告警并回退 free_mv 重构。逐期方法、快照日、陈旧日数、T+1 生效日和回退原因
写入 `batch_execution_stats.json/benchmark_quality`。

- 兼容：`official_frozen` 保留旧月度冻结口径；`reconstruct` 路径不变；旧值
  `official` 作为 `official_drift` 别名。
- 导出：`hqopt data index-weight --daily` 调用运行时同一实现生成
  `official_weight_daily.parquet`。`date` 是 T 日收盘观测日，新增
  `effective_date` 明确下一交易日生效，并附
  `anchor_date/snapshot_age_days/method/fallback_reason`。
- **影响**：全部 `index_enhance` 的基准权重、主动约束、优化权重和净值会变化；
  `alpha_max` 不受影响。
- **需重跑**：所有使用官方基准权重的指数增强历史优化、回测、归因和报告。

### ⚠️ 破坏性：回测首个调仓日的基准收益强制置零

`backtest/engine.py`

首个调仓日组合仍是全现金（目标收盘后才提交，最早 T+1 成交），`port_ret` 必为 0；
此前基准照常计息，导致整段超额被首日基准收益一次性侵蚀。现改为组合与基准统一
从首个调仓日**收盘**起算，`nav[0] = bm_nav[0] = excess_nav[0] = 1.0`。

- **影响**：全部含超额的指标（年化超额、IR、超额回撤、超额 Calmar、月度超额表）。
  首日基准 +1%、6 年回测约低估 0.16pct 年化超额；区间越短影响越大（3 年约 0.33pct）。
- **需重跑**：此前产出的**所有**回测结果。

### ⚠️ 破坏性：Alpha 默认做优化域内截面 z-score

`pipeline/universe.py::get_alpha_for_date`，配置项 `alpha.standardize`（默认 `true`）

目标函数 `w'α − γ·R(w)` 中 α 与风险/成本系数量纲耦合：同一因子排序乘 100 倍，
就能把 `γ=0.05` 下的 20 只分散组合压成 1 只全仓。标准化后 `risk_aversion` /
`turnover_penalty` / `diversification_penalty` 的默认标定才稳定，缺失票填 0 也才
等于「截面中性」。

- **影响**：只要传入的 Alpha 不是已标准化的截面 z-score，权重矩阵就会变化。
- **需重跑**：用未标准化因子（原始 PE、收益率等）跑过的结果。
  已自行标准化的因子设 `alpha.standardize: false` 可保持旧行为。

### ⚠️ 破坏性：Alpha 不可用时跳过该期，不再静默沿用

`pipeline/universe.py` + `pipeline/batch_optimize.py`

三种此前静默的情形现在会跳过该调仓期（账本继续执行旧目标）或直接报错：

| 情形 | 旧行为 | 新行为 |
|---|---|---|
| 陈旧超 `alpha.max_staleness_days` | 无限沿用最后一期信号 | 跳过该期（`null` 时仅告警） |
| 面板整体晚于回测区间 | 返回全零 alpha | 抛 `ValueError` |
| 对优化域零覆盖 | 返回全零 alpha | 跳过该期 |
| 截面无区分度（方差为 0） | 退化为纯风险最小化 | 抛 `AlphaZeroVarianceError`，跳过该期 |

- **影响**：Alpha 面板比回测区间短时，此前会用同一天的信号跑完剩余区间且毫无提示。
- **需重跑**：Alpha 覆盖区间短于回测区间的结果。
- `batch_execution_stats.json` 新增 `alpha_quality` 块，逐期记录 `as_of` 与
  `staleness_days`，可事后核查每期用的是哪天的信号。

### ⚠️ 破坏性：IR 与 Calmar 口径修正

`backtest/report.py::_annual_metrics`（同时影响 `<report>_data/yearly.parquet`）

- **IR**：分子由算术 `日均超额×252` 改为几何年化超额，与 `engine.calc_metrics`
  及 `docs/method.md` 的既有定义一致。此前年度行与全期行同列不可比，实测差 3%~59%。
- **Calmar**：年度与全期统一为**对应区间累计收益 / 同期最大回撤**，超额 Calmar
  使用全期累计几何超额。此前年度行会把不足一年的年份年化放大，而全期行又使用
  CAGR，导致同名指标口径不一致。新口径不做年化；多年区间累计收益会使数值显著
  高于常见的 CAGR Calmar，跨不同区间长度时不可直接横比。

- **影响**：`metrics.parquet`、报告摘要卡及 `yearly.parquet`；净值与权重不变。
- **需重跑**：只需重新生成报告，无需重跑优化。

### 非破坏性

- CNE6 风险面板改按需加载（`CNE6RiskModel(query_dates=...)`）：155 期回测峰值内存
  3.69 GB → 1.48 GB、查询 0.7 s → 0.2 s；结果与全量加载逐位一致。
- 两个优化器抽出 `optimizer/_common.py`；`run_batch_optimize` 与
  `RealisticBacktester.run` 拆分为多阶段。三处均以随机化场景验证输出逐位一致。
- CI 启用 ruff `E/F/I/B/UP/SIM/C4` 规则集与 85% 覆盖率门禁。
- `query_df` 对传输中断类异常（`IncompleteRead` / `RemoteDisconnected` /
  连接重置 / socket 超时）做指数退避重试（默认 4 次）；HTTP 4xx/5xx 属服务端
  确定性拒绝，不重试。此前大结果集分块导出中途断流会让已完成的工作全部作废。

---

## 更早

见 git log。此文件自 2026-07-28 起维护。
