# 变更记录

本文件只记录**影响回测结果或产物口径**的变更，以及需要重跑历史结果的范围。
纯重构、测试补充、文档更新不在此列（见 git log）。

格式：`⚠️ 破坏性` = 同一份权重/配置在新旧版本下会得到不同数字。

---

## 未发布（工作区，2026-07-29）

### 修复：交付安全、wheel 数据路径与分档数据锁

- CNE6 暴露查询缓存键新增数据源、处理合同和版本指纹，拒绝复用旧的日期同名缓存。
- ClickHouse 默认使用 HTTPS 并校验证书；明文 HTTP 仅允许显式危险开关，不自动降级。
- 包版本统一为 `2.1.0`；配置新增外部 `data.root`，wheel 不再依赖 site-packages
  内存在项目数据。成交状态机文档同步为“信号日盘中执行旧目标、收盘覆盖剩余订单”。
- 数据锁默认按 `data_bundle.<profile>.lock.json` 选择，默认档兼容旧锁文件；本地
  `default`、`attribution`、`long_risk`、`attribution_long` 四档均完成真实数据校验。

### ⚠️ 破坏性：量化选股基准统一为全市场等权

`alpha_max` 的回测和收益归因现在显式使用同一个 `equal_weight` 基准。此前回测读取
中证全指收益、归因却使用全市场等权持仓，两个主动收益不可直接核对。归因不再把
`csiall` 等无本地官方成分权重的指数名称静默解释为等权；误用会直接报错。

- **影响**：量化选股的基准收益、超额收益、信息比率和归因累计主动收益会变化。
- **需重跑**：量化选股回测与收益归因；指数增强不受影响。

### 新增：独立回测与归因生成 RunManifest

`hqopt backtest` 和 `hqopt attribute` 现在与 `run/optimize` 一样，在输出目录生成
不可覆盖的 `run.<run_id>.manifest.json` 及最新入口 `run.manifest.json`。清单绑定
代码状态、有效 CLI 参数、权重及已有 bundle 配套输入、运行环境和全部产物哈希；
失败运行同样保存 `status=failed`。独立命令没有配置绑定的数据锁时明确标记为
`not_verified`。此外，独立 `backtest` 未传 `--out-dir` 时会自动选择策略权重目录或
`output/backtest/`。

### ⚠️ 破坏性：收益归因改用真实执行收益并单列执行影响

`hqopt attribute` 现在把成交账本的 `daily_ret` 作为组合收益真值，不再只用昨收
实际权重乘收盘到收盘收益近似组合收益。VWAP→收盘时点差、费用和滑点统一记入
`执行影响(含费用)`；模型残差仍只检验风格、行业、Country 和特质收益分解。

- **影响**：归因逐日明细新增 `holding_portfolio_return`、`model_active_return` 和
  `execution_effect`，累计贡献与同参数回测 NAV 严格闭合；历史归因数值会变化。
- **需重跑**：历史收益归因；优化权重和既有回测 NAV 不受影响。
- 预载 `alpha_df` 也必须先通过 `alpha.source` 与布尔型 `alpha.synthetic` 校验；
  `source=synthetic` 与 `synthetic=false` 的矛盾配置直接报错。

### 新增：波动率 / 跟踪误差硬上限（`vol_upper` / `te_upper`）

指数增强新增 `te_upper`（年化跟踪误差上限）、量化选股新增 `vol_upper`
（年化组合波动率上限），把风险从**软惩罚**改为可直接指定的**硬约束**：

```
max w'α − 换手惩罚   s.t.  σ(w) ≤ 上限
```

- **启用后目标函数的风险惩罚项自动置 0**——风险已由硬约束限定，再叠加软惩罚
  等于对同一风险重复收费。因此与 `risk_aversion` **互斥**，同时设置会抛
  `ValueError`（刻意设计：二者是同一有效前沿的两种参数化，不能叠加）。
- 单位为**年化小数**（`0.05`=5%）；CNE6 的 F/δ 为日频方差，内部按 ×252 换算。
- 需 `optimize(risk_snapshot=...)`，否则报错；不退回 L2 代理。
- 仅上限：下限是反凸约束，凸优化框架内无法表达。
- 与其他硬约束一致做求解后校验，超出容差判不可行；不可行原因会点名本约束。

**不影响既有结果**：两个新参数默认 `None`，未显式启用时行为与此前完全一致。

### ⚠️ 破坏性：次新股阈值 60 → 120 自然日，并开放配置

`hqopt/data/real_adapter.py`、`hqopt/pipeline/batch_optimize.py`、两份默认配置

`RealMarketAdapter` 的次新判定（`list_days < new_listing_days`，禁止持仓）默认由
60 个自然日放宽到 120，并新增 `universe.new_listing_days` 配置项，不再是构造函数
里写死的值。上市 60~120 日的股票现在同样进入 `zero`（无持仓禁开仓）/`sell_only`
（有持仓只卖不买）。

- **影响**：两条策略的优化域与权重都会变化——被排除的次新票增多，其权重转移到
  其余候选票；净值随之变化。
- **需重跑**：全部历史优化、回测与归因。若要保持旧口径，在配置里显式写
  `universe.new_listing_days: 60`。

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
- **修复 `hqopt run` / `hqopt optimize` 无法启动**：数据包校验逻辑由
  `scripts/verify_data_bundle.py` 移入 `hqopt/io/data_bundle.py`。此前包内
  `run_manifest` 从 `scripts.` 导入，而 `scripts/` 既非包也不在 wheel 内，
  控制台入口下必然 `ModuleNotFoundError`；pytest 会把 rootdir 加进 `sys.path`
  故测试全绿而未暴露。`scripts/verify_data_bundle.py` 保留为 CLI 包装，
  用法与既有导入不变。新增 `tests/test_package_imports_standalone.py`
  以子进程在仓库外逐模块导入，守护同类问题。
- `batch_optimize.py` 1167 → 350 行，阶段二/三拆入 `hqopt/pipeline/batch/` 包
  （`types` / `config` / `execution_walk` / `periods` / `publish`）。阶段一因 46 处
  `monkeypatch.setattr(batch, ...)` 打在其命名空间上而留在原模块，理由见
  design.md §6.1。导入面不变；已验证拆分前后权重/成交统计/只卖矩阵逐位一致。
- 补齐求解器降级路径测试（`tests/test_solver_fallback.py`）：CLARABEL 抛异常或
  返回非最优时必须真的降级 SCS，两者都失败时优化器返回零权重的不可行结果而非
  半成品向量。`optimizer/index_enhance.py` 覆盖率 63% → 72%。

---

## 更早

见 git log。此文件自 2026-07-28 起维护。
