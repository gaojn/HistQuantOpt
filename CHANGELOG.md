# 变更记录

本文件只记录**影响回测结果或产物口径**的变更，以及需要重跑历史结果的范围。
纯重构、测试补充、文档更新不在此列（见 git log）。

格式：`⚠️ 破坏性` = 同一份权重/配置在新旧版本下会得到不同数字。

---

## 未发布（工作区，2026-07-29）

### 新增：报告增加"最后一期优化器目标持仓"表

`backtest/engine.py`（`BacktestResult` 新增 `target_weights` 字段）、
`backtest/report.py`（`_build_holdings_table`，`generate_html_report` 新增
`cache_dir`/`alpha_path` 参数）、`backtest/run.py`（`run_backtest` 新增 `alpha_path`
参数）、`cli.py`（`hqopt run` 自动透传 config 的 `alpha.path`；`hqopt backtest` 新增
`--alpha-path` 可选项）

`report.html` 末尾新增一节，展示最后一个**调仓日**的优化器目标持仓：代码/名称/
总市值(亿元)/行业(中信一级)/权重/Alpha值，按权重降序，权重≈0 的仓位不显示。数据源是
`target_weights`（优化器原始目标权重，未经撮合/涨跌停/停牌调整），不是执行后的
`actual_weights`——两者在被阻断的调仓期会不同，报告展示的是"优化器本来想要什么"。
名称/行业/市值取自本地行情缓存（`hqopt.io.data_panel.load_panel`）在该调仓日的截面，
天然 as-of、无前视风险；Alpha 列同样按"≤该调仓日最近一次信号"的 as-of 规则取值（与
`pipeline.universe.get_alpha_for_date` 口径一致），展示原始值，不做截面标准化；
`alpha_path` 未提供时该列显示"—"。当日行情缓存缺行的股票（如退市滞留仓）名称/行业/
市值显示"—"。同步落盘 `report_data/holdings.parquet`。

- **影响**：仅报告展示与新增产物文件，不影响净值/权重/绩效指标。
- **需重跑**：无需重跑优化或回测，重新生成报告即可（`hqopt backtest --alpha-path ...`
  或 `generate_html_report(..., alpha_path=...)`）。

### 新增：目标换手落盘，并区分现金重投与股票间调仓

此前逐期目标换手只走 `logger.info`，终端刷过去就无法事后审计；且它按
`Σ|w_target − w_prev|` 计算，而 `w_prev` 取自成交账本、不含现金（Σ<1），
于是把「把上期留存现金买回股票」这段必需买入也计入了换手——与优化器约束侧的
口径不一致（约束写作 `gross ≤ max_turnover + cash_gap`，实际受限的是净换手）。

`batch_execution_stats.json` 的 `optimization` 段新增 `target_turnover`，逐期落盘
`gross` / `cash_gap` / `net` 三口径及各自均值。终端汇总行改为同时打印净换手
（与 `max_turnover` 同口径）与含现金重投的换手；单期进度行的「换手」改为「净换手」。

- **影响**：终端日志的换手数字（现在显示净值口径，比原来小 `cash_gap` 的量）与
  `batch_execution_stats.json` 的新增字段。**权重、净值、报告数字均不受影响**。
- **需重跑**：无。若要为历史 bundle 补上该审计字段，需重跑优化。
- 判定 `max_turnover` 是否吃紧时应看 `net`，看 `gross` 会高估调仓强度。

### ⚠️ 破坏性：换手率报告口径修正（净值不变，换手数字变）

`result.turnover` 的 index 是**成交日**——一次调仓若在 T+1 未成交完、顺延到
T+2/T+3，会各占一行。但报告层的每一个消费者都把它当成**调仓期**，导致
`mean()` 分母偏大、平均换手被系统性低估，`再平衡次数` 也被多计。

`BacktestResult` 新增 `turnover_by_rebalance`（index=调仓日，把该期分次成交合并；
某期被完全阻断时记 0，不从分母消失）与 `rebalance_turnover` 属性。报告的换手卡片、
年度换手表、换手柱状图、副标题「再平衡次数」及 `report_data/turnover.parquet`
全部改用按期口径；顶层 `turnover.parquet` 保持逐成交日语义不变（文档原本即如此描述）。

- **影响**：`平均单期双边换手`、`再平衡次数`、`年均调仓次数`、年度换手表的
  `rebalance_count` / `avg_turnover`、换手柱状图分组、`report_data/turnover.parquet`
  行数。合成场景实测：平均单期换手 64.2% → **107.0%**（低估 1.67 倍），
  再平衡次数 10 → **6**。`年化双边换手率` 本就用 `sum/年数`，数值不变——此前
  它与平均单期换手互不自洽（年化÷单期隐含的调仓频率远高于真实频率），现已自洽。
- **净值、权重、绩效指标不受影响**：golden 基线的 `nav_digest` 与修正前逐位一致。
- **需重跑**：只需重新生成报告，无需重跑优化或回测。若曾依据报告的换手数字判断
  成本合理性或策略容量，结论需按新数字复核。

### 非破坏性：质量门禁（回测数值不变）

本节不改变任何回测口径——golden 基线由现行代码生成，全部 423 项测试在
3.12 / 3.14.6 上通过，净值与权重逐位不变。

- **新增回测 golden 回归**（`tests/test_golden_backtest.py`）：在解析式构造的合成
  面板（零随机数、零超越函数，注入停牌/涨停/跌停/退市/sell-only）上跑完整
  `run_backtest`，把 12 项绩效指标、执行统计与整条净值序列的 SHA-256 锁定到
  `tests/golden/backtest_baseline.json`。今后本 CHANGELOG 里的 ⚠️ 破坏性变更
  应当先在此处失败；有意变更时用 `HQOPT_UPDATE_GOLDEN=1` 重生成基线。
- **CI 新增 mypy 门禁**，范围 `hqopt` + `scripts`，零错误且未使用任何
  `# type: ignore`。首轮 62 个报错全部靠补实例属性注解、显式化跨函数不变量
  （`assert`）、消除分支间变量名复用来消除，未改动任何运行逻辑。
- **覆盖率统计纳入 `scripts/`**，门禁 85% → 82%。阈值下调是因为统计面从
  `hqopt`（约 90%）扩大到含数据导出脚本后整体为约 84%，属暴露现状而非放松要求；
  `scripts/build_alphas_vwap5.py`、`scripts/export_index_close.py` 仍无测试，
  补齐后应上调阈值。

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

### ⚠️ 破坏性：Calmar 改回行业标准口径（撤销上一条"累计口径"变更）

`backtest/engine.py::calc_metrics`、`backtest/report.py::_annual_metrics`

上一条变更把 Calmar 分子从年化收益改成累计收益，本意是防止不足一年的短样本
被年化外推放大，但副作用是多年区间的全期/年度 Calmar 被系统性抬高——累计收益
不封顶地随区间变长，分母最大回撤却是有界的（-100%~0%），区间越长虚高越严重。
实测 6.4 年回测：年化收益 16.13%、最大回撤 -24.74%，标准 Calmar 应为 0.65，
旧口径（累计收益/回撤）算出 6.10，偏差近 9.4 倍，且跨不同长度区间完全不可比。

现改回行业标准定义：**Calmar = 年化收益(CAGR) / 最大回撤**，年度行与全期行
统一，超额 Calmar 同理用年化几何超额。不足一年的分组年化后数值仍会偏大，
这是短样本年化的固有特性，不再通过更换公式掩盖，解读时结合样本天数判断即可。

- **影响**：`metrics.parquet`、报告摘要卡及 `yearly.parquet` 里的 `calmar` /
  `excess_calmar` 字段；净值、权重、其余指标不变。
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
