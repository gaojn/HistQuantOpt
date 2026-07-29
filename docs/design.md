# 多因子组合优化框架设计文档

## 1. 目标

基于 Barra CNE6 因子体系，构建适用于 A 股市场的多因子组合优化框架，
显式处理停牌、涨跌停、流动性等 A 股特有约束。

---

## 2. 因子体系

### 2.1 Alpha 因子（合成因子）

由研究员自行构建的预期收益信号，框架假设已有如下形式输入：

| 字段       | 类型            | 说明                     |
|----------|---------------|------------------------|
| `alpha`  | `(N,)` array  | 每只股票的预期超额收益（年化）        |

### 2.2 Barra CNE6 风格因子（S 20 个 / L 16 个）

来自 ClickHouse `test_barra_cne6_gao`，定义见
`hqopt/risk/cne6_risk.py::STYLE_FACTORS_S/STYLE_FACTORS_L`。
CNE6L 的 16 个核心风格为：

Size, MidCap, Beta, Momentum, ResidualVolatility, LongTermReversal, Liquidity,
Value, EarningsYield, Growth, Profitability, InvestmentQuality, EarningsQuality,
EarningsVariability, Leverage, DividendYield。

CNE6S 在上述核心风格上增加 AnalystSentiment、IndustryMomentum、Seasonality、
ShortTermReversal 4 个快策略风格，共 20 个。

风格暴露用于 `style_active_bound` 约束（支持按因子分别设定），因子协方差/特质风险
用于 `risk_aversion` 真跟踪误差目标项。

### 2.3 行业因子

参考 CITIC 一级行业分类，共 30 个行业虚拟变量（dummy variable）。空值、空字符串和
“未知”行业不进入模型；正常分类股票满足 $\sum_k B_{ik}^{ind}=1$，未分类股票行业暴露全为 0。

---

## 3. 风险模型

### 3.1 因子风险模型结构

$$\Sigma = B F B^\top + \Delta$$

| 符号        | 维度              | 含义                       |
|-----------|-----------------|--------------------------|
| $B$       | $(N \times K)$  | 因子载荷矩阵（风格+行业）            |
| $F$       | $(K \times K)$  | 因子协方差矩阵（正定）              |
| $\Delta$  | $(N \times N)$  | 特质风险矩阵（对角矩阵）             |
| $N$       | —               | 股票数量                     |
| $K$       | —               | S=20+Country+30=51；L=16+Country+30=47 |

### 3.2 组合风险分解

$$\sigma_p^2 = w^\top \Sigma w = \underbrace{w^\top B F B^\top w}_{\text{系统性风险}} + \underbrace{w^\top \Delta w}_{\text{特质风险}}$$

### 3.3 风险覆盖保护

CNE6 对齐后保留逐股票 `covered_mask`。缺失暴露填 0、缺失特质方差填中位数只用于
数值稳定，不代表该股票已被风险模型覆盖：未覆盖股票禁止新开仓，已有持仓只卖不买。
指数增强还按基准权重计算覆盖率；低于 `min_risk_coverage`（默认 90%）时跳过该期优化并告警（账本继续执行原目标），
避免把大块基准权重错误当成“零风险、零暴露”。

---

## 4. 优化问题

### 4.1 目标函数

$$\max_w \; \alpha^\top w - \lambda_{\text{risk}} \cdot w^\top \Sigma w - \lambda_{\text{tc}} \cdot \|w - w_0\|_1$$

| 参数                 | 含义      |
|--------------------|---------|
| $\alpha$           | Alpha 向量|
| $\lambda_{risk}$   | 风险厌恶系数  |
| $\lambda_{tc}$     | 交易成本系数  |
| $w_0$              | 当前持仓权重  |

### 4.2 约束集合

#### 基础约束

$$\mathbf{1}^\top w = 1 \quad \text{（满仓）}$$
$$w_i \in [lb_i,\ ub_i] \quad \forall i \quad \text{（个股上下界）}$$

#### 因子暴露约束

$$|B_{\text{style}}^\top w - e_{\text{target}}| \leq \varepsilon_{\text{style}}$$

#### 行业中性约束

$$|B_{\text{ind}}^\top w - B_{\text{ind}}^\top w_{\text{bm}}| \leq \varepsilon_{\text{ind}}$$

#### 换手率约束

$$\|w - w_0\|_1 \leq T_{\max}$$

$w_0$ 来自成交账本的实际股票权重，允许因未成交而留有现金、权重和小于 1。
此时现金缺口用于恢复仓位，不占用股票间换手预算：
$\|w-w_0\|_1\le T_{\max}+(1-\mathbf{1}^\top w_0)$。

冻结口径（批次2）：停牌持仓票不计入换手（$w_i=w_{\text{prev},i}$ 等式约束使 $|\Delta w_i|=0$）；
掉池/次新/ST 持仓票以 sell_only 约束（$w_i \le w_{\text{prev},i}$）卖出，正常计入换手。

#### 流动性约束

当前实现没有 ADV 参与率硬约束，而是通过 `turnover_penalty` 与
`build_cost_vector()` 对低流动性股票施加更高换手惩罚。

---

## 5. A 股特殊状态处理

### 5.1 交易状态枚举

| 状态            | 说明         | 优化器处理                                         |
|---------------|------------|-----------------------------------------------|
| `NORMAL`      | 正常交易       | 无特殊限制                                         |
| `SUSPENDED`   | 停牌         | 有持仓：$w_i = w_{\text{prev},i}$（冻结，不计换手）；无持仓：$w_i = 0$ |
| `LIMIT_UP`    | T 日涨停      | 不限制目标；执行日涨停时仅阻断买入                         |
| `LIMIT_DOWN`  | T 日跌停      | 不限制目标；执行日跌停时仅阻断卖出                         |
| `NEW_LISTING` | 上市首日/次新股  | $w_i = 0$（禁止持仓，规避炒作风险）                        |

### 5.2 目标权重与真实成交账本

候选调仓日只有在成功生成并提交新目标后才成为**有效调仓日**。此时旧目标只执行到
前一交易日，调仓日直接取消其 pending 状态、不再尝试旧单，当日仅按真实股数估值；
收盘以账本中的**实际持仓**作为本期 $w_{prev}$ 并提交新目标，最早在下一交易日
T+1 执行。若快照、风险覆盖或求解失败导致没有新权重行，则该日不构成有效调仓，
旧目标保持激活并正常尝试。优化器输出的是目标权重，不是成交后的持仓。

**停牌冻结口径**：
- T 日停牌且有持仓 → $w_i = w_{\text{prev},i}$，目标后处理不再改变该值；成交账本把
  股数冻结到本目标结束，即使 T+1 复牌也不产生订单；
- 停牌且无持仓 → $w_i = 0$，维持原状。

真实成交按 T+1 VWAP 先卖后买。账本为每只股票维护 `FROZEN / PENDING_SELL /
PENDING_BUY / FILLED / EXPIRED` 状态。每个执行日只对 pending 股票按最新 NAV 和
原始目标权重重算差额；已完整成交的股票立即锁定，不因随后价格漂移再次交易。

涨停阻断买入、跌停阻断卖出，停牌或无有效 VWAP 阻断双向；这些执行日仍消耗一次
尝试。订单最多尝试 T+1、T+2、T+3：T+3 执行后残余订单转为 `EXPIRED`，不再交易
直到下一次调仓。现金不足时，可交易买单按需求同比例部分成交并保持 pending；没有
已实现卖出且没有现金时不买。新目标提交时重置股票状态和三日尝试计数，其 T+1
从下一交易日开始。停牌期间用最近有效价格估值，NAV 不断裂。

**掉池/次新/ST 持仓票（sell_only 口径）**：`filter_universe` 将掉出候选池但当日有行情且
有持仓的股票携带入优化域（`MarketSnapshot.sell_only=True`），优化器施加 $w_i \le w_{\text{prev},i}$，
允许正常减仓并计换手，禁止加仓。实际持仓中缺少当日行情的票（退市/长期停牌滞留）
不进优化域，但按最近有效价格保留在真实账本中并告警：股数不变、不释放现金，也不
清零其权重或重新归一化其余目标。其余部分正常优化，账本只按真实现金成交，不会把
滞留市值误当现金再分配。冻结权重精确保留且不再二次归一化；预算超额只向下扣减
非冻结权重。单票 <1e-6 的求解器粉尘仅在累计也 ≤1e-6 时清零，否则保留，
避免改变冻结仓位、放大上限约束或累计破坏下限约束；微小预算缺口保留为现金。

---

## 6. 模块结构

```
hqopt/
├── data/
│   ├── real_adapter.py     # parquet 面板 → MarketSnapshot
│   ├── benchmark.py        # 官方快照 PIT 每日漂移（异常回退分级靠档重构）
│   ├── index_close.py      # 官方指数收盘价加载（回测基准）
│   ├── clickhouse_db.py    # ClickHouse 只读连接层
│   └── generator.py        # 合成数据/快照构件
├── io/
│   ├── data_panel.py       # load_panel 主入口
│   └── schema.py           # 行情字段定义
├── risk/
│   ├── cne6_risk.py        # CNE6 因子风险模型（暴露 X / 协方差 F / 特质 Δ）
│   │                       #   构造时传 query_dates 只加载所需调仓日截面
│   └── attribution_data.py # 因子收益 f(t) / 个股特质收益 u(t) 加载器（收益归因用）
├── optimizer/
│   ├── _common.py          # 两优化器共用：交易状态掩码/换手项/求解降级/结果基类
│   ├── alpha_max.py        # 量化选股 QP 优化器
│   └── index_enhance.py    # 指数增强 QP 优化器
├── backtest/
│   ├── engine.py           # 真实执行回测（T+1 VWAP/涨跌停/成本）+ 绩效指标
│   │                       #   run() = 对齐 → _replay_days 逐日重放 → 指标/统计
│   └── report.py           # Plotly HTML 报告
├── analysis/
│   ├── attribution.py      # 收益归因（风格/行业/Country/特质分解，Carino多期链接）
│   └── run.py              # 权重→归因，供 hqopt attribute 复用
└── pipeline/
    ├── batch_optimize.py   # 阶段一（输入准备）+ run_batch_optimize 编排
    │                       #   拆分边界与不可外移的原因见 §6.1
    ├── batch/              # 分阶段实现
    │   ├── types.py        #   阶段间数据结构（三阶段之间的契约）
    │   ├── config.py       #   YAML 解析 / Alpha 可信度参数（fail-closed）
    │   ├── execution_walk.py #  成交账本按日推进游标
    │   ├── periods.py      #   阶段二：逐期优化
    │   └── publish.py      #   阶段三：汇总与原子发布
    └── universe.py         # 候选池过滤 / 成本向量 / 合成 alpha
```

### 6.1 `batch_optimize.py` 的拆分边界

原为 1167 行单文件，扛了配置解析、输入准备、逐期优化、成交推进、汇总发布五段职责，
2026-07-29 拆为 `batch/` 包，主文件约 350 行。

**当前边界的设计依据是组合根，而不是测试 patch**：`batch_optimize.py` 负责把配置、
行情、风险模型、基准、优化器和成交账本装配成 `_BatchInputs`，是 pipeline 的
composition root；`batch/periods.py` 和 `batch/publish.py` 只消费已构造的依赖，
分别负责逐期状态推进和原子发布。这样依赖方向保持为“入口装配 → 业务阶段”，子模块
不反向读取入口模块的全局状态。

现有测试确有 46 处 `monkeypatch.setattr(batch, ...)`，覆盖
`CNE6RiskModel`、`AlphaMaxOptimizer`、`IndexEnhanceOptimizer`、
`IndexBenchmarkWeights`、`ExecutionLedger`。这是当前导入路径的**兼容债务**，
不是“阶段一永远不能外移”的架构理由。若未来把装配继续拆出，应先引入显式依赖注入，
或把测试 patch 迁移到真正拥有构造点的模块，并增加断言证明替身确实被调用。

| 模块 | 行数 | 内容 |
|---|---:|---|
| `batch_optimize.py` | 约 350 | composition root + `run_batch_optimize` + 4 个兼容 re-export |
| `batch/types.py` | 116 | `_AlphaPolicy` `_RunConfig` `_BatchInputs` `_PeriodContext` `_RunStats` `_PeriodOutcome` |
| `batch/config.py` | 127 | `load_config` `_parse_run_config` `_parse_style_bound` `_build_alpha_policy` 等 |
| `batch/execution_walk.py` | 136 | 成交日 helpers + `_ExecutionWalker` |
| `batch/periods.py` | 429 | 阶段二逐期优化 + `_clean_target_weights` |
| `batch/publish.py` | 143 | 阶段三 `_log_run_summary` `_publish_outputs` |

**对外契约不变**：`import hqopt.pipeline.batch_optimize as batch` 后按旧名访问的
符号继续可用。其中 4 个（`SYNTHETIC_ALPHA_WARNING_FILE`、`_DUST_WEIGHT_TOL`、
`_ExecutionWalker`、`_clean_target_weights`）本模块自身已不使用，在文件末尾以带
`# noqa: F401` 的导入显式 re-export，删除会破坏调用方。

**当前未继续拆分的原因**只是迁移范围控制：组合根仍然清晰，继续拆分不会直接改善业务
正确性，却需要同步调整现有替身注入方式。它不是被永久否决的方案；当装配逻辑继续增长时，
应以构造函数/工厂参数注入完成下一步，而不是继续依赖模块级 patch。

**现有验证证据**：`tests/test_batch_execution_feedback.py` 覆盖逐期优化、失败恢复和成交反馈，
`tests/test_execution_bundle_e2e.py` 覆盖 batch bundle 到回测/归因的重放契约。仓库当前没有
“拆分前后产物合并 SHA-256 对比”或“绕过 patch 的变异测试”，因此不把这两项写成已完成事实。

---

## 7. 数据流

```
行情面板 load_panel（data/cache）          CNE6 面板（data/barra_cne6_S[_L]）
    │                                            │
    ▼                                            ▼
RealMarketAdapter.build_snapshot         CNE6RiskModel.at(date)
  → MarketSnapshot                         → 暴露 X / 协方差 F / 特质 Δ / style_loading
  （tickers/行业/ADV/状态/市值/成分）              │
    │            ┌───────── Alpha 因子（alphas/*.parquet）
    ▼            ▼          │  get_alpha_for_date：asof + 陈旧度 + 截面 z-score
  filter_universe(prev_holdings)  ──►│   # 候选池 ∪ 上期持仓（carry=sell_only）
    │                                ▼
    └────────────────►  optimizer（alpha_max / index_enhance, cvxpy）
                                     │  共用 _common：状态掩码/换手项/求解降级
                                     │  约束：预算/上限(frozen豁免)/行业/风格/换手/冻结/sell_only
                            ▼
                    逐期权重矩阵 weight_df
                            │
                            ▼
                RealisticBacktester（T+1 VWAP / 涨跌停 / 成本）
                  基准：官方指数收盘价（index_close）
                            │
                            ▼
                generate_html_report → HTML + parquet
```

---

## 8. 关键设计决策

1. **风险项两档**：`risk_aversion` 设置时用 CNE6 真因子风险 `λ·(active'XFX'active+δ'active²)`；否则退回 L2 偏离惩罚 `γ·‖w−w_bm‖²`。
2. **成交状态闭环**：
   - 只有成功生成并提交新目标才形成有效调仓日；旧目标只推进到此前一交易日，
     有效调仓日直接取消其 pending 状态、仅估值，不再尝试旧目标。
   - `w_prev` 使用实际成交持仓，不使用上一期目标权重。收盘提交新目标并重置状态和
     计数，下一交易日作为新目标 T+1 开始三次尝试。
   - 候选调仓日若没有生成新权重行，则旧目标保持激活并在该日正常尝试。
   - T 日停牌有持仓 → 目标 `w=w_prev` 且整期冻结股数；停牌无持仓 → `w=0`。
   - 掉池/次新/ST 有持仓 → sell_only `w≤w_prev`（卖出计换手）；无持仓 → `w=0`。
   - T 日涨跌停不约束目标；执行日涨停仅阻断买入、跌停仅阻断卖出。
   - 只重试 pending 股票，已成交股票锁定；普通 pending 按最新差额决定方向，
     sell_only 可继续卖但绝不反向买；最多尝试 T+1/T+2/T+3，残余订单过期。
   - 先卖后买，现金不足时可交易买单同比例部分成交；现金作为权重缺口原样反馈给优化器。
   - 常规 `weights.parquet` 保存 `batch_execution_stats.json`；自定义权重文件保存
     `<weights stem>.batch_execution_stats.json`，同目录多个 bundle 互不覆盖。统计同时
     保存成交状态、优化成功/失败期数、`alpha_quality` 和 `benchmark_quality`
     （快照 as-of、陈旧自然日数、T+1 生效日、漂移/回退原因）。官方快照默认最多
     陈旧 30 个自然日，超限回退重构。写统计
     前推进到回测结束日；回测另存 `execution_stats.json`，归因另存
     `attribution_execution_stats.json`。
   - 批量 bundle 的三份产物先写 sibling 临时文件，再创建
     `<weights stem>.bundle.in_progress`；旧 manifest 失效后依次原子替换 sidecar、
     对应统计、weights，manifest v2 最后原子发布并绑定三者 SHA-256，成功后删除标记。
     正式替换全程持权重绝对路径对应的跨进程独占锁；回测/归因从校验到读完
     weights+sidecar 全程持同路径共享锁，避免混读与双 publisher 交错。读取端对
     发布中、缺失或错配的批量 bundle fail-closed；manifest v1 保持兼容。完全没有
     sidecar/manifest/marker 的外部权重仅告警运行，只要存在其中任一配套文件或标记，
     就必须满足相应 bundle 契约。
   - 项目级 RunManifest 覆盖 `run/optimize/backtest/attribute` 四个 CLI。独立
     `backtest/attribute` 没有来源 YAML 时，以规范化 CLI 参数作为有效配置身份，
     并绑定权重、已有 batch bundle 配套文件和全部输出 SHA-256；失败同样发布
     不可覆盖清单。因为没有配置绑定的数据锁，清单明确记录
     `data_lock.verified=false`，不伪装为已验证。Python API 的纯内存调用不自动落盘。
3. **风险覆盖保护**：未覆盖股票只卖不买；指数基准覆盖率低于阈值则跳过该期并告警，
   不生成新权重行，旧目标保持激活并在当日正常尝试，不中断整段回测。
4. **流动性为软惩罚（非硬约束）**：通过 `turnover_penalty` + 个股冲击成本向量（基于 ADV）软性压制换手，未实现 ADV 参与率硬约束。
5. **求解器**：优先 CLARABEL，失败降级 SCS 兜底。两个优化器共用
   `optimizer/_common.py`（交易状态掩码、个股上限、换手项、求解降级、结果基类），
   只在目标函数与基准相关约束上分叉，避免执行语义修正只改了一侧。
6. **Alpha 可信度前置校验**（`pipeline/universe.get_alpha_for_date`）：按 `<= 调仓日`
   取 as-of 截面（面板 `date` 语义为信号可得日，故不构成前视），并做三件事——
   文件型 Alpha 默认最多陈旧 15 个自然日，超 `alpha.max_staleness_days` 跳过该期；
   常量、全零或零方差截面同样跳过；其余有效截面在优化域内做 z-score。标准化是
   必需项而非润色：α 与风险/成本系数量纲耦合，同一因子排序乘 100 倍
   就能把分散组合压成单票全仓。面板整体晚于回测区间直接报错，零覆盖期跳过；
   质量统计随 batch bundle 持久化并由 manifest 绑定。
7. **风险面板按需加载**：`CNE6RiskModel(query_dates=...)` 只读回测真正 as-of 命中的
   调仓日截面并预先分区。完整暴露面板约 700 万行 × 50 列，整表常驻峰值内存
   约 3.7GB；按需加载在 155 期回测下降到约 1.5GB，单期查询由 ~4.8ms 降到 ~1.4ms。
   不传 `query_dates` 时退回整表加载（行为与优化前一致）。
8. **收益归因是独立分析层，不反向影响优化**：归因复用成交账本，把目标权重重放为
   逐日实际权重和真实组合日收益，再结合 ClickHouse `test_barra_cne6_gao` 对应 S/L
   的 `factor_return_* / specific_return_*` 事后拆解超额收益。VWAP 时点差、费用和
   滑点归入独立 `execution_effect`，归因累计与成交账本 NAV 严格闭合。方法与残差
   自检的已知局限见
   [method.md §9](method.md#9-收益归因return-attribution)。
9. **精确重放边界**：从调仓周期中途裁剪权重并以全现金启动，无法恢复此前股数和
   pending/filled 状态；当前接口不支持载入完整 checkpoint，精确重放必须从策略起点开始。
