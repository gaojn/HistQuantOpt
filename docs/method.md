# 组合优化方法

> 本项目两种优化策略的数学模型、约束、默认参数与用法。
> 对应代码：[`optimizer/alpha_max.py`](../hqopt/optimizer/alpha_max.py)（量化多头）、
> [`optimizer/index_enhance.py`](../hqopt/optimizer/index_enhance.py)（指数增强）。

---

## 1. 两种策略

| 维度 | 量化多头 `alpha_max` | 指数增强 `index_enhance` |
|---|---|---|
| 目标 | 绝对 alpha 收益最大 | 跑赢基准、控跟踪误差 |
| 候选池 | 全市场（剔北交所+ST）~5000 | 同左（不限宽基，靠成分下限托底） |
| 风险度量 | 组合分散度 / 因子风险 | 相对基准的主动风险（TE） |
| 行业约束 | 绝对上限 | 相对基准偏离 ±X% |
| 风格约束 | 绝对暴露 `style_bound` | 主动暴露 `style_active_bound` |
| 成分股约束 | 可选下限（默认 40%） | 目标指数成分 ≥ 下限（默认 80%） |

一句话：**量化多头偏绝对约束，指数增强偏相对基准约束。**

> 注：两策略候选池其实**相同**（全市场剔北交所+ST，~5000 只）。指数增强不另限宽基，
> 而是靠 `min_constituent_ratio`（默认 80%）把目标指数成分权重托起、剩余 ≤20% 配全市场
> ——属「**泛化指增**」（允许从非成分挖超额）。

---

## 2. 数学模型

统一目标函数（最大化）：

$$
\max_w \quad w^\top \alpha \;-\; \underbrace{R(w)}_{\text{风险项}} \;-\; \underbrace{\lambda_c \sum_i c_i\,|w_i - w_{\text{prev},i}|}_{\text{换手成本项}}
$$

- $w\in\mathbb{R}^N$：组合权重；$\alpha$：预期收益信号（推荐截面 z-score）
- 换手项：$c_i$ 为个股成本权重（默认按 $\sigma_i/\sqrt{ADV_i}$ 的冲击成本代理），软约束
- $w_{prev}$：成交账本在本调仓日收盘的实际股票权重；未成交现金保留为权重缺口，
  不会用上一期目标权重代替，也不会把股票权重重新归一化到 1

**风险项 $R(w)$ 有两种形态**，由是否配置 `risk_aversion` 决定：

| 策略 | 默认（L2 代理） | 启用 CNE6 风险模型 |
|---|---|---|
| 量化多头 | $\gamma\,\lVert w\rVert_2^2$（分散惩罚） | $\lambda\,(w^\top XFX^\top w + \delta^\top w^2)$ |
| 指数增强 | $\gamma\,\lVert w-w_{bm}\rVert_2^2$（偏离基准） | $\lambda\,(a^\top XFX^\top a + \delta^\top a^2),\ a=w-w_{bm}$ |

其中 $X$=因子暴露、$F$=因子协方差、$\delta$=特质方差，来自 CNE6 风险面板
（S=51 因子：20 风格 + Country + 30 行业；L=47 因子：16 风格 + Country + 30 行业，
详见 [design.md](design.md) 与
[`risk/cne6_risk.py`](../hqopt/risk/cne6_risk.py)）。
L2 形态无需协方差、简单稳健；CNE6 形态刻画真实因子相关性与个股特质风险差异。

> **指数增强的基准权重 $w_{bm}$** 默认使用 `official_drift`：取不晚于 T 日的
> 最近官方月度/调样快照 S，并用后复权收盘价直接漂移
> $w_i(T)\propto w_i(S)P_i^{adj}(T)/P_i^{adj}(S)$。快照日直接使用官方原值；
> 停牌/缺当日价格只向前取最近有效价，严禁使用下一快照插值。默认只允许快照陈旧
> 30 个自然日（`benchmark_max_snapshot_age_days`）；超过阈值、官方未覆盖、价格
> 不完整或每日成分名单与快照不一致，均告警并回退 free_mv 分级靠档重构，同时写入
> `batch_execution_stats.json/benchmark_quality`。旧冻结口径用 `official_frozen`，纯重构
> 用 `reconstruct`；兼容值 `official` 等同 `official_drift`。

---

## 3. 约束体系

两策略约束结构对称，差别在「绝对」还是「相对基准」：

| # | 约束 | 量化多头 | 指数增强 |
|---|---|---|---|
| 1 | 预算 | $\sum w_i = 1$ | 同 |
| 2 | 个股区间 | $0\le w_i\le W_{\max}$ | 同 |
| 2b | 单票主动偏离（可选） | — | $\lvert w_i - w_{bm,i}\rvert \le \delta$ |
| 3 | 成分股下限 | $\sum_{i\in C} w_i \ge R_{\min}$（可选） | $\sum_{i\in C_{\text{index}}} w_i \ge R_{\min}$ |
| 4 | 行业 | $\sum_{i\in k} w_i \le I_{\max}$ | $\lvert\sum_{i\in k}(w_i-w_{bm,i})\rvert \le I_{\text{act}}$ |
| 5 | 风格 | $\lvert B_k^\top w\rvert \le S_{\max,k}$ | $\lvert B_k^\top(w-w_{bm})\rvert \le S_{\text{act},k}$ |
| 6 | 换手（硬上限，可选） | $\lVert w-w_{\text{prev}}\rVert_1 \le T_{\max}+c$，$c=\max(0,1-\sum w_{prev})$ | 同 |
| 7 | 交易状态 | 见下 | 同 |

> 风格上限 $S$ 支持标量（统一）或 dict（按因子名分别约束，可带 `default` 兜底）。
> 因子已 z-score 标准化，$S=1$ 即组合在该因子上加权暴露不超过 ±1σ。

**约束 7 — A 股交易状态**（强制，贴近真实成交，停牌冻结口径）：

三类互斥掩码，优先级：frozen > zero > sell_only。

| 条件 | 掩码 | 数学约束 | 说明 |
|---|---|---|---|
| 停牌 & 有持仓（$w_{\text{prev},i}>0$） | **frozen** | $w_i = w_{\text{prev},i}$ | 冻结，不计换手、不释放资金 |
| 停牌/次新/ST & 无持仓 | **zero** | $w_i = 0$ | 禁止开仓 |
| 掉出候选池的持仓票、次新/ST 有持仓 | **sell_only** | $w_i \le w_{\text{prev},i}$ | 只卖不买，卖出正常计换手与成本 |

**T 日涨跌停不进入目标约束。** T 日收盘只能知道 T 日状态，不能据此推断 T+1
能否成交；因此优化器可以对 T 日涨停股生成买入目标，也可以对 T 日跌停股生成卖出
目标。实际能否成交只使用执行日（T+1/T+2/T+3）的行情判断。

**优化域 = 当期候选池 ∪ 上期持仓（有当日行情的票）**（`filter_universe` 的
`prev_holdings` 参数控制）。已有持仓若当日 panel 无行情行，则不进入优化域，但仍
按最近有效价格保留在真实成交账本中：股数不变、不释放现金，也不把其权重清零或将
其余目标重新归一化；系统同时输出场外滞留持仓告警。优化器只为当日可建模股票生成
目标，真实组合则继续包含该滞留持仓。

冻结票上界豁免：$W_{\max,i}^{\text{frozen}} = \max(W_{\max}, w_{\text{prev},i})$，避免漂移后持仓超上限导致 infeasible。
`active_weight_upper`（指增专属）对冻结票同样豁免，仅对非冻结票施加。

alpha 置零范围：frozen ∪ zero ∪ sell_only（避免干扰目标函数方向）。
冻结票因 $w_i = w_{\text{prev},i}$ 等式约束，$|w_i - w_{\text{prev},i}| = 0$，自动不消耗换手预算。

**风险覆盖保护**：CNE6 原始暴露和特质方差均完整才记为覆盖。未覆盖股票并入
`sell_only`，禁止新开仓、已有持仓只能减仓；指数增强按基准权重计算覆盖率，低于
`min_risk_coverage`（默认 0.5）跳过该期优化并告警（不中断整段回测），避免缺失值填 0 造成假中性。

状态判定见 [`RealMarketAdapter._compute_status`](../hqopt/data/real_adapter.py)；停牌优先级
高于 ST/次新，避免停牌持仓丢失冻结语义。次新阈值（上市不足 `new_listing_days`
个自然日）可通过 `universe.new_listing_days` 配置，默认 120，见操作指南 §3.2。

> ⚠️ 行业上限不可行陷阱：若所有行业上限之和 < 100% 则无解
> （30 个行业 × $I_{\max}$ 须 ≥ 1）。

---

## 4. 团队默认参数

完整模板见 [`configs/alpha_max_default.yaml`](../configs/alpha_max_default.yaml) 与
[`configs/index_enhance_default.yaml`](../configs/index_enhance_default.yaml)，以下为对照：

> **注**：下表数值来自 YAML 默认配置文件；Python 类（`IndexEnhanceConfig`/`AlphaMaxConfig`）的代码默认值以各类 docstring 为准，两者可能不同（如 `weight_upper`：YAML=0.01，Python 类默认=0.02/0.05）。

| 参数 | 量化多头 | 指数增强 | 含义 |
|---|---:|---:|---|
| `weight_upper` | 0.01 | 0.01 | 单票绝对上限，两策略目前一致 |
| `active_weight_upper` | — | 0.01 | 指增专属：单票主动偏离硬上限 ±1%，`null`=不约束 |
| `min_constituent_ratio` | 0.40 | 0.80 | 指增须大部分留在目标指数内 |
| 行业 | `industry_upper: 0.20` | `industry_active_bound: 0.05` | 绝对集中度 vs 相对偏离 |
| 风格默认上限 | `style_bound.default: 0.50` | `style_active_bound.default: 0.60` | 指增更怕风格漂移 |
| `Size` / `Beta` | 0.20 / 0.20 | 0.20 / 0.20 | 控市值、市场暴露 |
| `Momentum` | 0.30 | 0.20 | 指增显著更严，防动量偏离 |
| `Liquidity` / `ResidualVolatility` | 0.20 / 0.20 | 0.20 / 0.25 | 控流动性、高波动偏离 |
| `risk_aversion` | `null` | 10.0 | CNE6 风险项强度（量化多头默认关闭，退回 L2） |
| 风险代理 | `diversification_penalty: 0.05` | `tracking_penalty: 10.0` | 控集中度 vs 控偏离基准 |
| `max_turnover` / `turnover_penalty` | 0.40 / 0.01 | 0.40 / 0.01 | 一致 |

**设计要点**：不照搬「全因子中性」。保留 alpha 表达空间，只压住最关键的公共风格风险，
其余交给 `risk_aversion` 对应的风险项吸收。成分股下限让量化多头仍保留主流敞口
（流动性/容量），让指数增强仍「是指数增强」而非带基准约束的泛化选股。

---

## 5. Barra 风格因子约束建议（按 Universe）

原则：**先统一因子优先级，再按 universe 调强弱**，不为每个 universe 写死一套参数。

**因子分层**（L 为16个核心风格；S 额外包含4个快策略风格）：

| 层 | 因子 | 处理 |
|---|---|---|
| A 基础护栏 | `Size` `Liquidity` `ResidualVolatility` `Beta` | 几乎总要约束（最易把组合做歪） |
| B 多数约束 | `Momentum` | 中等约束（防风格切换回撤、推高换手） |
| C 视 alpha | `Value` `Growth` `EarningsYield` `Profitability`；S专属 `AnalystSentiment` `IndustryMomentum` `Seasonality` `ShortTermReversal` | 是收益风格本身，收太紧会削收益 |
| D 默认监控 | `MidCap` `LongTermReversal` `InvestmentQuality` `EarningsQuality` `EarningsVariability` `Leverage` `DividendYield` | 不建议先硬约束（多与主因子重复） |

**按 Universe 速查**：

| Universe | 必管 | 建议管 | 慎重管 |
|---|---|---|---|
| HS300/大盘 | `Size` `Liquidity` `Beta` `ResidualVolatility` | `Momentum` | `Value` `Growth` |
| ZZ500/中盘 | `Liquidity` `Beta` `ResidualVolatility` | `Size` `Momentum` | `Value` `Growth` |
| ZZ1000/小盘 | `Liquidity` `ResidualVolatility` | `Beta` `Momentum` | `Size` |
| 全市场 | `Size` `Liquidity` `ResidualVolatility` | `Beta` `Momentum` | `Value` `Growth` |
| 主题/行业 | `Liquidity` `Beta` `ResidualVolatility` | `Momentum` | 主题自带因子 |

要点：小盘/全市场重点防 `Size/Liquidity/Volatility` **风格下沉**（避免「看似 alpha，实为风格补偿」）；
大盘/指增重点防**风格漂移**；主题 universe 只约束「跑偏」，不抹平主题本身
（红利别收紧 `DividendYield`、成长别收紧 `Growth`）。落地时量化多头用 `style_bound`（绝对），
指数增强用 `style_active_bound`（主动）。

---

## 6. 求解与性能

- **类型**：凸二次规划（QP），求解器 **CLARABEL**（`max_iter=500`）；两个优化器（`index_enhance`/`alpha_max`）均在 CLARABEL 失败时降级 **SCS**（`max_iters=10000`）兜底
- **失败处理**：快照、风险覆盖或求解失败导致没有新权重行时，不提交新目标；该日不
  构成有效调仓，旧目标保持激活并正常尝试
- **数值稳定**：求解后 `clip(0)`，停牌无持仓精确置零、停牌持仓精确恢复
  $w_{\text{prev}}$；不再二次归一化，避免放大 sell_only、个股上限等硬约束，
  预算数值超额只向下扣减非冻结权重。单票粉尘只有在累计也不超过 $10^{-6}$ 时
  才清零，否则原样保留，避免累计破坏下限类硬约束；微小预算缺口保留为现金
- **规模**：全市场 ~5000 只（剔北交所+ST）约 1s/期
- **首个调仓日**：组合仍为全现金，组合与基准日收益都固定为 0，两者统一从首个调仓日收盘起算，避免首日基准收益一次性侵蚀超额。
- **延期成交**：每个目标只在 T+1、T+2、T+3 三个交易日尝试；每经过一个交易日
  都计一次（包括停牌、涨跌停或 VWAP 缺失）。T+3 完成最后一次尝试后，剩余订单
  标记 `EXPIRED`，之后不再交易，直到下一调仓目标重新激活
- **risk_aversion 置 0**：显式设 `risk_aversion: 0.0` 将完全关闭风险惩罚项（factor_risk/specific_risk 均不加入目标函数），等价于纯 alpha 最大化；与 `risk_aversion: null`（退回 L2 兜底）行为不同。

### 6.1 T+1 成交状态机

T 日收盘提交目标，最早 T+1 成交。股票级状态为 `FROZEN / PENDING_SELL /
PENDING_BUY / FILLED / EXPIRED`：

- T 日停牌股在整个目标生命周期内为 `FROZEN`：有持仓保持股数不变，无持仓不下单；
  即使 T+1 复牌也不交易。
- 每个执行日只处理仍 pending 的股票，并按当日最新 NAV 和原始目标权重重算差额；
  `FILLED` 和 `FROZEN` 股票不因价格漂移再次交易。普通 pending 可随最新差额变更
  买卖方向；制度性 `sell_only` 只能卖，差额转正时也不会反向买入。
- 先卖后买。涨停只阻断买入，跌停只阻断卖出；停牌或无有效 VWAP 阻断双向。
  买入资金只来自真实现金与当日已实现卖出。
- 现金不足时，对当日可交易的 pending 买单同比例部分成交，剩余继续 pending；
  没有卖出且没有现金时不做买入。
- T+3 后残余订单过期，已成交股数和部分成交股数均保留；累计过期笔数与未成交金额
  写入执行统计 JSON。
- 只有成功生成并提交新目标的日期才是有效调仓日：旧目标及其残余 pending 状态直接
  取消，当日不再为旧目标增加尝试次数，只按实际股数和最新价格估值；收盘提交新目标
  并重置股票状态，下一交易日才作为新目标的 T+1 开始三次尝试计数。
- 若候选调仓日因快照、风险覆盖或求解失败而没有新权重行，则不构成有效调仓；旧目标
  保持激活并在该日正常尝试。

批量优化把权重、sell_only sidecar 和批量执行统计作为一个 bundle 发布。常规
`weights.parquet` 使用 `batch_execution_stats.json`；自定义权重文件使用
`<weights stem>.batch_execution_stats.json`，使同目录多个 bundle 的统计互不覆盖。
全部产物先写入同目录临时文件，随后创建 `<weights stem>.bundle.in_progress`
（如 `weights.bundle.in_progress`），使旧 manifest 失效，再依次原子替换 sidecar、
对应统计和权重；最后原子发布 `<weights stem>.sell_only.manifest.json`。manifest v2
的 `schema_version=2`，通过 `weights_file/weights_sha256`、
`sell_only_file/sell_only_sha256` 和
`batch_execution_stats_file/batch_execution_stats_sha256` 绑定三份产物。统计文件除
成交状态外，还持久化 `optimization`（候选/成功/失败期数）和 `alpha_quality`
（标准化、陈旧阈值、最大陈旧度、跳过/零方差期数、逐期 as-of 日期）；发布成功后
才删除 in-progress 标记。正式替换的整个区间由该权重绝对路径派生的跨进程独占锁
保护；回测与归因从 manifest/sidecar 校验到权重与 sidecar 读取完成，全程持同路径
的共享锁，避免读取新旧混合 bundle，也避免两个 publisher 交错替换。
读取端看到 in-progress 标记，或发现批量 bundle 的 manifest 缺失、清单不完整、
哈希及结构错配时，均 fail-closed 拒绝读取。已有 manifest v1 继续按旧双文件契约
兼容读取。

外部权重若完全没有 sidecar、manifest 和 in-progress 标记，仍可在明确告警后仅执行
目标权重，不凭空推断制度性只卖限制；一旦存在任一配套文件或标记，就按 bundle
处理并要求契约完整有效。

执行优先级为：**股数冻结 / 已成交锁定 > pending 股票逼近目标 > 全组合精确目标**。

**`RealisticBacktester.run` 的内部分工**（`backtest/engine.py`）：

| 函数 | 职责 |
|---|---|
| `_normalize_weight_inputs` | 统一权重索引为 Timestamp、绑定 sell-only 矩阵、校验调仓日在交易日历内 |
| `_align_market_frames` | 各行情宽表裁剪对齐到「首个调仓日起」，产出只读 `_MarketFrames` |
| `_replay_days` | 逐日：撤旧目标 → 当日成交 → 估值 → 收盘提交新目标，产出 `_ReplayRecords` |
| `_resolve_benchmark_returns` | 对齐基准日收益（含首日置零，见下方口径说明） |
| `_stale_holding_value` | 末日靠 ffill 陈旧价估值的滞留持仓数与市值 |
| `_build_exec_stats` | 汇总可审计的执行统计 JSON |

`_MarketFrames` 同时持有 `adj_close_raw`（当日真实价，退市/缺行为 NaN）与
`adj_close_marked`（其 ffill 版本）：**估值走 marked、成交价走 raw**。这条分界是
退市估值不假摔、同时又不能假装能成交的关键——不要在新增代码里把两者混用。

**infeasible 常见原因**：成分股下限太高、行业约束太紧（某行业基准权重为 0 时易冲突）、
首期建仓却设了 `max_turnover`（应传 `None`，pipeline 已自动处理首期）。

**绩效指标口径**（`engine.calc_metrics`）：
- **首个调仓日基准收益强制置零**：该日组合仍是全现金（目标收盘后才提交，最早 T+1 成交），
  `port_ret` 必为 0；若基准照常计息，整段超额会被首日基准收益一次性侵蚀
  （首日基准 +1%、6 年回测约低估 0.16pct 年化超额）。组合与基准统一从首个调仓日
  **收盘**起算，`nav[0] = bm_nav[0] = excess_nav[0] = 1.0`。
- 年化超额收益采用**几何口径**：`(1+total_port)/(1+total_bm)-1` 后再年化，与 report 年度表、全期行一致。
- 月度超额收益表采用算术差（月组合收益 − 月基准收益），与年度/全期口径不同，报告中已注明。
- **Calmar 统一累计口径**：年度行 = 当年累计收益 / 当年最大回撤；全期行 =
  全期累计收益 / 全期最大回撤；超额 Calmar 同理使用全期累计几何超额。分子分母
  始终同期且不做年化外推。报告全期行的“收益”列仍显示年化收益（标 `*`），但
  Calmar 不使用该年化值。该口径是项目约定，跨不同长度区间比较时需谨慎。
- **IR 两处实现必须同口径**：`report._annual_metrics` 与 `engine.calc_metrics` 是两套
  独立实现，分子都必须是几何年化超额（不能用算术 `日均超额×252`，实测两者差 3%~59%）。
  该不变量由 `tests/test_metric_consistency.py` 锁定，同时覆盖年化收益/波动/Sharpe/
  最大回撤/TE/超额回撤/年化超额七项。
- 跟踪误差（TE）= 超额日收益标准差 × √252（算术，不改口径）。
- 信息比率 IR = 年化超额收益（几何） / 跟踪误差。

---

## 7. 参数调优速查

| 现象 | 调整方向 |
|---|---|
| 持仓太少(<30) | ↑ `diversification_penalty` 或 ↓ `weight_upper` |
| 持仓太多(>200) | ↓ `diversification_penalty` 或 ↑ `weight_upper` |
| 行业过度集中 | ↓ `industry_upper` / `industry_active_bound` |
| 组合偏小盘 | 收紧 `Size`（→0.20~0.25） |
| 波动高于预期 | 收紧 `Beta`/`ResidualVolatility`，↑ `risk_aversion` |
| 风格保守、超额不足 | 先放松 `Momentum/Value/Growth`，最后才放 `Size/Beta` |
| 换手过高 | 先加 `turnover_penalty`，再收 `max_turnover` |
| 跟踪误差太大（指增） | ↑ `tracking_penalty` → ↓ `industry_active_bound` → 收风格 → ↑ 成分股下限 |

---

## 8. API 用法

逐期批量优化通常直接走 [`pipeline/batch_optimize.py`](../hqopt/pipeline/batch_optimize.py)
（YAML 驱动，见 [操作指南.md](操作指南.md)）。`run_batch_optimize` 内部分三段，
需要复用其中某一段（如只做输入装配再自定义循环）可直接调用私有函数：

| 阶段 | 函数 | 职责 |
|---|---|---|
| 一 | `_prepare_inputs(config, panel, alpha_df)` | 解析配置、加载行情/Alpha、构造账本与优化器/风险模型，产出只读 `_BatchInputs` |
| 二 | `_run_periods(inputs)` | 逐期推进成交账本 → 构建优化域 → 取 Alpha → 求解 → 落账，产出 `_PeriodOutcome` |
| 三 | `_publish_outputs(inputs, outcome)` | 汇总权重矩阵，原子发布 bundle（weights + sidecar + 成交统计 + 清单） |

账本推进的游标状态收敛在 `_ExecutionWalker`：`open_signal_day` 补齐调仓日之前的
成交、信号日只估值；未发布新目标的调仓日由 `replay_signal_day` 恢复旧目标的当日
尝试；`finish` 在区间末尾补记最后一批 T+1/T+2/T+3 成交或过期。

单期直接调优化器：

**量化多头**

```python
from datetime import date
from hqopt.io.data_panel import load_panel
from hqopt import (
    RealMarketAdapter, CNE6RiskModel, AlphaMaxConfig, AlphaMaxOptimizer,
)

target = date(2026, 5, 21)
panel = load_panel(date(2026, 4, 1), target)             # 默认列已覆盖所需字段
snap = RealMarketAdapter().build_snapshot_from_panel(
    panel=panel, target_date=target, index="all", portfolio_value=1e8,
)
risk_snap = CNE6RiskModel().at(target, snap.tickers)      # None 则退回 L2 惩罚
cfg = AlphaMaxConfig(
    weight_upper=0.01, industry_upper=0.20, min_constituent_ratio=0.40,
    style_bound={"default": 0.50, "Size": 0.20, "Beta": 0.20},
    max_turnover=0.40, turnover_penalty=0.01, risk_aversion=None,
)
res = AlphaMaxOptimizer(cfg).optimize(
    alpha=alpha_vec, snapshot=snap,
    style_loading=risk_snap.style_loading(),
    prev_weight=None,            # 首期建仓
    risk_snapshot=risk_snap,
)
print(res.summary()); print(res.top_holdings(10))
```

**指数增强**

```python
from hqopt import IndexBenchmarkWeights, IndexEnhanceConfig, IndexEnhanceOptimizer
from hqopt.pipeline.universe import filter_universe

snap = RealMarketAdapter().build_snapshot_from_panel(panel, target, index="zz1000")
snap = filter_universe(snap, panel, target)               # 剔北交所+ST
bm = IndexBenchmarkWeights(index="zz1000", panel=panel)
bm.precompute(date(2026, 4, 1), target, panel=panel)
bm_w = bm.get_weights(target, tickers=snap.tickers).values
risk_snap = CNE6RiskModel().at(target, snap.tickers)
cfg = IndexEnhanceConfig(
    weight_upper=0.01, min_constituent_ratio=0.80, industry_active_bound=0.05,
    style_active_bound={"default": 0.60, "Size": 0.20, "Momentum": 0.20},
    tracking_penalty=10.0, max_turnover=0.40, risk_aversion=10.0,
    active_weight_upper=0.01,
)
res = IndexEnhanceOptimizer(cfg).optimize(
    alpha=alpha_vec, snapshot=snap, benchmark_weight=bm_w,
    style_loading=risk_snap.style_loading(), prev_weight=prev_w, risk_snapshot=risk_snap,
)
print(res.industry_active_weights())        # 行业相对基准偏离
print(res.style_active_exposure(risk_snap.style_loading()))
```

`alpha_vec` 需与 `snap.tickers` 对齐、缺失填 0，且**必须是截面 z-score**——
目标函数里 α 与风险/成本系数量纲耦合，同一因子排序乘 100 倍即可把 γ=0.05 下的
20 只分散组合压成 1 只全仓，`risk_aversion` / `turnover_penalty` 的标定随之失效。
直接调优化器时需自行标准化；走 pipeline 则由 `alpha.standardize`（默认 `true`）
统一处理，详见操作指南 §3.5。

---

## 9. 收益归因（Return Attribution）

> 对应代码：[`analysis/attribution.py`](../hqopt/analysis/attribution.py)、
> [`risk/attribution_data.py`](../hqopt/risk/attribution_data.py)。
> CLI：`hqopt attribute`（见 [操作指南.md](操作指南.md)）。

**动机**：一条超额净值曲线好看，不代表 alpha 干净——可能是优化器偷偷吃了某个
风格 beta（如小盘、低波）。归因把每期主动收益拆成风格 / 行业 / Country / 个股
特质（选股）贡献，回答"钱到底从哪来"。

**方法**：每个调仓期 $(T_i, T_{i+1}]$ 使用信号日 $T_i$ 的风险暴露快照；目标权重
先经共享成交账本按 T+1 VWAP、涨跌停、停牌和费用逐日重放，得到实际权重
$w_{p,t}$。主动权重 $w_{active,t}=w_{p,t}-w_{bm}$ 与主动暴露
$X_{active,t}=X^\top w_{active,t}$ 因实际成交和价格漂移逐日更新：

$$
\text{主动收益}(t) \approx \underbrace{X_{active,t}^\top f(t)}_{\text{风格+行业+Country}} + \underbrace{w_{active,t}^\top u(t)}_{\text{选股（特质）}}
$$

$f(t)$（因子收益）、$u(t)$（个股特质收益）取自 ClickHouse
`test_barra_cne6_gao.factor_return_S/L` / `specific_return_S/L`，与暴露 $X$
（`test_barra_cne6_gao.factor_exposure`）**同源且同模型**，保证残差
自检有意义。多期用 Carino(1999) 平滑系数链接，保证 $\Sigma$ 各归因项累计 =
真实几何累计主动收益。

**残差自检**：每日「主动收益 − (风格+行业+Country+特质)」理论上应 ≈0；非零
主要来自**覆盖缺口**——`test_barra_cne6_gao` 只覆盖其估计域（`univ_flag==1`，剔除次新/
极小市值等），组合或基准里在域外的持仓收益全部漏进残差。`daily["coverage_pct"]`
给出每期"被模型覆盖的主动权重占比"可供判断，但覆盖率与残差占比非线性关系
（未覆盖的常是高波动票，权重小也可能贡献不成比例的残差）。**残差占比高的
期间，归因结论需谨慎**；合成数据（完全覆盖）下残差严格为 0，证明分解算式
本身正确，见 `tests/test_attribution.py`。

**已知局限**：风险模型暴露快照仍按持有期冻结，调仓越不频繁，暴露与
`f(t)/u(t)` 实际估计所用的当日暴露之间的近似误差可能越大；成交发生在日内 VWAP，
而因子收益是收盘到收盘口径，因此成交日仍可能出现时点残差；`t统计` 为简单
`mean/std` 未做自相关调整（非 NW 稳健标准误）。此外，独立回测或归因若从调仓周期
中途裁剪权重并以全现金启动，无法还原此前的股数和订单状态。当前接口尚不支持载入
完整成交 checkpoint，因此精确重放必须从策略起点开始。

---

## 10. 已知局限

| 局限 | 说明 / 改进方向 |
|---|---|
| L2 风险代理 | 未启用 `risk_aversion` 时不含相关性结构；接 CNE6 风险面板即用真因子风险 |
| 行业按一级 | 中信 30 个一级行业；可扩展二级 |
| 无最小持仓数约束 | 仅约束单票上限；如需精确控持仓数需引入整数变量（MILP） |
| 泛化指增的非成分敞口 | 指增候选全市场、靠成分下限托底，剩余 ≤20% 非成分敞口可能引入额外跟踪偏离，靠 risk_aversion / 风格约束管住 |
