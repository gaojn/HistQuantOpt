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

### 2.2 Barra CNE6 风格因子（16 个）

来自 ClickHouse `cne6_risk`，定义见 `hqopt/risk/cne6_risk.py::STYLE_FACTORS`：

Size, MidCap, Beta, Momentum, ResidualVolatility, LongTermReversal, Liquidity,
Value, EarningsYield, Growth, Profitability, InvestmentQuality, EarningsQuality,
EarningsVariability, Leverage, DividendYield。

风格暴露用于 `style_active_bound` 约束（支持按因子分别设定），因子协方差/特质风险
用于 `risk_aversion` 真跟踪误差目标项。

### 2.3 行业因子

参考 CITIC 一级行业分类，共 30 个行业虚拟变量（dummy variable）。  
每只股票属于且仅属于一个行业，行业矩阵满足：$\sum_k B_{ik}^{ind} = 1$。

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
| $K$       | —               | 因子数量 = 16（风格）+ Country + 30（行业）= 47 |

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
| `LIMIT_UP`    | 涨停（无法买入）   | $w_i \leq w_{0,i}$（只能减仓或持有）                   |
| `LIMIT_DOWN`  | 跌停（无法卖出）   | $w_i \geq w_{0,i}$（只能加仓或持有，实际通常视为 SUSPENDED） |
| `NEW_LISTING` | 上市首日/次新股  | $w_i = 0$（禁止持仓，规避炒作风险）                        |

### 5.2 目标权重与真实成交账本

每个调仓日先把上一目标从 T+1 起逐日送入成交账本，再以账本中的**实际持仓**作为
本期 $w_{prev}$ 优化。优化器输出的是新目标权重，不是成交后的持仓。

**停牌冻结口径**：
- 停牌且有持仓 → $w_i = w_{\text{prev},i}$（冻结），不产生卖单，不计换手，不释放现金；
- 停牌且无持仓 → $w_i = 0$，维持原状。

真实成交按 T+1 VWAP 先卖后买。涨停买单、跌停卖单、停牌双向订单不会被丢弃；
账本在后续交易日按“当前实际持仓 → 最新目标”重新计算差额并继续尝试。若下一次调仓
产生新目标，则以新目标替换旧的未完成目标。停牌期间用最近有效价格估值，NAV 不断裂。

**掉池/次新/ST 持仓票（sell_only 口径）**：`filter_universe` 将掉出候选池但当日有行情且
有持仓的股票携带入优化域（`MarketSnapshot.sell_only=True`），优化器施加 $w_i \le w_{\text{prev},i}$，
允许正常减仓并计换手，禁止加仓。实际持仓中缺少当日行情的票（退市/长期停牌滞留）
不进优化域，按场外滞留资产处理并告警，其余部分正常优化——账本只按真实现金成交，
不会把滞留市值误当现金再分配。优化目标提交前会清除 <1e-6 的求解器数值粉尘并重归一，
避免账本买入微量仓位（此类仓位退市后即成滞留资产）。

---

## 6. 模块结构

```
hqopt/
├── data/
│   ├── real_adapter.py     # parquet 面板 → MarketSnapshot
│   ├── benchmark.py        # 指数成分权重（默认官方权重，缺则分级靠档重构）
│   ├── index_close.py      # 官方指数收盘价加载（回测基准）
│   ├── clickhouse_db.py    # ClickHouse 只读连接层
│   └── generator.py        # 合成数据/快照构件
├── io/
│   ├── data_panel.py       # load_panel 主入口
│   └── schema.py           # 行情字段定义
├── risk/
│   ├── cne6_risk.py        # CNE6 因子风险模型（暴露 X / 协方差 F / 特质 Δ）
│   └── attribution_data.py # 因子收益 f(t) / 个股特质收益 u(t) 加载器（收益归因用）
├── optimizer/
│   ├── alpha_max.py        # 量化选股 QP 优化器
│   └── index_enhance.py    # 指数增强 QP 优化器
├── backtest/
│   ├── engine.py           # 真实执行回测（T+1 VWAP/涨跌停/成本）+ 绩效指标
│   └── report.py           # Plotly HTML 报告
├── analysis/
│   ├── attribution.py      # 收益归因（风格/行业/Country/特质分解，Carino多期链接）
│   └── run.py              # 权重→归因，供 hqopt attribute 复用
└── pipeline/
    ├── batch_optimize.py   # 逐期批量优化（两策略）
    └── universe.py         # 候选池过滤 / 成本向量 / 合成 alpha
```

---

## 7. 数据流

```
行情面板 load_panel（data/cache）          CNE6 面板（data/barra_cne6[_L]）
    │                                            │
    ▼                                            ▼
RealMarketAdapter.build_snapshot         CNE6RiskModel.at(date)
  → MarketSnapshot                         → 暴露 X / 协方差 F / 特质 Δ / style_loading
  （tickers/行业/ADV/状态/市值/成分）              │
    │            ┌───────── Alpha 因子（alphas/*.parquet）
    ▼            ▼          │
  filter_universe(prev_holdings)  ──►│   # 候选池 ∪ 上期持仓（carry=sell_only）
    │                                ▼
    └────────────────►  optimizer（alpha_max / index_enhance, cvxpy）
                                     │  约束：预算/上限(frozen豁免)/行业/风格/换手/涨跌停/冻结/sell_only
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
   - 每期优化前先推进实际成交账本，`w_prev` 使用实际成交持仓，不使用上一期目标权重。
   - 停牌有持仓 → 冻结 `w=w_prev`（不计换手）；停牌无持仓 → `w=0`。
   - 掉池/次新/ST 有持仓 → sell_only `w≤w_prev`（卖出计换手）；无持仓 → `w=0`。
   - 涨停 `w≤w_prev`、跌停 `w≥w_prev`；alpha 对 frozen/zero/sell_only 均置 0。
   - 不可成交订单逐日重试；新目标替换旧的未完成目标；现金作为权重缺口原样反馈给优化器。
3. **风险覆盖保护**：未覆盖股票只卖不买；指数基准覆盖率低于阈值则跳过该期并告警，不中断整段回测。
4. **流动性为软惩罚（非硬约束）**：通过 `turnover_penalty` + 个股冲击成本向量（基于 ADV）软性压制换手，未实现 ADV 参与率硬约束。
5. **求解器**：优先 CLARABEL，失败降级 SCS 兜底。
6. **收益归因是独立分析层，不反向影响优化**：归因复用成交账本，把目标权重重放为
   逐日实际权重，再结合 ClickHouse `cne6_risk.factor_return/specific_return` 事后拆解
   超额收益。方法与残差自检的已知局限见
   [method.md §9](method.md#9-收益归因return-attribution)。
