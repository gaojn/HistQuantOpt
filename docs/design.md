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

来自 ClickHouse `cne6_risk`，定义见 `portfolio_optimizer/risk/cne6_risk.py::STYLE_FACTORS`：

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

当前实现按目标权重计算双边换手；停牌若被目标卖出，会计入目标换手，
真实成交由回测引擎延期处理。

#### 流动性约束

当前实现没有 ADV 参与率硬约束，而是通过 `turnover_penalty` 与
`build_cost_vector()` 对低流动性股票施加更高换手惩罚。

---

## 5. A 股特殊状态处理

### 5.1 交易状态枚举

| 状态            | 说明         | 优化器处理                                         |
|---------------|------------|-----------------------------------------------|
| `NORMAL`      | 正常交易       | 无特殊限制                                         |
| `SUSPENDED`   | 停牌         | 目标权重 $w_i = 0$，真实回测不可成交并延期处理                  |
| `LIMIT_UP`    | 涨停（无法买入）   | $w_i \leq w_{0,i}$（只能减仓或持有）                   |
| `LIMIT_DOWN`  | 跌停（无法卖出）   | $w_i \geq w_{0,i}$（只能加仓或持有，实际通常视为 SUSPENDED） |
| `NEW_LISTING` | 上市首日/次新股  | $w_i = 0$（禁止持仓，规避炒作风险）                        |

### 5.2 停牌股票的目标权重与真实成交

当前优化器输出的是目标权重，不是已经执行后的真实持仓。停牌股票在目标组合中
可被置为 0，表达“希望卖出”的意图；真实回测中停牌不可交易，卖单会延期到
复牌且可成交时执行。停牌期间仍按行情前值填充价格估值，避免 NAV 断裂。

---

## 6. 模块结构

```
portfolio_optimizer/
├── data/
│   ├── real_adapter.py     # parquet 面板 → MarketSnapshot
│   ├── benchmark.py        # 分级靠档指数成分权重
│   ├── index_close.py      # 官方指数收盘价加载（回测基准）
│   ├── clickhouse_db.py    # ClickHouse 只读连接层
│   └── generator.py        # 合成数据/快照构件
├── io/
│   ├── data_panel.py       # load_panel 主入口
│   └── schema.py           # 行情字段定义
├── factors/
│   └── alpha_factors.py    # Alpha 预处理（去极值/标准化）
├── risk/
│   └── cne6_risk.py        # CNE6 因子风险模型（暴露 X / 协方差 F / 特质 Δ）
├── optimizer/
│   ├── alpha_max.py        # 量化选股 QP 优化器
│   └── index_enhance.py    # 指数增强 QP 优化器
├── backtest/
│   ├── engine.py           # 真实执行回测（T+1 VWAP/涨跌停/成本）+ 绩效指标
│   └── report.py           # Plotly HTML 报告
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
  filter_universe  ────────►│
    │                       ▼
    └────────────►  optimizer（alpha_max / index_enhance, cvxpy）
                            │  约束：预算/单票上限/行业/风格(CNE6)/换手/涨跌停/停牌
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
2. **交易状态约束（实现口径）**：停牌/次新在优化中约束为 `w=0`（alpha 置 0）；涨停 `w≤w_prev`、跌停 `w≥w_prev`。真实回测中涨停不可买（留现金）、跌停不可卖（进延期队列）、停牌不可交易。
   > 说明：行情数据在停牌日对价格字段（adj_close / adj_vwap）做**前值填充**（仅 volume/amount 为 0），因此停牌持仓按最近价估值（NAV 不失真），停牌卖单 `exec_p>0` 会正确进入延期队列、复牌后成交。优化阶段目标 `w=0`（意图卖出）+ 回测延期卖出，整体自洽且贴近现实。
3. **流动性为软惩罚（非硬约束）**：通过 `turnover_penalty` + 个股冲击成本向量（基于 ADV）软性压制换手，未实现 ADV 参与率硬约束。
4. **求解器**：优先 CLARABEL，失败降级 SCS 兜底。
