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
| $K$       | —               | 因子数量 = 10（风格）+ 30（行业）= 40 |

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

$$\sum_{i \notin \mathcal{S}} |w_i - w_{0,i}| \leq T_{\max}$$

其中 $\mathcal{S}$ 为停牌股票集合（排除在换手率计算之外）。

#### 流动性约束

$$|w_i - w_{0,i}| \cdot V_p \leq \rho \cdot \text{ADV}_i \quad \forall i$$

其中 $V_p$ 为组合总市值，$\rho$ 为最大市场参与率（如 20%），$\text{ADV}_i$ 为近 20 日平均成交额。

---

## 5. A 股特殊状态处理

### 5.1 交易状态枚举

| 状态            | 说明         | 优化器处理                                         |
|---------------|------------|-----------------------------------------------|
| `NORMAL`      | 正常交易       | 无特殊限制                                         |
| `SUSPENDED`   | 停牌         | 强制 $w_i = w_{0,i}$（等式约束）                      |
| `LIMIT_UP`    | 涨停（无法买入）   | $w_i \leq w_{0,i}$（只能减仓或持有）                   |
| `LIMIT_DOWN`  | 跌停（无法卖出）   | $w_i \geq w_{0,i}$（只能加仓或持有，实际通常视为 SUSPENDED） |
| `NEW_LISTING` | 上市首日/次新股  | $w_i = 0$（禁止持仓，规避炒作风险）                        |

### 5.2 停牌股票的预算约束修正

停牌股票权重固定，设 $w_{\mathcal{S}}$ 为停牌股票总权重，则可交易股票满足：

$$\sum_{i \notin \mathcal{S}} w_i = 1 - \sum_{i \in \mathcal{S}} w_{0,i}$$

---

## 6. 模块结构

```
portfolio_optimizer/
├── data/
│   └── generator.py        # 随机生成股票数据、因子、ADV、交易状态
├── factors/
│   ├── alpha_factors.py    # Alpha 因子容器与标准化
│   └── barra_factors.py   # CNE6 风格 + 行业因子生成
├── risk/
│   └── risk_model.py       # 因子风险模型（B, F, Δ）
├── optimizer/
│   ├── constraints.py      # 约束构建器（可组合）
│   └── mean_variance.py   # cvxpy 均值方差优化器
└── portfolio/
    └── portfolio.py        # 优化结果、风险分解、归因
```

---

## 7. 数据流

```
随机数据生成
    │
    ├──► Alpha 因子 (N,)
    ├──► Barra 因子载荷 B (N×K)
    ├──► 因子协方差 F (K×K)
    ├──► 特质风险 Δ (N,)
    ├──► ADV (N,)  ← 流动性约束
    └──► 交易状态 (N,)  ← 停牌/涨跌停
            │
            ▼
        风险模型 Σ = BFB' + Δ
            │
            ▼
        约束构建器
        ├── TradingStatusConstraint   ← 停牌/涨跌停
        ├── LiquidityConstraint       ← ADV 参与率
        ├── FactorExposureConstraint  ← 因子中性
        ├── IndustryConstraint        ← 行业中性
        └── TurnoverConstraint        ← 换手率
            │
            ▼
        cvxpy 求解器
            │
            ▼
        Portfolio（权重 + 风险分解 + 成交金额）
```

---

## 8. 关键设计决策

1. **因子风险用 Cholesky 分解加速**：将 `quad_form(w, Sigma)` 拆成两个 `sum_squares`，避免大矩阵求逆。
2. **停牌股从优化变量中分离**：停牌股不进入优化，减少变量数量，预算约束相应调整。
3. **约束可组合**：每个约束类返回 `List[cp.Constraint]`，优化器统一收集。
4. **流动性约束用绝对金额**：以 ADV 的百分比限制单日换手，框架外部传入组合总市值。
