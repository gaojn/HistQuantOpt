# 量化多头选股优化方法

> 本文档详细说明本项目所采用的组合优化数学模型、约束体系、求解流程，
> 供策略研究员、风控人员、IT 实施人员参考。
>
> 对应代码：`portfolio_optimizer/optimizer/alpha_max.py`

---

## 1. 问题定义

### 1.1 任务背景

在 A 股全市场（约 5500 只股票）中，每个调仓日基于 alpha 信号选择一组股票，
构造满足风险与流动性约束的纯多头组合，使预期收益最大化。

### 1.2 与指数增强的区别

| 维度 | 量化多头（本框架） | 指数增强 |
|---|---|---|
| 目标 | 最大化绝对 alpha 收益 | 跑赢基准指数 |
| 风险度量 | 组合分散度（隐式） | 跟踪误差 TE |
| 行业约束 | 绝对权重上限 | 相对基准偏离 ±X% |
| 因子约束 | 绝对暴露上限 | 主动暴露 ±σ |
| 成分股约束 | 可选下限（如≥40%） | 强制下限（如≥80%） |
| 换手 | 偏高（年化 ~1500%） | 偏低（年化 ~500%） |

---

## 2. 数学模型

### 2.1 目标函数

$$
\max_{w} \quad w^\top \alpha - \gamma \cdot \|w\|_2^2
$$

**含义：**

- $w \in \mathbb{R}^N$：组合权重向量，$N$ 为投资域股票数（含禁止持仓的股票，但其权重将被约束为 0）
- $\alpha \in \mathbb{R}^N$：每只股票的预期收益信号（推荐使用横截面 z-score 标准化后的值）
- $w^\top \alpha$：组合预期收益（线性项）
- $\gamma \cdot \|w\|_2^2 = \gamma \sum_i w_i^2$：**L2 分散惩罚项**，作用等价于隐式假设了一个对角风险矩阵 $\Sigma_{\text{implicit}} = \gamma \cdot I$
- $\gamma$：分散度系数，越大越分散

### 2.2 为什么用 L2 惩罚而不是显式协方差矩阵

| 方案 | 优点 | 缺点 |
|---|---|---|
| 显式 $w^\top \Sigma w$ | 真实风险约束 | 需要 Sigma 估计，PCA/Barra 工程量大 |
| **L2 惩罚 $\|w\|_2^2$** | **无需 Sigma；自然分散** | **不考虑相关性结构** |
| L1 惩罚 $\|w\|_1$ | 产生稀疏解 | 偏离预算约束 sum(w)=1 |

本框架选择 **L2 惩罚**作为入门方案，简单且数学性质好（QP 可解）。
后续接入真实 Sigma 数据时可平滑切换到完整的 MV 框架。

### 2.3 完整约束体系（共 7 类）

```
约束 1: 预算约束
        sum(w) = 1                               # 满仓

约束 2: 个股权重区间
        0 ≤ w_i ≤ W_max                          # 纯多头 + 单票上限

约束 3: 行业权重绝对上限
        Σ_{i ∈ ind_k} w_i ≤ I_max     ∀ 行业 k    # 行业集中度控制

约束 4: 成分股权重下限（可选）
        Σ_{i ∈ constituent} w_i ≥ R_min          # 如 HS300 ≥ 40%

约束 5: 风格因子绝对暴露（可选）
        |B_style[:,k]^T · w| ≤ S_max  ∀ 风格 k    # 控制风格集中度

约束 6: 双边换手率（可选）
        ‖w − w_prev‖_1 ≤ T_max                   # 防止过度调仓

约束 7: 交易状态约束
        w_i = 0     if i ∈ {停牌, ST, 次新}        # A股特有限制
```

---

## 3. 约束详解

### 3.1 预算约束（强制）

$$
\sum_{i=1}^{N} w_i = 1
$$

**说明：** 组合权重之和为 1（满仓投资）。如需保留现金缓冲，可改为 `sum(w) ≤ 1 - cash_buffer`。

### 3.2 个股权重区间（强制）

$$
0 \leq w_i \leq W_{\max}, \quad \forall i
$$

**典型设置：**

| $W_{\max}$ | 持仓数 | 适用场景 |
|---:|---:|---|
| 0.5% | 200+ | 高分散，被动型增强 |
| 1.0% | 100+ | 标准多头 |
| **2.0%** | **50-60** | **本项目默认** |
| 5.0% | 20-40 | 集中持仓策略 |

**纯多头约束：** $w_i \geq 0$ 由 cvxpy 变量声明 `cp.Variable(n, nonneg=True)` 自动保证。

### 3.3 行业权重绝对上限（强制）

$$
\sum_{i \in \text{ind}_k} w_i \leq I_{\max}, \quad \forall \text{ 行业 } k
$$

**实现：** 遍历每个一级行业（中信31个），约束该行业所有股票权重之和。

**典型设置：**

| $I_{\max}$ | 说明 |
|---:|---|
| 10% | 强行业中性 |
| 15% | 中等约束 |
| **20%** | **本项目默认** |
| 30% | 较宽松 |
| 100% | 不约束行业 |

> ⚠️ 注意：若全部行业上限之和 < 100%，约束将不可行。
> 例如 31 个行业 × 3% = 93% < 100%，应保证 31 × $I_{\max}$ ≥ 1.

### 3.4 成分股权重下限（可选）

$$
\sum_{i \in \mathcal{C}} w_i \geq R_{\min}
$$

其中 $\mathcal{C}$ 为指数成分股集合（如沪深300、中证500、中证1000）。

**目的：** 即使是多头策略，也常需要保证一定比例的大盘股权重，原因：
1. **流动性保障**：成分股流动性好，便于建仓和清仓
2. **降低跟踪误差**：避免与主流指数过度偏离
3. **风控合规**：部分机构资金有最低市值要求

**典型设置：**

| $R_{\min}$ | 说明 |
|---:|---|
| 0% | 不约束（纯小盘策略） |
| 20% | 弱约束 |
| **40%** | **本项目默认（HS300）** |
| 60% | 中等约束 |
| 80% | 接近指数增强 |

### 3.5 风格因子绝对暴露（可选）

$$
\left| \mathbf{B}_{\text{style}}^\top \cdot w \right|_k \leq S_{\max,k}, \quad \forall \text{ 风格因子 } k
$$

其中 $\mathbf{B}_{\text{style}} \in \mathbb{R}^{N \times K}$ 为风格载荷矩阵，$K$ 为风格因子数（本项目用 CNE6 16 个）。
约束上限 $S_{\max,k}$ 支持**按因子分别设定**（`style_active_bound` 可写标量统一，或写 dict 按因子名分别约束，见第 7 节配置）。

**包含的 16 个 CNE6 风格因子：**

| 因子 | 含义 |
|---|---|
| Size | 规模 |
| MidCap | 中盘 |
| Beta | 市场 Beta |
| Momentum | 动量 |
| ResidualVolatility | 残差波动率 |
| LongTermReversal | 长期反转 |
| Liquidity | 流动性 |
| Value | 价值 |
| EarningsYield | 盈利收益 |
| Growth | 成长 |
| Profitability | 盈利能力 |
| InvestmentQuality | 投资质量 |
| EarningsQuality | 盈利质量 |
| EarningsVariability | 盈利波动 |
| Leverage | 杠杆 |
| DividendYield | 股息率 |

**典型设置：**

| $S_{\max}$ | 说明 |
|---:|---|
| 0.3 | 严格风格中性（接近指数增强水平） |
| 0.5 | 中等约束 |
| **1.0** | **本项目默认（允许 1σ 暴露）** |
| 2.0 | 弱约束 |
| None | 不约束 |

> 注：由于因子已 z-score 标准化，$S_{\max} = 1$ 意味着组合在该因子上的加权暴露不超过 ±1 个标准差。

### 3.6 双边换手率约束（可选）

$$
\sum_{i=1}^{N} |w_i - w_{\text{prev},i}| \leq T_{\max}
$$

**说明：** L1 范数约束，对应组合调整时**买卖总和**（双边）。

**典型设置：**

| $T_{\max}$ | 单边换手 | 说明 |
|---:|---:|---|
| 0.20 | 10% | 月频换仓建议 |
| **0.30** | **15%** | **本项目默认（5日换仓）** |
| 0.50 | 25% | 较宽松 |
| 2.00 | 100% | 完全自由 |
| None | — | 不约束（首期建仓必用） |

**重要：** 首期建仓时 `prev_weight=None`，此约束自动跳过。

### 3.7 交易状态约束（强制）

```
w_i = 0,  if status_i ∈ {SUSPENDED, NEW_LISTING, ST}
```

**A 股特有处理：**

| 状态 | 处理 | 原因 |
|---|---|---|
| 正常交易（NORMAL） | 不限制 | — |
| 停牌（SUSPENDED） | 禁止持仓 | 无法买卖 |
| 涨停（LIMIT_UP） | 允许持仓但禁止加仓 | 当前简化为允许（实际可加细约束） |
| 跌停（LIMIT_DOWN） | 允许持仓但禁止减仓 | 同上 |
| 次新股（NEW_LISTING）| 禁止持仓 | 上市 < 60 自然日 |
| ST/*ST | 禁止持仓 | 风险偏高 |

**实现：** 在 `RealMarketAdapter._compute_status()` 中按优先级判断，
再在优化器中对禁止持仓的股票添加硬约束 `w[i] == 0`，
同时其 `alpha[i]` 清零，避免影响目标函数方向。

---

## 4. 求解流程

### 4.1 算法选择

- **问题类型：** 凸二次规划（QP）
- **求解器：** [CLARABEL](https://github.com/oxfordcontrol/Clarabel.rs)（cvxpy 默认）
- **复杂度：** $O(N^2)$，5500 股票 ~ 1 秒
- **失败处理：** 返回 `infeasible`，外层调用方决定保持上期权重 / 跳过

### 4.2 数值稳定性

求解后做归一化以消除浮点误差：

```python
weights = np.clip(w.value, 0.0, None)
weights /= weights.sum()
```

### 4.3 性能数据（实测）

| 投资域规模 | 求解时间 | 内存 |
|---:|---:|---:|
| 300 只（HS300）| 0.1s | 50MB |
| 500 只（ZZ500）| 0.2s | 80MB |
| 1000 只（ZZ1000）| 0.4s | 150MB |
| **5500 只（全市场）** | **~1s** | **300MB** |

---

## 5. 参数调优指南

### 5.1 三类典型场景

#### 场景 A：均衡型（默认）

```python
AlphaMaxConfig(
    weight_upper=0.02,
    industry_upper=0.20,
    min_constituent_ratio=0.40,
    diversification_penalty=0.05,
    style_bound=1.0,
    max_turnover=0.30,
)
# 持仓 50-60 只，年化换手 ~1500%
```

#### 场景 B：高分散

```python
AlphaMaxConfig(
    weight_upper=0.005,
    industry_upper=0.15,
    min_constituent_ratio=0.60,
    diversification_penalty=0.20,    # 强分散
    style_bound=0.5,
    max_turnover=0.20,
)
# 持仓 200-300 只，年化换手 ~250%
```

#### 场景 C：行业中性

```python
AlphaMaxConfig(
    weight_upper=0.01,
    industry_upper=0.10,             # 行业紧约束
    diversification_penalty=0.10,
    style_bound=0.3,                 # 风格强中性
    max_turnover=0.25,
)
```

### 5.2 调参经验法则

| 现象 | 调整方向 |
|---|---|
| 持仓数太少（<30） | ↑ diversification_penalty 或 ↓ weight_upper |
| 持仓数太多（>200） | ↓ diversification_penalty 或 ↑ weight_upper |
| 行业过度集中 | ↓ industry_upper |
| 风格暴露偏大 | ↓ style_bound |
| 换手过高 | ↓ max_turnover 或检查 alpha 自相关 |
| 求解 infeasible | 放宽 min_constituent_ratio / industry_upper |

---

## 6. 与传统均值方差的对比

### 6.1 均值方差（Markowitz）

$$
\max_w \quad w^\top \mu - \frac{\lambda}{2} w^\top \Sigma w
$$

- $\mu$：预期收益向量
- $\Sigma$：协方差矩阵（需要 Barra 模型或样本估计）
- $\lambda$：风险厌恶系数

### 6.2 本框架（AlphaMax）

$$
\max_w \quad w^\top \alpha - \gamma \cdot w^\top I w
$$

**形式上等价于设置 $\Sigma = I$**（单位对角矩阵）。

### 6.3 何时升级到完整 MV

当满足以下条件时，应将 $\|w\|_2^2$ 替换为 $w^\top \Sigma w$：

1. 有可靠的 Barra/PCA 风险模型
2. Sigma 估计窗口稳定（如 120 个交易日）
3. 需要严格控制组合波动率而非仅分散

> 代码切换路径：在 `alpha_max.py` 中将
> `cp.sum_squares(w)` 替换为 `cp.quad_form(w, Sigma)`。
> 历史 MV 实现保留在 `legacy/portfolio_optimizer/optimizer/mean_variance.py`。

---

## 7. 输入数据接口

### 7.1 MarketSnapshot

封装目标日的市场截面信息（`portfolio_optimizer/data/generator.py`）：

```python
@dataclass
class MarketSnapshot:
    tickers:         list[str]            # N 只股票代码
    industry:        pd.Series            # 行业归属
    adv:             pd.Series            # 20日平均成交额（元）
    status:          pd.Series            # 交易状态（TradingStatus 枚举）
    prev_weight:     pd.Series            # 上期权重
    market_cap:      pd.Series            # 流通市值（元）
    portfolio_value: float                # 组合总市值
    is_constituent:  pd.Series            # 是否成分股（布尔）
```

### 7.2 Alpha 向量

```python
alpha: np.ndarray, shape (N,)
```

要求：
- 与 `snapshot.tickers` 对齐
- 推荐 z-score 标准化
- 缺失值填 0

### 7.3 风格载荷矩阵

```python
style_loading: pd.DataFrame, shape (N, K)
```

- index = tickers，columns = 风格因子名
- 已 z-score 标准化
- 缺失填 0

---

## 8. 输出结果

```python
@dataclass
class AlphaMaxResult:
    tickers:         list[str]
    weights:         np.ndarray
    status:          str               # "optimal" / "infeasible: ..."
    objective_value: float
    snapshot:        MarketSnapshot | None

    # 主要方法
    def is_feasible() -> bool
    def n_positions() -> int            # 持仓数
    def to_series() -> pd.Series        # ticker → weight
    def top_holdings(n=10) -> pd.DataFrame
    def industry_weights() -> pd.Series
    def style_exposures(B) -> pd.Series
    def summary() -> str
```

---

## 9. 与 alpha 因子的交互

### 9.1 因子前处理

进入优化器前，alpha 通常需要：

1. **去极值（winsorize）**：±3σ 截断
2. **截面标准化（z-score）**：均值=0，标准差=1
3. **缺失填充**：填 0（中性化处理）

工具：`portfolio_optimizer/factors/alpha_factors.py::AlphaFactors`

### 9.2 因子复合

多个因子合成 alpha 时：

```python
alpha = sum(w_k * factor_k for k in factors)
# w_k 可基于 IC、IR 或回归系数确定
alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-10)
```

### 9.3 因子衰减建模

合成因子时控制其跨期持续性（避免每期独立噪声）：

```python
f_t = decay * f_{t-1} + sqrt(1 - decay^2) * new_signal_t
```

| decay | 因子自相关 | 适用 |
|---:|---:|---|
| 0.70 | 0.69 | 短期反转 |
| **0.90** | **0.89** | **本项目默认** |
| 0.97 | 0.97 | 长周期价值 |

---

## 10. 已知局限与未来工作

### 10.1 当前局限

| 局限 | 说明 | 改进方向 |
|---|---|---|
| 无显式风险矩阵 | 使用 L2 近似 | 接入 Barra 协方差 $\Sigma = BFB' + \Delta$ |
| 涨跌停允许持仓 | 简化处理 | 加入"涨停日不可加仓"约束 |
| 行业按一级 | 31 个行业 | 支持二级行业约束 |
| 无最小持仓数 | 仅约束最大 | 加入 0-1 整数变量（MILP） |
| 不考虑 T+1 | 假设 T 日内可对冲 | 加入 T+1 锁定约束 |

### 10.2 扩展接口预留

代码已为以下扩展预留接口：

```python
# 1. 显式风险矩阵
optimizer.optimize(..., sigma=Sigma_matrix)

# 2. 自定义业绩约束
optimizer.add_constraint(custom_cp_constraint)

# 3. 多基准对比
optimizer.optimize(..., benchmark_list=[hs300, zz500])
```

---

## 11. 参考文献

1. Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
2. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*.
3. MSCI Barra (2010). *Barra China Equity Model (CNE6)*.
4. CVXPY 文档: https://www.cvxpy.org/
5. CLARABEL 求解器: https://clarabel.org/

---

## 附录 A：完整代码示例

```python
from datetime import date
import numpy as np
from pathlib import Path

from histquant.io.data_panel import load_panel
from portfolio_optimizer import (
    RealMarketAdapter,
    CNE6RiskModel,
    AlphaMaxConfig,
    AlphaMaxOptimizer,
)

# 1. 加载行情
target = date(2026, 5, 21)
panel = load_panel(
    date(2026, 4, 1), target,
    columns=["code", "date", "close", "adj_close",
             "limit_up", "limit_down", "amount",
             "float_mv", "free_mv", "total_mv",
             "free_turnover", "trade_status",
             "industry_l1", "list_days",
             "is_hs300", "is_zz500", "is_zz1000", "is_st"],
)

# 2. 构建市场快照
snapshot = RealMarketAdapter().build_snapshot_from_panel(
    panel=panel,
    target_date=target,
    index="hs300",
    portfolio_value=1e8,
)

# 3. 加载 CNE6 风险模型（16 风格因子暴露 + 协方差），取目标日快照
risk_snap = CNE6RiskModel().at(target, snapshot.tickers)
style_loading = risk_snap.style_loading()

# 4. Alpha（实际场景从研究员模型导入）
alpha = your_alpha_signal  # shape (N,)

# 5. 配置约束
config = AlphaMaxConfig(
    weight_upper=0.02,
    industry_upper=0.20,
    min_constituent_ratio=0.40,
    diversification_penalty=0.05,
    style_bound=1.0,
    max_turnover=0.30,
)

# 6. 求解
result = AlphaMaxOptimizer(config).optimize(
    alpha=alpha,
    snapshot=snapshot,
    style_loading=style_loading,
    prev_weight=None,   # 首期建仓
)

# 7. 查看结果
print(result.summary())
print(result.top_holdings(10))
print(result.industry_weights().head())
print(result.style_exposures(style_loading))
```

---

**文档版本**：v1.0
**适用代码版本**：见 git commit hash
**维护人**：策略团队
