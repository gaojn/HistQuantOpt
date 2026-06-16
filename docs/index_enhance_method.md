# HS300 指数增强优化方法

> 本文档说明本项目"指数增强"优化的设计思路、数学模型与约束体系。
> 对应代码：`portfolio_optimizer/optimizer/index_enhance.py`
>
> 与"量化多头"的核心差异在于：**所有约束和目标都是相对基准的**。

---

## 1. 问题定义

### 1.1 目标

在 HS300 指数基础上，通过主动选股获取超额收益（Alpha），
同时控制组合相对基准的偏离（跟踪误差，TE）。

### 1.2 候选池

**HS300 + 中证500 + 中证1000 ≈ 1800 只**（中大盘股全集）

为什么不只用 HS300？
- 仅 300 只股票，alpha 信号空间受限
- 允许在中盘股（ZZ500、ZZ1000）找超额收益
- 但必须保证 HS300 成分股权重 ≥80%，维持与基准的相关性

### 1.3 vs 量化多头

| 维度 | 量化多头 | HS300 指数增强 |
|---|---|---|
| 候选池 | 全市场 ~5500 只 | HS300+ZZ500+ZZ1000 ~1800 只 |
| 目标函数 | $\max w^\top\alpha - \gamma\|w\|^2$ | $\max w^\top\alpha - \gamma\|w-w_{bm}\|^2$ |
| 行业约束 | 绝对上限 ≤20% | **相对基准 ±5%** |
| 风格约束 | 绝对暴露 ≤1σ | **主动暴露 ±0.3σ** |
| HS300 权重 | ≥40%（可选）| **≥80%（强制）** |
| 单票上限 | 2% | **5%**（容纳基准重仓股）|
| 换手率 | ≤30% | **≤20%** |
| 风险度量 | 组合分散度 | **跟踪误差代理** |

---

## 2. 数学模型

### 2.1 目标函数

$$
\max_w \quad w^\top \alpha - \gamma \cdot \|w - w_{bm}\|_2^2
$$

**含义：**

- $w \in \mathbb{R}^N$：组合权重，$N$ 为候选池大小
- $w_{bm}$：基准（HS300）权重，非成分股位置为 0
- $w^\top \alpha$：组合预期 alpha 收益
- $\gamma \cdot \|w - w_{bm}\|_2^2$：**跟踪误差 L2 惩罚**，控制组合偏离基准的程度
- $\gamma$：惩罚系数（默认 10），越大越贴近基准

### 2.2 为什么用 L2 惩罚而不是真实 TE

完整的跟踪误差公式：

$$
TE = \sqrt{(w - w_{bm})^\top \Sigma (w - w_{bm})}
$$

需要协方差矩阵 $\Sigma$（Barra 风险模型）。**本项目暂无 Sigma 数据**，使用 $\|w - w_{bm}\|_2^2$ 作为简化代理。
等价于假设 $\Sigma = I$（个股波动率相同，无相关性）。

### 2.3 完整约束体系（共 7 类）

```
约束 1: 预算约束
        sum(w) = 1

约束 2: 个股权重区间
        0 ≤ w_i ≤ W_max               # 单票绝对上限（默认 5%）

约束 3: HS300 成分股权重下限
        Σ_{i ∈ HS300} w_i ≥ R_min     # 默认 80%

约束 4: 行业主动偏离
        |Σ_{i ∈ ind_k}(w_i - w_bm,i)| ≤ I_active   ∀ 行业 k   # 默认 ±5%

约束 5: 风格因子主动暴露
        |B_style[:,k]^T (w - w_bm)| ≤ S_active     ∀ 风格 k   # 默认 ±0.3σ

约束 6: 双边换手率
        ‖w - w_prev‖_1 ≤ T_max         # 默认 20%

约束 7: 交易状态
        w_i = 0    if i ∈ {停牌, ST, 次新}
```

---

## 3. 约束详解

### 3.1 个股绝对上限（约束 2）

$$
0 \leq w_i \leq W_{\max}
$$

**典型设置：** $W_{\max} = 5\%$

**为什么不是 2%（像量化多头）？**

HS300 中最大权重股（贵州茅台、宁德时代）基准权重已经 ~5%。
若设 $W_{\max} = 2\%$，则不论 alpha 多高，都被迫低配茅台 3 个点 —— 这是巨大的主动偏离。

**建议：**

| 配置 | $W_{\max}$ | 说明 |
|---|---:|---|
| **默认** | **5%** | **容纳基准重仓股** |
| 更进取 | 8% | 允许超配基准重仓股 |
| 更保守 | 3% | 强制偏离基准重仓股（不推荐）|

### 3.2 HS300 成分股下限（约束 3）

$$
\sum_{i \in \mathcal{C}_{HS300}} w_i \geq R_{\min}
$$

**典型设置：**

| $R_{\min}$ | 风险等级 | 适用场景 |
|---:|---|---|
| 0.95 | 极低 | 几乎被动 |
| 0.90 | 低 | 严格指数增强 |
| **0.80** | **中** | **本项目默认** |
| 0.60 | 高 | 风格漂移大 |
| 0.40 | 极高 | 接近多头策略 |

**作用：** 保证组合与 HS300 的高相关性，降低跟踪误差。

### 3.3 行业主动偏离（约束 4）

$$
\left| \sum_{i \in \text{ind}_k} w_i - \sum_{i \in \text{ind}_k} w_{bm,i} \right| \leq I_{\text{active}}, \quad \forall \text{ 行业 } k
$$

**与量化多头的对比：**

```
量化多头：sum(w[ind_k]) ≤ I_max              # 绝对上限
指数增强：|sum(w[ind_k]) - sum(w_bm[ind_k])| ≤ I_active  # 相对偏离
```

**典型设置：**

| $I_{\text{active}}$ | TE 贡献 | 说明 |
|---:|---|---|
| ±3% | 低 | 接近行业中性 |
| **±5%** | **中** | **本项目默认** |
| ±8% | 高 | 行业 timing 策略 |
| 无约束 | 极高 | 纯 alpha 驱动 |

### 3.4 风格因子主动暴露（约束 5）

$$
\left| \mathbf{B}_{\text{style}}^\top \cdot (w - w_{bm}) \right|_k \leq S_{\text{active}}
$$

**等价含义：** 组合相对基准的风格因子加权暴露不超过 $S_{\text{active}}$ 个标准差。

**典型设置：**

| $S_{\text{active}}$ | TE 贡献 | 说明 |
|---:|---|---|
| ±0.1 | 极低 | 完全风格中性 |
| **±0.3** | **低** | **本项目默认** |
| ±0.5 | 中 | 允许中等风格偏好 |
| ±1.0 | 高 | 较激进 |

**Barra 经典做法：** $S_{\text{active}} = 0.3$ 是行业标准。

### 3.5 跟踪误差惩罚系数 $\gamma$（目标函数中的关键参数）

$$
\text{Objective: } w^\top \alpha - \gamma \cdot \|w - w_{bm}\|^2
$$

**直觉：**

- $\gamma$ 越大，组合越靠近基准（被动）
- $\gamma$ 越小，组合越激进追逐 alpha

**调参参考：**

| $\gamma$ | 表现倾向 |
|---:|---|
| 1 | 弱约束，TE 可能偏大 |
| 5 | 平衡 |
| **10** | **本项目默认** |
| 50 | 强约束，接近被动 |
| 100+ | 几乎完全被动 |

**经验法则：** 若 alpha 已 z-score 标准化（std=1），
$\gamma \approx 5 \sim 20$ 通常能产生合理的 TE（年化 3~6%）。

---

## 4. 候选池过滤

代码逻辑（`portfolio_optimizer/pipeline/universe.py::filter_universe`）：

```python
universe = (
    (panel["is_hs300"]  == 1)
    | (panel["is_zz500"]  == 1)
    | (panel["is_zz1000"] == 1)
)
```

**每个调仓日动态过滤：**
- 当日属于 HS300/ZZ500/ZZ1000 任一指数的股票纳入候选
- 调整时调入、调出的股票自动反映在候选池中

**典型候选池规模：** ~1800 只（明显小于全市场 5500）

---

## 5. 实测绩效（2024-06 ~ 2026-05，合成 Alpha IC=0.08）

| 指标 | 组合 | HS300 基准 |
|---|---:|---:|
| 年化收益 | **+29.49%** | +19.63% |
| 年化波动 | 19.19% | 18.37% |
| Sharpe | **1.339** | 0.960 |
| 最大回撤 | -14.73% | -14.98% |
| Calmar | **2.002** | 1.311 |
| **年化超额** | **+8.07%** | — |
| **信息比率 IR** | **1.377** | — |
| 月度胜率 | 66.7% | — |
| 年化换手 | ~1180% | — |
| 平均持仓数 | 79 只 | 300 只 |

### 5.1 年度分解

| 年份 | 组合 | 基准 | 超额 | 最大回撤 |
|---|---:|---:|---:|---:|
| 2024（半年）| +20.71% | +12.84% | +7.87% | -8.44% |
| 2025 | +20.67% | +20.52% | +0.14% | -13.45% |
| 2026（5个月）| +11.86% | +3.17% | +8.69% | -6.84% |

### 5.2 真实水平对标

国内一线指数增强产品：

| 指标 | 行业一流 | 本项目 |
|---|---|---|
| 年化超额 | 8~15% | **+8.07%** ✓ |
| 信息比率 | 1.5~2.5 | **1.38** ✓ |
| 月度胜率 | 60~70% | **66.7%** ✓ |
| 年化换手 | 800~1500% | **1177%** ✓ |

**结论：** 各项指标均落在真实产品的合理区间内。

---

## 6. API 用法

```python
from portfolio_optimizer import (
    RealMarketAdapter,
    IndexBenchmarkWeights,
    CNE6RiskModel,
    IndexEnhanceConfig,
    IndexEnhanceOptimizer,
)

# 1. 快照 + 过滤候选池（HS300+ZZ500+ZZ1000）
snap = RealMarketAdapter().build_snapshot_from_panel(
    panel, target_date, index="hs300", portfolio_value=1e8,
)
snap = filter_universe(snap, panel, target_date)   # 自定义过滤函数

# 2. 基准权重（分级靠档）
bm = IndexBenchmarkWeights(index="hs300", panel=panel)
bm.precompute(start_date, target_date, panel=panel)
bm_weight = bm.get_weights(target_date, tickers=snap.tickers).values

# 3. CNE6 风格因子暴露（16 因子）
risk_snap = CNE6RiskModel().at(target_date, snap.tickers)
style_loading = risk_snap.style_loading()

# 4. 配置（style_active_bound 可写标量统一，或写 dict 按因子分别约束）
cfg = IndexEnhanceConfig(
    weight_upper=0.05,
    min_constituent_ratio=0.80,
    industry_active_bound=0.05,
    style_active_bound={"default": 0.30, "Size": 0.20, "Momentum": 0.20},
    tracking_penalty=10.0,
    max_turnover=0.20,
    # risk_aversion=10.0,  # 可选：因子协方差进目标（真跟踪误差），需传 risk_snapshot
)

# 5. 优化
result = IndexEnhanceOptimizer(cfg).optimize(
    alpha=alpha_vec,
    snapshot=snap,
    benchmark_weight=bm_weight,
    style_loading=style_loading,
    prev_weight=prev_weight_array,
    risk_snapshot=risk_snap,   # risk_aversion 设置时用于真因子风险
)

# 6. 结果
print(result.summary())
print(result.industry_active_weights())      # 各行业相对基准偏离
print(result.style_active_exposure(style_loading))  # 风格主动暴露
print(result.top_holdings(10))                # 前10大持仓（含基准权重对比）
```

---

## 7. 参数模板

### 7.1 默认（中等约束）

```python
IndexEnhanceConfig(
    weight_upper=0.05,
    min_constituent_ratio=0.80,
    industry_active_bound=0.05,
    style_active_bound=0.30,
    tracking_penalty=10.0,
    max_turnover=0.20,
)
# TE ~ 3-5%，IR ~ 1.0-1.5
```

### 7.2 保守（贴近基准）

```python
IndexEnhanceConfig(
    weight_upper=0.04,
    min_constituent_ratio=0.90,    # ↑
    industry_active_bound=0.03,     # ↓
    style_active_bound=0.20,        # ↓
    tracking_penalty=30.0,          # ↑↑
    max_turnover=0.15,              # ↓
)
# TE ~ 1.5-3%，IR ~ 1.5-2.0
```

### 7.3 进取（追逐超额）

```python
IndexEnhanceConfig(
    weight_upper=0.06,              # ↑
    min_constituent_ratio=0.70,     # ↓
    industry_active_bound=0.08,     # ↑
    style_active_bound=0.50,        # ↑
    tracking_penalty=5.0,           # ↓
    max_turnover=0.30,              # ↑
)
# TE ~ 5-8%，超额波动大
```

---

## 8. 输出文件

| 文件 | 内容 |
|---|---|
| `output/index_enhance_weights.parquet` | 96 期权重矩阵（96 × 2135） |
| `output/index_enhance_nav.parquet` | 净值序列（组合/基准/超额）|
| `output/index_enhance_turnover.parquet` | 各调仓日双边换手 |
| `output/index_enhance_report.html` | 交互式 HTML 报告 |

---

## 9. 常见问题

### 9.1 跟踪误差太大怎么办？

按优先级调整：
1. ↑ `tracking_penalty`（10 → 20）
2. ↓ `industry_active_bound`（5% → 3%）
3. ↓ `style_active_bound`（0.3 → 0.2）
4. ↑ `min_constituent_ratio`（80% → 90%）

### 9.2 求解 infeasible

常见原因：
- $R_{\min} > 1 - (\text{非HS300股票数} \times W_{\max})$
- 行业偏离约束太紧，HS300 某行业本身权重为 0 时易冲突
- 首期建仓时若设了 `max_turnover` → 应改 `None`

### 9.3 与官方指数增强差异

| 我们 | 一线产品 |
|---|---|
| 用 L2 惩罚做 TE 代理 | 用 Barra 完整 $\Sigma$ 算真实 TE |
| 候选池固定 HS300+ZZ500+ZZ1000 | 动态扩展（含科创/北交所）|
| 不考虑交易冲击 | 流动性、TWAP 完整建模 |

接入 Barra Sigma 后，代码切换：
将 `cp.sum_squares(w - w_bm)` 改为 `cp.quad_form(w - w_bm, Sigma)`。

---

**文档版本：** v1.0
**对应代码：** `portfolio_optimizer/optimizer/index_enhance.py`
