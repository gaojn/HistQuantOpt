# HistQuantOpt 操作手册

> 版本：2026-05  
> 覆盖两条策略流水线：**量化多头**（绝对收益）和**指数增强**（相对基准超额）

---

## 1. 项目结构

```
HistQuantOpt/
├── portfolio_optimizer/          ← 核心框架
│   ├── data/
│   │   ├── generator.py          # MarketSnapshot 数据类、TradingStatus 枚举
│   │   ├── real_adapter.py       # parquet 行情 → MarketSnapshot
│   │   └── benchmark.py          # 分级靠档指数权重计算器
│   ├── factors/
│   │   ├── alpha_factors.py      # Alpha 预处理（去极值/标准化）
│   │   └── jy_barra.py           # 聚源 9 风格因子加载器
│   ├── optimizer/
│   │   ├── alpha_max.py          # 量化多头 QP 优化器
│   │   └── index_enhance.py      # 指数增强 QP 优化器
│   └── backtest/
│       ├── engine.py             # 向量化回测引擎
│       └── report.py             # Plotly HTML 报告生成器
│   └── io/
│       ├── data_panel.py         # load_panel 主入口
│       └── schema.py             # 行情字段定义与单位说明
├── examples/                     ← 可直接运行的 demo
│   ├── demo_synthetic_alpha.py       # 量化多头：生成合成 Alpha
│   ├── demo_alpha_max.py             # 量化多头：单日优化
│   ├── demo_batch_alpha_max.py       # 量化多头：批量优化
│   ├── demo_backtest.py              # 量化多头：回测 + HTML 报告
│   ├── demo_index_enhance_single.py  # 指数增强：单日优化验证
│   ├── demo_batch_index_enhance.py   # 指数增强：批量优化
│   ├── demo_index_enhance_backtest.py# 指数增强：回测 + HTML 报告
│   └── compare_te_control.py         # TE 控制对比工具
├── data/                         ← 本地数据（不入版本控制）
│   ├── cache/ashare_daily_2023.parquet
│   ├── cache/ashare_daily_2024.parquet
│   ├── cache/ashare_daily_2025.parquet
│   ├── cache/ashare_daily_2026.parquet
│   └── jy_stylefactor_000985_CSI_20230209_20260522.parquet
├── output/                       ← 结果统一输出
└── docs/                         ← 文档
    ├── manual.md                 ← 本文档（操作手册）
    ├── README.md                 ← 文档索引
    ├── optimization_method.md    ← 量化多头优化方法详解
    └── index_enhance_method.md   ← 指数增强方法详解
```

---

## 2. 两条策略流水线总览

| 维度 | A. 量化多头 | B. 指数增强 |
|---|---|---|
| **目标** | 绝对收益最大化 | 超越基准指数（ZZ500 / HS300 / ZZ1000） |
| **候选池** | 全市场 ~5000 只（剔除北交所 + ST） | 同上 |
| **优化器** | `AlphaMaxOptimizer` | `IndexEnhanceOptimizer` |
| **目标函数** | `max w'α − γ‖w‖²` | `max w'α − γ‖w−w_bm‖²` |
| **基准参照** | 无（或外部传入） | 分级靠档基准权重 |
| **TE 管控** | 无显式 TE 约束 | 行业/风格/L2 硬约束 + γ 惩罚 |
| **批量优化入口** | `demo_batch_alpha_max.py` | `demo_batch_index_enhance.py` |
| **回测入口** | `demo_backtest.py` | `demo_index_enhance_backtest.py` |

---

## 3. 流水线 A：量化多头

### 3.1 端到端运行顺序

```bash
# Step 1：生成合成 Alpha
python examples/demo_synthetic_alpha.py
# → output/synthetic_alpha.parquet

# Step 2（可选）：单日优化验证
python examples/demo_alpha_max.py

# Step 3：批量优化（96 期 × 5天）
python examples/demo_batch_alpha_max.py
# → output/batch_weights_alpha_max.parquet

# Step 4：回测 + HTML 报告
python examples/demo_backtest.py
# → output/nav.parquet
# → output/backtest_report.html
```

### 3.2 合成 Alpha 参数

AR(1) 衰减模型，每期 IC 来自 N(ic_mean, ic_std²)：

```python
# demo_synthetic_alpha.py 关键参数
IC_MEAN = 0.10   # 因子 IC 均值
IC_STD  = 0.07   # IC 标准差
DECAY   = 0.90   # AR(1) 衰减（控制因子换手）
```

| decay | 因子日自相关 | 持仓换手参考 | 类比因子类型 |
|------:|----------:|----------:|---|
| 0.70 | 0.69 | ~63% | 短期反转 |
| **0.90** | **0.89** | **~38%** | **动量/成长（推荐）** |
| 0.97 | 0.97 | ~21% | 价值/质量 |

### 3.3 优化器配置

```python
from portfolio_optimizer.optimizer.alpha_max import AlphaMaxConfig, AlphaMaxOptimizer

config = AlphaMaxConfig(
    weight_upper=0.02,              # 个股绝对权重上限
    industry_upper=0.20,            # 行业绝对权重上限
    min_constituent_ratio=0.40,     # 成分股（如 HS300）权重下限
    diversification_penalty=0.05,   # γ：L2 分散惩罚
    style_bound=1.0,                # 风格因子绝对暴露上限（σ）
    max_turnover=0.30,              # 双边换手率上限
)
```

**目标函数与约束：**

```
max  w'α  −  γ · ‖w‖²
s.t.
  sum(w) = 1
  0 ≤ w_i ≤ weight_upper
  sum(w[ind==k]) ≤ industry_upper          ∀ 行业 k
  sum(w[constituent]) ≥ min_const_ratio
  |B[:,k]' w| ≤ style_bound                ∀ 风格 k
  ‖w − w_prev‖₁ ≤ max_turnover
  w[停牌/ST/次新] = 0
```

---

## 4. 流水线 B：指数增强

### 4.1 端到端运行顺序

```bash
# Step 1：批量优化（144 期 × 5天，全市场候选池）
python examples/demo_batch_index_enhance.py
# → output/zz500_enhance_weights.parquet

# Step 2：回测 + HTML 报告
python examples/demo_index_enhance_backtest.py
# → output/zz500_enhance_nav.parquet
# → output/zz500_enhance_report.html
```

### 4.2 关键全局参数

`demo_batch_index_enhance.py` 顶部：

```python
INDEX          = "zz500"           # 基准指数：hs300 / zz500 / zz1000
BACKTEST_START = date(2023, 6, 1)
BACKTEST_END   = date(2026, 5, 22)
REBAL_FREQ     = 5                 # 调仓频率（交易日）
PORTFOLIO_VAL  = 1e8              # 组合规模（元，用于 ADV 约束）

# 合成 Alpha 参数（与流水线 A 共用相同生成逻辑）
IC_MEAN  = 0.08
IC_STD   = 0.10
DECAY    = 0.80
```

### 4.3 优化器配置

```python
from portfolio_optimizer.optimizer.index_enhance import IndexEnhanceConfig, IndexEnhanceOptimizer

# 推荐配置（ZZ500，全市场候选池）
config = IndexEnhanceConfig(
    weight_upper=0.02,              # 单票绝对上限（ZZ500 最大成分 ~1.5%）
    weight_lower=0.0,               # 单票下限
    min_constituent_ratio=0.80,     # 目标指数成分股 ≥80%
    industry_active_bound=0.07,     # 行业相对基准偏离 ±7%
    style_active_bound=0.50,        # 风格主动暴露 ±0.5σ
    tracking_penalty=10.0,          # γ：TE 代理惩罚（越大越贴近基准）
    max_turnover=0.40,              # 单次双边换手上限
    weight_diff_l2_bound=None,      # ‖w−w_bm‖₂ 硬约束（None=不启用）
)
```

**weight_upper 参考值：**

| 指数 | 推荐值 | 原因 |
|---|---|---|
| HS300 | 0.05 | 茅台等重仓股 ~5% |
| ZZ500 | 0.02 | 最大成分 ~1.5% |
| ZZ1000 | 0.01 | 最大成分 <1% |

**目标函数与约束：**

```
max  w'α  −  γ · ‖w − w_bm‖²
s.t.
  sum(w) = 1
  0 ≤ w_i ≤ weight_upper
  sum(w[constituent]) ≥ min_constituent_ratio
  |sum(w[ind==k]) − sum(w_bm[ind==k])| ≤ industry_active_bound   ∀ 行业 k
  |B[:,k]'(w − w_bm)| ≤ style_active_bound                        ∀ 风格 k
  ‖w − w_prev‖₁ ≤ max_turnover
  ‖w − w_bm‖₂ ≤ weight_diff_l2_bound    （可选硬约束）
  w[停牌/ST/次新] = 0
```

求解器：CLARABEL（max_iter=500），失败自动降级 SCS（max_iters=10000）。

### 4.4 候选池过滤

`filter_universe` 函数：全市场剔除北交所（.BJ）和 ST 股票，约剩 ~4500 只。

### 4.5 基准权重：分级靠档

`IndexBenchmarkWeights` 按中证编制方案计算每日权重：

```
f = free_mv / total_mv
A = ceil(f × 10) / 10，当 f > 80% 时取 A = 1.0（上限截断）
A ≥ 0.1（下限截断）
adj_mv = total_mv × A
w_i = adj_mv_i / Σ adj_mv_j
```

### 4.6 实测绩效（合成 Alpha，ZZ500，2023-06 ~ 2026-05）

| 指标 | 组合 | ZZ500 基准 |
|---|---:|---:|
| 年化收益 | +98.19% | +16.92% |
| 年化超额 | **+53.12%** | — |
| 跟踪误差 TE | **8.18%** | — |
| 信息比率 IR | **6.497** | — |
| Sharpe | 2.885 | 0.710 |
| 最大回撤 | -21.85% | -27.25% |
| 超额最大回撤 | **-5.14%** | — |
| 超额 Calmar | **10.325** | — |
| 月度胜率 | **88.9%** | — |

> 注：使用合成 Alpha（含前瞻信息），真实因子超额通常 10~20%/年。

---

## 5. TE 控制方案

当实际 TE 超出目标时，可调整以下参数（按效果从快到慢）：

### 方案一：提高 γ（tracking_penalty）

γ 从 10 → 40，TE 约下降 1.5%，但 Alpha 捕获小幅下降。IR 通常提升（信噪比改善）。

```python
config = IndexEnhanceConfig(..., tracking_penalty=40.0)
```

### 方案二：L2 硬约束

直接约束 ‖w−w_bm‖₂ ≤ budget，是最精确的 TE 代理控制。

```python
# TE_L2 与实际 TE 换算关系（ZZ500 全市场，经验值）
# 实际 TE ≈ TE_L2 × 68%
# 目标 TE=6% → TE_L2_budget ≈ 0.088
config = IndexEnhanceConfig(..., weight_diff_l2_bound=0.088)
```

### 方案三：收紧行业/风格约束

间接减少主动偏离，TE 改善幅度较小（~0.4%）。

```python
config = IndexEnhanceConfig(
    industry_active_bound=0.05,   # 7% → 5%
    style_active_bound=0.30,      # 0.5 → 0.3
    ...
)
```

### 对比实验结果（ZZ500，2023-06 ~ 2026-05）

| 方案 | TE | 年化超额 | IR | 超额最大回撤 |
|---|---:|---:|---:|---:|
| 基准（γ=10） | 8.19% | +53.56% | 6.54 | -5.14% |
| 方案3（收紧行业/风格） | 7.77% | +51.60% | 6.64 | -4.77% |
| 方案1（γ=20） | 7.62% | +52.65% | 6.91 | -4.53% |
| 方案1（γ=40） | 6.69% | +48.99% | 7.33 | -3.52% |
| 方案2（L2≤0.088） | 6.69% | +49.13% | 7.34 | -3.95% |

**推荐：** γ=40 效果与 L2 硬约束相当，且超额 Calmar 更优（13.93 vs 12.43），是更灵活的软约束。

---

## 6. 回测引擎

### 6.1 逻辑

```
调仓日：按目标权重建仓，扣除单边成本 cost_one_way
非调仓日：权重随价格自然漂移
每日组合收益 = Σ w_i(漂移后) × r_i
```

### 6.2 调用

```python
from portfolio_optimizer.backtest.engine import Backtester

bt = Backtester(
    cost_one_way=0.0015,   # 单边 15bp（佣金 + 冲击）
    risk_free=0.02,        # 年化无风险利率
)
result = bt.run(
    weight_df=weight_df,         # DataFrame：index=调仓日，columns=ticker
    adj_close=adj_wide,          # DataFrame：index=所有交易日，columns=ticker
    benchmark_ret=bm_ret,        # Series：基准日收益率（None=等权全股）
)
```

### 6.3 BacktestResult 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `nav` | Series | 组合净值（起始=1） |
| `bm_nav` | Series | 基准净值 |
| `excess_nav` | Series | 超额净值（几何：nav/bm_nav） |
| `daily_ret` | Series | 组合日收益 |
| `bm_ret` | Series | 基准日收益 |
| `excess_ret` | Series | 超额日收益 |
| `turnover` | Series | 调仓日双边换手率 |
| `portfolio_metrics` | PerformanceMetrics | 组合绩效指标 |
| `benchmark_metrics` | PerformanceMetrics | 基准绩效指标 |

### 6.4 PerformanceMetrics 字段

```python
annual_return        # 年化收益
annual_vol           # 年化波动
sharpe               # Sharpe 比率
max_drawdown         # 最大回撤
calmar               # Calmar = 年化收益 / |最大回撤|
annual_excess_return # 年化超额（日均×252）
tracking_error       # 跟踪误差 TE（超额日收益年化标准差）
info_ratio           # 信息比率 IR = 年化超额 / TE
excess_max_drawdown  # 超额净值最大回撤（几何）
excess_calmar        # 年化超额 / |超额最大回撤|
win_rate_monthly     # 月度超额胜率
avg_monthly_excess   # 月均超额收益
```

---

## 7. HTML 报告

`generate_html_report` 生成交互式 Plotly 报告，包含：

1. **总体绩效卡片**（12 项）：年化收益/波动/Sharpe/最大回撤/Calmar/年化超额/TE/IR/超额最大回撤/超额Calmar/月度胜率/月均超额
2. **年度绩效分解表**：各年累计收益（几何超额）+ 全期年化，含超额TE和超额回撤列
3. **净值与回撤图**：组合 vs 基准 vs 超额净值，下方组合回撤填充
4. **超额净值与超额回撤图**：单独展示超额曲线（蓝色）+ 超额回撤填充
5. **月度超额明细表格**：越大越红、越小越绿热力着色，含年度/月度胜率
6. **调仓换手率柱状图**：各期双边换手 + 均值参考线

所有图表横坐标使用 YYYYMMDD 数字格式。

```python
from portfolio_optimizer.backtest.report import generate_html_report

report_path = generate_html_report(
    result,
    output_path="output/my_report.html",
    title="ZZ500 指数增强组合回测报告",
)
```

---

## 8. 绩效指标口径

| 指标 | 计算方法 |
|---|---|
| 年化收益 | `(1 + total) ^ (252/n_days) − 1`（几何年化） |
| 区间累计收益 | `∏(1 + daily_ret) − 1` |
| 超额收益（年内/分年） | `(1 + 组合累计) / (1 + 基准累计) − 1`（几何，行业标准） |
| 年化超额（全期） | `excess_daily.mean() × 252` |
| 跟踪误差 TE | `(ret − bm_ret).std() × √252` |
| 信息比率 IR | `年化超额 / TE` |
| 超额最大回撤 | `excess_nav / excess_nav.cummax() − 1` 的最小值 |

**为什么超额用几何方法？**  
算术差 `组合 − 基准` 在极端年份会放大误差。2025 年组合 +137.63%、基准 +35.98%：
- 算术差：+101.65%（高估）
- 几何差：`(1+1.3763)/(1+0.3598)−1 = +74.76%`（真实超额倍率）

---

## 9. 常见问题

### 优化报 infeasible

1. `min_constituent_ratio` 与 `industry_active_bound` 冲突（成分股集中于少数行业）→ 放宽行业约束
2. `style_active_bound` 过紧，与基准固有风格暴露冲突 → 放宽到 0.5σ
3. `max_turnover` 首期太严 → 首期自动设为 None（代码已内置）
4. 全市场 5000+ 股票时 CLARABEL 迭代不足 → 已内置 SCS 兜底

### 换手持续触及上限

- 因子 `decay` 过低（信号翻转太快）→ 调到 0.85~0.95
- `max_turnover` 过严 → 放宽到 0.35~0.50

### 持仓数太少（< 30）

- 量化多头：`diversification_penalty` 调大到 0.1~0.3
- 指数增强：`tracking_penalty` 调小（减小向基准靠拢的力度）

### CLARABEL UserWarning "Solution may be inaccurate"

属于精度警告，结果通常仍可用（状态为 `optimal_inaccurate`）。如需更高精度，可增大 `max_iter` 到 1000 或使用 SCS 求解。

---

## 10. 输出文件一览

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| `synthetic_alpha.parquet` | demo_synthetic_alpha | 合成 Alpha 矩阵（date × ticker） |
| `batch_weights_alpha_max.parquet` | demo_batch_alpha_max | 量化多头权重矩阵 |
| `nav.parquet` | demo_backtest | 量化多头净值/基准净值/日收益 |
| `turnover.parquet` | demo_backtest | 量化多头换手率 |
| `backtest_report.html` | demo_backtest | 量化多头回测报告 |
| `zz500_enhance_weights.parquet` | demo_batch_index_enhance | ZZ500 增强权重矩阵（144×5281） |
| `zz500_enhance_nav.parquet` | demo_index_enhance_backtest | ZZ500 增强净值/超额/日收益 |
| `zz500_enhance_turnover.parquet` | demo_index_enhance_backtest | ZZ500 增强换手率 |
| `zz500_enhance_report.html` | demo_index_enhance_backtest | ZZ500 增强回测报告 |

---

## 11. 数据依赖

### 行情面板（`load_panel`）

缓存位置：`data/cache/ashare_daily_{year}.parquet`，覆盖 2023~2026 年。

主要字段：

| 字段 | 说明 |
|---|---|
| `adj_close` | 复权收盘价 |
| `free_mv`, `total_mv` | 自由流通市值、总市值（万元） |
| `amount` | 成交额（千元） |
| `trade_status` | 交易状态 |
| `industry_l1` | 中信一级行业（31个） |
| `is_hs300`, `is_zz500`, `is_zz1000` | 指数成分标志 |
| `is_st`, `list_days` | ST 标志、上市天数 |

### 聚源风格因子

文件：`data/jy_stylefactor_000985_CSI_20230209_20260522.parquet`

9 个 Barra 风格因子（已 z-score 标准化）：

| 因子代码 | 含义 |
|---|---|
| siz | 规模 |
| vol | 波动率 |
| liq | 流动性 |
| mom | 动量 |
| qua | 质量/盈利 |
| val | 估值 |
| gro | 成长 |
| sen | 情绪 |
| divid | 股息 |

---

## 12. CNE6 因子风险模型 + config 驱动批量优化

这是另一套与第 3/4 节 demo 脚本并行的流水线：用 YAML 配置 + CLI 驱动
`portfolio_optimizer/pipeline/batch_optimize.py`，并接入真实 CNE6 因子风险模型
（替代第 3/4 节里默认的 L2 惩罚 / 聚源 9 因子风格暴露）。

### 12.1 数据来源与刷新

CNE6 风险面板由 [scripts/export_cne6_panels.py](../scripts/export_cne6_panels.py)
从 ClickHouse `the_quant.cne6_risk`（因子暴露/协方差/特质风险）+ 本地
`data/cache/ashare_daily_<year>.parquet`（行业 one-hot）拉取，写入：

| 输出目录 | 来源 | 用途 |
|---|---|---|
| `data/barra_cne6/` | `factor_cov_S` + `specific_risk_S` | CNE6S，短周期 hl=63（默认） |
| `data/barra_cne6_L/` | `factor_cov_L` + `specific_risk_L` | CNE6L，长周期 hl=252 |

各含 `exposure_panel.parquet`（rebal_date, code, 47因子, spec_var）和
`factor_cov_panel.parquet`（rebal_date, factor, 47因子协方差）。

**47 因子** = 16 风格 + `Country`（全市场恒为1）+ 30 行业（CITIC L1）。
16 风格因子定义见 `portfolio_optimizer/risk/cne6_risk.py` 的 `STYLE_FACTORS`。

exposure 按 `univ_flag==1`（当日可交易 + 上市满期）过滤；`Country`/行业
不计入 `style_active_bound` 约束。

刷新（ClickHouse 数据更新后重跑；仅读，需只读密码）：

```bash
CLICKHOUSE_PASSWORD=... python scripts/export_cne6_panels.py

# 其余连接参数可选覆盖（默认 the_quant/dw_player）：
# CLICKHOUSE_HOST / CLICKHOUSE_PORT / CLICKHOUSE_DB / CLICKHOUSE_USER
```

连接层见 `portfolio_optimizer/data/clickhouse_db.py`；密码只走环境变量，不入代码/git。
行情缓存 `data/cache/ashare_daily_<year>.parquet` 的同步见
[scripts/sync_ashare_cache.py](../scripts/sync_ashare_cache.py)
（`CLICKHOUSE_PASSWORD=... python scripts/sync_ashare_cache.py --years 2026`）。

### 12.2 CNE6RiskModel 用法

```python
from portfolio_optimizer.risk import CNE6RiskModel

rm = CNE6RiskModel()                       # 默认 CNE6S（data/barra_cne6/）
rm_l = CNE6RiskModel(data_dir="data/barra_cne6_L")  # CNE6L

print(rm.coverage)        # (起始调仓日, 截止调仓日)
snap = rm.at(target_date, tickers)   # ≤ target_date 最近调仓日；早于覆盖范围返回 None
# snap.X (N,47) 暴露, snap.F (47,47) 因子协方差, snap.delta (N,) 特质方差
# snap.style_loading() -> DataFrame(N, 16)，供 style_active_bound 约束使用
```

组合风险：`V = X F Xᵀ + diag(δ)`。

### 12.3 config 驱动批量优化（CLI）

```bash
python scripts/run_batch_optimize.py configs/zz500_enhance_cne6_horizon.yaml
```

CLI 参数（均为可选覆盖项）：

| 参数 | 作用 |
|---|---|
| `--alpha-file PATH` | 用外部 Alpha parquet 替代合成 Alpha（见 12.4） |
| `--output PATH` | 覆盖 `output.weights` 路径 |
| `--cne6-dir PATH` | 覆盖 `optimizer.cne6_data_dir`（如 `data/barra_cne6_L`） |
| `--risk-aversion FLOAT` | 覆盖 `optimizer.risk_aversion`（λ） |

config 关键字段（`optimizer` 段）：

| 字段 | 说明 |
|---|---|
| `use_cne6_risk` | `true` 启用 CNE6 因子风险模型；`false` 走第3/4节默认 L2 惩罚 + 聚源9因子 |
| `cne6_data_dir` | `null`→CNE6S（`data/barra_cne6/`）；`"data/barra_cne6_L"`→CNE6L |
| `risk_aversion` | 风险厌恶系数 λ，目标函数中 `λ·wᵀΣw` |

`use_cne6_risk=true` 时，调仓日早于 CNE6 面板覆盖范围（见 `rm.coverage`）会被跳过。

### 12.4 自定义 Alpha 因子

config 的 `alpha` 段新增 `source`：

```yaml
alpha:
  source: file                          # 默认 "synthetic"（合成 Alpha，见 3.2）
  path: "output/my_alpha.parquet"       # 宽表：index=date, columns=ticker
```

`path` 指向的 parquet 与权重矩阵同一约定——`index` 为日期，`columns` 为股票代码，
值为因子分数（截面相对大小即可，不要求标准化）。优化时按调仓日取
`index <= 调仓日` 的最近一行，缺失股票填0。

也可只用 `--alpha-file` 临时覆盖任意 config 的 alpha 来源，不用改 YAML。

---

## 13. VWAP5 合成 Alpha 网格 + 批量回测

用于批量生成一组覆盖不同 IC / ICIR / 换手率的合成 Alpha（基于未来 H=5 日
`adj_vwap` 涨跌幅构造），并跑通"批量优化 + 真实回测"全流程，作为优化器/
因子合成管线的输入样本测试。

⚠️ 这些 Alpha 含未来信息（前视），仅用于管线测试/IR标定，不可用于实盘。

### 13.1 生成因子网格

```bash
python examples/build_alphas_vwap5.py
```

构造方法与 `research/signal_grid.SignalGridRunner` 一致（AR(1) 衰减 + 截面
z-score 混入未来收益），但 `price_col="adj_vwap"`：

```
sig_t = ρ_t · zscore(r_{t+5}) + √(1-ρ_t²) · ε,   ρ_t ~ N(ic_mean, ic_std²)
f_t   = decay · f_{t-1} + √(1-decay²) · sig_t     (截面 z-score)
```

网格 = `signal_grid` 默认网格（IC×ICIR @ decay=0.8，+ decay 单独扫描），去重后
28 组。输出：

| 路径 | 内容 |
|---|---|
| `alphas/alpha_vwap5_ic{IC}_icir{ICIR}_decay{decay}.parquet` | 长表 `(date, code, alpha)` |
| `alphas/_summary.csv` | 各因子输入参数 + 实测 IC/ICIR/自相关/换手 |

`alphas/` 已加入 `.gitignore`（28 个文件共约 2GB，可随时用上述脚本重新生成）。

### 13.2 单因子批量回测

```bash
python examples/run_zz1000_enhance_vwap5_backtest.py
```

依赖 `configs/zz1000_enhance_vwap5_test.yaml`（`alpha.source: file`，指向
单个因子的宽表 parquet），跑中证1000指数增强的批量优化 + 真实执行回测。

### 13.3 全网格批量回测

```bash
python examples/run_alpha_grid_pipeline.py
```

对 `alphas/alpha_vwap5_*.parquet` 逐个跑批量优化 + 真实回测，结果写入
`output/vwap5_grid/<因子名>/{weights,nav_realistic}.parquet`、
`report_realistic.html`，并汇总到 `output/vwap5_grid/summary.csv`
（年化超额、IR、最大回撤、换手等）。

- 支持断点续跑：已存在 `weights.parquet`/`nav_realistic.parquet` 的因子会跳过。
- `run_batch_optimize()` 新增 `panel` / `alpha_df` 可选参数，传入预加载数据可
  避免网格扫描时重复加载行情/重复构造 Alpha。
