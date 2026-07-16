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

**风险项 $R(w)$ 有两种形态**，由是否配置 `risk_aversion` 决定：

| 策略 | 默认（L2 代理） | 启用 CNE6 风险模型 |
|---|---|---|
| 量化多头 | $\gamma\,\lVert w\rVert_2^2$（分散惩罚） | $\lambda\,(w^\top XFX^\top w + \delta^\top w^2)$ |
| 指数增强 | $\gamma\,\lVert w-w_{bm}\rVert_2^2$（偏离基准） | $\lambda\,(a^\top XFX^\top a + \delta^\top a^2),\ a=w-w_{bm}$ |

其中 $X$=因子暴露、$F$=因子协方差、$\delta$=特质方差，来自 CNE6 风险面板
（47 因子 = 16 风格 + Country + 30 中信一级行业，详见 [design.md](design.md) 与
[`risk/cne6_risk.py`](../hqopt/risk/cne6_risk.py)）。
L2 形态无需协方差、简单稳健；CNE6 形态刻画真实因子相关性与个股特质风险差异。

> **指数增强的基准权重 $w_{bm}$** 默认取指数**官方成分权重**（由 wind_db 导出，
> 月度快照，按调仓日 asof 取 ≤当日最近快照）；官方未覆盖的日期/指数自动回退
> free_mv 分级靠档**重构**。可在 YAML 设 `optimizer.benchmark_weight_source: reconstruct`
> 切回纯重构。导出脚本见 [操作指南.md](操作指南.md) 数据准备。

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
| 6 | 换手（硬上限，可选） | $\lVert w-w_{\text{prev}}\rVert_1 \le T_{\max}$ | 同 |
| 7 | 交易状态 | 见下 | 同 |

> 风格上限 $S$ 支持标量（统一）或 dict（按因子名分别约束，可带 `default` 兜底）。
> 因子已 z-score 标准化，$S=1$ 即组合在该因子上加权暴露不超过 ±1σ。

**约束 7 — A 股交易状态**（强制，贴近真实成交，停牌冻结口径）：

三类互斥掩码，优先级：frozen > sell_only > zero。

| 条件 | 掩码 | 数学约束 | 说明 |
|---|---|---|---|
| 停牌 & 有持仓（$w_{\text{prev},i}>0$） | **frozen** | $w_i = w_{\text{prev},i}$ | 冻结，不计换手、不释放资金 |
| 停牌/次新/ST & 无持仓 | **zero** | $w_i = 0$ | 禁止开仓 |
| 掉出候选池的持仓票、次新/ST 有持仓 | **sell_only** | $w_i \le w_{\text{prev},i}$ | 只卖不买，卖出正常计换手与成本 |
| 涨停 LIMIT_UP | — | $w_i \le w_{\text{prev},i}$ | 不可加仓（与 sell_only 方向相同） |
| 跌停 LIMIT_DOWN | — | $w_i \ge w_{\text{prev},i}$ | 不可减仓 |

**优化域 = 当期候选池 ∪ 上期持仓（有当日行情的票）**（`filter_universe` 的 `prev_holdings` 参数控制）。
真退市票（当日 panel 无行情行）无法进优化域，按原有逻辑清零+归一并输出告警日志。

冻结票上界豁免：$W_{\max,i}^{\text{frozen}} = \max(W_{\max}, w_{\text{prev},i})$，避免漂移后持仓超上限导致 infeasible。
`active_weight_upper`（指增专属）对冻结票同样豁免，仅对非冻结票施加。

alpha 置零范围：frozen ∪ zero ∪ sell_only（避免干扰目标函数方向）。
冻结票因 $w_i = w_{\text{prev},i}$ 等式约束，$|w_i - w_{\text{prev},i}| = 0$，自动不消耗换手预算。

涨跌停约束需传 `prev_weight`；状态判定见 [`RealMarketAdapter._compute_status`](../hqopt/data/real_adapter.py)。

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

**因子分层**（16 个 CNE6 风格）：

| 层 | 因子 | 处理 |
|---|---|---|
| A 基础护栏 | `Size` `Liquidity` `ResidualVolatility` `Beta` | 几乎总要约束（最易把组合做歪） |
| B 多数约束 | `Momentum` | 中等约束（防风格切换回撤、推高换手） |
| C 视 alpha | `Value` `Growth` `EarningsYield` `Profitability` | 是收益风格本身，收太紧会削收益 |
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
- **失败处理**：返回 `infeasible`，pipeline 保持上期权重（重新归一）或跳过
- **数值稳定**：求解后 `clip(0)` + 归一化消除浮点误差
- **规模**：全市场 ~5000 只（剔北交所+ST）约 1s/期
- **首个调仓日**：组合日收益固定为 0（首日全仓现金，T+1 才建仓），基准当日有收益，存在一日不对等（对 Sharpe 影响极小，但长期回测首日图形可见跳变）。
- **risk_aversion 置 0**：显式设 `risk_aversion: 0.0` 将完全关闭风险惩罚项（factor_risk/specific_risk 均不加入目标函数），等价于纯 alpha 最大化；与 `risk_aversion: null`（退回 L2 兜底）行为不同。

**infeasible 常见原因**：成分股下限太高、行业约束太紧（某行业基准权重为 0 时易冲突）、
首期建仓却设了 `max_turnover`（应传 `None`，pipeline 已自动处理首期）。

**绩效指标口径**（`engine.calc_metrics`）：
- 年化超额收益采用**几何口径**：`(1+total_port)/(1+total_bm)-1` 后再年化，与 report 年度表、全期行一致。
- 月度超额收益表采用算术差（月组合收益 − 月基准收益），与年度/全期口径不同，报告中已注明。
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
（YAML 驱动，见 [操作指南.md](操作指南.md)）。单期直接调优化器：

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

`alpha_vec` 需与 `snap.tickers` 对齐、推荐 z-score、缺失填 0。

---

## 9. 收益归因（Return Attribution）

> 对应代码：[`analysis/attribution.py`](../hqopt/analysis/attribution.py)、
> [`risk/attribution_data.py`](../hqopt/risk/attribution_data.py)。
> CLI：`hqopt attribute`（见 [操作指南.md](操作指南.md)）。

**动机**：一条超额净值曲线好看，不代表 alpha 干净——可能是优化器偷偷吃了某个
风格 beta（如小盘、低波）。归因把每期主动收益拆成风格 / 行业 / Country / 个股
特质（选股）贡献，回答"钱到底从哪来"。

**方法**：每个调仓期 $(T_i, T_{i+1}]$ 内，主动权重 $w_{active}=w_p-w_{bm}$ 与主动
暴露 $X_{active}=X^\top w_{active}$ 固定不变（信号日 $T_i$ 的 as-of 值，与优化器
决策时用的暴露一致）；$T_i$ 当天仍按上一期权重计入收益，$T_i$ 决定的新权重从
$T_i+1$ 起才实际持有（T+1 执行，与回测引擎时序一致）。持有期内逐日：

$$
\text{主动收益}(t) \approx \underbrace{X_{active}^\top f(t)}_{\text{风格+行业+Country}} + \underbrace{w_{active}^\top u(t)}_{\text{选股（特质）}}
$$

$f(t)$（因子收益）、$u(t)$（个股特质收益）取自 ClickHouse `cne6_risk.factor_return` /
`specific_return`，与暴露 $X$（`cne6_risk.factor_exposure`）**同源**，保证残差
自检有意义。多期用 Carino(1999) 平滑系数链接，保证 $\Sigma$ 各归因项累计 =
真实几何累计主动收益。

**残差自检**：每日「主动收益 − (风格+行业+Country+特质)」理论上应 ≈0；非零
主要来自**覆盖缺口**——`cne6_risk` 只覆盖其估计域（`univ_flag==1`，剔除次新/
极小市值等），组合或基准里在域外的持仓收益全部漏进残差。`daily["coverage_pct"]`
给出每期"被模型覆盖的主动权重占比"可供判断，但覆盖率与残差占比非线性关系
（未覆盖的常是高波动票，权重小也可能贡献不成比例的残差）。**残差占比高的
期间，归因结论需谨慎**；合成数据（完全覆盖）下残差严格为 0，证明分解算式
本身正确，见 `tests/test_attribution.py`。

**已知局限**：暴露按持有期冻结（不随日频更新），调仓越不频繁，暴露与
`f(t)/u(t)` 实际估计所用的当日暴露之间的近似误差可能越大；`t统计` 为简单
`mean/std` 未做自相关调整（非 NW 稳健标准误）。

---

## 10. 已知局限

| 局限 | 说明 / 改进方向 |
|---|---|
| L2 风险代理 | 未启用 `risk_aversion` 时不含相关性结构；接 CNE6 风险面板即用真因子风险 |
| 行业按一级 | 中信 30 个一级行业；可扩展二级 |
| 无最小持仓数约束 | 仅约束单票上限；如需精确控持仓数需引入整数变量（MILP） |
| 泛化指增的非成分敞口 | 指增候选全市场、靠成分下限托底，剩余 ≤20% 非成分敞口可能引入额外跟踪偏离，靠 risk_aversion / 风格约束管住 |
