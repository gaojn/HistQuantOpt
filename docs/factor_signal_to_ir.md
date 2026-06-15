# 因子信号强度 → 策略 IR 标定

> ⚠️ 本文涉及的合成因子使用了**未来收益（前视）**，是**标定工具，不可交易**。
> 它回答一个问题：**「假如我挖到一个 IC=X、ICIR=Y、衰减=Z 的真因子，在我这套约束下大概能做出多少 IR？」**

## 一、理论锚点：主动管理基本法则

Grinold 基本法则：

```
策略IR = IC × √BR × TC
  IC = 信息系数（因子与未来收益的截面相关）
  BR = 广度（每年独立下注次数）
  TC = 转换系数（约束把信号翻译成持仓的效率，0~1）
```

对**固定调仓周期**的截面策略，可化简成一条可直接套用的式子：

```
年化IR ≈ ICIR × √(年调仓次数) × TC
       = (IC / IC_std) × √(年调仓次数) × TC
```

推导：若每期主动收益 active_t ≈ k·IC_t，则
`年化IR = mean(active)/std(active) × √(年期数) = [mean(IC)/std(IC)] × √(年期数) = ICIR × √(年期数)`，
再乘上约束折损 TC 即得。

**关键：决定 IR 的不是 IC 绝对值，而是 ICIR（IC 的稳定性）。**

## 二、三层 IR 口径（约束依次增强，IR 依次衰减）

| 口径 | 含义 | 对应列 |
|---|---|---|
| `ir_theory` | ICIR × √(年调仓次数)，纯信息论上限，无组合构建 | 标定脚本 |
| `ls_ir` | 分位多空组合年化 IR，含组合构建、无约束 | 标定脚本 |
| `full_ir` | 全管线（行业/风格中性 + TE + 换手 + long-only） | 真实回测 |

转换系数：`TC = full_ir / ir_theory`。

## 三、用已跑结果标定 TC（本项目）

锚点：合成因子 IC=0.08 / IC_std=0.10（ICIR=0.8）/ decay=0.8，5 日调仓（≈50 次/年）。

| 真实回测 | full_IR |
|---|---|
| HS300 指数增强 | 2.85 |
| ZZ500 指数增强 | 3.01 |
| ZZ1000 指数增强 | 3.30 |

理论上限 = 0.8 × √50 = 5.66 → **TC ≈ 3.05 / 5.66 ≈ 0.54**。

**经验法则（本项目 5 日调仓、全市场、行业+风格中性约束下）：**

```
年化IR ≈ ICIR × √50 × 0.54 ≈ 3.9 × ICIR ≈ 3.9 × (IC / IC_std)
```

> 例：挖到 IC=0.04、IC_std=0.08（ICIR=0.5）的真因子 → 预期策略 IR ≈ 1.9。

## 四、参数与可测因子统计的对应

合成器（`research/signal_grid.py` 与 `pipeline/build_synthetic_alpha`）的三个旋钮，
恰好对应你评估真因子时能测到的三个量：

| 合成参数 | 含义 | 你测真因子时对应 |
|---|---|---|
| `ic_mean` | 每期截面 IC 的均值 | 因子月度/周度 IC 均值 |
| `ic_std`  | 每期 IC 的波动 → ICIR = ic_mean/ic_std | IC 标准差 |
| `decay`   | 因子 AR(1) 自相关 | 相邻调仓期因子自相关（换手反比） |

⚠️ 输入参数 ≠ 实测值：z-score 与 decay 混合会让实测 Spearman IC/ICIR 略偏离输入，
查找表以**实测值**为准（脚本同时记录 `in_*` 与实测列）。

## 五、decay（衰减）的双刃剑

- **高 decay**：换手低（省成本），但携带的旧信号与当期未来收益相关性弱 → 稀释当期 IC → 实测 ICIR 下降。
- **低 decay**：信息新鲜、ICIR 高，但换手大、成本拖累重。

标定脚本的 decay 扫描会量化这个权衡（实测 ICIR、自相关、换手随 decay 的变化）。

## 六、使用方法

```bash
python examples/run_signal_grid.py
```

产出：
- `output/signal_grid_results.parquet` —— 全网格实测指标
- `output/signal_grid_ir.html` —— IC×ICIR → 估算 IR 热图

**查表流程**：测出真因子的 (IC, IC_std, decay) → 算 ICIR → 在热图/查找表定位 → 读估算 IR。
或直接套经验法则 `IR ≈ 3.9 × ICIR`。

锚点（`KNOWN_ANCHORS`）随真实回测更新可在 `examples/run_signal_grid.py` 顶部修改，TC 会自动重标。
