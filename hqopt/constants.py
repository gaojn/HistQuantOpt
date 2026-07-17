"""项目级共享常量。"""

from __future__ import annotations

# 涨跌停封板判定容差（相对涨跌停价的比例）。
# close 落在涨停价的 (1 - LIMIT_TOL) 以上 / 跌停价的 (1 + LIMIT_TOL) 以下即视为封板。
# 仅用于吸收原始价与涨跌停价的浮点/取整误差；回测引擎与数据适配器共用同一值，
# 避免两处阈值不一致。偏保守（宁可多判封板 → 不可成交，贴近真实）。
LIMIT_TOL = 1e-3

# 合成 Alpha 产物必须携带的统一水印。默认配置使用由未来收益构造的
# VWAP5 标定信号，即使 alpha.source=file 也必须显式标记并传播到报告。
SYNTHETIC_ALPHA_WARNING_FILE = "SYNTHETIC_ALPHA_WARNING.txt"
SYNTHETIC_ALPHA_WARNING_TEXT = (
    "本产物使用了含未来信息的合成 Alpha 信号生成。\n"
    "回测业绩完全不可信，绝不可用于实盘或对外披露。\n"
)
