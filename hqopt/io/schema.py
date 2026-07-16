"""A 股行情面板的 schema 定义（纯常量 + 文档，无外部依赖）。

来源:
    Wind A 股日频行情 (ClickHouse 视图 ``vw_ashare_daily_backtest``)

设计原则:
    - 列名映射: Wind 原始列 → 团队统一标准列 (snake_case)
    - 单位: 沿用 Wind 惯例 (价格元、volume 股、amount 千元、市值万元)
    - 派生列: 视图无原字段, 由标准列计算 (free_mv / adj_vwap / list_days)

各模块的引用:
    - cache_builder.py: 用 SCHEMA / OUTPUT_COLUMNS 拉数据 + 写缓存
    - data_panel.py:    用 OUTPUT_COLUMNS 校验本地缓存的列
    - PITFALLS.md:      引用 TRADE_STATUS_DOCS / CDR_NOTES 解释边界 case
"""

from __future__ import annotations

# 输出列顺序及含义（Wind 原始列 → 标准列名 → 说明）
# 单位沿用 Wind 惯例，因子计算前请确认是否需要换算
SCHEMA: list[tuple[str, str, str]] = [
    # --- 标识 ---
    ("s_info_windcode", "code", "Wind 股票代码，如 000001.SZ"),
    ("trade_dt", "date", "交易日期"),
    ("s_info_name", "name", "证券简称"),
    # --- 原始行情（不复权，单位：价格=元，volume=股，amount=千元）---
    ("s_dq_preclose", "pre_close", "前收盘价（元，不复权）"),
    ("s_dq_open", "open", "开盘价（元，不复权）"),
    ("s_dq_high", "high", "最高价（元，不复权）"),
    ("s_dq_low", "low", "最低价（元，不复权）"),
    ("s_dq_close", "close", "收盘价（元，不复权）"),
    ("s_dq_limit", "limit_up", "涨停价（元，不复权，ClickHouse s_dq_limit）"),
    ("s_dq_stopping", "limit_down", "跌停价（元，不复权，ClickHouse s_dq_stopping）"),
    ("s_dq_pctchange", "pct_change", "涨跌幅（%）"),
    ("s_dq_volume", "volume", "成交量（股）"),
    ("s_dq_amount", "amount", "成交额（千元）"),
    # --- 复权行情（后复权，单位：元；adj_price = raw_price × adj_factor）---
    # 依据：最新交易日 adj_close 显著大于 close（如 000001.SZ 2026-05-21：10.7 vs 883.9），
    # 且 adj_close ≈ close × adj_factor，符合后复权（价格向后累积调整）特征
    ("s_dq_adjpreclose", "adj_pre_close", "前收盘价（元，后复权）"),
    ("s_dq_adjopen", "adj_open", "开盘价（元，后复权）"),
    ("s_dq_adjhigh", "adj_high", "最高价（元，后复权）"),
    ("s_dq_adjlow", "adj_low", "最低价（元，后复权）"),
    ("s_dq_adjclose", "adj_close", "收盘价（元，后复权）"),
    ("s_dq_adjfactor", "adj_factor", "后复权因子（adj_price = raw_price × adj_factor）"),
    # --- 交易状态（Wind s_dq_tradestatus，详见 TRADE_STATUS_DOCS）---
    ("s_dq_tradestatus", "trade_status", "交易状态：交易/停牌/N/XR/XD/DR，含义见 TRADE_STATUS_DOCS"),
    ("s_dq_avgprice", "vwap", "成交量加权均价 VWAP（元，不复权）"),
    # --- 市值与换手（total_mv/float_mv=万元，turnover=%）---
    ("s_val_mv", "total_mv", "总市值（万元）"),
    ("s_dq_mv", "float_mv", "A 股流通市值（万元）"),
    ("s_dq_turn", "turnover", "换手率（%）"),
    ("s_dq_freeturnover", "free_turnover", "自由流通换手率（%）"),
    # --- 股本（万股）---
    ("tot_shr_today", "total_shares", "总股本（万股）"),
    ("float_a_shr_today", "float_shares", "流通 A 股股本（万股）"),
    ("free_shares_today", "free_shares", "自由流通股本（万股）"),
    # --- 行业（中信）---
    ("citics_ind_name_l1", "industry_l1", "中信一级行业"),
    ("citics_ind_name_l2", "industry_l2", "中信二级行业"),
    ("citics_ind_name_l3", "industry_l3", "中信三级行业"),
    # --- 上市信息 ---
    ("s_info_listdate", "list_date", "上市日期，YYYYMMDD"),
    ("s_info_delistdate", "delist_date", "退市日期，YYYYMMDD；在市通常为 20991231"),
    # --- 成分股 / ST 标记（0/1）---
    ("in_hs300", "is_hs300", "是否沪深 300 成分股"),
    ("in_zz500", "is_zz500", "是否中证 500 成分股"),
    ("in_zz1000", "is_zz1000", "是否中证 1000 成分股"),
    ("in_st", "is_st", "是否 ST / *ST"),
]


# trade_status 取值说明（Wind A 股日行情 s_dq_tradestatus）
# 除权除息日的不复权价格会跳空，回测收益计算应优先用后复权列（adj_*）
TRADE_STATUS_DOCS: dict[str, str] = {
    "交易": "正常交易；有成交，价格有效",
    "停牌": "停牌；当日无正常交易（仍可能有行，量价可能为 0 或沿用前值）",
    "N": "上市首日（New）；新股上市第一个交易日，简称可能尚无 XD/XR 前缀",
    "XR": "除权（Ex-Rights）；送股、转增、配股等导致股本变动，当日不复权价向下跳空",
    "XD": "除息（Ex-Dividend）；现金分红导致不复权价向下跳空",
    "DR": "除权除息（Ex-Rights & Ex-Dividend）；同日既有送配转增又有现金分红",
}


# CDR 特殊标的说明
# 689009.SH（九号公司-WD）是 A 股首只科创板 CDR，市值/股本口径与普通 A 股不同：
# - 转换比例：1 股基础证券 = 10 份 CDR（1 份 CDR = 0.1 股基础证券）
# - 普通 A 股：总市值 ≈ 总股本(股) × 股价
# - CDR：需先将 CDR 份数按转换比例折算为基础证券股数，再 × CDR 每份价格
# 数据源中 total_mv/float_mv 通常已是系统换算后的展示值，故与 total_shares×close 简单相乘
# 会出现约 10 倍偏差；free_mv（= free_shares × close）不受此影响。因子回测时可直接用库中市值字段。
CDR_CODES: frozenset[str] = frozenset({"689009.SH"})
CDR_NOTES: dict[str, str] = {
    "689009.SH": (
        "九号公司-WD，科创板 CDR；1 基础股=10 份 CDR。"
        "total_mv/float_mv 为换算后市值，勿用 total_shares×close 验算。"
    ),
}


# 派生列（视图中无原字段，由标准列计算；插入位置见 OUTPUT_COLUMNS）
DERIVED_SCHEMA: list[tuple[str, str]] = [
    ("free_mv", "自由流通市值（万元，= free_shares × close；与 float_mv 同单位，已用 000001.SZ 交叉验证）"),
    ("adj_vwap", "后复权 VWAP（元，= vwap × adj_factor）"),
    ("list_days", "上市天数（自然日，= date - list_date；上市首日 = 0）"),
]


COLUMN_RENAME_MAP: dict[str, str] = {wind: std for wind, std, _ in SCHEMA}
COLUMN_DOCS: dict[str, str] = {std: doc for _, std, doc in SCHEMA} | {
    name: doc for name, doc in DERIVED_SCHEMA
}


OUTPUT_COLUMNS: list[str] = [
    "code", "date", "name",
    "pre_close", "open", "high", "low", "close",
    "limit_up", "limit_down",
    "pct_change", "volume", "amount",
    "adj_pre_close", "adj_open", "adj_high", "adj_low", "adj_close",
    "adj_vwap", "adj_factor",
    "trade_status", "vwap",
    "total_mv", "float_mv", "free_mv",
    "turnover", "free_turnover",
    "total_shares", "float_shares", "free_shares",
    "industry_l1", "industry_l2", "industry_l3",
    "list_date", "list_days", "delist_date",
    "is_hs300", "is_zz500", "is_zz1000", "is_st",
]


# 单因子回测不加载/不导出的列（IC 用 fwd_ret = price(T+H+1)/price(T+1)-1，不用 pct_change）
BACKTEST_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"pct_change"})


# 回测时序：
#   T 日收盘后：用 signal 组列算因子、过滤 universe（剔除 ST/停牌/新股等）
#   T+1 日     ：用 exec_limit 组列成交（open / vwap / close），并用 limit_up/down 价判断涨跌停
# 涨跌停判断优先用库中 limit_up/limit_down（s_dq_limit/s_dq_stopping）
# 注意：exec_limit 列在面板里仍是「当日」字段，回测代码里对同一 code 做 shift(-1)
#       取下一交易日即为 T+1 数据；或使用 merge_asof 对齐。
BACKTEST_COLUMN_GROUPS: dict[str, list[str]] = {
    "core": ["code", "date"],
    "signal": ["adj_close", "volume", "amount", "turnover", "float_mv"],
    "universe": ["is_st", "trade_status", "list_date", "list_days", "delist_date", "industry_l1"],
    "exec_limit": [
        "limit_up", "limit_down",
        "pre_close", "open", "high", "low", "close",
        "vwap", "volume", "amount", "trade_status",
    ],
}


BACKTEST_PRESETS: dict[str, list[str]] = {
    "factor_backtest": [
        "code", "date",
        "adj_close", "volume", "amount", "turnover", "float_mv",
        "industry_l1", "is_st", "trade_status",
        "list_date", "list_days",
        "is_hs300", "is_zz500", "is_zz1000",
        "pre_close", "open", "high", "low", "close",
        "limit_up", "limit_down",
        "vwap", "adj_vwap",
    ],
    "signal_only": [
        "code", "date", "adj_close", "volume", "float_mv",
        "is_st", "trade_status", "industry_l1",
    ],
}


BACKTEST_COLUMN_DOCS: dict[str, str] = {
    "factor_backtest": (
        "T 日选股 + T+1 成交回测。"
        "T 日用 adj_close/volume/is_st/trade_status 算因子并过滤；"
        "T+1 日用 adj_vwap/open/close 成交；"
        "IC/分层收益用 fwd_ret（adj_vwap 价格序列，小数口径），不含 pct_change。"
        "涨跌停用库中 limit_up/limit_down 与 open/close 比较。"
    ),
    "signal_only": "仅 IC/分组检验，不含成交价",
}


__all__ = [
    "SCHEMA",
    "TRADE_STATUS_DOCS",
    "CDR_CODES",
    "CDR_NOTES",
    "DERIVED_SCHEMA",
    "COLUMN_RENAME_MAP",
    "COLUMN_DOCS",
    "OUTPUT_COLUMNS",
    "BACKTEST_EXCLUDE_COLUMNS",
    "BACKTEST_COLUMN_GROUPS",
    "BACKTEST_PRESETS",
    "BACKTEST_COLUMN_DOCS",
]
