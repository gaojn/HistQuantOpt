"""
候选池过滤与合成 Alpha 生成。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from hqopt.data.generator import MarketSnapshot


def build_cost_vector(
    tickers: list[str],
    panel: pl.DataFrame,
    target_date: date,
    lookback: int = 20,
) -> np.ndarray:
    """
    计算个股冲击成本代理向量，归一化使中位数 = 1。

    公式：c_i = σ_i / sqrt(ADV_i)

    来源：Almgren-Chriss 冲击模型，单笔冲击成本 ≈ σ × sqrt(Q/ADV)。
    在下单量占组合比例固定时，c_i ∝ σ_i / sqrt(ADV_i)：
    波动大或流动性差的股票冲击成本更高。

    Parameters
    ----------
    tickers     : 目标股票列表
    panel       : 行情面板（需含 adj_close、amount、date、code）
    target_date : 调仓日（取该日及之前 lookback 个交易日）
    lookback    : 滚动窗口（交易日数），默认 20

    Returns
    -------
    np.ndarray, shape (N,)
        归一化成本权重，中位数=1，缺失/异常值填 1.0（等权）
    """
    # 取目标日之前（含）lookback 个交易日
    avail_dates = sorted(
        panel.filter(pl.col("date") <= target_date)
        .select("date").unique()["date"].to_list()
    )
    window_dates = avail_dates[-lookback:]
    if len(window_dates) < 5:
        return np.ones(len(tickers))

    sub = (
        panel.filter(
            (pl.col("date").is_in(window_dates)) &
            (pl.col("code").is_in(tickers))
        )
        .select(["date", "code", "adj_close", "amount"])
        .to_pandas()
        .pivot(index="date", columns="code", values=["adj_close", "amount"])
        .sort_index()
    )

    adj   = sub["adj_close"].reindex(columns=tickers)
    amt   = sub["amount"].reindex(columns=tickers)

    # 年化波动率（20日滚动标准差 × √252）
    daily_ret = adj.pct_change(fill_method=None)
    vol = daily_ret.std(ddof=1) * np.sqrt(252)          # pd.Series, index=ticker

    # ADV：窗口内日均成交额（千元）
    adv = amt.mean()                                     # pd.Series, index=ticker

    # c_i = σ_i / sqrt(ADV_i)，对零/NaN 做保护
    adv_safe = adv.clip(lower=1.0)
    c_raw = vol / np.sqrt(adv_safe)

    # 归一化：除以中位数，使中位数股票成本权重 = 1
    median = c_raw.median()
    if median > 1e-12:
        c_raw = c_raw / median

    # 缺失/异常 → 填 1.0（等权，不额外惩罚）
    c_raw = c_raw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    c_raw = c_raw.clip(lower=0.1, upper=10.0)          # 防极端值

    return c_raw.reindex(tickers).fillna(1.0).values.astype(float)


def filter_universe(
    snapshot: MarketSnapshot,
    panel: pl.DataFrame,
    target_date: date,
    exclude_bj: bool = True,
    exclude_st: bool = True,
    top_n: int | None = None,
    prev_holdings: pd.Series | None = None,
) -> MarketSnapshot:
    """
    候选池过滤：剔除北交所、ST，可选按市值截取 TOP_N。
    并将掉出候选池但有持仓（且当日有行情）的票携带进优化域，
    标记 sell_only=True（只卖不买）。

    Parameters
    ----------
    snapshot      : 原始市场快照（含全市场行情）
    panel         : 行情面板（含 is_st 字段）
    target_date   : 调仓日
    exclude_bj    : 是否剔除北交所（.BJ 结尾）
    exclude_st    : 是否剔除 ST 股票
    top_n         : 按自由流通市值取前 N 只，None = 不限制
    prev_holdings : 优化器上期持仓权重（index=ticker，value=weight）；
                    None 时不做 carry 处理（行为与旧版完全一致）
    """
    tickers = snapshot.tickers
    keep = list(tickers)

    if exclude_bj:
        keep = [t for t in keep if not t.endswith(".BJ")]

    if exclude_st:
        st_df = (
            panel.filter(pl.col("date") == target_date)
            .select(["code", "is_st"])
            .to_pandas()
            .set_index("code")
        )
        st_set = set(st_df[st_df["is_st"] == 1].index)
        keep = [t for t in keep if t not in st_set]

    if top_n is not None and len(keep) > top_n:
        cap = snapshot.market_cap.reindex(keep).fillna(0.0)
        keep = cap.nlargest(top_n).index.tolist()

    # carry 逻辑：掉出候选池但有持仓且当日有行情的票并回优化域
    # 真退市票（当日 snapshot 无行情行）自然不在 snapshot.tickers 里，不并回
    if prev_holdings is not None:
        snapshot_ticker_set = set(tickers)
        keep_set = set(keep)
        carry = [
            t for t in prev_holdings.index
            if prev_holdings[t] > 1e-8
            and t not in keep_set
            and t in snapshot_ticker_set
        ]
    else:
        carry = []

    all_tickers = keep + carry

    # sell_only：carry 票标 True，正常候选票标 False；无 carry 时为 None（兼容旧行为）
    if carry:
        carry_flags = pd.Series(False, index=all_tickers, dtype=bool)
        for t in carry:
            carry_flags[t] = True
        sell_only: pd.Series | None = carry_flags
    else:
        sell_only = None

    return replace(
        snapshot,
        tickers=all_tickers,
        industry=snapshot.industry.reindex(all_tickers),
        adv=snapshot.adv.reindex(all_tickers),
        status=snapshot.status.reindex(all_tickers),
        prev_weight=snapshot.prev_weight.reindex(all_tickers).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(all_tickers),
        is_constituent=(
            snapshot.is_constituent.reindex(all_tickers)
            if snapshot.is_constituent is not None else None
        ),
        sell_only=sell_only,
    )


# 风格对冲候选：每个风格因子正/负载荷两端各保留的股票数。
# 60 只 × weight_upper 1~5% 提供 0.6~3.0 的单因子暴露调节容量，
# 远超 ±0.2σ~±0.6σ 约束所需；实测该值下瘦身解与全池解逐票一致。
_STYLE_HEDGE_PER_TAIL = 60


def candidate_pool_mask(
    alpha: np.ndarray,
    snapshot: MarketSnapshot,
    prev_weight: np.ndarray | None,
    benchmark_weight: np.ndarray | None,
    top_m: int,
    style_loading: pd.DataFrame | None = None,
) -> np.ndarray:
    """候选池瘦身掩码：alpha top-M ∪ 约束相关股票。

    纯多头 + 线性 alpha + 凸惩罚下，深度负 alpha 的股票不会进入最优解的
    支撑集；但**约束相关**股票即使 alpha 平庸也可能被最优解选中，必须
    显式保留，否则瘦身会悄悄改变问题：

    - 当前持仓：卖出/冻结/只卖约束都挂在它身上；
    - 基准权重非零（指增）：主动偏离与成分下限约束才有意义；
    - 成分股按 alpha 取前 ``0.6·M``：保证 ``min_constituent_ratio`` 有充足
      容量（0.6M × weight_upper 远大于 40%/80% 下限）；
    - 风格载荷两端极值股票：绝对风格约束（如 Size ≤ ±0.2σ）是紧约束时，
      最优解会持有 alpha 排名靠后的「对冲票」把组合暴露拉回界内——实测
      漏掉它们会造成 ~4% 权重、0.16 L1 的解漂移。

    注意：传入的 ``alpha`` 应是全池截面 z-score；**瘦身后不要重新标准化**——
    重新标准化会改变 alpha 与风险/换手惩罚的量纲耦合，破坏与全池解的等价性。
    """
    if top_m <= 0:
        raise ValueError(f"top_m 必须为正，收到 {top_m}")
    rank = pd.Series(alpha).rank(ascending=False, method="first").values
    keep = rank <= top_m
    if prev_weight is not None:
        keep |= np.asarray(prev_weight, dtype=float) > 1e-12
    if benchmark_weight is not None:
        keep |= np.asarray(benchmark_weight, dtype=float) > 1e-10
    cmask = snapshot.constituent_mask
    if cmask.any() and not cmask.all():
        const_rank = (
            pd.Series(np.where(cmask, alpha, -np.inf))
            .rank(ascending=False, method="first")
            .values
        )
        keep |= (const_rank <= int(top_m * 0.6)) & cmask
    if style_loading is not None:
        loadings = style_loading.reindex(snapshot.tickers).fillna(0.0)
        for column in loadings.columns:
            order = np.argsort(loadings[column].values)
            keep[order[:_STYLE_HEDGE_PER_TAIL]] = True
            keep[order[-_STYLE_HEDGE_PER_TAIL:]] = True
    return keep


def subset_snapshot(snapshot: MarketSnapshot, keep_idx: np.ndarray) -> MarketSnapshot:
    """按位置索引切取快照子集（保持原有顺序，全部字段按 ticker 重对齐）。"""
    keep_tickers = [snapshot.tickers[i] for i in keep_idx]
    return replace(
        snapshot,
        tickers=keep_tickers,
        industry=snapshot.industry.reindex(keep_tickers),
        adv=snapshot.adv.reindex(keep_tickers),
        status=snapshot.status.reindex(keep_tickers),
        prev_weight=snapshot.prev_weight.reindex(keep_tickers).fillna(0.0),
        market_cap=snapshot.market_cap.reindex(keep_tickers),
        is_constituent=(
            snapshot.is_constituent.reindex(keep_tickers)
            if snapshot.is_constituent is not None else None
        ),
        sell_only=(
            snapshot.sell_only.reindex(keep_tickers).fillna(False)
            if snapshot.sell_only is not None else None
        ),
    )


def build_synthetic_alpha(
    panel: pl.DataFrame,
    fwd_days: int = 5,
    ic_mean: float = 0.08,
    ic_std: float = 0.10,
    decay: float = 0.80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    ⚠️ 警告：合成 Alpha 使用未来收益（shift(-fwd_days)）构造，含前视信息。
    绝不可用于实盘交易或真实业绩评估！仅用于流程验证与功能测试。

    生成合成 Alpha 矩阵。

    Returns
    -------
    pd.DataFrame  index=date, columns=ticker
    """
    rng = np.random.default_rng(seed)
    adj = (
        panel.select(["date", "code", "adj_close"]).to_pandas()
        .pivot(index="date", columns="code", values="adj_close").sort_index()
    )
    fwd_ret = adj.shift(-fwd_days) / adj - 1
    dates = fwd_ret.index[fwd_ret.notna().sum(axis=1) > 50]

    rows: dict = {}
    f_prev: pd.Series | None = None
    for dt in dates:
        r = fwd_ret.loc[dt].dropna()
        if len(r) < 50:
            continue
        mu, sig = r.mean(), r.std()
        if sig < 1e-8:
            continue
        z_r = (r - mu) / sig
        rho = float(np.clip(rng.normal(ic_mean, ic_std), -0.95, 0.95))
        eps = rng.standard_normal(len(r))
        new_sig = pd.Series(
            rho * z_r.values + np.sqrt(max(1 - rho**2, 0)) * eps,
            index=r.index,
        )
        new_sig = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        if f_prev is None or decay == 0.0:
            f = new_sig
        else:
            common = f_prev.index.intersection(new_sig.index)
            f = new_sig.copy()
            if len(common) > 0:
                f[common] = (
                    decay * f_prev[common]
                    + np.sqrt(max(1 - decay**2, 0)) * new_sig[common]
                )
        f = (f - f.mean()) / (f.std() + 1e-10)
        f_prev = f
        rows[dt] = f

    alpha_df = pd.DataFrame(rows).T
    alpha_df.index.name = "date"
    return alpha_df


def _parse_dates_flexible(values) -> pd.Index:
    """
    将多种日期表示形式统一解析为 DatetimeIndex。

    支持：
      - pandas Timestamp / datetime64（原生）
      - datetime.date 对象
      - 日期字符串（如 "2024-01-02"、"2024/01/02"）
      - 整数/字符串形式的 YYYYMMDD（如 20240102），
        若按默认方式解析（视为纳秒时间戳）会出现明显错误的日期，
        因此对纯 8 位数字单独按 "%Y%m%d" 解析。
    """
    s = pd.Series(values)

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.DatetimeIndex(pd.to_datetime(s))

    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        ints = s.astype("Int64")
        if ints.notna().all() and ((ints >= 10_000_101) & (ints <= 99_991_231)).all():
            return pd.DatetimeIndex(
                pd.to_datetime(ints.astype(int).astype(str), format="%Y%m%d")
            )
        return pd.DatetimeIndex(pd.to_datetime(s))

    str_s = s.astype(str)
    if str_s.str.fullmatch(r"\d{8}").all():
        return pd.DatetimeIndex(pd.to_datetime(str_s, format="%Y%m%d"))

    try:
        return pd.DatetimeIndex(pd.to_datetime(s))
    except ValueError:
        return pd.DatetimeIndex(pd.to_datetime(s, format="mixed"))


# parquet schema metadata 键：因子文件的机器可读前视标记。
# 生成含前视信号的脚本必须写入 b"true"；加载端读到后强制其可信度声明
# 只能加严不能放松（见 alpha_panel_synthetic_marker 的调用方）。
ALPHA_SYNTHETIC_METADATA_KEY = b"hqopt.alpha.synthetic"


def save_alpha_panel(
    alpha_df: pd.DataFrame,
    path: str | Path,
    *,
    synthetic: bool,
) -> Path:
    """保存 Alpha 因子矩阵（长表或宽表），并嵌入机器可读的前视标记。

    ``synthetic=True`` 表示信号构造使用了未来信息（如 shift(-H) 未来收益）。
    该标记写入 parquet schema metadata，随文件流转；加载端
    （``_load_alpha_matrix``）读到 True 而配置声明 false 时会直接拒绝运行，
    防止 ``--alpha-file`` 把含前视的因子静默当作真实信号。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(alpha_df)
    metadata = dict(table.schema.metadata or {})
    metadata[ALPHA_SYNTHETIC_METADATA_KEY] = b"true" if synthetic else b"false"
    table = table.replace_schema_metadata(metadata)
    output = Path(path)
    pq.write_table(table, output)
    return output


def alpha_panel_synthetic_marker(path: str | Path) -> bool | None:
    """读取 Alpha parquet 的前视标记；无标记（历史文件/外部来源）返回 None。"""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(ALPHA_SYNTHETIC_METADATA_KEY)
    if raw is None:
        return None
    value = raw.decode("utf-8").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(
            f"Alpha 文件 {path} 的 {ALPHA_SYNTHETIC_METADATA_KEY!r} 标记非法：{raw!r}"
        )
    return value == "true"


def load_alpha_panel(path: str | Path) -> pd.DataFrame:
    """
    从 parquet 加载外部 Alpha 因子矩阵。

    支持两种格式：
      - 宽表：index=date，columns=ticker（与权重矩阵
        `weight_df.to_parquet()` 同一约定）
      - 长表：列为 (date, code, alpha)，自动 pivot 为宽表

    `date` 列/索引支持多种输入形式（Timestamp、datetime.date、
    日期字符串、YYYYMMDD 整数等），统一解析为 DatetimeIndex，
    详见 `_parse_dates_flexible`。

    值为因子分数（截面排序/打分皆可，优化器只用其截面相对大小）。

    Returns
    -------
    pd.DataFrame  index=date（DatetimeIndex），columns=ticker
    """
    alpha_df = pd.read_parquet(path)

    if set(alpha_df.columns) >= {"date", "code", "alpha"}:
        alpha_df["date"] = _parse_dates_flexible(alpha_df["date"])
        alpha_df = alpha_df.pivot(index="date", columns="code", values="alpha")
    else:
        alpha_df.index = _parse_dates_flexible(alpha_df.index)

    alpha_df.index.name = "date"
    return alpha_df.sort_index()


@dataclass(frozen=True)
class AlphaSlice:
    """某调仓日实际取用的 Alpha 截面及其可信度元数据。"""

    values: np.ndarray      # (N,) 对齐 tickers 的 alpha 向量
    as_of: date             # 实际取用的 alpha 面板日期（≤ 请求日）
    staleness_days: int     # 请求日 − as_of 的自然日数，0 表示当日信号
    n_valid: int            # 优化域内有非缺失 alpha 的股票数
    standardized: bool      # 是否已做截面 z-score


class AlphaZeroVarianceError(ValueError):
    """Alpha 截面没有横截面区分度，继续优化会静默退化为纯风险最小化。"""


def get_alpha_for_date(
    alpha_df: pd.DataFrame,
    target_date: date,
    tickers: list[str],
    *,
    max_staleness_days: int | None = None,
    standardize: bool = True,
    min_valid: int = 1,
) -> AlphaSlice | None:
    """取 ≤ target_date 的最近一期 Alpha，对齐到 tickers。

    Alpha 面板的 `date` 语义是**信号可得日**：T 日的行必须只用 T 日及之前的
    信息构造。优化器在 T 日收盘后据此下单、T+1 成交，因此取 `<= target_date`
    不构成前视；若研究员把"预测目标日"写进 date 列，则会引入前视——这个约定
    由 alpha 生产方保证，本函数不做（也无法做）检测。

    Parameters
    ----------
    max_staleness_days : int | None
        允许的最大陈旧自然日数。超过则返回 None，由调用方跳过该期，避免用
        早已过期的信号继续"赚钱"。None（默认）不做硬限制，仅在返回值的
        ``staleness_days`` 中如实记录，由调用方决定告警。
    standardize : bool
        是否对优化域内的 alpha 做截面 z-score，默认 True。目标函数
        ``w'α − γ·R(w)`` 里 α 与风险/成本系数量纲耦合：同一因子排序，α 乘以
        100 倍就能把 γ=0.05 下的 20 只分散组合压成 1 只全仓。标准化后
        ``risk_aversion`` / ``turnover_penalty`` / ``diversification_penalty``
        的默认标定才有稳定含义，缺失值填 0 也才等于"截面中性"。
    min_valid : int
        优化域内有效 alpha 的最少股票数，低于则视为该期不可用（返回 None）。
        默认 1，即只拒绝"该期 Alpha 对优化域零覆盖"——那会让 alpha 恒为 0、
        优化退化成纯风险最小化。覆盖偏低但非零的情形不在此拦截（小 universe 下
        属正常），由调用方按覆盖率告警。

    Returns
    -------
    AlphaSlice | None
        None 表示该期 Alpha 不可用（过于陈旧或有效值不足），调用方应跳过。

    Raises
    ------
    AlphaZeroVarianceError
        有效 Alpha 截面为常量（包括全零）。这种信号没有横截面信息，调用方应
        跳过该期；不得把全零向量继续送入优化器。
    ValueError
        Alpha 面板内不存在任何 ≤ target_date 的日期——通常是面板整体晚于
        回测区间的配置错误，静默返回全零会让优化退化成纯风险最小化。
    """
    ts = pd.Timestamp(target_date)
    avail = alpha_df.index[alpha_df.index <= ts]
    if len(avail) == 0:
        earliest = alpha_df.index.min()
        raise ValueError(
            f"Alpha 面板无 {target_date} 及之前的数据（最早 {earliest}）。"
            "请检查 alpha.path 与回测区间是否匹配——返回全零 alpha 会让优化器"
            "退化为纯风险最小化，业绩不可解释。"
        )

    as_of_ts = avail[-1]
    staleness = (ts - as_of_ts).days
    if max_staleness_days is not None and staleness > max_staleness_days:
        return None

    raw = alpha_df.loc[as_of_ts].reindex(tickers)
    raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_valid = int(raw.notna().sum())
    if n_valid < min_valid:
        return None

    sigma = float(raw.std(ddof=0))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise AlphaZeroVarianceError(
            f"{target_date} 的 Alpha 截面无区分度"
            f"（as_of={as_of_ts.date()}，有效股票={n_valid}）"
        )

    if standardize:
        mu = float(raw.mean())
        values = (raw - mu) / sigma
    else:
        values = raw

    return AlphaSlice(
        values=values.fillna(0.0).to_numpy(dtype=float),
        as_of=as_of_ts.date(),
        staleness_days=int(staleness),
        n_valid=n_valid,
        standardized=bool(standardize),
    )
