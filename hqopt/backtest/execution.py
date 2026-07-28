"""状态化 T+1 成交账本。

同一账本同时供逐期优化和回测使用，确保下一期优化看到的是实际成交持仓。
每个目标最多在 T+1、T+2、T+3 三个交易日执行；单只股票完成成交后即锁定，
不会因后续价格漂移被重复交易。T 日停牌股票可在提交目标时冻结股数。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from hqopt.constants import LIMIT_TOL

MAX_EXECUTION_ATTEMPTS = 3
LEGACY_SELL_ONLY_MANIFEST_VERSION = 1
SELL_ONLY_MANIFEST_VERSION = 2
SUPPORTED_SELL_ONLY_MANIFEST_VERSIONS = {
    LEGACY_SELL_ONLY_MANIFEST_VERSION,
    SELL_ONLY_MANIFEST_VERSION,
}


def sell_only_path_for_weights(weight_path: str | Path) -> Path:
    """返回目标权重对应的只卖不买元数据文件路径。"""
    path = Path(weight_path)
    return path.with_name(f"{path.stem}.sell_only.parquet")


def sell_only_manifest_path_for_weights(weight_path: str | Path) -> Path:
    """返回权重与 sell-only 文件的内容绑定清单路径。"""
    path = Path(weight_path)
    return path.with_name(f"{path.stem}.sell_only.manifest.json")


def batch_execution_stats_path_for_weights(weight_path: str | Path) -> Path:
    """返回批量优化成交统计路径，并隔离同目录内的不同权重 bundle。"""
    weights = Path(weight_path)
    if weights.stem == "weights":
        return weights.parent / "batch_execution_stats.json"
    return weights.with_name(f"{weights.stem}.batch_execution_stats.json")


def bundle_in_progress_path_for_weights(weight_path: str | Path) -> Path:
    """返回批量 bundle 发布中的 fail-closed 标记路径。"""
    path = Path(weight_path)
    return path.with_name(f"{path.stem}.bundle.in_progress")


def _bundle_lock_path(weight_path: str | Path) -> Path:
    """返回跨进程读写锁路径；放在用户专属临时目录，兼容只读权重目录。"""
    canonical = str(Path(weight_path).expanduser().resolve(strict=False))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    root = Path(tempfile.gettempdir()) / f"hqopt-bundle-locks-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{digest}.lock"


@contextmanager
def bundle_io_lock(
    weight_path: str | Path,
    *,
    exclusive: bool,
) -> Iterator[None]:
    """为同一权重路径提供跨进程共享读锁或独占发布锁。"""
    lock_path = _bundle_lock_path(weight_path)
    with lock_path.open("a+b") as stream:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(stream.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalidate_sell_only_manifest(weight_path: str | Path) -> None:
    """在改写权重 bundle 前使旧清单失效，避免中断时误读陈旧 sidecar。"""
    sell_only_manifest_path_for_weights(weight_path).unlink(missing_ok=True)


def mark_bundle_in_progress(weight_path: str | Path) -> Path:
    """原子创建发布中标记；读取端只看标记是否存在并 fail-closed。"""
    weights = Path(weight_path)
    marker = bundle_in_progress_path_for_weights(weights)
    temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": 1,
        "weights_file": weights.name,
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def clear_bundle_in_progress(weight_path: str | Path) -> None:
    """成功发布完整 manifest 后删除发布中标记。"""
    bundle_in_progress_path_for_weights(weight_path).unlink(missing_ok=True)


def ensure_bundle_not_in_progress(weight_path: str | Path) -> None:
    """拒绝读取正在发布或曾中断发布的批量 bundle。"""
    marker = bundle_in_progress_path_for_weights(weight_path)
    if marker.exists():
        raise ValueError(f"权重 bundle 发布未完成，存在 in-progress 标记：{marker}")


def write_sell_only_manifest(
    weight_path: str | Path,
    *,
    batch_execution_stats_path: str | Path | None = None,
) -> Path:
    """原子发布内容绑定清单。

    未传成交统计时写 v1，兼容自行提供 weights+sidecar 的外部权重；
    批量优化必须传成交统计并写 v2，将三项产物绑定为同一 bundle。
    """
    weights = Path(weight_path)
    sell_only = sell_only_path_for_weights(weights)
    if not weights.is_file() or not sell_only.is_file():
        raise FileNotFoundError("写入 sell-only 清单前，权重文件和 sidecar 必须同时存在")

    payload = {
        "schema_version": LEGACY_SELL_ONLY_MANIFEST_VERSION,
        "weights_file": weights.name,
        "weights_sha256": _file_sha256(weights),
        "sell_only_file": sell_only.name,
        "sell_only_sha256": _file_sha256(sell_only),
    }
    if batch_execution_stats_path is not None:
        stats = Path(batch_execution_stats_path)
        expected_stats = batch_execution_stats_path_for_weights(weights)
        if stats != expected_stats:
            raise ValueError(
                "批量成交统计路径必须与权重位于同一输出目录且命名为 "
                f"{expected_stats.name}"
            )
        if not stats.is_file():
            raise FileNotFoundError(f"写入 v2 清单前缺少批量成交统计：{stats}")
        payload.update(
            {
                "schema_version": SELL_ONLY_MANIFEST_VERSION,
                "batch_execution_stats_file": stats.name,
                "batch_execution_stats_sha256": _file_sha256(stats),
            }
        )
    manifest = sell_only_manifest_path_for_weights(weights)
    temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def validate_sell_only_manifest(
    weight_path: str | Path,
    *,
    allow_in_progress: bool = False,
) -> Path:
    """验证清单文件名与内容哈希，拒绝错配或写入中断后的 sidecar。"""
    weights = Path(weight_path)
    if not allow_in_progress:
        ensure_bundle_not_in_progress(weights)
    sell_only = sell_only_path_for_weights(weights)
    manifest = sell_only_manifest_path_for_weights(weights)
    if not manifest.is_file():
        raise ValueError(f"sell-only 元数据缺少内容绑定清单：{manifest}")
    if not weights.is_file() or not sell_only.is_file():
        raise ValueError("sell-only 清单存在，但权重文件或 sidecar 缺失")

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"sell-only 内容绑定清单无法读取：{manifest}") from exc

    expected_names = {
        "weights_file": weights.name,
        "sell_only_file": sell_only.name,
    }
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SELL_ONLY_MANIFEST_VERSIONS:
        raise ValueError(f"sell-only 内容绑定清单版本不支持：{manifest}")
    for key, expected in expected_names.items():
        if payload.get(key) != expected:
            raise ValueError(f"sell-only 内容绑定清单文件名不匹配：{manifest}")

    expected_hashes = {
        "weights_sha256": _file_sha256(weights),
        "sell_only_sha256": _file_sha256(sell_only),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            raise ValueError(f"sell-only 内容绑定清单哈希不匹配：{manifest}")
    if schema_version == SELL_ONLY_MANIFEST_VERSION:
        stats = batch_execution_stats_path_for_weights(weights)
        if payload.get("batch_execution_stats_file") != stats.name:
            raise ValueError(f"sell-only 内容绑定清单文件名不匹配：{manifest}")
        if not stats.is_file():
            raise ValueError(f"sell-only v2 清单存在，但批量成交统计缺失：{stats}")
        if payload.get("batch_execution_stats_sha256") != _file_sha256(stats):
            raise ValueError(f"sell-only 内容绑定清单哈希不匹配：{manifest}")
    return manifest


def normalize_sell_only_matrix(
    frame: pd.DataFrame,
    *,
    context: str = "sell-only 元数据",
) -> pd.DataFrame:
    """校验 sell-only 矩阵的索引、列和值，并统一为布尔宽表。"""
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.index.name = "date"
    if not normalized.index.is_unique:
        raise ValueError(f"{context}日期不唯一")
    if not normalized.columns.is_unique:
        raise ValueError(f"{context}股票列不唯一")
    for column in normalized.columns:
        values = normalized[column]
        is_bool = pd.api.types.is_bool_dtype(values.dtype)
        is_binary_number = (
            pd.api.types.is_numeric_dtype(values.dtype)
            and values.notna().all()
            and values.isin([0, 1]).all()
        )
        if values.isna().any() or not (is_bool or is_binary_number):
            raise ValueError(
                f"{context}只能包含无缺失的布尔值或 0/1：列={column}"
            )
    return normalized.astype(bool).sort_index()


def align_sell_only_matrix(
    metadata: pd.DataFrame,
    weight_df: pd.DataFrame,
    *,
    context: str = "sell-only 元数据",
) -> pd.DataFrame:
    """严格按完整调仓日期和股票列绑定 sell-only 矩阵与权重。"""
    normalized = normalize_sell_only_matrix(metadata, context=context)
    weight_dates = pd.DatetimeIndex(pd.to_datetime(weight_df.index))
    if not weight_dates.is_unique:
        raise ValueError("权重文件调仓日期不唯一，无法绑定 sell-only 元数据")
    if not weight_df.columns.is_unique:
        raise ValueError("权重文件股票列不唯一，无法绑定 sell-only 元数据")

    metadata_dates = pd.DatetimeIndex(normalized.index)
    missing_dates = weight_dates.difference(metadata_dates)
    extra_dates = metadata_dates.difference(weight_dates)
    missing_columns = weight_df.columns.difference(normalized.columns)
    extra_columns = normalized.columns.difference(weight_df.columns)
    if not missing_dates.empty or not extra_dates.empty:
        raise ValueError(
            f"{context}日期与权重文件不一致："
            f"缺少={list(missing_dates.strftime('%Y-%m-%d'))}，"
            f"多出={list(extra_dates.strftime('%Y-%m-%d'))}"
        )
    if not missing_columns.empty or not extra_columns.empty:
        raise ValueError(
            f"{context}股票列与权重文件不一致："
            f"缺少={missing_columns.tolist()}，多出={extra_columns.tolist()}"
        )
    return normalized.loc[weight_dates, weight_df.columns]


def publish_batch_bundle(
    weight_df: pd.DataFrame,
    sell_only_df: pd.DataFrame,
    execution_stats: Mapping[str, Any],
    weight_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """原子暂存并 fail-closed 发布批量优化 bundle。

    暂存阶段不触碰旧 bundle；正式替换开始后先创建 in-progress 标记，
    manifest 最后发布且通过内部校验后才删除标记。任何中断都会阻止读取。
    """
    weights = Path(weight_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    sell_only = sell_only_path_for_weights(weights)
    stats = batch_execution_stats_path_for_weights(weights)
    aligned_sell_only = align_sell_only_matrix(sell_only_df, weight_df)
    # 校验时统一成 DatetimeIndex，落盘则保持权重文件原索引类型，兼容既有消费者。
    aligned_sell_only.index = weight_df.index.copy()
    aligned_sell_only.index.name = weight_df.index.name

    token = uuid.uuid4().hex
    weight_temporary = weights.with_name(
        f".{weights.stem}.{token}.tmp{weights.suffix}"
    )
    sell_only_temporary = sell_only.with_name(
        f".{sell_only.stem}.{token}.tmp{sell_only.suffix}"
    )
    stats_temporary = stats.with_name(f".{stats.stem}.{token}.tmp{stats.suffix}")
    temporary_paths = (weight_temporary, sell_only_temporary, stats_temporary)

    try:
        weight_df.to_parquet(weight_temporary)
        aligned_sell_only.to_parquet(sell_only_temporary)
        stats_temporary.write_text(
            json.dumps(
                dict(execution_stats),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=lambda value: value.item(),
            )
            + "\n",
            encoding="utf-8",
        )

        with bundle_io_lock(weights, exclusive=True):
            mark_bundle_in_progress(weights)
            invalidate_sell_only_manifest(weights)
            sell_only_temporary.replace(sell_only)
            stats_temporary.replace(stats)
            weight_temporary.replace(weights)
            manifest = write_sell_only_manifest(
                weights,
                batch_execution_stats_path=stats,
            )
            validate_sell_only_manifest(weights, allow_in_progress=True)
            clear_bundle_in_progress(weights)
        return weights, sell_only, stats, manifest
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _valid_price(value: object) -> float:
    """将有效正价格转为 float，无效值返回 0。"""
    if value is None or pd.isna(value):
        return 0.0
    price = float(value)
    return price if price > 0 else 0.0


class OrderState(str, Enum):
    """单只股票在当前目标中的执行状态。"""

    FROZEN = "frozen"
    PENDING_SELL = "pending_sell"
    PENDING_BUY = "pending_buy"
    FILLED = "filled"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ExecutionDayResult:
    """单日成交结果。"""

    turnover: float = 0.0
    buy_fail_count: int = 0
    sell_defer_count: int = 0
    target_pending: bool = False
    filled_count: int = 0
    pending_count: int = 0
    expired_count: int = 0
    expired_notional: float = 0.0
    attempt_number: int = 0


class ExecutionLedger:
    """维护股票份额、现金、估值价格和当前目标的股票级订单状态。"""

    def __init__(
        self,
        initial_value: float,
        cost_buy: float = 0.001,
        cost_sell: float = 0.002,
        min_notional: float = 1.0,
        max_attempts: int = MAX_EXECUTION_ATTEMPTS,
    ) -> None:
        if initial_value <= 0:
            raise ValueError("initial_value 必须为正数")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, Integral)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts 必须为正整数")
        self.cash = float(initial_value)
        self.cost_buy = float(cost_buy)
        self.cost_sell = float(cost_sell)
        self.min_notional = float(min_notional)
        self.max_attempts = int(max_attempts)
        self.shares: dict[str, float] = {}
        self.last_price: dict[str, float] = {}

        self.pending_target: pd.Series | None = None
        self.pending_tickers: set[str] = set()
        self.frozen_tickers: set[str] = set()
        self.sell_only_tickers: set[str] = set()
        self.order_states: dict[str, OrderState] = {}
        self.target_attempts = 0

        self.buy_fail_count = 0
        self.sell_defer_count = 0
        self.expired_order_count = 0
        self.expired_notional = 0.0

    def submit_target(
        self,
        target_weight: pd.Series,
        *,
        frozen_tickers: Collection[str] = (),
        sell_only_tickers: Collection[str] = (),
    ) -> None:
        """提交新目标并替换旧目标；从下一次 ``step`` 开始执行。

        ``frozen_tickers`` 来自信号日 T 的停牌状态。它们在本目标生命周期内不生成
        任何订单，即使 T+1 已复牌，持仓股数也保持不变。
        ``sell_only_tickers`` 可随最新 NAV 重算卖出量，但绝不允许转为买单。
        """
        target = pd.to_numeric(target_weight, errors="coerce").fillna(0.0).astype(float)
        target.index = target.index.map(str)
        target = target.clip(lower=0.0)
        if not target.index.is_unique:
            target = target.groupby(level=0).sum()
        total = float(target.sum())
        if total > 1.0 + 1e-10:
            raise ValueError(f"目标权重和不能超过 1，当前为 {total:.8f}")

        candidates = set(target[target > 0.0].index)
        candidates.update(ticker for ticker, shares in self.shares.items() if shares > 1e-10)
        frozen = {str(ticker) for ticker in frozen_tickers} & candidates
        sell_only = (
            {str(ticker) for ticker in sell_only_tickers} & candidates
        ) - frozen

        current_weight = self.actual_weights()
        self.pending_target = target
        self.frozen_tickers = frozen
        self.sell_only_tickers = sell_only
        self.pending_tickers = candidates - frozen
        self.order_states = dict.fromkeys(frozen, OrderState.FROZEN)
        for ticker in self.pending_tickers:
            target_w = float(target.get(ticker, 0.0))
            current_w = float(current_weight.get(ticker, 0.0))
            self.order_states[ticker] = (
                OrderState.PENDING_BUY
                if ticker not in sell_only and target_w >= current_w
                else OrderState.PENDING_SELL
            )
        self.target_attempts = 0

        if not self.pending_tickers:
            self.pending_target = None

    def cancel_pending_target(self) -> None:
        """在新调仓日取消旧目标，不成交、不计过期并重置目标级状态。"""
        self.pending_target = None
        self.pending_tickers.clear()
        self.frozen_tickers.clear()
        self.sell_only_tickers.clear()
        self.order_states.clear()
        self.target_attempts = 0

    def _update_marks(self, adj_close: pd.Series) -> None:
        for ticker, value in adj_close.items():
            price = _valid_price(value)
            if price > 0:
                self.last_price[str(ticker)] = price

    def mark_to_market(self, adj_close: pd.Series) -> None:
        """仅更新估值价格，不执行或消耗当前目标的尝试次数。"""
        self._update_marks(adj_close)

    def position_values(self, prices: pd.Series | None = None) -> pd.Series:
        """计算持仓市值；传入价格时优先使用，否则回退到最近有效收盘价。"""
        values = {}
        for ticker, shares in self.shares.items():
            if shares <= 1e-10:
                continue
            price = _valid_price(prices.get(ticker)) if prices is not None else 0.0
            if price <= 0:
                price = self.last_price.get(ticker, 0.0)
            if price > 0:
                values[ticker] = shares * price
        return pd.Series(values, dtype=float)

    @property
    def nav(self) -> float:
        return float(self.cash + self.position_values().sum())

    def actual_weights(self) -> pd.Series:
        """返回实际股票权重；现金不伪装成股票，因此权重和允许小于 1。"""
        nav = self.nav
        if nav <= 1e-12:
            return pd.Series(dtype=float, name="actual_weight")
        return (self.position_values() / nav).rename("actual_weight")

    @staticmethod
    def _at_limit(
        ticker: str,
        close_raw: pd.Series,
        limit_price: pd.Series,
        side: str,
    ) -> bool:
        close = _valid_price(close_raw.get(ticker))
        limit = _valid_price(limit_price.get(ticker))
        if close <= 0 or limit <= 0:
            return False
        if side == "up":
            return close >= limit * (1 - LIMIT_TOL)
        return close <= limit * (1 + LIMIT_TOL)

    def _delta_for(
        self,
        tickers: Collection[str],
        prices: pd.Series,
    ) -> tuple[pd.Series, float]:
        """按当日 VWAP 和最新 NAV 计算指定待处理股票相对原目标的差额。"""
        current_values = self.position_values(prices)
        total_val = float(self.cash + current_values.sum())
        index = pd.Index(sorted(tickers), dtype=object)
        if self.pending_target is None or index.empty:
            return pd.Series(dtype=float), total_val
        target_values = self.pending_target.reindex(index, fill_value=0.0) * total_val
        current = current_values.reindex(index, fill_value=0.0)
        return target_values - current, total_val

    def _mark_filled(self, ticker: str) -> None:
        self.pending_tickers.discard(ticker)
        self.order_states[ticker] = OrderState.FILLED

    def _finish_target(self) -> None:
        self.pending_target = None
        self.pending_tickers.clear()
        self.frozen_tickers.clear()
        self.sell_only_tickers.clear()

    def _expire_pending(self, prices: pd.Series) -> tuple[int, float]:
        delta, _ = self._delta_for(self.pending_tickers, prices)
        expired = list(self.pending_tickers)
        notional = float(delta.abs().sum())
        for ticker in expired:
            self.order_states[ticker] = OrderState.EXPIRED
        self.expired_order_count += len(expired)
        self.expired_notional += notional
        self._finish_target()
        return len(expired), notional

    def step(
        self,
        *,
        adj_close: pd.Series,
        adj_vwap: pd.Series,
        close_raw: pd.Series,
        limit_up: pd.Series,
        limit_down: pd.Series,
        trade_status: pd.Series,
    ) -> ExecutionDayResult:
        """推进一个交易日，只执行仍为 pending 的股票。"""
        self._update_marks(adj_close)
        if self.pending_target is None or not self.pending_tickers:
            return ExecutionDayResult()

        self.target_attempts += 1
        attempt_number = self.target_attempts
        delta, total_val = self._delta_for(self.pending_tickers, adj_vwap)
        if total_val <= 1e-12:
            self._finish_target()
            return ExecutionDayResult(attempt_number=attempt_number)

        def suspended(ticker: str) -> bool:
            return trade_status.get(ticker) == "停牌"

        def exec_price(ticker: str) -> float:
            return _valid_price(adj_vwap.get(ticker))

        filled_count = 0
        sell_total = 0.0
        buy_total = 0.0
        deferred_sells = 0
        failed_buys = 0

        # 先卖：每成功卖出一只就按最新 NAV 重算。卖费可能使原待买单转为待卖单，
        # 尤其在 T+3 必须同日继续卖完，不能直接过期。
        blocked_sells: set[str] = set()
        while self.pending_tickers:
            sell_delta, _ = self._delta_for(self.pending_tickers, adj_vwap)
            for ticker, value in sell_delta.items():
                value = float(value)
                if abs(value) <= self.min_notional:
                    self._mark_filled(ticker)
                    filled_count += 1
                elif value < 0:
                    self.order_states[ticker] = OrderState.PENDING_SELL
                elif ticker in self.sell_only_tickers:
                    self._mark_filled(ticker)
                    filled_count += 1
                else:
                    self.order_states[ticker] = OrderState.PENDING_BUY

            sold_one = False
            for ticker, value in sell_delta.items():
                if (
                    ticker not in self.pending_tickers
                    or ticker in blocked_sells
                    or self.order_states[ticker] != OrderState.PENDING_SELL
                    or value >= -self.min_notional
                ):
                    continue
                price = exec_price(ticker)
                blocked = (
                    suspended(ticker)
                    or self._at_limit(ticker, close_raw, limit_down, "down")
                    or price <= 0
                )
                if blocked:
                    blocked_sells.add(ticker)
                    continue

                held_shares = self.shares.get(ticker, 0.0)
                shares_to_sell = min(-float(value) / price, held_shares)
                if shares_to_sell > 1e-10:
                    self.shares[ticker] = held_shares - shares_to_sell
                    proceeds = shares_to_sell * price
                    self.cash += proceeds * (1.0 - self.cost_sell)
                    sell_total += proceeds
                self._mark_filled(ticker)
                filled_count += 1
                sold_one = True
                break

            if not sold_one:
                break
        deferred_sells = len(blocked_sells)

        # 卖出完成后，用真实现金和最新 NAV 重新计算剩余 pending 股票的买入差额。
        delta_after_sell, _ = self._delta_for(self.pending_tickers, adj_vwap)
        tradable_buys: dict[str, float] = {}
        for ticker, value in delta_after_sell.items():
            value = float(value)
            if abs(value) <= self.min_notional:
                self._mark_filled(ticker)
                filled_count += 1
                continue
            if value < 0:
                self.order_states[ticker] = OrderState.PENDING_SELL
                continue
            if ticker in self.sell_only_tickers:
                self._mark_filled(ticker)
                filled_count += 1
                continue

            self.order_states[ticker] = OrderState.PENDING_BUY
            price = exec_price(ticker)
            blocked = (
                suspended(ticker)
                or self._at_limit(ticker, close_raw, limit_up, "up")
                or price <= 0
            )
            if blocked:
                failed_buys += 1
            else:
                tradable_buys[ticker] = value

        buy_demand = float(sum(tradable_buys.values()))
        affordable = self.cash / (1.0 + self.cost_buy)
        scale = min(1.0, affordable / buy_demand) if buy_demand > 0 else 0.0
        executed_buys: set[str] = set()
        for ticker, buy_value in tradable_buys.items():
            actual_buy = buy_value * scale
            price = exec_price(ticker)
            if actual_buy < self.min_notional or price <= 0:
                continue
            shares_bought = actual_buy / price
            self.shares[ticker] = self.shares.get(ticker, 0.0) + shares_bought
            self.last_price.setdefault(ticker, price)
            self.cash -= actual_buy * (1.0 + self.cost_buy)
            buy_total += actual_buy
            executed_buys.add(ticker)

        if self.cash < -self.min_notional:
            raise RuntimeError(f"成交后现金为负：{self.cash:.2f}")
        self.cash = max(self.cash, 0.0)
        self.shares = {ticker: shares for ticker, shares in self.shares.items() if shares > 1e-10}

        # 完整买单直接完成；同比例部分成交只有真正达到目标后才完成。
        if executed_buys:
            post_delta, _ = self._delta_for(executed_buys, adj_vwap)
            for ticker in executed_buys:
                full_order = scale >= 1.0 - 1e-12
                reached_target = (
                    float(post_delta.get(ticker, 0.0)) <= self.min_notional
                )
                if full_order or reached_target:
                    self._mark_filled(ticker)
                    filled_count += 1

        expired_count = 0
        expired_notional = 0.0
        if not self.pending_tickers:
            self._finish_target()
        elif attempt_number >= self.max_attempts:
            expired_count, expired_notional = self._expire_pending(adj_vwap)

        self.buy_fail_count += failed_buys
        self.sell_defer_count += deferred_sells
        turnover = (sell_total + buy_total) / total_val
        return ExecutionDayResult(
            turnover=float(turnover),
            buy_fail_count=failed_buys,
            sell_defer_count=deferred_sells,
            target_pending=self.pending_target is not None,
            filled_count=filled_count,
            pending_count=len(self.pending_tickers),
            expired_count=expired_count,
            expired_notional=expired_notional,
            attempt_number=attempt_number,
        )
