"""YAML 配置解析与 Alpha 可信度参数。

集中所有"把配置字典变成强类型运行参数"的逻辑，并在此处 fail-closed：策略名、
alpha.source、max_staleness_days 等非法值一律抛错，不做默认兜底——漏配比报错
更危险（如 alpha.source 缺失时误用前视合成信号）。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hqopt.pipeline.batch.types import _AlphaPolicy, _RunConfig

logger = logging.getLogger(__name__)

_INDEX_NAMES = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}
_STRATEGIES = {"index_enhance", "alpha_max"}
_ALPHA_SOURCES = {"file", "synthetic"}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Alpha 陈旧告警阈值下限（自然日），防止高频调仓时阈值过小而刷屏。
_MIN_ALPHA_STALENESS_WARN_DAYS = 7

def _parse_style_bound(v: Any) -> float | dict[str, float]:
    """解析风格约束：dict（按因子分别约束）或标量（统一）。"""
    if isinstance(v, dict):
        return {str(k): float(val) for k, val in v.items()}
    return float(v)


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    """解析可空数值配置；显式 0.0 必须保留。"""
    value = config.get(key)
    return None if value is None else float(value)


def _validate_alpha_config(alpha_cfg: dict[str, Any]) -> tuple[str, bool]:
    """校验 Alpha 来源与可信度元数据，返回 ``(source, synthetic)``。"""
    if not isinstance(alpha_cfg, dict):
        raise ValueError("alpha 配置必须是对象")
    if "source" not in alpha_cfg:
        raise ValueError(
            "配置缺少 alpha.source 字段。必须显式指定 'file' 或 'synthetic'"
        )
    source = alpha_cfg["source"]
    if source not in _ALPHA_SOURCES:
        raise ValueError(
            f"alpha.source 须为 {sorted(_ALPHA_SOURCES)} 之一，当前为 {source!r}"
        )
    if source == "file" and "synthetic" not in alpha_cfg:
        raise ValueError(
            "alpha.source=file 时必须显式设置 alpha.synthetic 为 true/false；"
            "框架不能从文件名或 --alpha-file 推断信号是否含前视"
        )
    synthetic = alpha_cfg.get("synthetic", source == "synthetic")
    if not isinstance(synthetic, bool):
        raise ValueError("alpha.synthetic 必须是布尔值 true/false")
    if source == "synthetic" and not synthetic:
        raise ValueError(
            "alpha.source=synthetic 与 alpha.synthetic=false 矛盾；"
            "合成 Alpha 必须标记为 synthetic=true"
        )
    return source, synthetic


def _synthetic_alpha_enabled(alpha_cfg: dict[str, Any]) -> bool:
    """返回 Alpha 是否含前视，并完整校验来源与可信度元数据。"""
    return _validate_alpha_config(alpha_cfg)[1]


def _alpha_staleness_warn_days(rebalance_freq: int) -> int:
    """告警阈值：正常情况下每个调仓日都应有当期 Alpha，容忍约两个调仓间隔。

    交易日 → 自然日按 1.5 倍折算（含周末），下限 7 天避免高频调仓时过度告警。
    """
    return max(int(rebalance_freq * 2 * 1.5), _MIN_ALPHA_STALENESS_WARN_DAYS)


def _resolve_project_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def normalize_config_paths(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """返回路径已归一化的配置副本。

    ``data.root`` 是运行数据根目录；写在 YAML 中时相对 YAML 所在目录解析，
    直接传入 dict 时相对当前工作目录解析。未配置时保留源码仓库根目录兼容行为。
    项目数据、Alpha 与输出路径随后统一相对 ``data.root`` 解析，使 wheel 无需
    依赖 site-packages 内存在 ``data/``。
    """
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是对象")
    normalized = deepcopy(config)
    data_cfg = normalized.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("data 配置必须是对象")

    config_base = (
        Path(config_path).expanduser().resolve().parent
        if config_path is not None
        else Path.cwd().resolve()
    )
    root_value = data_cfg.get("root")
    data_root = (
        _resolve_project_path(root_value, config_base)
        if root_value is not None
        else _PROJECT_ROOT
    )
    data_cfg["root"] = str(data_root)

    for key in ("manifest", "lock"):
        if data_cfg.get(key):
            data_cfg[key] = str(_resolve_project_path(data_cfg[key], data_root))

    alpha_cfg = normalized.get("alpha", {})
    if isinstance(alpha_cfg, dict) and alpha_cfg.get("path"):
        alpha_cfg["path"] = str(
            _resolve_project_path(alpha_cfg["path"], data_root)
        )

    optimizer_cfg = normalized.get("optimizer", {})
    if isinstance(optimizer_cfg, dict) and optimizer_cfg.get("cne6_data_dir"):
        optimizer_cfg["cne6_data_dir"] = str(
            _resolve_project_path(
                optimizer_cfg["cne6_data_dir"],
                data_root,
            )
        )

    output_cfg = normalized.get("output", {})
    if isinstance(output_cfg, dict) and output_cfg.get("weights"):
        output_cfg["weights"] = str(
            _resolve_project_path(output_cfg["weights"], data_root)
        )
    return normalized


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return normalize_config_paths(config, config_path=path)


def _build_alpha_policy(alpha_cfg: dict[str, Any], rebal_freq: int) -> _AlphaPolicy:
    """解析 Alpha 可信度控制参数。

    陈旧信号会让回测"继续赚钱"却毫无依据；量纲未标准化则使 risk_aversion /
    turnover_penalty 的标定失真。
    """
    default_staleness = 15 if alpha_cfg.get("source") == "file" else None
    max_staleness_days = alpha_cfg.get("max_staleness_days", default_staleness)
    if max_staleness_days is not None:
        if isinstance(max_staleness_days, bool) or not isinstance(
            max_staleness_days, int
        ):
            raise ValueError("alpha.max_staleness_days 必须是非负整数或 null")
        if max_staleness_days < 0:
            raise ValueError("alpha.max_staleness_days 必须是非负整数或 null")

    policy = _AlphaPolicy(
        max_staleness_days=max_staleness_days,
        standardize=bool(alpha_cfg.get("standardize", True)),
        stale_warn_days=_alpha_staleness_warn_days(rebal_freq),
    )
    hard_skip = (
        "关闭" if policy.max_staleness_days is None
        else f">{policy.max_staleness_days}日"
    )
    logger.info(
        f"  Alpha 截面标准化={'是（z-score）' if policy.standardize else '否（原始量纲）'}  "
        f"陈旧告警>{policy.stale_warn_days}日  硬跳过={hard_skip}"
    )
    return policy


def _parse_run_config(cfg: dict[str, Any]) -> _RunConfig:
    """校验策略并解析回测区间，打印运行头。"""
    strategy = cfg["strategy"]          # "index_enhance" | "alpha_max"
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"strategy 须为 {sorted(_STRATEGIES)} 之一，当前为 {strategy!r}"
        )
    index    = cfg["index"]
    bt_cfg   = cfg["backtest"]
    uni_cfg  = cfg["universe"]
    run_cfg = _RunConfig(
        strategy=strategy,
        index=index,
        start_date=date.fromisoformat(bt_cfg["start_date"]),
        end_date=date.fromisoformat(bt_cfg["end_date"]),
        rebal_freq=int(bt_cfg["rebalance_freq"]),
        initial_value=float(bt_cfg["initial_value"]),
        universe_cfg=uni_cfg,
        optimizer_cfg=cfg["optimizer"],
        alpha_cfg=cfg["alpha"],
        execution_cfg=cfg.get("execution", {}),
        output_path=Path(cfg["output"]["weights"]),
        data_root=Path(cfg["data"]["root"]),
    )

    index_name = _INDEX_NAMES.get(index, index.upper())
    logger.info(f"\n{'='*65}")
    logger.info(
        f"  {index_name} {strategy} 批量优化  "
        f"{run_cfg.start_date} ~ {run_cfg.end_date}"
    )
    logger.info(f"  调仓={run_cfg.rebal_freq}日  候选池: 剔除北交所+ST"
          + (f"  TOP_N={uni_cfg['top_n']}" if uni_cfg.get("top_n") else "  全市场"))
    logger.info(f"{'='*65}")
    return run_cfg
