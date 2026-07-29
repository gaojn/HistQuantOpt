"""数据包清单校验：资产存在性、列/日期覆盖与锁文件哈希比对。

**为什么在库里而不在 ``scripts/``**：``hqopt run`` / ``hqopt optimize`` 会在加载
行情前强制执行这套校验（见 :mod:`hqopt.io.run_manifest`），它已是运行时依赖而非
一次性脚本。放在 ``scripts/`` 会让包内代码 ``from scripts...`` 导入——而 scripts
既不是包也不在 wheel 里，控制台入口下必然 ModuleNotFoundError。

``scripts/verify_data_bundle.py`` 现为本模块的命令行包装。
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data_manifest.yaml"
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class AssetResult:
    name: str
    paths: tuple[Path, ...]
    min_date: date | None
    max_date: date | None
    errors: tuple[str, ...]


def load_manifest(path: Path) -> dict[str, Any]:
    """读取并做最低限度的清单结构校验。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manifest_version") != 1:
        raise ValueError("仅支持 manifest_version: 1")
    for key in ("profiles", "assets"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"清单缺少映射字段: {key}")
    return data


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"清单路径必须位于项目根目录内: {value}")
    return path


def expand_asset_paths(asset: dict[str, Any], root: Path) -> tuple[Path, ...]:
    """展开单文件或含占位符的路径模板。"""
    if "path" in asset:
        return (root / _safe_relative_path(str(asset["path"])),)

    template = asset.get("path_template")
    values = asset.get("values")
    if not isinstance(template, str) or not isinstance(values, dict) or not values:
        raise ValueError("资产必须配置 path，或配置 path_template + values")

    keys = list(values)
    combinations = itertools.product(*(values[key] for key in keys))
    paths = []
    for combination in combinations:
        relative = template.format(**dict(zip(keys, combination, strict=True)))
        paths.append(root / _safe_relative_path(relative))
    return tuple(paths)


def _scan(path: Path, file_format: str) -> pl.LazyFrame:
    if file_format == "parquet":
        return pl.scan_parquet(path)
    if file_format == "csv":
        return pl.scan_csv(path)
    raise ValueError(f"不支持的数据格式: {file_format}")


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法识别日期值: {value!r}")


def inspect_asset(
    name: str,
    asset: dict[str, Any],
    column_sets: dict[str, list[str]],
    root: Path,
) -> AssetResult:
    """检查一个资产的文件、字段和合并日期覆盖。"""
    errors: list[str] = []
    try:
        paths = expand_asset_paths(asset, root)
    except (KeyError, TypeError, ValueError) as exc:
        return AssetResult(name, (), None, None, (f"{name}: {exc}",))

    missing = [path for path in paths if not path.is_file()]
    for path in missing:
        errors.append(f"{name}: 缺少文件 {path.relative_to(root)}")

    existing = tuple(path for path in paths if path.is_file())
    required_columns = set(asset.get("required_columns", []))
    for set_name in asset.get("required_column_sets", []):
        if set_name not in column_sets:
            errors.append(f"{name}: 未定义字段组 {set_name}")
        else:
            required_columns.update(column_sets[set_name])

    minima: list[date] = []
    maxima: list[date] = []
    file_format = str(asset.get("format", ""))
    date_column = asset.get("date_column")
    for path in existing:
        relative = path.relative_to(root)
        try:
            frame = _scan(path, file_format)
            columns = set(frame.collect_schema().names())
            absent = sorted(required_columns - columns)
            if absent:
                errors.append(f"{name}: {relative} 缺少字段 {', '.join(absent)}")
            if date_column:
                stats = frame.select(
                    pl.col(date_column).min().alias("min_date"),
                    pl.col(date_column).max().alias("max_date"),
                ).collect()
                min_value, max_value = stats.row(0)
                if min_value is None or max_value is None:
                    errors.append(f"{name}: {relative} 日期列为空")
                else:
                    minima.append(_as_date(min_value))
                    maxima.append(_as_date(max_value))
        except Exception as exc:  # 文件损坏、格式错误也应汇总，而不是中断其余检查
            errors.append(f"{name}: 无法读取 {relative}: {exc}")

    min_date = min(minima) if minima else None
    max_date = max(maxima) if maxima else None
    if min_date is not None and asset.get("required_start"):
        required_start = _as_date(asset["required_start"])
        if min_date > required_start:
            errors.append(f"{name}: 起始日 {min_date} 晚于要求 {required_start}")
    if max_date is not None and asset.get("required_end"):
        required_end = _as_date(asset["required_end"])
        if max_date < required_end:
            errors.append(f"{name}: 截止日 {max_date} 早于要求 {required_end}")

    return AssetResult(name, paths, min_date, max_date, tuple(errors))


def verify_profile(
    manifest: dict[str, Any], profile: str, root: Path
) -> tuple[list[AssetResult], list[str]]:
    """校验一个交接 profile，返回逐资产结果及全部错误。"""
    profiles = manifest["profiles"]
    if profile not in profiles:
        raise ValueError(f"未知 profile: {profile}；可选: {', '.join(sorted(profiles))}")

    assets = manifest["assets"]
    column_sets = manifest.get("column_sets", {})
    results: list[AssetResult] = []
    errors: list[str] = []
    for name in profiles[profile].get("assets", []):
        if name not in assets:
            errors.append(f"profile {profile}: 未定义资产 {name}")
            continue
        result = inspect_asset(name, assets[name], column_sets, root)
        results.append(result)
        errors.extend(result.errors)
    return results, errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def create_lock(results: list[AssetResult], profile: str, root: Path) -> dict[str, Any]:
    """为已通过结构校验的数据文件生成稳定的 SHA-256 清单。"""
    files = []
    unique_paths = sorted({path for result in results for path in result.paths})
    for path in unique_paths:
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"lock_version": 1, "profile": profile, "files": files}


def verify_lock(
    lock: dict[str, Any],
    profile: str,
    root: Path,
    expected_paths: set[Path] | None = None,
) -> list[str]:
    """核验接收方文件大小和 SHA-256。"""
    errors: list[str] = []
    if lock.get("lock_version") != 1:
        return ["仅支持 lock_version: 1"]
    if lock.get("profile") != profile:
        errors.append(f"锁文件 profile={lock.get('profile')}，当前 profile={profile}")
    items = lock.get("files")
    if not isinstance(items, list):
        return [*errors, "锁文件 files 必须是列表"]

    locked_paths: set[Path] = set()
    for item in items:
        if not isinstance(item, dict):
            errors.append("锁文件 files 元素必须是映射")
            continue
        try:
            relative = _safe_relative_path(str(item["path"]))
        except (KeyError, ValueError) as exc:
            errors.append(f"锁文件路径无效: {exc}")
            continue
        if relative in locked_paths:
            errors.append(f"锁文件路径重复: {relative}")
            continue
        locked_paths.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"锁文件要求的文件不存在: {item.get('path')}")
            continue
        if path.stat().st_size != item.get("size"):
            errors.append(f"文件大小不匹配: {item.get('path')}")
            continue
        if sha256_file(path) != item.get("sha256"):
            errors.append(f"SHA-256 不匹配: {item.get('path')}")

    if expected_paths is not None:
        expected_relative = {path.relative_to(root) for path in expected_paths}
        for relative in sorted(expected_relative - locked_paths):
            errors.append(f"锁文件漏列必需文件: {relative}")
        for relative in sorted(locked_paths - expected_relative):
            errors.append(f"锁文件包含 profile 外文件: {relative}")
    return errors
