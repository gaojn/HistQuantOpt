"""运行前数据锁校验与项目级可复现性清单。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hqopt.io.data_bundle import (
    load_manifest,
    verify_lock,
    verify_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_MANIFEST = PROJECT_ROOT / "data_manifest.yaml"
DEFAULT_DATA_LOCK = PROJECT_ROOT / "data_bundle.lock.json"
PROFILE_DATA_LOCK_TEMPLATE = "data_bundle.{profile}.lock.json"
RUN_MANIFEST_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def infer_data_profile(config: dict[str, Any]) -> str:
    """从显式配置或风险面板目录推导数据 profile。"""
    data_cfg = config.get("data", {})
    if data_cfg.get("profile"):
        return str(data_cfg["profile"])
    risk_dir = str(config.get("optimizer", {}).get("cne6_data_dir") or "")
    return "long_risk" if "barra_cne6_L" in risk_dir else "default"


@dataclass(frozen=True)
class DataLockEvidence:
    profile: str
    manifest_path: str
    manifest_sha256: str
    lock_path: str
    lock_sha256: str
    verified: bool = True


def verify_data_bundle_lock(
    config: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> DataLockEvidence:
    """按配置 profile 做结构、覆盖范围、大小和 SHA-256 的强制校验。"""
    data_cfg = config.get("data", {})
    base_root = (project_root or PROJECT_ROOT).resolve()
    root = _resolve_path(data_cfg.get("root", base_root), base_root)
    profile = infer_data_profile(config)
    manifest_path = _resolve_path(
        data_cfg.get("manifest", root / "data_manifest.yaml"),
        root,
    )
    if data_cfg.get("lock"):
        lock_path = _resolve_path(data_cfg["lock"], root)
    else:
        profile_lock = root / PROFILE_DATA_LOCK_TEMPLATE.format(profile=profile)
        legacy_lock = root / "data_bundle.lock.json"
        lock_path = (
            legacy_lock
            if profile == "default"
            and not profile_lock.is_file()
            and legacy_lock.is_file()
            else profile_lock
        )
    if not lock_path.is_file():
        raise FileNotFoundError(f"缺少数据锁文件：{lock_path}")

    manifest = load_manifest(manifest_path)
    results, errors = verify_profile(manifest, profile, root)
    if not errors:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据锁文件不是合法 JSON：{lock_path}") from exc
        expected_paths = {path for result in results for path in result.paths}
        errors.extend(verify_lock(lock, profile, root, expected_paths))
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"数据锁校验失败（profile={profile}）：\n{detail}")

    return DataLockEvidence(
        profile=profile,
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        lock_path=str(lock_path),
        lock_sha256=sha256_file(lock_path),
    )


def _run_git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout


def collect_repository_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """绑定提交、已跟踪差异及未跟踪源码文件。"""
    root = root.resolve()
    commit = _run_git(root, "rev-parse", "HEAD").decode().strip()
    branch = _run_git(root, "branch", "--show-current").decode().strip()
    diff = _run_git(root, "diff", "--binary", "HEAD")
    untracked_raw = _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = [Path(value.decode("utf-8")) for value in untracked_raw.split(b"\0") if value]

    digest = hashlib.sha256()
    digest.update(diff)
    untracked = []
    for relative in sorted(untracked_paths):
        path = root / relative
        file_hash = sha256_file(path)
        digest.update(str(relative).encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
        untracked.append({"path": str(relative), "sha256": file_hash})

    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(diff or untracked),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "worktree_sha256": digest.hexdigest(),
        "untracked_files": untracked,
    }


def _runtime_state() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "clarabel",
        "cvxpy",
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "PyYAML",
        "scs",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(root))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class RunManifestRecorder:
    """记录一次 CLI 运行；唯一清单永不覆盖，latest 清单仅作入口。"""

    run_id: str
    mode: str
    output_dir: Path
    project_root: Path
    started_at_utc: str
    base_payload: dict[str, Any]
    in_progress_path: Path

    @classmethod
    def start(
        cls,
        *,
        mode: str,
        config: dict[str, Any],
        config_path: str | Path | None,
        output_dir: str | Path,
        command: list[str],
        data_lock: DataLockEvidence | None,
        input_files: Iterable[tuple[str, str | Path]] = (),
        project_root: Path = PROJECT_ROOT,
    ) -> RunManifestRecorder:
        root = project_root.resolve()
        # CLI 相对输出路径沿用进程当前目录；数据锁路径才相对项目根。
        out = Path(output_dir).expanduser().resolve()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:12]
        started = datetime.now(UTC).isoformat()
        source_config = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else None
        )
        alpha_cfg = config.get("alpha", {})
        inputs: list[dict[str, Any]] = []
        seen_inputs: set[Path] = set()
        if alpha_cfg.get("source") == "file":
            alpha_path = Path(alpha_cfg["path"]).expanduser().resolve()
            if not alpha_path.is_file():
                raise FileNotFoundError(f"Alpha 文件不存在：{alpha_path}")
            alpha_record = _file_record(alpha_path, root)
            alpha_record["role"] = "alpha"
            inputs.append(alpha_record)
            seen_inputs.add(alpha_path)
        for role, value in input_files:
            input_path = Path(value).expanduser().resolve()
            if not input_path.is_file():
                raise FileNotFoundError(f"运行输入文件不存在：{input_path}")
            if input_path in seen_inputs:
                continue
            input_record = _file_record(input_path, root)
            input_record["role"] = str(role)
            inputs.append(input_record)
            seen_inputs.add(input_path)

        # 值既有字符串也有嵌套的 effective 配置字典，故不是 dict[str, str | None]。
        config_record: dict[str, Any] = {
            "source_path": str(source_config) if source_config is not None else None,
            "source_sha256": (
                sha256_file(source_config) if source_config is not None else None
            ),
            "effective_sha256": _canonical_sha256(config),
        }
        if source_config is None:
            config_record["effective"] = config
        data_lock_record = (
            asdict(data_lock)
            if data_lock is not None
            else {
                "profile": None,
                "manifest_path": None,
                "manifest_sha256": None,
                "lock_path": None,
                "lock_sha256": None,
                "verified": False,
                "reason": "standalone_command_without_config_bound_data_lock",
            }
        )

        payload = {
            "manifest_version": RUN_MANIFEST_VERSION,
            "run_id": run_id,
            "mode": mode,
            "status": "running",
            "started_at_utc": started,
            "command": command,
            "repository": collect_repository_state(root),
            "config": config_record,
            "data_lock": data_lock_record,
            "inputs": inputs,
            "runtime": _runtime_state(),
            "alpha": {
                "source": alpha_cfg.get("source"),
                "synthetic": bool(alpha_cfg.get("synthetic", False))
                or alpha_cfg.get("source") == "synthetic",
            },
        }
        in_progress = out / f".run.{run_id}.in_progress.json"
        _atomic_write_json(in_progress, payload)
        return cls(
            run_id=run_id,
            mode=mode,
            output_dir=out,
            project_root=root,
            started_at_utc=started,
            base_payload=payload,
            in_progress_path=in_progress,
        )

    def finalize(
        self,
        *,
        status: str,
        artifacts: Iterable[Path],
        quality_checks: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> tuple[Path, Path]:
        if status not in {"complete", "failed"}:
            raise ValueError("run manifest status 只能是 complete 或 failed")
        artifact_records = [
            _file_record(path, self.project_root)
            for path in sorted({Path(path).resolve() for path in artifacts})
            if Path(path).is_file()
        ]
        payload = {
            **self.base_payload,
            "status": status,
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": artifact_records,
            "quality_checks": quality_checks or {},
        }
        if error is not None:
            payload["error"] = error

        unique = self.output_dir / f"run.{self.run_id}.manifest.json"
        latest = self.output_dir / "run.manifest.json"
        if unique.exists():
            raise FileExistsError(f"唯一运行清单已存在，拒绝覆盖：{unique}")
        _atomic_write_json(unique, payload)
        _atomic_write_json(latest, payload)
        self.in_progress_path.unlink(missing_ok=True)
        return unique, latest


def execution_bundle_inputs(
    weights_path: str | Path,
) -> list[tuple[str, Path]]:
    """返回独立回测/归因应绑定的权重 bundle 输入文件。"""
    from hqopt.backtest.execution import (
        batch_execution_stats_path_for_weights,
        sell_only_manifest_path_for_weights,
        sell_only_path_for_weights,
    )

    weights = Path(weights_path).expanduser().resolve()
    candidates = [
        ("weights", weights),
        ("sell_only", sell_only_path_for_weights(weights)),
        ("batch_execution_stats", batch_execution_stats_path_for_weights(weights)),
        ("execution_bundle_manifest", sell_only_manifest_path_for_weights(weights)),
    ]
    return [
        (role, path)
        for role, path in candidates
        if role == "weights" or path.is_file()
    ]


def expected_standalone_artifacts(
    output_dir: str | Path,
    *,
    mode: str,
) -> list[Path]:
    """返回独立 backtest/attribute 必需产物，并拒绝静默缺失。"""
    out = Path(output_dir).expanduser().resolve()
    if mode == "backtest":
        relative_paths = [
            "report.html",
            "nav.parquet",
            "turnover.parquet",
            "actual_weights.parquet",
            "execution_stats.json",
            "report_data/timeseries.parquet",
            "report_data/turnover.parquet",
            "report_data/metrics.parquet",
            "report_data/yearly.parquet",
            "report_data/monthly_excess.parquet",
        ]
    elif mode == "rotate":
        relative_paths = [
            "report.html",
            "nav.parquet",
            "turnover.parquet",
            "actual_weights.parquet",
            "trades.parquet",
            "execution_stats.json",
            "report_data/timeseries.parquet",
            "report_data/turnover.parquet",
            "report_data/metrics.parquet",
            "report_data/yearly.parquet",
            "report_data/monthly_excess.parquet",
        ]
    elif mode == "attribute":
        relative_paths = [
            "attribution_report.html",
            "attribution_summary.csv",
            "attribution_daily.parquet",
            "attribution_factor_daily.parquet",
            "attribution_execution_stats.json",
        ]
    else:
        raise ValueError(f"独立运行清单模式不支持：{mode!r}")

    artifacts = [out / relative for relative in relative_paths]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "独立运行成功但缺少必需产物："
            + ", ".join(str(path) for path in missing)
        )
    return artifacts


def expected_run_artifacts(
    weights_path: str | Path,
    *,
    include_backtest: bool,
) -> list[Path]:
    """返回一次 optimize/run 必须生成的固定产物，并拒绝静默缺失。"""
    from hqopt.backtest.execution import (
        batch_execution_stats_path_for_weights,
        sell_only_manifest_path_for_weights,
        sell_only_path_for_weights,
        synthetic_alpha_warning_path_for_weights,
    )

    weights = Path(weights_path).resolve()
    artifacts = [
        weights,
        sell_only_path_for_weights(weights),
        batch_execution_stats_path_for_weights(weights),
        sell_only_manifest_path_for_weights(weights),
    ]
    if include_backtest:
        artifacts.extend(
            weights.parent / name
            for name in (
                "actual_weights.parquet",
                "execution_stats.json",
                "nav.parquet",
                "report.html",
                "turnover.parquet",
            )
        )
    warning = synthetic_alpha_warning_path_for_weights(weights)
    if warning.is_file():
        artifacts.append(warning)
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "运行成功但缺少必需产物：" + ", ".join(str(path) for path in missing)
        )
    return artifacts


def load_run_quality_checks(weights_path: str | Path) -> dict[str, Any]:
    """从批量统计读取求解后门禁摘要，写入顶层运行清单。

    除求解后校验外，还提取 Alpha 可信度与基准回退治理指标：合成 Alpha、
    陈旧/跳过期数、official_drift 中途回退口径的期数。这些异常在逐期日志里
    会被刷掉，只有进入顶层清单才能被审计工具和人可靠地看到。
    """
    from hqopt.backtest.execution import batch_execution_stats_path_for_weights

    stats_path = batch_execution_stats_path_for_weights(weights_path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    alpha_quality = payload.get("alpha_quality", {})
    benchmark_quality = payload.get("benchmark_quality", {})
    return {
        "data_lock": "passed",
        "post_solve_validation": payload.get("optimization", {}).get("post_solve_validation", {}),
        "alpha_quality": {
            "synthetic": alpha_quality.get("synthetic"),
            "max_observed_staleness_days": alpha_quality.get(
                "max_observed_staleness_days"
            ),
            "stale_warning_period_count": alpha_quality.get(
                "stale_warning_period_count"
            ),
            "skipped_period_count": alpha_quality.get("skipped_period_count"),
            "zero_variance_period_count": alpha_quality.get(
                "zero_variance_period_count"
            ),
        },
        "benchmark_quality": {
            "source": benchmark_quality.get("source"),
            "fallback_period_count": benchmark_quality.get(
                "fallback_period_count"
            ),
            "roster_change_period_count": benchmark_quality.get(
                "roster_change_period_count"
            ),
            "stale_snapshot_period_count": benchmark_quality.get(
                "stale_snapshot_period_count"
            ),
        },
    }


__all__ = [
    "DataLockEvidence",
    "RunManifestRecorder",
    "collect_repository_state",
    "execution_bundle_inputs",
    "expected_run_artifacts",
    "expected_standalone_artifacts",
    "infer_data_profile",
    "load_run_quality_checks",
    "sha256_file",
    "verify_data_bundle_lock",
]
