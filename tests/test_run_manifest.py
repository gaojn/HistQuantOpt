from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from hqopt.io import run_manifest
from hqopt.io.run_manifest import (
    DataLockEvidence,
    RunManifestRecorder,
    execution_bundle_inputs,
    sha256_file,
    verify_data_bundle_lock,
)
from scripts.verify_data_bundle import create_lock, load_manifest, verify_profile


def _write_data_contract(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "date": [20200102, 20260102],
            "code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    ).write_parquet(data_dir / "sample.parquet")
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(
        """
manifest_version: 1
profiles:
  default:
    assets: [sample]
assets:
  sample:
    path: data/sample.parquet
    format: parquet
    required_columns: [date, code, value]
    date_column: date
    required_start: "2020-01-02"
    required_end: "2026-01-02"
""".lstrip(),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    results, errors = verify_profile(manifest, "default", root)
    assert errors == []
    lock_path = root / "data.lock.json"
    lock_path.write_text(
        json.dumps(create_lock(results, "default", root), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, lock_path


def test_data_lock_preflight_passes_and_rejects_changed_input(tmp_path: Path) -> None:
    manifest_path, lock_path = _write_data_contract(tmp_path)
    config = {
        "data": {
            "profile": "default",
            "manifest": str(manifest_path),
            "lock": str(lock_path),
        },
        "optimizer": {},
    }

    evidence = verify_data_bundle_lock(config, project_root=tmp_path)

    assert evidence.verified
    assert evidence.profile == "default"
    assert evidence.lock_sha256 == sha256_file(lock_path)

    with (tmp_path / "data" / "sample.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="数据锁校验失败"):
        verify_data_bundle_lock(config, project_root=tmp_path)


def test_run_manifest_binds_identity_inputs_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("strategy: alpha_max\n", encoding="utf-8")
    alpha_path = tmp_path / "alpha.parquet"
    alpha_path.write_bytes(b"alpha")
    artifact = tmp_path / "output" / "weights.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"weights")
    config = {
        "strategy": "alpha_max",
        "alpha": {
            "source": "file",
            "path": str(alpha_path),
            "synthetic": False,
        },
    }
    monkeypatch.setattr(
        run_manifest,
        "collect_repository_state",
        lambda root: {
            "root": str(root),
            "commit": "abc123",
            "branch": "main",
            "dirty": True,
            "tracked_diff_sha256": "d" * 64,
            "worktree_sha256": "e" * 64,
            "untracked_files": [],
        },
    )
    evidence = DataLockEvidence(
        profile="default",
        manifest_path="/manifest.yaml",
        manifest_sha256="a" * 64,
        lock_path="/data.lock.json",
        lock_sha256="b" * 64,
    )

    recorder = RunManifestRecorder.start(
        mode="optimize",
        config=config,
        config_path=config_path,
        output_dir=artifact.parent,
        command=["hqopt", "optimize", str(config_path)],
        data_lock=evidence,
        project_root=tmp_path,
    )
    unique, latest = recorder.finalize(
        status="complete",
        artifacts=[artifact],
        quality_checks={"data_lock": "passed"},
    )

    payload = json.loads(unique.read_text(encoding="utf-8"))
    assert latest.read_bytes() == unique.read_bytes()
    assert payload["status"] == "complete"
    assert payload["repository"]["commit"] == "abc123"
    assert payload["config"]["source_sha256"] == sha256_file(config_path)
    assert payload["inputs"][0]["sha256"] == sha256_file(alpha_path)
    assert payload["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert payload["quality_checks"]["data_lock"] == "passed"
    assert not recorder.in_progress_path.exists()


def test_standalone_manifest_binds_weights_without_config_or_data_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "weights.parquet"
    weights.write_bytes(b"weights")
    artifact = tmp_path / "output" / "report.html"
    artifact.parent.mkdir()
    artifact.write_bytes(b"report")
    monkeypatch.setattr(
        run_manifest,
        "collect_repository_state",
        lambda root: {
            "root": str(root),
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
            "tracked_diff_sha256": "d" * 64,
            "worktree_sha256": "e" * 64,
            "untracked_files": [],
        },
    )

    recorder = RunManifestRecorder.start(
        mode="backtest",
        config={
            "weights": str(weights),
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
        },
        config_path=None,
        output_dir=artifact.parent,
        command=["hqopt", "backtest", "--weights", str(weights)],
        data_lock=None,
        input_files=[("weights", weights)],
        project_root=tmp_path,
    )
    unique, latest = recorder.finalize(
        status="complete",
        artifacts=[artifact],
        quality_checks={"data_lock": "not_verified"},
    )

    payload = json.loads(unique.read_text(encoding="utf-8"))
    assert latest.read_bytes() == unique.read_bytes()
    assert payload["mode"] == "backtest"
    assert payload["config"]["source_path"] is None
    assert payload["config"]["source_sha256"] is None
    assert payload["config"]["effective"] == {
        "weights": str(weights),
        "start_date": "2024-01-02",
        "end_date": "2024-01-31",
    }
    assert payload["data_lock"]["verified"] is False
    assert payload["inputs"][0]["role"] == "weights"
    assert payload["inputs"][0]["sha256"] == sha256_file(weights)
    assert payload["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert not recorder.in_progress_path.exists()


def test_execution_bundle_inputs_include_all_existing_companions(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "custom.parquet"
    companions = {
        "weights": weights,
        "sell_only": tmp_path / "custom.sell_only.parquet",
        "batch_execution_stats": (
            tmp_path / "custom.batch_execution_stats.json"
        ),
        "execution_bundle_manifest": (
            tmp_path / "custom.sell_only.manifest.json"
        ),
    }
    for role, path in companions.items():
        path.write_bytes(role.encode("utf-8"))

    discovered = dict(execution_bundle_inputs(weights))

    assert discovered == companions
