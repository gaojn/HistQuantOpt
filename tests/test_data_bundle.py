from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.verify_data_bundle import (
    create_lock,
    load_manifest,
    verify_lock,
    verify_profile,
)


def _write_fixture(root: Path) -> Path:
    data_dir = root / "data"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "date": [20200102, 20260102],
            "code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    ).write_parquet(data_dir / "sample.parquet")
    manifest = root / "manifest.yaml"
    manifest.write_text(
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
    return manifest


def test_verify_profile_and_lock_roundtrip(tmp_path: Path) -> None:
    manifest = load_manifest(_write_fixture(tmp_path))

    results, errors = verify_profile(manifest, "default", tmp_path)

    assert errors == []
    assert results[0].min_date.isoformat() == "2020-01-02"
    lock = create_lock(results, "default", tmp_path)
    expected_paths = set(results[0].paths)
    assert verify_lock(lock, "default", tmp_path, expected_paths) == []

    lock["files"] = []
    assert verify_lock(lock, "default", tmp_path, expected_paths) == [
        "锁文件漏列必需文件: data/sample.parquet"
    ]


def test_verify_profile_reports_schema_and_coverage_errors(tmp_path: Path) -> None:
    manifest = load_manifest(_write_fixture(tmp_path))
    asset = manifest["assets"]["sample"]
    asset["required_columns"].append("missing")
    asset["required_end"] = "2027-01-01"

    _, errors = verify_profile(manifest, "default", tmp_path)

    assert any("缺少字段 missing" in error for error in errors)
    assert any("截止日 2026-01-02 早于要求 2027-01-01" in error for error in errors)


def test_verify_lock_detects_changed_file(tmp_path: Path) -> None:
    manifest = load_manifest(_write_fixture(tmp_path))
    results, _ = verify_profile(manifest, "default", tmp_path)
    lock = create_lock(results, "default", tmp_path)

    with (tmp_path / "data/sample.parquet").open("ab") as stream:
        stream.write(b"changed")

    assert verify_lock(lock, "default", tmp_path) == ["文件大小不匹配: data/sample.parquet"]
