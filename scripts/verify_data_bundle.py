"""数据包校验命令行入口。

实现在 :mod:`hqopt.io.data_bundle`——该逻辑已被 ``hqopt run`` 在运行时调用，
属库代码；此处只保留参数解析与退出码，并原样 re-export 供既有导入方使用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hqopt.io.data_bundle import (  # noqa: E402
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    AssetResult,
    create_lock,
    expand_asset_paths,
    inspect_asset,
    load_manifest,
    sha256_file,
    verify_lock,
    verify_profile,
)

__all__ = [
    "AssetResult",
    "DEFAULT_MANIFEST",
    "PROJECT_ROOT",
    "create_lock",
    "expand_asset_paths",
    "inspect_asset",
    "load_manifest",
    "main",
    "sha256_file",
    "verify_lock",
    "verify_profile",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    lock_group = parser.add_mutually_exclusive_group()
    lock_group.add_argument("--write-lock", type=Path, metavar="PATH")
    lock_group.add_argument("--lock", type=Path, metavar="PATH")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    try:
        manifest = load_manifest(args.manifest)
        results, errors = verify_profile(manifest, args.profile, root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[FAIL] {exc}")
        return 2

    for result in results:
        coverage = ""
        if result.min_date is not None:
            coverage = f"，覆盖 {result.min_date} ~ {result.max_date}"
        print(f"[{'OK' if not result.errors else 'FAIL'}] {result.name}: {len(result.paths)} 个文件{coverage}")

    if args.lock and not errors:
        try:
            lock = json.loads(args.lock.read_text(encoding="utf-8"))
            expected_paths = {path for result in results for path in result.paths}
            errors.extend(verify_lock(lock, args.profile, root, expected_paths))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"无法读取锁文件: {exc}")

    if errors:
        for error in errors:
            print(f"  - {error}")
        print(f"[FAIL] 数据包校验失败，共 {len(errors)} 项")
        return 1

    if args.write_lock:
        lock = create_lock(results, args.profile, root)
        args.write_lock.parent.mkdir(parents=True, exist_ok=True)
        args.write_lock.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[OK] 已生成 SHA-256 锁文件: {args.write_lock}")

    print(f"[OK] 数据包可用于 profile={args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
