"""包内模块必须能脱离仓库工作目录导入。

pytest 会把 rootdir 塞进 ``sys.path``，于是包内 ``from scripts...`` 这类导入在
测试下完全正常，却会让控制台入口 ``hqopt`` 直接 ModuleNotFoundError——``scripts/``
既不是包（无 ``__init__.py``），也不在 wheel 里（``pyproject`` 只打包 ``hqopt*``）。

2026-07-29 实际发生过：``hqopt/io/run_manifest.py`` 从 ``scripts.verify_data_bundle``
导入，330 个测试全绿，但 ``hqopt run`` 与 ``hqopt optimize`` 全线崩溃。

因此这里用**子进程 + 仓库外 cwd + 剥离 sys.path 注入**来复现真实入口条件，
断言每个包内模块都能独立导入。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 逐个导入而非只导顶层包：顶层 __init__ 未必触及所有子模块，惰性导入的
# 问题模块（如 run_manifest 只在 cli 命令里才 import）会被漏掉。
_MODULES = sorted(
    ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
    for path in (PROJECT_ROOT / "hqopt").rglob("*.py")
    if path.name != "__main__.py"
)


@pytest.mark.parametrize("module", _MODULES)
def test_module_imports_outside_repo(module: str) -> None:
    """在仓库外的 cwd 导入包内模块，不得依赖 sys.path 里的仓库根目录。"""
    assert _MODULES, "未发现任何 hqopt 模块，路径推导有误"
    with tempfile.TemporaryDirectory() as outside:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=outside,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, (
        f"{module} 无法脱离仓库目录导入——控制台入口 hqopt 下会崩溃。\n"
        f"常见原因：包内 `from scripts...` 导入了未打包的目录。\n"
        f"{result.stderr[-800:]}"
    )
