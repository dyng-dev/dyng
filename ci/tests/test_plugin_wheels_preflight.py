# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The toolkit checks of ci/plugin_wheels.sh, which run before anything is built.

Every plugin's CUDA toolkit must have ``bin/nvcc`` (the build) and ``bin/cuobjdump`` (the check
of the module's SASS and PTX after the repair). A toolkit without either is refused with a
message naming what is missing, before the core wheel is built and before the output directory
is cleared (M6a acceptance: a conda-forge toolkit without ``cuda-cuobjdump`` failed only after
the ten-minute build). The script runs with fake tools here: it must exit before using them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ci" / "plugin_wheels.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho 'fake tool must not run' >&2\nexit 99\n")
    path.chmod(0o755)


def _toolkit(tmp: Path, name: str, *, nvcc: bool = True, cuobjdump: bool = True) -> Path:
    root = tmp / name
    (root / "bin").mkdir(parents=True)
    if nvcc:
        _executable(root / "bin" / "nvcc")
    if cuobjdump:
        _executable(root / "bin" / "cuobjdump")
    return root


def _run(tmp: Path, plugins: str, **roots: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    toolchain = tmp / "toolchain"
    tools = tmp / "tools"
    for tool in ("x86_64-conda-linux-gnu-g++", "x86_64-conda-linux-gnu-gcc"):
        _executable(toolchain / "bin" / tool)
    for tool in ("auditwheel", "twine"):
        _executable(tools / "bin" / tool)
    out = tmp / "out"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp),
        "DYNG_SCRATCH": str(tmp / "scratch"),
        "DYNG_PLUGINS": plugins,
        "DYNG_PLUGIN_OUT": str(out),
        "DYNG_WHEEL_TOOLCHAIN": str(toolchain),
        "DYNG_WHEEL_TOOLS": str(tools),
        "DYNG_WHEEL_PYTHONS": "python3",
        # Never the machine's toolkit: every test names the roots it uses.
        "DYNG_CUDA12_ROOT": str(roots.get("cu12", tmp / "absent-cu12")),
        "DYNG_CUDA13_ROOT": str(roots.get("cu13", tmp / "absent-cu13")),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    return proc, out


def test_missing_cuobjdump_is_refused_before_the_build(tmp_path: Path) -> None:
    cu12 = _toolkit(tmp_path, "cuda-12.9", cuobjdump=False)
    cu13 = _toolkit(tmp_path, "cuda-13.1")
    proc, out = _run(tmp_path, "cu13 cu12", cu12=cu12, cu13=cu13)
    assert proc.returncode == 1, proc.stderr
    assert f"cu12: {cu12}/bin/cuobjdump is missing" in proc.stderr
    assert "cuda-cuobjdump" in proc.stderr
    assert "fake tool must not run" not in proc.stderr
    assert "==>" not in proc.stdout  # no step started (the core build is the first)
    assert not out.exists()


def test_missing_nvcc_is_refused_before_the_build(tmp_path: Path) -> None:
    cu13 = _toolkit(tmp_path, "cuda-13.1", nvcc=False)
    proc, out = _run(tmp_path, "cu13", cu13=cu13)
    assert proc.returncode == 1, proc.stderr
    assert "cu13 needs its CUDA toolkit: set DYNG_CUDA13_ROOT" in proc.stderr
    assert not out.exists()


def test_cu12_without_a_root_is_refused(tmp_path: Path) -> None:
    proc, out = _run(tmp_path, "cu12")
    assert proc.returncode == 1, proc.stderr
    assert "cu12 needs its CUDA toolkit: set DYNG_CUDA12_ROOT" in proc.stderr
    assert not out.exists()


def test_unknown_plugin_is_refused(tmp_path: Path) -> None:
    proc, out = _run(tmp_path, "cu11")
    assert proc.returncode == 2, proc.stderr
    assert "unknown plugin cu11" in proc.stderr
    assert not out.exists()
