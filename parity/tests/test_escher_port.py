# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The transplant of ESCHER-GPU's CBST store is reproducible (M8a step 3, ADR 0038): the first
port commit equals parity/escher_port/transplant.py's output from the pinned original."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load():
    path = REPO / "parity" / "escher_port" / "transplant.py"
    spec = importlib.util.spec_from_file_location("transplant", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["transplant"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_rename_map_is_complete_and_consistent() -> None:
    """Every file part names an existing line range shape; every rename is to a snake_case
    identifier that is not renamed again; the kernels end in _kernel."""
    t = load()
    cfg = t.load_map()
    assert cfg["commit"].startswith("abcf9b2")
    targets = [f["target"] for f in cfg["file"]]
    assert len(targets) == len(set(targets))
    for f in cfg["file"]:
        for part in f["parts"]:
            if "lines" in part:
                first, last = part["lines"]
                assert 1 <= first <= last
    names = cfg["names"]
    for old, new in names.items():
        assert new == new.lower(), (old, new)
        assert new not in names or names[new] == new, (old, new)
        if old.endswith(("Kernel", "_thread", "_warp", "_block")) or old in (
            "applyReuse",
            "buildEmptyBinaryTree",
        ):
            assert new.endswith("_kernel"), (old, new)


def c1_commit() -> str | None:
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "--format=%H", "--grep", r"(M8a c1)", "--fixed-strings"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return out[-1] if out else None


def test_the_transplant_commit_is_reproducible() -> None:
    t = load()
    original = t.default_original()
    if not (original / ".git").exists():
        pytest.skip(f"the original's clone is not at {original}")
    if shutil.which("clang-format") is None:
        pytest.skip("clang-format is not on PATH")
    rev = c1_commit()
    if rev is None:
        pytest.skip("no transplant commit in this history")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "parity" / "escher_port" / "transplant.py"),
            "--check",
            "--rev",
            rev,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "identical" in proc.stdout
