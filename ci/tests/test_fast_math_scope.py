# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/fast_math_scope.py (M10a criterion 12): the rule on synthetic databases, and on the
parity-cuda build of the worktree when there is one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ci"))

import fast_math_scope as fms  # noqa: E402


def entry(file: str, *flags: str) -> dict:
    return {"file": f"/x/{file}", "arguments": ["nvcc", *flags, "-c", file]}


LP = "cpp/src/algorithms/label_propagation/propagation.cu"


def test_the_scope_holds() -> None:
    db = [entry(LP, "-O3", "-fmad=true", "--use_fast_math"), entry("cpp/src/graph/x.cu", "-O3")]
    assert fms.check(db, require_fmad=True) == []


def test_fast_math_elsewhere_is_refused() -> None:
    db = [entry(LP, "-fmad=true", "--use_fast_math"), entry("cpp/src/x/y.cu", "--use_fast_math")]
    assert any("outside" in p for p in fms.check(db, require_fmad=True))


def test_the_lp_tus_need_fast_math_and_one_fmad() -> None:
    assert any("no --use_fast_math" in p for p in fms.check([entry(LP, "-fmad=true")], True))
    two = [entry(LP, "-fmad=true", "--use_fast_math", "-fmad=true")]
    assert any("2 -fmad flags" in p for p in fms.check(two, False))
    assert any("-fmad=true missing" in p for p in fms.check([entry(LP, "--use_fast_math")], True))
    assert fms.check([entry(LP, "--use_fast_math")], False) == []
    assert any("no cpp/src" in p for p in fms.check([], False))


def test_the_parity_cuda_build() -> None:
    db = REPO / "build" / "parity-cuda" / "compile_commands.json"
    if not db.exists():
        pytest.skip("no parity-cuda build in this worktree")
    assert fms.check(json.loads(db.read_text()), require_fmad=True) == []
