# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/label_propagation_sass.py (M10a step 1, ADR 0066) that need no compiler: dynG's
nvcc line keeps every flag when it is retargeted to one of O's translation units, and the record
committed by the step says the go/no-go passed with a negative control that failed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))

import label_propagation_sass as ls  # noqa: E402


def test_retarget_keeps_every_flag() -> None:
    line = ["nvcc", "-I/d/include", "-O3", "-fmad=true", "--use_fast_math", "-x", "cu", "-c",
            "probe.cu", "-o", "probe.o"]  # fmt: skip
    out = ls.retarget(line, "/o/src/x.cu", "/out/x.o", "/o/headers")
    assert out == ["nvcc", "-I/o/headers", "-I/d/include", "-O3", "-fmad=true", "--use_fast_math",
                   "-x", "cu", "-c", "/o/src/x.cu", "-o", "/out/x.o"]  # fmt: skip


def test_the_committed_record_is_a_go_with_a_failing_negative_control() -> None:
    rec = json.loads((REPO / "parity/results/M10a-sass-go-no-go.json").read_text())
    assert rec["identical"] and not rec["differ"]
    assert set(rec["tus"]) == set(ls.TUS)
    assert sum(t["count"] for t in rec["tus"].values()) == 38
    assert not rec["negative_control"]["identical"] and rec["negative_control"]["differ"]
    for tu in ls.TUS:
        assert "--use_fast_math" in rec["lines"]["dyng"][tu]
        assert "-fmad=true" in rec["lines"]["dyng"][tu]
