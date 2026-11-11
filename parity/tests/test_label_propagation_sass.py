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


def test_the_fp_sequence_keeps_the_operands_and_the_conversions() -> None:
    """M10a review: the opcodes alone pass a reversed association; the conversions from integers
    are floating-point instructions too."""
    a = [
        "FADD R4, R2, R3 ;",
        "SHFL.BFLY PT, R5, R4, 0x1, 0x1f ;",
        "@P0 FMUL R6, R4, c[0x0][0x160] ;",
    ]
    b = [
        "FADD R4, R3, R2 ;",
        "SHFL.BFLY PT, R5, R4, 0x10, 0x1f ;",
        "@P0 FMUL R6, R4, c[0x0][0x168] ;",
    ]
    assert ls.fp_sequence(a) == ls.fp_sequence(b) == ["FADD", "FMUL"]
    assert ls.fp_sequence(a, operands=True) != ls.fp_sequence(b, operands=True)
    assert ls.fp_sequence(["I2FP.F32.S32 R1, R2 ;", "I2F.F64 R2, R3 ;", "FLO.U32 R1, R2 ;"]) == [
        "I2FP.F32.S32",
        "I2F.F64",
    ]
    # Parameter offsets are blanked, nothing else.
    c = ["@P0 FMUL R6, R4, c[0x0][0x168] ;"]
    assert ls.fp_sequence(a[2:], operands=True) == ls.fp_sequence(c, operands=True)


def test_the_port_gate_fails_on_any_difference_beyond_the_parameter_offsets(
    tmp_path, monkeypatch, capsys
) -> None:
    """The port gate passes only SASS equal but for the parameter offsets: a kernel whose
    floating-point opcodes are equal but whose shuffle offset (or an operand) differs fails, as
    does a floating-point (or conversion) instruction in the graph module's device apply."""
    import argparse

    o_name, p_name = next(iter(ls.PORT_KERNELS.items()))
    base = [
        "FADD R4, R2, R3 ;",
        "SHFL.BFLY PT, R5, R4, 0x1, 0x1f ;",
        "FMUL R6, R4, c[0x0][0x160] ;",
    ]

    def run(port_code: list[str], graph_code: list[str]) -> dict:
        def kernels_of(objs):
            first = Path(objs[0]).name
            if first.endswith(".cu.o") and "apply_slotted" not in first:
                return {k2: (port_code if k2 == p_name else ["NOP ;"]) for k2 in
                        ls.PORT_KERNELS.values()}, {}  # fmt: skip
            if "apply_slotted" in first:
                return {"append_kernel": graph_code}, {}
            return {k1: (base if k1 == o_name else ["NOP ;"]) for k1 in ls.PORT_KERNELS}, {}

        monkeypatch.setattr(ls, "kernels_of", kernels_of)
        monkeypatch.setattr(ls, "cuobjdump", lambda: "true")
        out = tmp_path / "rec.json"
        args = argparse.Namespace(
            original_obj=tmp_path / "o",
            port_obj=tmp_path / "p",
            graph_obj=tmp_path / "apply_slotted_device.cu.o",
            json=out,
        )
        rc = ls.port_cmd(args)
        rec = json.loads(out.read_text())
        rec["rc"] = rc
        return rec

    same = run([*base[:2], "FMUL R6, R4, c[0x0][0x168] ;"], ["IADD3 R1, R2, R3, RZ ;"])
    assert same["rc"] == 0 and same["equal"], same["problems"]
    shuffled = run(["FADD R4, R2, R3 ;", "SHFL.BFLY PT, R5, R4, 0x10, 0x1f ;", base[2]], ["NOP ;"])
    assert shuffled["rc"] == 1
    assert any("beyond the parameter offsets" in p for p in shuffled["problems"])
    swapped = run(["FADD R4, R3, R2 ;", *base[1:]], ["NOP ;"])
    assert swapped["rc"] == 1
    assert any("floating-point instructions differ" in p for p in swapped["problems"])
    converted = run(list(base), ["I2FP.F32.S32 R1, R2 ;"])
    assert converted["rc"] == 1 and any("graph module" in p for p in converted["problems"])
    capsys.readouterr()
