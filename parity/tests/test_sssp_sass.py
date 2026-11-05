# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/sssp_sass.py (M11b criterion 2): only names are normalized, so a change of one
instruction (an opcode, a register, an immediate, a predicate, a constant-bank offset, a branch
target) changes a kernel's digest, while two builds' build-derived names do not."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))
spec = importlib.util.spec_from_file_location("sssp_sass", REPO / "parity/sssp_sass.py")
sssp_sass = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sssp_sass)

NAME = "_ZN4dyng6detail14sssp_operators11pull_kernelIiiiEEvPKT_S3_NS0_12sssp_kernels10device_csr"
STREAM = [
    (0x00, "MOV R1, c[0x0][0x28]"),
    (0x10, "S2R R0, SR_CTAID.X"),
    (0x20, "ISETP.GE.AND P0, PT, R0, c[0x0][0x168], PT"),
    (0x30, "@P0 EXIT"),
    (0x40, "IADD3 R2, R0, 0x1, RZ"),
    (0x50, f"CALL.REL.NOINC `({NAME})"),
    (0x60, "BRA 0x40"),
]


def digest(stream, name=NAME) -> str:
    import hashlib

    text = "\n".join(
        f"{addr:06x} {sssp_sass.normalize_instruction(ins, name)}" for addr, ins in stream
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_every_one_instruction_change_is_caught() -> None:
    base = digest(STREAM)
    plants = {
        2: "ISETP.GT.AND P0, PT, R0, c[0x0][0x168], PT",  # the comparison
        3: "@!P0 EXIT",  # the predicate
        4: "IADD3 R2, R0, 0x2, RZ",  # an immediate
        0: "MOV R1, c[0x0][0x2c]",  # a constant-bank offset (a kernel parameter)
        6: "BRA 0x30",  # a branch target
        1: "S2R R3, SR_CTAID.X",  # a register
    }
    for i, ins in plants.items():
        planted = list(STREAM)
        planted[i] = (planted[i][0], ins)
        assert digest(planted) != base, ins
    assert digest(STREAM[:-1]) != base  # a dropped instruction


def test_only_names_are_normalized() -> None:
    a = "_ZN4dyng6detail47_GLOBAL__N__7_cuda_cu_d0616ebf_26297926nativeERKNS_9resourcesE"
    b = "_ZN4dyng6detail48_GLOBAL__N__7_cuda_cu_68d7eaa2_283232961nativeERKNS_9resourcesE"
    assert sssp_sass.normalize_name(a) == sssp_sass.normalize_name(b)
    # The kernel's own name is <self>; a call of another function keeps that function's name.
    ins = sssp_sass.normalize_instruction(f"CALL.REL.NOINC `({NAME})", NAME)
    assert ins == "CALL.REL.NOINC `(<self>)"
    other = sssp_sass.normalize_instruction("CALL.REL.NOINC `(_Z5otherv)", NAME)
    assert "_Z5otherv" in other
    # Immediates and addresses stay (triad_sass.normalize would replace them).
    assert sssp_sass.normalize_instruction("IADD3 R2, R0, 0x1, RZ", NAME) == "IADD3 R2, R0, 0x1, RZ"


def test_the_objects_are_sssps_and_mosps() -> None:
    names = {Path(o).parent.name + "/" + Path(o).name for o in sssp_sass.OBJECTS}
    for algo in ("dyng_sssp.dir", "dyng_mosp.dir"):
        assert f"{algo}/cuda.cu.o" in names
    assert "dyng_sssp.dir/operators.cu.o" in names
    for o in sssp_sass.OBJECTS:
        rel = o.split(".dir/", 1)[1].removesuffix(".o")
        base = (
            "tools/compat"
            if "compat" in o
            else "cpp/src/algorithms/" + ("sssp" if "dyng_sssp" in o else "mosp")
        )
        assert (REPO / base / rel).is_file(), o


def test_the_planted_line_is_in_the_shared_pull_once() -> None:
    text = (REPO / "cpp/src/algorithms/sssp" / sssp_sass.PLANT_FILE).read_text()
    assert text.count(sssp_sass.PLANT_OLD) == 1


def test_debug_sections_are_told_apart() -> None:
    for name in (".debug_line", ".nv_debug_ptx_txt", ".rel.debug_line", ".strtab", ".symtab"):
        assert sssp_sass.DEBUG_SECTION.search(name), name
    for name in (".text._Z1kv", ".nv.info._Z1kv", ".nv.constant0._Z1kv", ".nv.shared._Z1kv"):
        assert not sssp_sass.DEBUG_SECTION.search(name), name
