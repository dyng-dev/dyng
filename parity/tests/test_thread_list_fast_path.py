# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/thread_list_fast_path.py (M11b, ADR 0094): region_push_back's fast path is
push_back's on the compiler at hand, a planted change to it is caught, and the path search and the
signature read the instructions as documented."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "thread_list_fast_path", REPO / "parity/thread_list_fast_path.py"
)
fast_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_path)


def ins(addr: int, op: str, args: str = "", reloc: str | None = None) -> dict:
    return {"addr": addr, "op": op, "args": args, "reloc": reloc}


def test_the_fast_path_avoids_calls_and_relocated_branches() -> None:
    # GCC's push_back: the full list branches to a call; the path to ret without it is the
    # fast path.
    body = [
        ins(0x0, "sub", "$0x18,%rsp"),
        ins(0x4, "mov", "%esi,0xc(%rsp)"),
        ins(0x8, "mov", "0x8(%rdi),%rsi"),
        ins(0xC, "cmp", "0x10(%rdi),%rsi"),
        ins(0x10, "je", "30 <f+0x30>"),
        ins(0x12, "mov", "0xc(%rsp),%eax"),
        ins(0x16, "add", "$0x4,%rsi"),
        ins(0x1A, "mov", "%eax,-0x4(%rsi)"),
        ins(0x1D, "mov", "%rsi,0x8(%rdi)"),
        ins(0x21, "add", "$0x18,%rsp"),
        ins(0x25, "ret"),
        ins(0x30, "lea", "0xc(%rsp),%rdx"),
        ins(0x35, "call", "3a <f+0x3a>", reloc="_M_realloc_insert"),
        ins(0x3A, "add", "$0x18,%rsp"),
        ins(0x3E, "ret"),
    ]
    path = fast_path.fast_path(body)
    assert [i["addr"] for i in path] == [
        0x0,
        0x4,
        0x8,
        0xC,
        0x10,
        0x12,
        0x16,
        0x1A,
        0x1D,
        0x21,
        0x25,
    ]
    sig = fast_path.signature(path)
    assert sig == {
        "header": ["load+16", "load+8", "store+8"],
        "other_memory": 1,
        "conditional_branches": 1,
        "calls": 0,
    }
    # A branch to a relocated target (the cold section) is not followed.
    cold = [
        ins(0x0, "mov", "0x8(%rdi),%rax"),
        ins(0x4, "je", "6 <g+0x6>", reloc=".text.unlikely"),
        ins(0x6, "ret"),
    ]
    assert [i["addr"] for i in fast_path.fast_path(cold)] == [0x0, 0x4, 0x6]


def test_a_capacity_compare_is_a_load_of_8_and_a_compare_with_16_of_one_register() -> None:
    body = [
        ins(0x0, "mov", "0x8(%rbx),%rax"),
        ins(0x4, "cmp", "%rax,0x10(%rbx)"),
        ins(0x8, "je", "20 <f+0x20>"),
        ins(0xA, "mov", "0x8(%rcx),%rdx"),
        ins(0xE, "cmp", "0x10(%rbx),%rdx"),  # another register's offset 16: not one
        ins(0x12, "jne", "20 <f+0x20>"),
    ]
    assert fast_path.capacity_compares(body) == 1


@pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64")
    or shutil.which("objdump") is None
    or shutil.which("c++") is None,
    reason="needs x86-64, objdump and a C++ compiler",
)
@pytest.mark.parametrize("flags", ["-O2", "-O3"])
def test_region_push_back_has_push_backs_fast_path_and_a_planted_change_fails(flags) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(REPO / "parity/thread_list_fast_path.py"),
            "probe",
            "--plant",
            f"--flags={flags}",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode == 2:
        pytest.skip(run.stderr.strip() or "the probe could not be built")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "the planted change is caught" in run.stdout
