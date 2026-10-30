# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of parity/sass_identity.py (M11a criterion 20): the objects it compares and the names of
anonymous namespaces, whose hash depends on the source tree's path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "parity"))
spec = importlib.util.spec_from_file_location("sass_identity", REPO / "parity/sass_identity.py")
sass_identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_identity)


def test_the_objects_are_the_03_cuda_code_the_criterion_names() -> None:
    names = [Path(o).name for o in sass_identity.OBJECTS]
    assert "escher_store.cu.o" in names and "sorted_merge.cu.o" in names
    assert {"cuda.cu.o", "counters.cu.o"} <= set(names)
    for o in sass_identity.OBJECTS:
        source = o.split(".dir/", 1)[1].removesuffix(".o")
        base = "cpp/src/algorithms/triad_count" if "triad_count" in o else "cpp"
        assert (REPO / base / source).is_file(), o


def test_an_anonymous_namespaces_hash_is_not_part_of_a_name() -> None:
    a = "_ZN4dyng6detail6escher48_GLOBAL__N__ce64ea3e_15_sorted_merge_cu_9763098c20narrow_kernelEv"
    b = "_ZN4dyng6detail6escher48_GLOBAL__N__ef5a956d_15_sorted_merge_cu_9763098c20narrow_kernelEv"
    assert sass_identity.ANONYMOUS.sub("_GLOBAL__N__", a) == sass_identity.ANONYMOUS.sub(
        "_GLOBAL__N__", b
    )
