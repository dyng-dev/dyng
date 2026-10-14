# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/provenance_check.py: the CBST identifiers of the store port (M8a review)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "provenance_check", REPO / "ci" / "provenance_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["provenance_check"] = module
    spec.loader.exec_module(module)
    return module


provenance = _load()


def test_every_explicit_rename_of_the_store_port_is_a_symbol():
    originals, ported = provenance.rename_map_symbols()
    assert "chainedRowLengths" in originals and "HostTree" in originals
    for name in originals + ported:
        assert name in provenance.SYMBOLS, name
    # The dynG names of the kernels and internal functions, not the types the store's users name.
    assert "insert_node_thread_kernel" in ported
    assert "construct_cbst" in ported
    assert "cbst_store" not in provenance.SYMBOLS
    assert "cbst_context" not in provenance.SYMBOLS


def test_a_planted_original_name_without_a_header_is_reported():
    for body in (
        "/* unfillCBST */\n",
        "findNode<<<1, 32, 0, s>>>(); chainedRowLengths(x);\n",
        "__global__ void insert_node_thread_kernel(); walk_chain(); compact_cbst();\n",
    ):
        verdict, what = provenance.check_text("cpp/src/hypergraph/plant.cu", body)
        assert verdict == "missing", body
        assert what
    header = "// Derived from ESCHER-GPU@abcf9b2:kernel/payload.cu\n"
    verdict, _ = provenance.check_text(
        "cpp/src/hypergraph/plant.cu", header + "__global__ void insert_node_thread_kernel();\n"
    )
    assert verdict == "ok"
    # A caller of the store names its types only.
    verdict, _ = provenance.check_text(
        "cpp/src/hypergraph/container.cpp", "cbst_store trees_; cbst_context ctx;\n"
    )
    assert verdict == "ok"


def test_the_rename_map_must_have_its_explicit_block():
    try:
        provenance.rename_map_symbols("[names]\nfoo = 'bar'\n")
    except SystemExit:
        return
    raise AssertionError("a rename map without its '# explicit' block was accepted")
