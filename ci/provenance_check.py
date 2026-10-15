#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Provenance check (PLAN Sections 3.4 and 6.3 step 5).

Every file that ports code from one of the six original repositories carries the header line

    // Derived from <repo>@<sha>:<path> (...)

This check fails when a library or tool source file names a function, class or tool of an
original (the symbols below) but has no such line. Files that only mention an original to
describe a format or a behaviour they do not copy are listed in NOT_DERIVED, each with its reason,
so an exemption is a reviewed decision and not an accident.

    python3 ci/provenance_check.py          # exit 1 and list the files on failure
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories whose sources are library or tool code (tests and exporters are harness code).
SCOPES = ("cpp/src/", "cpp/include/", "tools/")
SUFFIXES = (".hpp", ".cpp", ".cuh", ".cu", ".h")

# Identifiers of the originals (MOSP-OpenMP@c352151, MOSP-CUDA@e220ee2 for M1a;
# CycleEnumeration-GPU@0a976ad for M2). Extend this list when a port of another repository lands.
SYMBOLS = [
    "sospUpdateCpu",
    "sospFromScratchCpu",
    "sospUpdateGpu",
    "sequentialSOSPUpdate",
    "SospWorkspace",
    "nextGeneration",
    "defaultDelta",
    "choosePacking",
    "canonicalizeTree",
    "mospUpdate",
    "applyChangeBatch",
    "updateGraphCSR",
    "transposeCsrGraph",
    "readChangeBatch",
    "readCsrGraph",
    "readIntFile",
    "readValuesFile",
    "writeCsrGraph",
    "runConcurrently",
    "ListGather",
    "TextWriter",
    "TextScanner",
    "mtxToCsr",
    "dijkstraCsrGraph",
    "checkSospTree",
    "combinedGraphSospCpu",
    "buildHostGraph",
    # MOSP-CUDA@e220ee2 / MOSP-OpenMP@c352151, the mosp port (M7)
    "combinedGraphSospGpu",
    "combinedGraphReference",
    "mospPathCosts",
    "preferenceScale",
    "combinedEdgeWeight",
    "countEdgesKernel",
    "fillEdgesKernel",
    "CombineWorkspace",
    # CycleEnumeration-GPU@0a976ad (M2)
    "prepare_batch",
    "build_directed_graph",
    "DirectedGraph",
    "EdgeBatch",
    "BatchParams",
    "generate_batch",
    "sort_and_dedup",
    "read_temporal_graph",
    "read_graph_view",
    "group_edges",
    "parse_matrix_market_banner",
    "count_simple_cycles_bruteforce",
    "oracle_simple_cycles",
    "parallel_parts",
    "CycleHistogram",
    "count_simple_cycles_johnson",
    "update_static_histogram",
    "count_cycles_through_edge",
    "ChangedEdgeIndex",
    "apply_histogram_delta",
    "JohnsonSearch",
    "circuit_bounded",
    "extend_prefix",
    "update_histogram",
    # ESCHER-GPU@abcf9b2 / MOSP_ESCHER@4b86159, the CBST store (M8a)
    "CBSTOperations",
    "CBSTContext",
    "CBSTNode",
    "constructCBST",
    "rebuildCBSTFromDevice",
    "insertCBST",
    "fillCBST",
    "deleteCBST",
    "unfillCBST",
    "compactCBST",
    "cbstRankOfPosition",
    "initTailMetadata",
    "ensurePayloadCapacity",
    "InsertMapping",
    "DegreeBinLaunches",
    # ESCHER-GPU@abcf9b2, the store port (M8a step 3; parity/escher_port/rename_map.toml)
    "applyReuse",
    "applyReuse_warp",
    "applyReuse_block",
    "insertNode_thread",
    "insertNode_warp",
    "insertNode_block",
    "unfillRowSerial",
    "unfill_thread",
    "unfill_warp",
    "unfill_block",
    "locateReusableSlots",
    "locateDeleteTargets",
    "applyDeletes",
    "reduceAvailLevel",
    "extractSlotCapacities",
    "extractKeysFromPositions",
    "lowerBoundKernel",
    "pairMatches",
    "buildEmptyBinaryTree",
    "storeItemsIntoNodes",
    "placeNodeRecords",
    "rankOrderNodes",
    "recordKeysAndStarts",
    "allocateSpace",
    "fixupOverflowMetadata",
    "computeNextMultipleOf4",
    "updatePartialSolution",
    "reserveScratch",
    "recomputeSubtreeAvail",
    "checkTreeRows",
    "flatten2DVector",
    "RowLocator",
    "RowView",
    "buildRowLocatorKernel",
    "TempBuffer",
    "DegreeBins",
    # ESCHER-GPU@abcf9b2, the wedge engine and the counters (M9a; rename_map_triad.toml)
    "computeMotifCounts",
    "runFrontierPhase",
    "MotifDeltaAccumulator",
    "MotifOp",
    "Choose3Op",
    "Type2Op",
    "Type3Op",
    "countWedgeShape",
    "forEachWedge",
    "TriangleBody",
    "MotifCounter",
    "Type1Counter",
    "DeviceBins",
    "count_wedge_shape",
    "for_each_wedge",
    "triangle_body",
    "motif_op",
    "choose3_op",
    "type2_op",
    "type3_op",
    "compute_motif_index",
    # MOSP_ESCHER@4b86159, its behaviours merged into the store (M8a step 4)
    "readFlat",
    "setInitialOccupancy",
    "warpWalkRow",
    "compactSizesKernel",
    "compactCopyKernel",
    "ensureScratchCapacity",
    "checkSubtreeAvail",
    "EscherError",
]

RENAME_MAP = REPO / "parity" / "escher_port" / "rename_map.toml"


def rename_map_symbols(text: str | None = None) -> tuple[list[str], list[str]]:
    """The ESCHER-GPU identifiers the store port renamed explicitly, and the distinctive dynG names
    they became (the `# explicit` block of parity/escher_port/rename_map.toml's [names]).

    The original names are symbols of an original like every other in SYMBOLS. Of the dynG names,
    the kernels (`*_kernel`) and the store's internal free functions are listed too, so that port
    code moved out of cpp/src/hypergraph/escher under its dynG names still needs its header; the
    types the store's users name (cbst_store, cbst_context, cbst_node, insert_mapping, ...) are
    not (a caller of the store does not copy it).
    """
    if text is None:
        text = RENAME_MAP.read_text(encoding="utf-8")
    block = re.search(r"^\[names\]\s*\n# explicit\n(.*?)^# camelCase", text, re.M | re.S)
    if not block:
        raise SystemExit(f"{RENAME_MAP}: no '# explicit' block in [names]")
    originals, ported = [], []
    entry = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]+)"', re.M)
    for key, value in entry.findall(block.group(1)):
        originals.append(key)
        if value.endswith("_kernel") or value in PORT_FUNCTIONS:
            ported.append(value)
    return originals, ported


# The store port's internal free functions (dynG names; M8a review): distinctive, and called only by
# the port itself and its tests.
PORT_FUNCTIONS = {
    "construct_cbst",
    "rebuild_cbst_from_device",
    "insert_cbst",
    "fill_cbst",
    "delete_cbst",
    "unfill_cbst",
    "flatten_rows",
    "for_each_chained_entry",
}
# dynG names of MOSP_ESCHER's merged behaviours (M8a step 4) that the rename map does not list.
PORT_EXTRA = ["compact_cbst", "walk_chain", "compact_sizes_kernel", "compact_copy_kernel"]

_ORIGINAL_NAMES, _PORT_NAMES = rename_map_symbols()
SYMBOLS = sorted(set(SYMBOLS) | set(_ORIGINAL_NAMES) | set(_PORT_NAMES) | set(PORT_EXTRA))

PINNED = re.compile(
    r"\b(MOSP-OpenMP|MOSP-CUDA|MOSP_ESCHER|ESCHER-GPU|LabelPropagation-CUDA|"
    r"CycleEnumeration-GPU)@[0-9a-f]{7,40}"
)
SYMBOL = re.compile(r"\b(" + "|".join(SYMBOLS) + r")\b")
HEADER = re.compile(r"^// Derived from [A-Za-z_-]+@[0-9a-f]{7,40}:", re.M)

# Files that name an original without copying its code: path -> reason.
NOT_DERIVED = {
    "tools/compat/escher/escher_compat.cpp": "converts ESCHER-GPU's recorded batches to dynG files "
    "and replays them; no copied code (its command-line parsing is escher_dump.cpp's)",
    "cpp/include/dyng/sssp.hpp": "public API; documents the semantics the port reproduces",
    "cpp/include/dyng/graph/graph.hpp": "public API; documents the MOSP apply it ports",
    "cpp/include/dyng/graph/graph_properties.hpp": "public API; names the MOSP semantics preset",
    "cpp/include/dyng/io/batch_io.hpp": "public API of batch_io.cpp (which carries the header)",
    "cpp/include/dyng/io/csr_triplet.hpp": "public API of csr_triplet.cpp (header there)",
    "cpp/include/dyng/testing/check_sssp.hpp": "public API of check_sssp.cpp (header there)",
    "cpp/include/dyng/testing/dijkstra.hpp": "public API of dijkstra.cpp (header there)",
    "cpp/include/dyng/io/edge_list_io.hpp": "public API of edge_list_io.cpp (header there)",
    "cpp/include/dyng/testing/cycle_oracle.hpp": "public API of cycle_oracle.cpp (header there)",
    "cpp/src/graph/graph_impl.hpp": "describes the apply_delta classification; no copied code",
    "cpp/include/dyng/cycle_count.hpp": "public API; names the CycleEnum functions it ports",
    "cpp/include/dyng/io/result_io.hpp": "public API of result_io.cpp (header there)",
    "cpp/src/framework/workspace.hpp": "new pool; names MOSP's shared SospWorkspace it mirrors",
    "cpp/include/dyng/mosp.hpp": "public API; documents the semantics the port reproduces",
    "cpp/include/dyng/testing/mosp_oracle.hpp": "public API of mosp_oracle.cpp (header there)",
    "cpp/include/dyng/testing/cbst_host_model.hpp": "public API of cbst_host_model.cpp (header "
    "there)",
}


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "-C", str(REPO), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.startswith(SCOPES) and p.endswith(SUFFIXES)]


def check_text(path: str, text: str) -> tuple[str, str]:
    """The verdict on one file: ("ok", ""), ("missing", the name it uses) or ("stale", "")."""
    names_original = SYMBOL.search(text) or PINNED.search(text)
    if not names_original:
        return ("stale", "") if path in NOT_DERIVED else ("ok", "")
    if HEADER.search(text) or path in NOT_DERIVED:
        return ("ok", "")
    return ("missing", names_original.group(0))


def main() -> int:
    missing = []
    stale = []
    for path in tracked():
        text = (REPO / path).read_text(encoding="utf-8", errors="replace")
        verdict, what = check_text(path, text)
        if verdict == "stale":
            stale.append(path)
        elif verdict == "missing":
            missing.append((path, what))
    for path, what in missing:
        print(
            f"{path}: names '{what}' of an original but has no '// Derived from <repo>@<sha>:"
            "<path>' line (add it, or list the file in NOT_DERIVED with a reason)"
        )
    for path in stale:
        print(f"{path}: listed in NOT_DERIVED but names no original any more (remove the entry)")
    if missing or stale:
        return 1
    print(f"provenance: OK ({len(tracked())} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
