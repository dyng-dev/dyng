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

# Identifiers of the originals (MOSP-OpenMP@c352151, MOSP-CUDA@e220ee2 for M1a). Extend this list
# when a port of another repository lands.
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
]
PINNED = re.compile(
    r"\b(MOSP-OpenMP|MOSP-CUDA|MOSP_ESCHER|ESCHER-GPU|LabelPropagation-CUDA|"
    r"CycleEnumeration-GPU)@[0-9a-f]{7,40}"
)
SYMBOL = re.compile(r"\b(" + "|".join(SYMBOLS) + r")\b")
HEADER = re.compile(r"^// Derived from [A-Za-z_-]+@[0-9a-f]{7,40}:", re.M)

# Files that name an original without copying its code: path -> reason.
NOT_DERIVED = {
    "cpp/include/dyng/sssp.hpp": "public API; documents the semantics the port reproduces",
    "cpp/include/dyng/graph/graph.hpp": "public API; documents the MOSP apply it ports",
    "cpp/include/dyng/graph/graph_properties.hpp": "public API; names the MOSP semantics preset",
    "cpp/include/dyng/io/batch_io.hpp": "public API of batch_io.cpp (which carries the header)",
    "cpp/include/dyng/io/csr_triplet.hpp": "public API of csr_triplet.cpp (header there)",
    "cpp/include/dyng/testing/check_sssp.hpp": "public API of check_sssp.cpp (header there)",
    "cpp/include/dyng/testing/dijkstra.hpp": "public API of dijkstra.cpp (header there)",
    "cpp/src/graph/graph_impl.hpp": "describes the apply_delta classification; no copied code",
}


def tracked() -> list[str]:
    out = subprocess.check_output(["git", "-C", str(REPO), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.startswith(SCOPES) and p.endswith(SUFFIXES)]


def main() -> int:
    missing = []
    stale = []
    for path in tracked():
        text = (REPO / path).read_text(encoding="utf-8", errors="replace")
        names_original = SYMBOL.search(text) or PINNED.search(text)
        if not names_original:
            if path in NOT_DERIVED:
                stale.append(path)
            continue
        if HEADER.search(text) or path in NOT_DERIVED:
            continue
        missing.append((path, names_original.group(0)))
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
