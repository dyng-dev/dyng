#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of MOSP_ESCHER@4b86159 (PLAN Sections 6.4.5 and 8.3; ADRs 0013 and 0035).
# Called by parity/build_reference.sh with the PATCHED scratch copy (an archive of the COMMITTED
# tree; the clone's uncommitted changes never reach it) as its only argument, after the copy was
# built with the original Makefile (make all). It changes no tracked file: it relinks the copy's
# unchanged objects and build/lib/libescher_core.a of every executable that links the CBST with
# the CBST trace recorder (parity/exporters/escher: cbst_recorder.cpp + recorder_mosp_escher.cpp)
# through -Xlinker --wrap=<symbol> for every symbol of wrap_symbols.txt (and, where the executable links
# the batch entry points, the hooks of wrap_hooks.txt for the batch dumps), and installs the twins in
# place of the executables, so tests/run_tests.sh (make test) runs them
# (parity/exporters/escher/relink_twins.sh). The originals are kept as parity_export/orig/<name>.
# tests/local/local_tests (host core only, built by run_tests.sh) links no CBST.
#
# Recording is on only when DYNG_CBST_TRACE_DIR is set (ADR 0036); otherwise a twin behaves as
# the original (parity/exporters/escher/check_twins.py compares their outputs).
set -euo pipefail
copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${here}/../../exporters/escher/relink_twins.sh" "${copy}" recorder_mosp_escher.cpp \
  "${here}/wrap_symbols.txt" recorder_mosp_escher_hooks.cpp "${here}/wrap_hooks.txt" build/hypergraph/src/DynamicHypergraph.cpp.o "escher/include escher/kernel hypergraph/include" -- \
  bin/main bin/stressTest bin/parallelStressTest bin/hsospBench bin/hsospStress \
  bin/test_cbst_smoke bin/test_cbst_ops bin/test_dynamicgraph_roundtrip \
  bin/test_snapshot_matches_updateCSR bin/test_h2h_construction bin/test_h2h_delta \
  bin/test_hsosp_matches_dijkstra bin/test_hsosp_scale bin/test_mosp_update
