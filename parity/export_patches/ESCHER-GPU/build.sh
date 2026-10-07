#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of ESCHER-GPU@abcf9b2 (PLAN Sections 6.4.5 and 8.3; ADRs 0013 and 0035).
# Called by parity/build_reference.sh with the PATCHED scratch copy as its only argument, after
# the copy was built with the original Makefile (make all-types build/cbst_test). It changes no
# tracked file: it relinks the copy's unchanged objects of every executable that links the CBST
# (build/main, build/type1, build/type2, build/type3, build/coarseTriangle) with the CBST trace
# recorder (parity/exporters/escher: cbst_recorder.cpp + recorder_escher_gpu.cpp) through
# -Xlinker --wrap=<symbol> for every symbol of wrap_symbols.txt (and, where the executable links
# the batch entry points, the hooks of wrap_hooks.txt for the batch dumps), and installs the twins in place
# of the executables, so tests/run_tests.sh runs them (parity/exporters/escher/relink_twins.sh).
# The originals are kept as parity_export/orig/<name>. build/cbst_test links only build_tree.o
# and insert_reuse.o (no CBSTOperations function) and stays as built.
#
# Recording is on only when DYNG_CBST_TRACE_DIR is set (ADR 0036); otherwise a twin behaves as
# the original (parity/exporters/escher/check_twins.py compares their outputs).
set -euo pipefail
copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${here}/../../exporters/escher/relink_twins.sh" "${copy}" recorder_escher_gpu.cpp \
  "${here}/wrap_symbols.txt" recorder_escher_gpu_hooks.cpp "${here}/wrap_hooks.txt" build/dynamic_hypergraph.o "include" -- \
  build/main build/type1 build/type2 build/type3 build/coarseTriangle
