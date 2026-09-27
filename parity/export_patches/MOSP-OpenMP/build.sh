#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of MOSP-OpenMP@c352151 (PLAN Section 8.3). Called by
# parity/build_reference.sh with the PATCHED scratch copy as its only argument, after the copy was
# built with the original Makefile. It adds two programs under <copy>/parity_export/ and changes no
# original file:
#
#   parity_export/bin/export_graph_io   generateGraphCSR, generateChangedEdges, applyChangeBatch
#   parity_export/bin/export_sssp       the file-based sequentialSOSPUpdate / parallelSOSPUpdate
#
# The sources come from parity/exporters/mosp and are copied into the copy, so the copy records
# exactly what was built. Flags follow the original Makefile (-std=c++17 -O3 -fopenmp).
set -euo pipefail

copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exporters="$(cd "${here}/../../exporters/mosp" && pwd)"
dst="${copy}/parity_export"
mkdir -p "${dst}/src" "${dst}/bin"
for f in export_graph_io.cpp export_sssp.cpp; do
  # cp -p keeps the timestamp, so an unchanged source is not rebuilt.
  cmp -s "${exporters}/${f}" "${dst}/src/${f}" || cp -p "${exporters}/${f}" "${dst}/src/${f}"
done

cxx="${CXX:-g++}"
flags=(-std=c++17 -O3 -fopenmp -I"${copy}/headers")
src="${copy}/src"
build() { # <output> <sources...>
  local out="$1"
  shift
  local newest
  newest="$(ls -t "$@" | head -n 1)"
  if [ ! -x "${out}" ] || [ "${newest}" -nt "${out}" ]; then
    "${cxx}" "${flags[@]}" "$@" -o "${out}"
  fi
}
build "${dst}/bin/export_graph_io" "${dst}/src/export_graph_io.cpp" "${src}/csrGraph.cpp" \
  "${src}/generateGraphCSR.cpp" "${src}/generateChangedEdges.cpp" "${src}/read.cpp"
build "${dst}/bin/export_sssp" "${dst}/src/export_sssp.cpp" "${src}/csrGraph.cpp" \
  "${src}/read.cpp" "${src}/stageTimer.cpp" "${src}/sospUpdateCpu.cpp" \
  "${src}/sequentialSOSPUpdate.cpp" "${src}/parallelSOSPUpdate.cpp"
echo "export tools: ${dst}/bin/export_graph_io ${dst}/bin/export_sssp"
