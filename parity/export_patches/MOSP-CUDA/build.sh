#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of MOSP-CUDA@e220ee2 (PLAN Section 8.3). Called by
# parity/build_reference.sh with the PATCHED scratch copy as its only argument, after the copy was
# built with the original Makefile. It adds two programs under <copy>/parity_export/ and changes no
# original file:
#
#   parity_export/bin/export_graph_io   generateGraphCSR, generateChangedEdges, applyChangeBatch
#                                       of the CUDA repository (host code)
#   parity_export/bin/export_sssp       the file-based sequentialSOSPUpdate / parallelSOSPUpdate
#                                       (sospUpdateGpu) of the CUDA repository, linked against the
#                                       objects the original Makefile built (build/*.o)
#
# Flags follow the original Makefile (nvcc -std=c++17 --extended-lambda -arch=sm_86 -O3
# -lineinfo).
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
f=export_graph_io.cpp

nvcc="${NVCC:-/usr/local/cuda-13.1/bin/nvcc}"
arch="${CUDA_ARCH:-sm_86}"
src="${copy}/src"
sources=("${dst}/src/${f}" "${src}/csrGraph.cu" "${src}/generateGraphCSR.cu"
  "${src}/generateChangedEdges.cu" "${src}/read.cu")
out="${dst}/bin/export_graph_io"
newest="$(ls -t "${sources[@]}" | head -n 1)"
if [ ! -x "${out}" ] || [ "${newest}" -nt "${out}" ]; then
  # The exporter itself is host C++; -x cu lets nvcc compile it next to the .cu sources.
  "${nvcc}" -std=c++17 --extended-lambda -arch="${arch}" -O3 -I"${copy}/headers" \
    -x cu "${sources[@]}" -o "${out}"
fi

# export_sssp: the Makefile's BASE_SRCS objects plus the two file-based updates (the objects of
# bin/parallelStressTest without its main).
base=(generateGraph generateGraphCSR generateChangedEdges updateGraphCSR generateTestCases Dijkstra
  read csrGraph stageTimer validation changeGenerator sospUpdateGpu combinedGraphGpu deviceGraph
  mospUpdate sequentialSOSPUpdate parallelSOSPUpdate)
objects=()
for o in "${base[@]}"; do
  objects+=("${copy}/build/${o}.o")
done
out2="${dst}/bin/export_sssp"
newest="$(ls -t "${dst}/src/export_sssp.cpp" "${objects[@]}" | head -n 1)"
if [ ! -x "${out2}" ] || [ "${newest}" -nt "${out2}" ]; then
  "${nvcc}" -std=c++17 --extended-lambda -arch="${arch}" -O3 -lineinfo -I"${copy}/headers" \
    -x cu -c "${dst}/src/export_sssp.cpp" -o "${dst}/src/export_sssp.o"
  "${nvcc}" -arch="${arch}" "${dst}/src/export_sssp.o" "${objects[@]}" -o "${out2}"
fi
echo "export tools: ${out} ${out2}"
