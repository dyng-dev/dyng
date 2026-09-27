#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of MOSP-CUDA@e220ee2 (PLAN Section 8.3). Called by
# parity/build_reference.sh with the PATCHED scratch copy as its only argument, after the copy was
# built with the original Makefile. It adds one program under <copy>/parity_export/ and changes no
# original file:
#
#   parity_export/bin/export_graph_io   generateGraphCSR, generateChangedEdges, applyChangeBatch
#                                       of the CUDA repository (host code)
#
# Flags follow the original Makefile (nvcc -std=c++17 --extended-lambda -arch=sm_86 -O3).
set -euo pipefail

copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exporters="$(cd "${here}/../../exporters/mosp" && pwd)"
dst="${copy}/parity_export"
mkdir -p "${dst}/src" "${dst}/bin"
f=export_graph_io.cpp
cmp -s "${exporters}/${f}" "${dst}/src/${f}" || cp -p "${exporters}/${f}" "${dst}/src/${f}"

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
echo "export tools: ${out}"
