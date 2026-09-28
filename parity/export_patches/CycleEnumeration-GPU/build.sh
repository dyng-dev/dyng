#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of CycleEnumeration-GPU@0a976ad (PLAN Section 8.3). Called by
# parity/build_reference.sh with the PATCHED scratch copy as its only argument, after the copy was
# built with the original CMake build. It adds one program under <copy>/parity_export/ and changes
# no original file:
#
#   parity_export/bin/export_cycle_enum   the parser (read_temporal_graph, read_graph_view),
#                                         build_directed_graph, prepare_batch, apply_batch,
#                                         generate_batch and the tests' subset-DP / brute-force
#                                         oracles (see parity/exporters/cycle_enum)
#
# The source comes from parity/exporters/cycle_enum and is copied into the copy, so the copy
# records exactly what was built. It is compiled together with the original's unchanged sources
# with the flags of the original's Release build (-std=c++17 -O3 -DNDEBUG, std::thread).
set -euo pipefail

copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exporters="$(cd "${here}/../../exporters/cycle_enum" && pwd)"
dst="${copy}/parity_export"
mkdir -p "${dst}/src" "${dst}/bin"
f=export_cycle_enum.cpp
# cp -p keeps the timestamp, so an unchanged source is not rebuilt.
cmp -s "${exporters}/${f}" "${dst}/src/${f}" || cp -p "${exporters}/${f}" "${dst}/src/${f}"

cxx="${CXX:-g++}"
flags=(-std=c++17 -O3 -DNDEBUG -pthread -I"${copy}/include" -I"${copy}/src/core" -I"${copy}/tests")
src="${copy}/src"
sources=("${dst}/src/${f}" "${src}/core/graph.cpp" "${src}/core/graph_builder.cpp"
  "${src}/core/histogram.cpp" "${src}/core/timestamp.cpp" "${src}/dynamic/directed_graph.cpp"
  "${src}/dynamic/edge_change.cpp" "${src}/dynamic/batch_generator.cpp"
  "${src}/sequential/bruteforce.cpp")
out="${dst}/bin/export_cycle_enum"
newest="$(ls -t "${sources[@]}" | head -n 1)"
if [ ! -x "${out}" ] || [ "${newest}" -nt "${out}" ]; then
  "${cxx}" "${flags[@]}" "${sources[@]}" -o "${out}"
fi
echo "export tool: ${out}"
