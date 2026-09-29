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
#                                         generate_batch, the tests' subset-DP / brute-force
#                                         oracles, the sequential Johnson and OpenMP counters and
#                                         update_static_histogram[_openmp] (see
#                                         parity/exporters/cycle_enum)
#
#   parity_export/bin/export_cycle_enum_cuda
#                                         the same program with the original's CUDA backend
#                                         (count_simple_cycles_johnson[_work_queue],
#                                         update_static_histogram_cuda), linked against the copy's
#                                         CUDA build (only when that build compiled the kernels)
#
# The source comes from parity/exporters/cycle_enum and is copied into the copy, so the copy
# records exactly what was built. It is compiled together with the original's unchanged sources
# with the flags of the original's Release build (-std=c++17 -O3 -DNDEBUG, std::thread, OpenMP
# with CYCLE_ENUM_OPENMP_ENABLED=1 as the original's CMake defines it).
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
flags=(-std=c++17 -O3 -DNDEBUG -pthread -fopenmp -DCYCLE_ENUM_OPENMP_ENABLED=1 -I"${copy}/include"
  -I"${copy}/src/core" -I"${copy}/tests")
src="${copy}/src"
sources=("${dst}/src/${f}" "${src}/core/graph.cpp" "${src}/core/graph_builder.cpp"
  "${src}/core/histogram.cpp" "${src}/core/timestamp.cpp" "${src}/dynamic/directed_graph.cpp"
  "${src}/dynamic/edge_change.cpp" "${src}/dynamic/batch_generator.cpp"
  "${src}/sequential/bruteforce.cpp" "${src}/sequential/johnson.cpp"
  "${src}/openmp/openmp_johnson.cpp" "${src}/openmp/openmp_config.cpp"
  "${src}/dynamic/cycles_through_edge.cpp" "${src}/dynamic/update_sequential.cpp"
  "${src}/dynamic/update_openmp.cpp")
out="${dst}/bin/export_cycle_enum"
newest="$(ls -t "${sources[@]}" | head -n 1)"
if [ ! -x "${out}" ] || [ "${newest}" -nt "${out}" ]; then
  "${cxx}" "${flags[@]}" "${sources[@]}" -o "${out}"
fi
echo "export tool: ${out}"

# The same source with the original's CUDA backend (M2b): linked against the static libraries of
# the copy's own CUDA build (-DCYCLE_ENUM_ENABLE_CUDA=ON), whose kernels were compiled by its CMake
# build with nvcc; only when that build made them.
build="${copy}/build"
cuda_libs=("${build}/libcycle_enum_dynamic.a" "${build}/libcycle_enum_cuda.a"
  "${build}/libcycle_enum_openmp.a" "${build}/libcycle_enum_sequential.a"
  "${build}/libcycle_enum_core.a")
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.1}"
if [ -f "${build}/libcycle_enum_cuda.a" ] &&
  [ -f "${build}/CMakeFiles/cycle_enum_cuda.dir/src/cuda/cuda_static_kernels.cu.o" ]; then
  out_cuda="${dst}/bin/export_cycle_enum_cuda"
  newest="$(ls -t "${dst}/src/${f}" "${cuda_libs[@]}" | head -n 1)"
  if [ ! -x "${out_cuda}" ] || [ "${newest}" -nt "${out_cuda}" ]; then
    "${cxx}" "${flags[@]}" -DCYCLE_ENUM_CUDA_ENABLED=1 "${dst}/src/${f}" "${cuda_libs[@]}" \
      -L"${cuda_home}/lib64" -Wl,-rpath,"${cuda_home}/lib64" -lcudart -o "${out_cuda}"
  fi
  echo "export tool (cuda): ${out_cuda}"
fi
