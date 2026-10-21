#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The documented way to use an installed dynG from CMake, run against a build: the `cmake` block
# of docs/getting_started/install.md ("Use dynG from CMake"; README.md's "Using the library from
# C++" must be the same block) is the CMakeLists.txt of a small consumer, configured and built
# against `cmake --install <build> --prefix <tmp>`; the consumer prints the linked library's
# version, which must be VERSION. It fails when the documented find_package() version no longer
# finds the installed package (SameMinorVersion before 1.0, cmake/install.cmake), as
# `find_package(dyng 0.1 REQUIRED)` did with 0.2.0rc1 (R020).
#
# Two more consumers of the installed package (M9b, ADR 0058: experimental::count_local_patterns):
#
#   - host_only: includes <dyng/experimental/local_patterns.hpp> (never the extension header
#     local_patterns.cuh), reads a hypergraph's rows with experimental::access_rows() and counts
#     with triad_count, configured as a CXX-only project: a program that does not use an Op needs
#     no CUDA, whatever the installed build;
#   - op (CUDA builds only): a user Op compiled by nvcc against the installed package
#     (find_package(dyng), project(... CXX CUDA), no flag of dynG's own build), counted with
#     count_local_patterns on cuda and on the host backends, updated under a batch and compared
#     with triad_count and with a fresh compute(). It runs on the GPU of CUDA_VISIBLE_DEVICES
#     (ci/gpu_local.sh runs it on GPU 1); without a visible GPU it runs the host backends only.
#
#   ci/cmake_consumer.sh [build directory]     # default build/cpu-only (ci/check.sh, step consumer)
#   ci/cmake_consumer.sh build/dev-cuda        # also the nvcc consumer (ci/gpu_local.sh)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$(realpath "${1:-${repo_root}/build/cpu-only}")"
if [ ! -f "${build}/CMakeCache.txt" ]; then
  echo "ci/cmake_consumer.sh: ${build} is not a configured build (build the cpu-only preset)" >&2
  exit 2
fi
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

cmake --install "${build}" --prefix "${tmp}/prefix" >"${tmp}/install.log"
mkdir -p "${tmp}/consumer"
python3 - "${repo_root}" "${tmp}/consumer" <<'EOF'
import re
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])


def block(path: Path) -> str:
    """The first ```cmake block of `path` that calls find_package(dyng ...)."""
    for body in re.findall(r"```cmake\n(.*?)```", path.read_text(), re.S):
        if "find_package(dyng" in body:
            return body
    raise SystemExit(f"{path}: no cmake block with find_package(dyng ...)")


install = block(repo / "docs/getting_started/install.md")
readme = block(repo / "README.md")
if install != readme:
    raise SystemExit("README.md and docs/getting_started/install.md show different CMake blocks")
(out / "CMakeLists.txt").write_text(
    "cmake_minimum_required(VERSION 3.30)\nproject(dyng_consumer CXX)\n"
    "set(CMAKE_CXX_STANDARD 17)\nadd_executable(my_app main.cpp)\n" + install
)
(out / "main.cpp").write_text(
    "#include <dyng/dyng.hpp>\n#include <cstdio>\n"
    'int main() { std::printf("%s\\n", dyng::library_version().string); return 0; }\n'
)
print(f"cmake_consumer: the documented block:\n{install}", end="")
EOF
cmake -S "${tmp}/consumer" -B "${tmp}/consumer/build" -G Ninja \
  -DCMAKE_PREFIX_PATH="${tmp}/prefix" -DCMAKE_BUILD_TYPE=Release >"${tmp}/configure.log" 2>&1 || {
  cat "${tmp}/configure.log"
  echo "cmake_consumer: the documented find_package() does not find the installed build" >&2
  exit 1
}
cmake --build "${tmp}/consumer/build" >"${tmp}/build.log" 2>&1 || {
  cat "${tmp}/build.log"
  exit 1
}
got="$("${tmp}/consumer/build/my_app")"
want="$(cat "${repo_root}/VERSION")"
if [ "${got}" != "${want}" ]; then
  echo "cmake_consumer: the consumer linked dynG ${got}, VERSION is ${want}" >&2
  exit 1
fi
echo "cmake_consumer: OK (the documented find_package() finds and links dynG ${got})"

major_minor="$(cut -d. -f1-2 "${repo_root}/VERSION")"
src="${repo_root}/ci/consumer"

# The host-only consumer: the experimental host header, no CUDA language.
mkdir -p "${tmp}/host_only"
cat >"${tmp}/host_only/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.30)
project(dyng_host_only CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(dyng ${major_minor} REQUIRED)
add_executable(host_only "${src}/host_only.cpp")
target_link_libraries(host_only PRIVATE dyng::dyng)
CMAKE
cmake -S "${tmp}/host_only" -B "${tmp}/host_only/build" -G Ninja \
  -DCMAKE_PREFIX_PATH="${tmp}/prefix" -DCMAKE_BUILD_TYPE=Release >"${tmp}/host_only.log" 2>&1 &&
  cmake --build "${tmp}/host_only/build" >>"${tmp}/host_only.log" 2>&1 || {
  cat "${tmp}/host_only.log"
  echo "cmake_consumer: the host-only consumer of the experimental header does not build" >&2
  exit 1
}
"${tmp}/host_only/build/host_only" || {
  echo "cmake_consumer: the host-only consumer failed" >&2
  exit 1
}
echo "cmake_consumer: OK (a host-only program uses the installed experimental host header)"

# The triad_count example against the installed package (PLAN 9.6, M9b): examples/cpp/
# triad_count_update.cpp built with find_package(dyng) and run on examples/data, its output
# compared with the committed expected output by examples/cpp/compare_output.cmake (the in-tree
# CTest example.triad_count_update.* does the same in the build tree). Backends: sequential,
# openmp when the build has it, cuda when it has CUDA and a device is visible (77 = skipped).
mkdir -p "${tmp}/triad_example"
cat >"${tmp}/triad_example/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.30)
project(dyng_triad_example CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(dyng ${major_minor} REQUIRED)
add_executable(triad_count_update "${repo_root}/examples/cpp/triad_count_update.cpp")
target_link_libraries(triad_count_update PRIVATE dyng::dyng)
CMAKE
cmake -S "${tmp}/triad_example" -B "${tmp}/triad_example/build" -G Ninja \
  -DCMAKE_PREFIX_PATH="${tmp}/prefix" -DCMAKE_BUILD_TYPE=Release >"${tmp}/triad_example.log" 2>&1 &&
  cmake --build "${tmp}/triad_example/build" >>"${tmp}/triad_example.log" 2>&1 || {
  cat "${tmp}/triad_example.log"
  echo "cmake_consumer: examples/cpp/triad_count_update.cpp does not build against the" \
    "installed package" >&2
  exit 1
}
data="${repo_root}/examples/data"
backends="sequential"
grep -q '^DYNG_ENABLE_OPENMP:BOOL=ON' "${build}/CMakeCache.txt" && backends="${backends} openmp"
grep -q '^DYNG_ENABLE_CUDA:BOOL=ON' "${build}/CMakeCache.txt" && backends="${backends} cuda"
for b in ${backends}; do
  rc=0
  cmake "-DEXE=${tmp}/triad_example/build/triad_count_update" \
    "-DARGS=${data}/triad_count.hg|${data}/triad_count.dgt|${b}" \
    "-DEXPECTED=${data}/triad_count_update.expected.txt" \
    -P "${repo_root}/examples/cpp/compare_output.cmake" >"${tmp}/triad_example.${b}.log" 2>&1 || rc=$?
  if [ "${rc}" -eq 77 ] || grep -q 'skipped (exit code 77)' "${tmp}/triad_example.${b}.log"; then
    echo "cmake_consumer: triad_count_update ${b}: skipped (no CUDA device visible)"
  elif [ "${rc}" -ne 0 ]; then
    cat "${tmp}/triad_example.${b}.log"
    echo "cmake_consumer: the installed triad_count example differs from its expected output" \
      "on ${b}" >&2
    exit 1
  fi
done
echo "cmake_consumer: OK (examples/cpp/triad_count_update.cpp, built against the installed" \
  "package, prints its expected output on: ${backends})"

# The nvcc consumer: a user Op against the installed package (CUDA builds only).
if ! grep -q '^DYNG_ENABLE_CUDA:BOOL=ON' "${build}/CMakeCache.txt"; then
  echo "cmake_consumer: ${build} has no CUDA; the nvcc consumer of count_local_patterns is skipped"
  exit 0
fi
nvcc="$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "${build}/CMakeCache.txt")"
mkdir -p "${tmp}/op"
cat >"${tmp}/op/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.30)
project(dyng_op_consumer CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(dyng ${major_minor} REQUIRED)
add_executable(op "${src}/op.cu")
set_target_properties(op PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(op PRIVATE dyng::dyng)
CMAKE
cmake -S "${tmp}/op" -B "${tmp}/op/build" -G Ninja -DCMAKE_CUDA_COMPILER="${nvcc}" \
  -DCMAKE_PREFIX_PATH="${tmp}/prefix" -DCMAKE_BUILD_TYPE=Release >"${tmp}/op.log" 2>&1 &&
  cmake --build "${tmp}/op/build" >>"${tmp}/op.log" 2>&1 || {
  cat "${tmp}/op.log"
  echo "cmake_consumer: a user Op does not build with nvcc against the installed package" >&2
  exit 1
}
"${tmp}/op/build/op" || {
  echo "cmake_consumer: the user Op's program failed" >&2
  exit 1
}
echo "cmake_consumer: OK (a user Op compiled by nvcc against the installed package counts on" \
  "every backend)"
