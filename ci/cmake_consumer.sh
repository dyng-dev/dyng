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
#   ci/cmake_consumer.sh [build directory]     # default build/cpu-only (ci/check.sh, step consumer)
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
