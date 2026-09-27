#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Doxygen check of the public headers (PLAN Section 9.1): Doxygen with warnings as errors (every
# public entity documented, every parameter and return value described), then
# ci/doxygen_coverage.py on its XML (a @brief everywhere, every namespace-scope entity in a
# group, @backends / @determinism / @paper on compute() and update()). Needs a configured build
# tree for the generated version.hpp / config.hpp (default: build/cpu-only; override with
# DYNG_BUILD_DIR). Also available as the CMake target `docs-doxygen`.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

build_dir="${DYNG_BUILD_DIR:-build/cpu-only}"
if [ ! -f "${build_dir}/cpp/include/dyng/version.hpp" ]; then
  echo "ci/docs.sh: ${build_dir} is not configured; run: cmake --preset cpu-only" >&2
  exit 1
fi

export DYNG_GENERATED_INCLUDE="${build_dir}/cpp/include"
export DYNG_DOXYGEN_OUTPUT="${DYNG_DOXYGEN_OUTPUT:-build/doxygen}"
mkdir -p "${DYNG_DOXYGEN_OUTPUT}"
doxygen docs/Doxyfile
python3 ci/doxygen_coverage.py "${DYNG_DOXYGEN_OUTPUT}/xml"
echo "Doxygen XML written to ${DYNG_DOXYGEN_OUTPUT}/xml"
