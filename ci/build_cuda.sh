#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Compile-only CUDA build (PLAN Section 8.8), the body of .github/workflows/cuda-build.yml and its
# local equivalent: configure and build a CUDA preset for the release architecture list with
# warnings as errors (library, tests, tools; nothing is run, no GPU is needed), then report the
# kernels' resource usage (registers, stack, shared memory per architecture: cuobjdump
# --dump-resource-usage) and the size of the library (the input of the wheel size budget, PLAN
# Section 7.7).
#
#   ci/build_cuda.sh                    # preset ci-cuda13 with the CUDA of CUDACXX / PATH
#   ci/build_cuda.sh ci-cuda12          # in the CUDA 12 container
#   CUDACXX=/path/to/cuda-12.9/bin/nvcc ci/build_cuda.sh ci-cuda12   # locally, another toolkit
#
# Writes build/<preset>/cuda_build_report.md (and appends it to $GITHUB_STEP_SUMMARY in Actions).
# On the development machine the build takes the shared perf lock and runs niced (as every heavy
# command there); set DYNG_PERF_LOCK= (empty) to skip the lock.
set -euo pipefail

preset="${1:-ci-cuda13}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [ -z "${CI:-}" ] && [ -f scripts/dev_env.sh ]; then
  cudacxx_override="${CUDACXX:-}"
  # shellcheck disable=SC1091
  source scripts/dev_env.sh
  if [ -n "${cudacxx_override}" ]; then
    export CUDACXX="${cudacxx_override}"
  fi
fi

lock="${DYNG_PERF_LOCK-${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}/perf.lock}"
heavy() {
  if [ -n "${lock}" ] && [ -e "${lock}" ]; then
    flock -s "${lock}" nice -n 10 "$@"
  else
    "$@"
  fi
}

build_dir="build/${preset}"
report="${build_dir}/cuda_build_report.md"

echo "==> ${preset}: configure (CUDACXX=${CUDACXX:-nvcc on PATH})"
heavy cmake --preset "${preset}"
echo "==> ${preset}: build (compile only)"
heavy cmake --build --preset "${preset}"

nvcc_bin="$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "${build_dir}/CMakeCache.txt")"
cuobjdump="$(dirname "${nvcc_bin}")/cuobjdump"
[ -x "${cuobjdump}" ] || cuobjdump="$(command -v cuobjdump || true)"
library="$(find "${build_dir}/cpp" -maxdepth 1 -name 'libdyng.so*' -type f | head -n 1)"
[ -n "${library}" ] || library="$(find "${build_dir}/cpp" -maxdepth 1 -name 'libdyng.a' | head -n 1)"

{
  echo "### CUDA compile-only build: ${preset}"
  echo
  echo "- toolkit: $("${nvcc_bin}" --version | sed -n 's/.*release \([0-9.]*\), \(V[0-9.]*\).*/CUDA \1 (\2)/p')"
  echo "- host compiler: $("$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${build_dir}/CMakeCache.txt")" \
    --version | head -n 1)"
  echo "- architectures: $(sed -n 's/^DYNG_CUDA_ARCHITECTURES_RESOLVED:[A-Z]*=//p' \
    "${build_dir}/CMakeCache.txt")"
  stripped="$(mktemp)"
  strip -o "${stripped}" "${library}" 2>/dev/null || cp "${library}" "${stripped}"
  echo "- library: $(basename "${library}"), $(du -k "${library}" | cut -f1) KiB," \
    "$(du -k "${stripped}" | cut -f1) KiB stripped"
  rm -f "${stripped}"
  echo
  echo "Kernel resource usage (cuobjdump --dump-resource-usage; REG = registers per thread):"
  echo
  echo '```'
  if [ -n "${cuobjdump}" ]; then
    "${cuobjdump}" --dump-resource-usage "${library}" 2>/dev/null |
      awk '/arch = / {arch=$3} /Function / {fn=$2; sub(":$","",fn)}
           /REG:/ {print arch, $1, $2, $3, $4, fn}' |
      c++filt | sort -u || true
  else
    echo "cuobjdump not found"
  fi
  echo '```'
} >"${report}"
cat "${report}"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "${report}" >>"${GITHUB_STEP_SUMMARY}"
fi
