#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The host sanitizers (PLAN Sections 7.3 and 8.8; docs/developer/robustness.md): configure, build
# and run the CPU tests (ctest -L cpu) of the sanitizer presets. .github/workflows/sanitizers.yml
# runs one preset per job; locally the script runs the ones named, or all three.
#
#   ci/sanitizers.sh                    # asan, tsan and tsan-openmp, one after the other
#   ci/sanitizers.sh asan               # one preset
#
# Presets:
#   asan         AddressSanitizer + UndefinedBehaviorSanitizer (leak checks on), the OpenMP
#                backends included; any compiler (CXX, default the system's)
#   tsan         ThreadSanitizer with OpenMP off: the std::thread code (the parallel text parsers,
#                the concurrent jobs of util/concurrent.hpp without OpenMP, the profiler); any
#                compiler (GCC's libgomp is not instrumented, hence OpenMP off)
#   tsan-openmp  ThreadSanitizer with the OpenMP backends: Clang and its OpenMP runtime libomp,
#                whose OMPT tool Archer tells TSan about OpenMP's barriers, reductions and
#                schedules (without it every parallel region is reported as a race). CXX defaults
#                to clang++-18 (else clang++); the tests run with OMP_TOOL_LIBRARIES set to
#                Archer and OMP_NUM_THREADS=4 (the test preset's environment)
#
# Environment: CXX (the C++ compiler of the configure step), DYNG_ARCHER_LIBRARY (tsan-openmp:
# the path of libarcher.so; default: the one next to the compiler's libraries, as
# `$CXX -print-file-name=libarcher.so` finds it; Debian/Ubuntu ship it in libomp5-<N>),
# DYNG_SANITIZER_JOBS (parallel tests, default nproc). Exit status 0 when every preset's tests
# passed; a sanitizer report fails its test (halt_on_error=1 in the test presets).
#
# Heavy (a full Debug build per preset, then the test suite under the sanitizer): on the shared
# development machine run it under the shared perf lock and niced:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/sanitizers.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if ! command -v cmake >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${repo_root}/scripts/dev_env.sh"
fi

presets=()
for arg in "$@"; do
  case "${arg}" in
    asan | tsan | tsan-openmp) presets+=("${arg}") ;;
    -h | --help)
      sed -n '4,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "ci/sanitizers.sh: unknown preset '${arg}' (asan, tsan or tsan-openmp; try --help)" >&2
      exit 2
      ;;
  esac
done
if [ "${#presets[@]}" -eq 0 ]; then
  presets=(asan tsan tsan-openmp)
fi
jobs="${DYNG_SANITIZER_JOBS:-$(nproc)}"

failed=()
for preset in "${presets[@]}"; do
  cxx="${CXX:-}"
  if [ "${preset}" = tsan-openmp ]; then
    if [ -z "${cxx}" ]; then
      if command -v clang++-18 >/dev/null 2>&1; then
        cxx=clang++-18
      else
        cxx=clang++
      fi
    fi
    if ! "${cxx}" --version 2>/dev/null | grep -qi clang; then
      echo "ci/sanitizers.sh: tsan-openmp needs Clang (CXX=clang++), got '${cxx}'" >&2
      exit 2
    fi
    archer="${DYNG_ARCHER_LIBRARY:-$("${cxx}" -print-file-name=libarcher.so)}"
    if [ "${archer#/}" = "${archer}" ] || [ ! -f "${archer}" ]; then
      echo "ci/sanitizers.sh: Archer (libarcher.so) not found for ${cxx}: install the OpenMP" \
        "runtime of that Clang (Debian/Ubuntu: libomp-<N>-dev) or set DYNG_ARCHER_LIBRARY" >&2
      exit 2
    fi
    export OMP_TOOL_LIBRARIES="${archer}"
    echo "==> Archer: ${archer}"
  fi

  printf '\n==> preset %s: configure, build, test (label cpu)\n' "${preset}"
  configure=(cmake --preset "${preset}")
  if [ -n "${cxx}" ]; then
    configure=(env "CXX=${cxx}" "${configure[@]}")
  fi
  if "${configure[@]}" && cmake --build --preset "${preset}" &&
    ctest --preset "${preset}" -L cpu -j "${jobs}"; then
    echo "preset ${preset}: OK"
  else
    failed+=("${preset}")
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/sanitizers.sh: FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "ci/sanitizers.sh: all presets passed (${presets[*]})"
