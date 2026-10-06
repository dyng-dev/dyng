#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The reader fuzzers (PLAN Sections 5.7 and 8.1; cpp/fuzz/README.md): build the libFuzzer
# targets with the preset `fuzz` (Clang: -fsanitize=fuzzer,address,undefined), replay the
# committed seed corpus and crash reproducers once (ctest --preset fuzz), then run every target
# (or the ones named) for a bounded time.
#
#   ci/fuzz.sh                        # every target, 60 s each
#   ci/fuzz.sh --time 600 edge_list   # one target, 10 minutes
#   ci/fuzz.sh --build-only           # build and replay the corpus only
#
# Environment: CXX (default clang++-18 if it exists, else clang++), DYNG_FUZZ_JOBS (parallel
# fuzzing processes per target, default 1), DYNG_FUZZ_OUT (default build/fuzz-runs): each
# target's working corpus starts as a copy of cpp/fuzz/corpus/<target> in <out>/corpus/<target>
# (new inputs never touch the repository), and a crash is written to <out>/artifacts/<target>/.
# Exit status 0 when every target ran its time without a crash; on a crash the reproducer's path
# is printed: fix the reader, add a regression test (cpp/tests/io/fuzz_regression_test.cpp) and
# commit the reproducer to cpp/fuzz/regressions/<target>/, which every test build replays.
#
# Heavy (a configure and a build, then CPU-bound fuzzing): on the shared development machine run
# it under the shared perf lock and niced:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/fuzz.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if ! command -v cmake >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${repo_root}/scripts/dev_env.sh"
fi

seconds=60
build_only=0
targets=()
while [ $# -gt 0 ]; do
  case "$1" in
    --time)
      seconds="$2"
      shift 2
      ;;
    --build-only)
      build_only=1
      shift
      ;;
    -h | --help)
      sed -n '4,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*)
      echo "ci/fuzz.sh: unknown option $1" >&2
      exit 2
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done

if [ -z "${CXX:-}" ]; then
  if command -v clang++-18 >/dev/null 2>&1; then
    export CXX=clang++-18
  else
    export CXX=clang++
  fi
fi
echo "==> compiler: $(command -v "${CXX}") ($("${CXX}" --version | head -n 1))"

build="${repo_root}/build/fuzz"
cmake --preset fuzz
cmake --build --preset fuzz
echo "==> the seed corpus and the crash reproducers, once each (ctest --preset fuzz)"
ctest --preset fuzz

if [ "${#targets[@]}" -eq 0 ]; then
  for exe in "${build}"/cpp/fuzz/fuzz_*; do
    [ -x "${exe}" ] && targets+=("$(basename "${exe}" | sed 's/^fuzz_//')")
  done
fi
if [ "${build_only}" = "1" ]; then
  exit 0
fi

out="${DYNG_FUZZ_OUT:-${repo_root}/build/fuzz-runs}"
jobs="${DYNG_FUZZ_JOBS:-1}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
failed=()
for target in "${targets[@]}"; do
  exe="${build}/cpp/fuzz/fuzz_${target}"
  if [ ! -x "${exe}" ]; then
    echo "ci/fuzz.sh: no fuzz target ${target} (cpp/fuzz/README.md lists them)" >&2
    exit 2
  fi
  corpus="${out}/corpus/${target}"
  artifacts="${out}/artifacts/${target}/"
  mkdir -p "${corpus}" "${artifacts}"
  cp -n "cpp/fuzz/corpus/${target}"/* "${corpus}/" 2>/dev/null || true
  if [ -d "cpp/fuzz/regressions/${target}" ]; then
    cp -n "cpp/fuzz/regressions/${target}"/* "${corpus}/" 2>/dev/null || true
  fi
  echo "==> fuzz_${target}: ${seconds} s (${jobs} job(s)), corpus ${corpus}"
  # -rss_limit_mb / -malloc_limit_mb: a reader that allocates more than 2 GB for an input of a
  # few kilobytes is a finding; -timeout: a single input that takes 10 s is one too.
  args=(-max_total_time="${seconds}" -rss_limit_mb=2048 -malloc_limit_mb=2048 -timeout=10
    -max_len=4096 -artifact_prefix="${artifacts}" -print_final_stats=1)
  if [ "${jobs}" -gt 1 ]; then
    args+=(-fork="${jobs}" -ignore_crashes=0)
  fi
  if ! "${exe}" "${args[@]}" "${corpus}" 2>&1 | tail -n 40; then
    failed+=("${target}")
  elif compgen -G "${artifacts}*" >/dev/null; then
    failed+=("${target}")
  fi
  if compgen -G "${artifacts}*" >/dev/null; then
    echo "!! fuzz_${target} found: $(ls "${artifacts}")"
  fi
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/fuzz.sh: FAILED: ${failed[*]} (reproducers under ${out}/artifacts)"
  exit 1
fi
echo "ci/fuzz.sh: every target ran ${seconds} s without a finding"
