#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# MEASUREMENT EXPERIMENTS, not references: copies of CycleEnumeration-GPU@0a976ad with one or more
# of the patches next to this script, for the isolation table of parity/results/M2a.md
# ("Improvements", PLAN 8.6: improvements are reported separately so that they never mask a
# regression). Timed with parity/perf_ab.py cycle_count run --baseline-exe <printed path>.
#
#   dense_histogram.patch  the OpenMP static counter (src/openmp/openmp_johnson.cpp) counts into a
#                          dense per-thread array, as dynG's port does, instead of calling
#                          CycleHistogram::increment() (a std::map lookup and an overflow check in
#                          another translation unit) once per cycle found
#   stage_timers.patch     instrumentation only: the CLI prints read_seconds= (read_graph_view: the
#                          parser and the CSR/CSC build) and count_seconds= (the static count) on
#                          standard error, so the count can be compared with dynG's
#                          cycle_count.compute stage (the unpatched original times only updates)
#
#   parity/experiments/cycle_enum/build_variant.sh dense_histogram
#   parity/experiments/cycle_enum/build_variant.sh stage_timers
#   parity/experiments/cycle_enum/build_variant.sh dense_histogram stage_timers
#
# Each variant is a fresh `git archive` of the pinned commit under
# $DYNG_SCRATCH/ref/CycleEnumeration-GPU@0a976ad/experiment-<patch>[+<patch>...], patched, and built
# with the build command of parity/references.toml (the cycle-enum target only); the script prints
# the binary's path. Idempotent. Run it under the shared perf lock (it compiles).
set -euo pipefail

[ "$#" -ge 1 ] || { echo "usage: build_variant.sh <patch name>..." >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../../.." && pwd)"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
origin="${HOME}/Projects/CycleEnumeration-GPU"
commit="0a976adfa801a712135bf1adb51a228f353a0751"
name="$(IFS=+; echo "$*")"
dir="${scratch}/ref/CycleEnumeration-GPU@${commit:0:7}/experiment-${name//_/-}"
log="${dir}.build.log"

build="$(python3 - "${repo_root}/parity/references.toml" <<'PY'
import sys, tomllib
for r in tomllib.load(open(sys.argv[1], "rb"))["reference"]:
    if r.get("name") == "CycleEnumeration-GPU":
        print(r["build"].replace("cmake --build build -j{jobs}",
                                 "cmake --build build -j16 --target cycle-enum"))
PY
)"
[ -n "${build}" ] || { echo "no build command for CycleEnumeration-GPU in references.toml" >&2; exit 1; }

stamp=""
for p in "$@"; do
  [ -f "${here}/${p}.patch" ] || { echo "no patch ${here}/${p}.patch" >&2; exit 2; }
  stamp+="${p}=$(sha256sum "${here}/${p}.patch" | cut -d' ' -f1) "
done
if [ ! -f "${dir}/.dyng-experiment" ] || ! grep -qxF "patches=${stamp}" "${dir}/.dyng-experiment"; then
  rm -rf "${dir}"
  mkdir -p "${dir}"
  git -C "${origin}" archive --format=tar "${commit}" | tar -x -C "${dir}"
  for p in "$@"; do
    (cd "${dir}" && patch -p1 --quiet <"${here}/${p}.patch")
  done
  { echo "commit=${commit}"; echo "patches=${stamp}"; echo "build=${build}"; } >"${dir}/.dyng-experiment"
fi
{ echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) CXX=/usr/bin/g++ ${build}"; } >>"${log}"
(cd "${dir}" && env CXX=/usr/bin/g++ bash -c "${build}") >>"${log}" 2>&1 ||
  { tail -n 30 "${log}" >&2; echo "build failed: ${log}" >&2; exit 1; }
echo "${dir}/build/cycle-enum"
