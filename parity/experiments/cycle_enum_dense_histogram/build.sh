#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# A MEASUREMENT EXPERIMENT, not a reference: CycleEnumeration-GPU@0a976ad with ONE change,
# dense_histogram.patch, which makes the OpenMP static counter (src/openmp/openmp_johnson.cpp)
# count into a dense per-thread array (as dynG's port does) instead of calling
# CycleHistogram::increment() (a std::map lookup and an overflow check in another translation
# unit) once per cycle found. Timing this copy against the unpatched original isolates the effect
# of the histogram; timing dynG against it isolates everything else in the port (the DFS, the
# graph layout), which PLAN 8.6 requires to be reported separately so that an improvement never
# masks a regression (parity/results/M2a.md, "Improvements").
#
#   parity/experiments/cycle_enum_dense_histogram/build.sh     # prints the binary's path
#
# The copy is a fresh `git archive` of the pinned commit under
# $DYNG_SCRATCH/ref/CycleEnumeration-GPU@0a976ad/experiment-dense-histogram, patched, and built
# with the build command of parity/references.toml (only the cycle-enum target). Idempotent.
# Run it under the shared perf lock (it compiles).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../../.." && pwd)"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
origin="${HOME}/Projects/CycleEnumeration-GPU"
commit="0a976adfa801a712135bf1adb51a228f353a0751"
dir="${scratch}/ref/CycleEnumeration-GPU@${commit:0:7}/experiment-dense-histogram"
log="${dir}.build.log"
patch_file="${here}/dense_histogram.patch"

build="$(python3 - "${repo_root}/parity/references.toml" <<'PY'
import sys, tomllib
doc = tomllib.load(open(sys.argv[1], "rb"))
refs = doc.get("reference", doc)
for r in (refs if isinstance(refs, list) else refs.values()):
    if isinstance(r, dict) and r.get("name") == "CycleEnumeration-GPU":
        print(r["build"].replace("{jobs}", "16").replace("cmake --build build -j16",
                                                          "cmake --build build -j16 --target cycle-enum"))
        break
PY
)"
[ -n "${build}" ] || { echo "no build command for CycleEnumeration-GPU in references.toml" >&2; exit 1; }

stamp="patch_sha256=$(sha256sum "${patch_file}" | cut -d' ' -f1)"
if [ ! -f "${dir}/.dyng-experiment" ] || ! grep -qx "${stamp}" "${dir}/.dyng-experiment"; then
  rm -rf "${dir}"
  mkdir -p "${dir}"
  git -C "${origin}" archive --format=tar "${commit}" | tar -x -C "${dir}"
  (cd "${dir}" && patch -p1 --quiet <"${patch_file}")
  { echo "commit=${commit}"; echo "${stamp}"; echo "build=${build}"; } >"${dir}/.dyng-experiment"
fi
{ echo; echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) CXX=/usr/bin/g++ ${build}"; } >>"${log}"
(cd "${dir}" && env CXX=/usr/bin/g++ bash -c "${build}") >>"${log}" 2>&1 ||
  { tail -n 30 "${log}" >&2; echo "build failed: ${log}" >&2; exit 1; }
echo "${dir}/build/cycle-enum"
