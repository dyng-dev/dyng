#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed fixtures of generators::legacy::mosp_changes (cpp/tests/data/
# mosp_changes) with MOSP's own change generator (`mospPrep changes`), from the UNPATCHED scratch
# copies that parity/build_reference.sh builds (the original repositories are only read with
# `git archive`).
#
#   parity/fixtures/generators/make_generator_fixtures.sh
#
# Environment: DYNG_SCRATCH (persistent work area, default $HOME/Projects/dyng-work).
#
# The graph is a 24 x 24 grid with a few long-range edges (written here as a symmetric Matrix
# Market pattern), converted with `mospPrep mtx2csr` (K = 3 weights in [1, 100], seed 12345). Every
# case of cases.txt runs `mospPrep changes` of MOSP-OpenMP@c352151 and writes insert.txt,
# delete.txt and the report line; MOSP-CUDA@e220ee2's mospPrep must produce the same files (the two
# generators differ only where an "increase" change would overflow 2^31 - 1, which MOSP-OpenMP
# saturates and no case here reaches). The script is deterministic: running it again must leave
# `git status` clean.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
out="${repo_root}/cpp/tests/data/mosp_changes"
tmp="${scratch}/runs/generator-fixtures-tmp"

"${repo_root}/parity/build_reference.sh" --variant unpatched MOSP-OpenMP MOSP-CUDA >/dev/null
openmp="$("${repo_root}/parity/build_reference.sh" --variant unpatched --print-dir MOSP-OpenMP)"
cuda="$("${repo_root}/parity/build_reference.sh" --variant unpatched --print-dir MOSP-CUDA)"

rm -rf "${out}" "${tmp}"
mkdir -p "${out}/graph" "${tmp}"

# --- The graph -----------------------------------------------------------------------------------
side=24
awk -v s="${side}" 'BEGIN {
  n = s * s; m = 0
  for (y = 0; y < s; ++y) for (x = 0; x < s; ++x) {
    v = y * s + x + 1
    if (x + 1 < s && (x * 7 + y * 3) % 11 != 5) { e[++m] = (v + 1) " " v }
    if (y + 1 < s && (x * 5 + y * 13) % 17 != 9) { e[++m] = (v + s) " " v }
    if ((x * y) % 29 == 5) { w = ((v * 37) % n) + 1; if (w != v) e[++m] = (w > v ? w " " v : v " " w) }
  }
  print "%%MatrixMarket matrix coordinate pattern symmetric"
  print n " " n " " m
  for (i = 1; i <= m; ++i) print e[i]
}' >"${tmp}/grid.mtx"
"${openmp}/bin/mospPrep" mtx2csr "${tmp}/grid.mtx" "${out}/graph/graphCsr" 3 1 100 12345 >/dev/null

# --- The cases (name, then the options of `mospPrep changes`) ------------------------------------
cat >"${out}/cases.txt" <<'CASES'
uniform --changes 200 --ins 50 --seed 1
uniform_safe --changes 300 --ins 50 --seed 777 --safe
uniform_all_inserts --changes 60 --ins 100 --seed 2
uniform_all_deletes --changes 60 --ins 0 --seed 3 --wmin 7 --wmax 9
local_safe --changes 100 --ins 50 --seed 777 --local 4 --safe
local --changes 120 --ins 30 --seed 4 --local 3 --source 100
targeted --mode targeted --changes 150 --ins 60 --seed 5
targeted_local_safe --mode targeted --changes 80 --ins 50 --seed 9 --local 5 --source 7 --safe
reweight --mode reweight --changes 120 --seed 11 --wmin 5 --wmax 500
reweight_local --mode reweight --changes 40 --seed 12 --local 2
increase --mode increase --changes 60 --seed 13 --wmax 50
increase_local --mode increase --changes 30 --seed 14 --local 6 --source 300
safe_heavy --changes 1600 --ins 10 --seed 21 --safe
local_safe_heavy --changes 400 --ins 20 --seed 22 --local 3 --safe --source 200
targeted_safe_heavy --mode targeted --changes 700 --ins 5 --seed 23 --safe
CASES

while read -r name args; do
  [ -n "${name}" ] || continue
  mkdir -p "${out}/${name}"
  # shellcheck disable=SC2086
  "${openmp}/bin/mospPrep" changes "${out}/graph/graphCsr" "${out}/${name}" ${args} \
    | sed -n 's/^changes: //p' >"${out}/${name}/report.txt"
  # shellcheck disable=SC2086
  "${cuda}/bin/mospPrep" changes "${out}/graph/graphCsr" "${tmp}/${name}" ${args} >/dev/null
  for f in insert.txt delete.txt; do
    if ! cmp -s "${out}/${name}/${f}" "${tmp}/${name}/${f}"; then
      echo "MOSP-CUDA and MOSP-OpenMP disagree on ${name}/${f}" >&2
      exit 1
    fi
  done
done <"${out}/cases.txt"

rm -rf "${tmp}"
du -sh "${out}"
