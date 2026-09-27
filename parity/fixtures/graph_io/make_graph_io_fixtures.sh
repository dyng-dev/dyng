#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed graph/io test fixtures (cpp/tests/data/mosp_graph_io) from the pinned
# original MOSP-OpenMP@c352151. The original repository is only read with `git archive`; the copy
# is built out of tree under the work directory.
#
#   parity/fixtures/graph_io/make_graph_io_fixtures.sh [work-dir]
#
# Environment:
#   MOSP_OPENMP_REPO  the original repository   (default: $HOME/Projects/MOSP-OpenMP)
#   DYNG_SCRATCH      persistent work area      (default: $HOME/Projects/dyng-work)
#
# The script is deterministic: running it again must leave `git status` clean.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
commit=c352151
origin="${MOSP_OPENMP_REPO:-${HOME}/Projects/MOSP-OpenMP}"
work="${1:-${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}/runs/graph-io-fixtures}"
ref="${work}/MOSP-OpenMP@${commit}"
out="${repo_root}/cpp/tests/data/mosp_graph_io"

# --- 1. Reference copy and tools -------------------------------------------------------------
if [ ! -f "${ref}/Makefile" ]; then
  mkdir -p "${ref}"
  git -C "${origin}" archive "${commit}" | tar -x -C "${ref}"
fi
make -C "${ref}" -j8 bin/mospPrep >/dev/null
exporter="${work}/export_graph_io"
g++ -std=c++17 -O2 -fopenmp -I"${ref}/headers" \
  "${repo_root}/parity/fixtures/graph_io/export_graph_io.cpp" \
  "${ref}/src/csrGraph.cpp" "${ref}/src/generateGraphCSR.cpp" \
  "${ref}/src/generateChangedEdges.cpp" "${ref}/src/read.cpp" -o "${exporter}"
prep="${ref}/bin/mospPrep"

rm -rf "${out}"
mkdir -p "${out}"
cases=()

apply_case() { # <case dir>
  mkdir -p "$1/applied"
  "${exporter}" apply "$1/graphCsr" "$1/insert.txt" "$1/delete.txt" "$1/applied/graphCsr"
}

# --- 2. The 10 generateTestCases cases (tracked in the pinned commit) -------------------------
for i in $(seq 0 9); do
  src="${ref}/tests/testCase${i}"
  dst="${out}/testCase${i}"
  mkdir -p "${dst}/updateGraphCSR"
  for f in RowPtr ColInd Values; do
    cp "${src}/originalGraph/graphCsr${f}.txt" "${dst}/graphCsr${f}.txt"
    cp "${src}/updatedGraph/updatedGraphCsr${f}.txt" "${dst}/updateGraphCSR/graphCsr${f}.txt"
  done
  cp "${src}/changedEdges/insert.txt" "${src}/changedEdges/delete.txt" "${dst}/"
  apply_case "${dst}"
  cases+=("testCase${i}")
done

# --- 3. Random cases: generateGraphCSR + generateChangedEdges ---------------------------------
#   name  n   m    K  count ins del exist dup self seed
random_cases=(
  "r00 25 90 3 30 50 50 1 1 0 101"
  "r01 25 90 3 30 50 50 1 1 1 102"
  "r02 40 200 1 60 70 30 0 1 1 103"
  "r03 40 200 2 60 30 70 1 0 0 104"
  "r04 60 300 4 80 50 50 1 1 1 105"
  "r05 10 60 2 40 80 20 1 1 1 106"
  "r06 12 100 3 50 100 0 0 1 1 107"
  "r07 30 120 1 40 0 100 1 1 0 108"
  "r08 8 40 5 30 60 40 1 1 1 109"
  "r09 50 49 1 30 50 50 1 0 0 110"
  "r10 20 100 32 40 50 50 1 1 1 111"
  "r11 5 20 2 20 100 0 0 1 1 112"
)
for spec in "${random_cases[@]}"; do
  read -r name n m k count ins del exist dup self seed <<<"${spec}"
  dst="${out}/${name}"
  mkdir -p "${dst}"
  "${exporter}" gen-graph "${dst}/graphCsr" "${n}" "${m}" "${k}" 1 30 "${seed}" >/dev/null
  "${exporter}" gen-changes "${dst}/graphCsr" "${dst}/insert.txt" "${dst}/delete.txt" "${k}" \
    "${n}" "${count}" "${ins}" "${del}" 1 30 "${exist}" "${dup}" "${self}" "$((seed + 1000))" \
    >/dev/null
  apply_case "${dst}"
  cases+=("${name}")
done

# --- 4. Hand-written cases: parallel edges, self-loops, unsorted rows, delete-all, empty batch --
write_case() { # <name> <rowptr> <colind> <values> <insert> <delete>
  local dst="${out}/$1"
  mkdir -p "${dst}"
  printf '%b' "$2" >"${dst}/graphCsrRowPtr.txt"
  printf '%b' "$3" >"${dst}/graphCsrColInd.txt"
  printf '%b' "$4" >"${dst}/graphCsrValues.txt"
  printf '%b' "$5" >"${dst}/insert.txt"
  printf '%b' "$6" >"${dst}/delete.txt"
  apply_case "${dst}"
  cases+=("$1")
}
# h0: unsorted rows with parallel edges and self-loops (K = 2). Deletions remove the FIRST
# remaining (u,v); upserts overwrite the first remaining match; the last insertion wins.
write_case h0 \
  '0\n5\n7\n7\n10\n' \
  '2\n1\n2\n0\n1\n3\n3\n0\n2\n0\n' \
  '10 11\n12 13\n14 15\n16 17\n18 19\n20 21\n22 23\n24 25\n26 27\n28 29\n' \
  '0 2 50 50\n0 1 7 99\n0 0 5 5\n2 3 1 1\n2 3 9 2\n1 3 4 40\n3 3 8 8\n' \
  '0 2\n3 0\n3 0\n3 0\n1 1\n2 3\n'
# h1: delete every edge (K = 1).
write_case h1 \
  '0\n2\n3\n4\n' \
  '1\n2\n2\n0\n' \
  '5\n6\n7\n8\n' \
  '' \
  '0 1\n0 2\n1 2\n2 0\n'
# h2: an empty batch (K = 3).
write_case h2 \
  '0\n1\n3\n3\n' \
  '2\n0\n2\n' \
  '1 2 3\n4 5 6\n7 8 9\n' \
  '' \
  ''
# h3: rows without out-edges, re-inserting a deleted edge, equal / lower / mixed weights (K = 3).
write_case h3 \
  '0\n3\n3\n5\n5\n5\n' \
  '1\n2\n4\n0\n4\n' \
  '10 10 10\n20 20 20\n30 30 30\n40 40 40\n50 50 50\n' \
  '0 1 10 10 10\n0 2 5 5 5\n0 4 31 29 30\n1 0 3 3 3\n4 0 1 2 3\n0 1 11 9 10\n2 0 40 41 39\n' \
  '0 1\n2 0\n4 3\n'
# h4: blank lines and trailing spaces in the batch files (accepted by the original reader).
write_case h4 \
  '0\n2\n2\n' \
  '1\n0\n' \
  '3 4\n5 6\n' \
  '\n1 0 1 1 \n\n  0 1 9 9\n' \
  '\n0 1\n\n'
printf '%s\n' "${cases[@]}" >"${out}/cases.txt"

# --- 5. mospPrep mtx2csr: seeded weights ---------------------------------------------------------
mtx="${out}/mtx"
mkdir -p "${mtx}"
printf '%s\n' '%%MatrixMarket matrix coordinate pattern symmetric' '% a comment' \
  '6 6 9' '2 1' '3 1' '3 2' '4 4' '5 3' '6 5' '6 1' '3 1' '5 4' >"${mtx}/m0_symmetric.mtx"
printf '%s\n' '%%MatrixMarket matrix coordinate integer general' '% first comment' \
  '%second comment' '5 5 11' '1 2 7' '2 3 -1' '3 1 4' '5 5 1' '4 2 9' '1 2 3' '2 5 2' \
  '5 1 8' '3 4 6' '4 3 5' '1 5 1' >"${mtx}/m1_general.mtx"
"${prep}" mtx2csr "${mtx}/m0_symmetric.mtx" "${mtx}/m0_k3_seed12345_" 3 1 100 12345 >/dev/null
"${prep}" mtx2csr "${mtx}/m1_general.mtx" "${mtx}/m1_k1_seed12345_" 1 1 100 12345 >/dev/null
"${prep}" mtx2csr "${mtx}/m1_general.mtx" "${mtx}/m1_k4_seed7_" 4 1 2147483647 7 >/dev/null

du -sh "${out}"
echo "fixtures written to ${out}"
