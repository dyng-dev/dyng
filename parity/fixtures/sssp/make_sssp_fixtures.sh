#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed sssp test fixtures (cpp/tests/data/mosp_sssp) from the pinned original
# MOSP-OpenMP@c352151. The original repository is only read with `git archive`; the copy is built
# out of tree under the work directory.
#
#   parity/fixtures/sssp/make_sssp_fixtures.sh [work-dir]
#
# Environment:
#   MOSP_OPENMP_REPO  the original repository   (default: $HOME/Projects/MOSP-OpenMP)
#   DYNG_SCRATCH      persistent work area      (default: $HOME/Projects/dyng-work)
#
# For every case it writes the initial trees (`mospPrep init`, Dijkstra with lowest-id ties) and
# the updated trees of the in-memory driver `mosp` (mospUpdate -> sospUpdateCpu), and records the
# deterministic `invalidated` counter per objective. Before anything is written it checks that
# four original implementations agree byte for byte: `mosp`, the file-based OpenMP update
# (parallelSOSPUpdate), the legacy sequential update (sequentialSOSPUpdate) and Dijkstra on the
# updated graph (`mospPrep expected`); `mosp --validate` must pass as well.
#
# The script is deterministic: running it again must leave `git status` clean.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
commit=c352151
origin="${MOSP_OPENMP_REPO:-${HOME}/Projects/MOSP-OpenMP}"
work="${1:-${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}/runs/sssp-fixtures}"
ref="${work}/MOSP-OpenMP@${commit}"
data="${repo_root}/cpp/tests/data"
out="${data}/mosp_sssp"
tmp="${work}/tmp"

# --- 1. Reference copy and tools -------------------------------------------------------------
if [ ! -f "${ref}/Makefile" ]; then
  mkdir -p "${ref}"
  git -C "${origin}" archive "${commit}" | tar -x -C "${ref}"
fi
make -C "${ref}" -j8 bin/mosp bin/mospPrep >/dev/null
src="${ref}/src"
common=("${src}/csrGraph.cpp" "${src}/read.cpp" "${src}/stageTimer.cpp" "${src}/sospUpdateCpu.cpp")
g++ -std=c++17 -O2 -fopenmp -I"${ref}/headers" \
  "${repo_root}/parity/fixtures/graph_io/export_graph_io.cpp" \
  "${src}/csrGraph.cpp" "${src}/generateGraphCSR.cpp" "${src}/generateChangedEdges.cpp" \
  "${src}/read.cpp" -o "${work}/export_graph_io"
g++ -std=c++17 -O2 -fopenmp -I"${ref}/headers" \
  "${repo_root}/parity/fixtures/sssp/export_sssp.cpp" "${common[@]}" \
  "${src}/sequentialSOSPUpdate.cpp" "${src}/parallelSOSPUpdate.cpp" -o "${work}/export_sssp"
generator="${work}/export_graph_io"
exporter="${work}/export_sssp"
mosp="${ref}/bin/mosp"
prep="${ref}/bin/mospPrep"
# A fixed thread count: the outputs do not depend on it, the counters recorded here neither.
export OMP_NUM_THREADS=4

rm -rf "${out}" "${tmp}"
mkdir -p "${out}" "${tmp}"
manifest=()

num_weights() { # <csr prefix>: K of the Values file (0 for a graph without edges)
  awk 'NF > 0 { print NF; found = 1; exit } END { if (!found) print 0 }' "$1Values.txt"
}

# export_case <name> <input dir relative to cpp/tests/data> [K]
#   The input directory holds graphCsr{RowPtr,ColInd,Values}.txt, insert.txt and delete.txt.
export_case() {
  local name="$1" rel="$2"
  local input="${data}/${rel}"
  local prefix="${input}/graphCsr"
  local k="${3:-$(num_weights "${prefix}")}"
  local dst="${out}/${name}" scratch="${tmp}/${name}"
  mkdir -p "${dst}" "${scratch}"
  "${prep}" init "${prefix}" "${dst}/init" -k "${k}" >/dev/null
  # The in-memory driver (the reference output), with the original's own validation.
  "${mosp}" --graph "${prefix}" --changes "${input}" --init "${dst}/init" -k "${k}" \
    --out "${scratch}/mosp" --validate >"${scratch}/mosp.log"
  if grep -q "FAIL" "${scratch}/mosp.log"; then
    echo "mosp --validate failed on ${name}" >&2
    exit 1
  fi
  "${prep}" expected "${prefix}" "${input}" "${scratch}/expected" -k "${k}" >/dev/null
  : >"${dst}/stats.txt"
  for ((obj = 0; obj < k; ++obj)); do
    local init="${dst}/init/obj${obj}" upd="${scratch}/mosp/obj${obj}"
    for impl in sequential parallel; do
      "${exporter}" "${impl}" "${prefix}" "${init}/distancesOriginal.txt" \
        "${init}/SSSPTreeOriginal.txt" "${input}/insert.txt" "${input}/delete.txt" "${obj}" 0 \
        "${scratch}/${impl}/obj${obj}/distances.txt" "${scratch}/${impl}/obj${obj}/tree.txt" \
        >/dev/null
      cmp "${upd}/distancesUpdated.txt" "${scratch}/${impl}/obj${obj}/distances.txt"
      cmp "${upd}/SSSPTreeUpdated.txt" "${scratch}/${impl}/obj${obj}/tree.txt"
    done
    cmp "${upd}/distancesUpdated.txt" "${scratch}/expected/obj${obj}/distancesUpdated.txt"
    cmp "${upd}/SSSPTreeUpdated.txt" "${scratch}/expected/obj${obj}/SSSPTreeUpdated.txt"
    mkdir -p "${dst}/updated/obj${obj}"
    cp "${upd}/distancesUpdated.txt" "${upd}/SSSPTreeUpdated.txt" "${dst}/updated/obj${obj}/"
    # "obj0   SOSP update 0.012 ms (invalidated 3, iterations ...)" -> "obj0 invalidated 3"
    sed -n "s/^obj${obj} .*(invalidated \([0-9]*\),.*/obj${obj} invalidated \1/p" \
      "${scratch}/mosp.log" >>"${dst}/stats.txt"
  done
  manifest+=("${name} ${rel} ${k}")
}

# --- 2. The graph/io cases: the 10 generateTestCases cases, random and hand-written cases ------
while read -r name; do
  export_case "${name}" "mosp_graph_io/${name}"
done <"${data}/mosp_graph_io/cases.txt"

# --- 3. Count-to-infinity regressions (mospTest runRegressions; stress-test seeds) --------------
#   name  n   m   K  changes ins% maxWeight graphSeed changeSeed
regressions=(
  "c2i_0 6 10 2 6 55 50 621705 250813"
  "c2i_1 13 22 3 6 26 50 770968 694580"
  "c2i_2 9 17 2 6 81 50 115080 943676"
)
for spec in "${regressions[@]}"; do
  read -r name n m k changes ins wmax gseed cseed <<<"${spec}"
  dst="${out}/${name}/input"
  mkdir -p "${dst}"
  "${generator}" gen-graph "${dst}/graphCsr" "${n}" "${m}" "${k}" 1 "${wmax}" "${gseed}" >/dev/null
  # exist = 1, duplicate = 1, self-loop = 0, as in runRegressions().
  "${generator}" gen-changes "${dst}/graphCsr" "${dst}/insert.txt" "${dst}/delete.txt" "${k}" \
    "${n}" "${changes}" "${ins}" "$((100 - ins))" 1 "${wmax}" 1 1 0 "${cseed}" >/dev/null
  export_case "${name}" "mosp_sssp/${name}/input"
done

# --- 4. Hand-written cases (MOSP_ESCHER tests/unit/test_mosp_update.cu and more) ---------------
write_input() { # <name> <rowptr> <colind> <values> <insert> <delete>
  local dst="${out}/$1/input"
  mkdir -p "${dst}"
  printf '%b' "$2" >"${dst}/graphCsrRowPtr.txt"
  printf '%b' "$3" >"${dst}/graphCsrColInd.txt"
  printf '%b' "$4" >"${dst}/graphCsrValues.txt"
  printf '%b' "$5" >"${dst}/insert.txt"
  printf '%b' "$6" >"${dst}/delete.txt"
}
# escher_disconnect: 0->1 (1), 0->3 (50), 1->2 (1), 2->1 (1), 3->1 (50); delete 0->1. Vertices 1
# and 2 stay reachable only through 3: d = 100, 101 (the original loop counted to infinity).
write_input escher_disconnect '0\n2\n3\n4\n5\n' '1\n3\n2\n1\n1\n' '1\n50\n1\n1\n50\n' '' '0 1\n'
export_case escher_disconnect mosp_sssp/escher_disconnect/input
# escher_delete_all: every edge is deleted; only the source stays reachable.
write_input escher_delete_all '0\n2\n3\n3\n4\n4\n' '1\n3\n2\n4\n' '3\n2\n4\n7\n' '' \
  '0 1\n1 2\n0 3\n3 4\n'
export_case escher_delete_all mosp_sssp/escher_delete_all/input 1
# escher_ties: 0->1 (5), 0->2 (1), 2->3 (1), 1->4 (2), 3->4 (2); insert 2->1 (1). Vertex 4 gets a
# second tight path (via 1); its parent must become the lower id, 1.
write_input escher_ties '0\n2\n3\n4\n5\n5\n' '1\n2\n4\n3\n4\n' '5\n1\n2\n1\n2\n' '2 1 1\n' ''
export_case escher_ties mosp_sssp/escher_ties/input
# ties_k2: equal-distance parents in two objectives, a weight increase on a tree edge, a parallel
# edge and a re-inserted deleted edge (K = 2).
write_input ties_k2 '0\n3\n5\n7\n9\n9\n' '1\n2\n1\n3\n4\n4\n3\n4\n0\n' \
  '2 1\n2 1\n5 5\n1 3\n2 2\n1 2\n1 1\n1 1\n9 9\n' '0 2 1 9\n1 3 1 3\n2 4 1 1\n0 1 2 1\n' \
  '0 1\n3 4\n'
export_case ties_k2 mosp_sssp/ties_k2/input

printf '%s\n' "${manifest[@]}" >"${out}/cases.txt"
rm -rf "${tmp}"
du -sh --apparent-size "${out}"
echo "fixtures written to ${out}"
