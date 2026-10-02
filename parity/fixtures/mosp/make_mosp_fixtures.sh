#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed mosp test fixtures (cpp/tests/data/mosp_combined) from the pinned
# originals MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2, using the scratch copies that
# parity/build_reference.sh builds (the original repositories are only read with `git archive`).
#
#   parity/fixtures/mosp/make_mosp_fixtures.sh
#
# Environment: DYNG_SCRATCH (persistent work area, default $HOME/Projects/dyng-work);
# CUDA_VISIBLE_DEVICES (default 1, the development GPU) for MOSP-CUDA's `mosp`.
#
# For every case it runs the original `mosp` driver with a preference vector (`--pref`) and keeps
# its combinedGraph/ outputs (the combined graph's distances in units of 1/L, its SOSP tree and the
# MOSP path costs). The golden corpus (parity/export_goldens.py) holds the default preferences
# only; these cases add non-trivial preferences, the worked example of thesis Chapter 4 with both
# of its preference vectors, and `-k K` below the graph's number of weight columns (mospCosts.txt
# keeps every column). Before anything is written it checks that MOSP-OpenMP's and MOSP-CUDA's
# `mosp` agree byte for byte on every output and that `mosp --validate` passes (the MOSP tree
# against host Dijkstra on the reference combined graph).
#
# The script is deterministic: running it again must leave `git status` clean.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
data="${repo_root}/cpp/tests/data"
out="${data}/mosp_combined"
tmp="${scratch}/runs/mosp-fixtures-tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_MODULE_LOADING=EAGER
export OMP_NUM_THREADS=4

# --- 1. Reference copies --------------------------------------------------------------------------
"${repo_root}/parity/build_reference.sh" --variant patched MOSP-OpenMP >/dev/null
"${repo_root}/parity/build_reference.sh" --variant patched MOSP-CUDA >/dev/null
omp_ref="$("${repo_root}/parity/build_reference.sh" --variant patched --print-dir MOSP-OpenMP)"
cuda_ref="$("${repo_root}/parity/build_reference.sh" --variant patched --print-dir MOSP-CUDA)"

rm -rf "${out}" "${tmp}"
mkdir -p "${out}" "${tmp}"
manifest=()

# --- 2. The thesis worked example: mospTest writes its graph, batch and initial trees -------------
"${omp_ref}/bin/mospTest" --seed 1 --work "${tmp}/mospTest" --only thesis-example \
  | grep -q "all checks passed"
thesis="${tmp}/mospTest/thesis-example"
mkdir -p "${out}/thesis/input"
cp "${thesis}/graph/graphCsrRowPtr.txt" "${thesis}/graph/graphCsrColInd.txt" \
  "${thesis}/graph/graphCsrValues.txt" "${out}/thesis/input/"
cp "${thesis}/changes/insert.txt" "${thesis}/changes/delete.txt" "${out}/thesis/input/"
"${omp_ref}/bin/mospPrep" init "${out}/thesis/input/graphCsr" "${out}/thesis/init" >/dev/null

# export_case <name> <input dir> <init dir> <K> <pref>   (dirs relative to cpp/tests/data)
export_case() {
  local name="$1" input="${data}/$2" init="${data}/$3" k="$4" pref="$5"
  local work="${tmp}/${name}"
  mkdir -p "${work}"
  for impl in omp cuda; do
    local ref="${omp_ref}"
    [ "${impl}" = cuda ] && ref="${cuda_ref}"
    "${ref}/bin/mosp" --graph "${input}/graphCsr" --changes "${input}" --init "${init}" -k "${k}" \
      --pref "${pref}" --out "${work}/${impl}" --validate >"${work}/${impl}.log"
    if grep -q "FAIL" "${work}/${impl}.log"; then
      echo "mosp --validate (${impl}) failed on ${name}" >&2
      exit 1
    fi
  done
  diff -r "${work}/omp" "${work}/cuda" >/dev/null
  mkdir -p "${out}/${name}/combined"
  cp "${work}/omp/combinedGraph/distancesCsr.txt" "${work}/omp/combinedGraph/SSSPTreeCsr.txt" \
    "${work}/omp/combinedGraph/mospCosts.txt" "${out}/${name}/combined/"
  manifest+=("${name} $2 $3 ${k} ${pref}")
}

#           name            input                         init                     K  pref
export_case thesis_414      mosp_combined/thesis/input    mosp_combined/thesis/init 3 4,1,4
export_case thesis_441      mosp_combined/thesis/input    mosp_combined/thesis/init 3 4,4,1
export_case testCase9_123   mosp_graph_io/testCase9       mosp_sssp/testCase9/init 3 1,2,3
export_case c2i_0_31        mosp_sssp/c2i_0/input         mosp_sssp/c2i_0/init     2 3,1
export_case h0_25           mosp_graph_io/h0              mosp_sssp/h0/init        2 2,5
export_case r04_1234        mosp_graph_io/r04             mosp_sssp/r04/init       4 1,2,3,4
export_case r04_k2          mosp_graph_io/r04             mosp_sssp/r04/init       2 6,4
export_case ties_k2_77      mosp_sssp/ties_k2/input       mosp_sssp/ties_k2/init   2 7,7

printf '%s\n' "${manifest[@]}" >"${out}/cases.txt"
rm -rf "${tmp}"
