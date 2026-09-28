#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates the committed fixtures of the CycleEnumeration-GPU@0a976ad ports (graph under
# graph_properties::cycle_enum_compatible(), io::read_edge_list, generators::legacy::
# cycle_enum_batch, dyng::testing cycle oracles) in cpp/tests/data/cycle_enum, with the exporter
# of the PATCHED scratch copy that parity/build_reference.sh builds (the original repository is
# only read with `git archive`).
#
#   parity/fixtures/cycle_enum/make_cycle_enum_fixtures.sh [--datasets]
#
# --datasets also recomputes cpp/tests/data/cycle_enum/datasets.txt, the digests of the parser
# output, the CSR, generated batches and the graphs after them on the TUDataset graphs under
# $DYNG_SCRATCH/datasets/cycle (read by the parity-labelled dataset test).
#
# Environment: DYNG_SCRATCH (persistent work area, default $HOME/Projects/dyng-work).
# The script is deterministic: running it again must leave `git status` clean.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
out="${repo_root}/cpp/tests/data/cycle_enum"
datasets=0
[ "${1:-}" = "--datasets" ] && datasets=1

"${repo_root}/parity/build_reference.sh" --variant patched CycleEnumeration-GPU >/dev/null
ref="$("${repo_root}/parity/build_reference.sh" --variant patched --print-dir CycleEnumeration-GPU)"
x="${ref}/parity_export/bin/export_cycle_enum"

keep_datasets=""
if [ "${datasets}" -eq 0 ] && [ -f "${out}/datasets.txt" ]; then
  keep_datasets="$(cat "${out}/datasets.txt")"
fi
rm -rf "${out}"
mkdir -p "${out}/parser" "${out}/errors" "${out}/cases" "${out}/generator"

# --- 1. Parser inputs and the original's output -------------------------------------------------
cp "${ref}/tests/data/sample_temporal.txt" "${out}/parser/sample_temporal.txt"
cp "${ref}/tests/data/reference_sample.txt" "${out}/parser/reference_sample.txt"
# TUDataset *_A.txt layout: "  u,  v" per line, 1-based, both directions of every edge.
awk 'BEGIN {
  s = 7
  for (i = 0; i < 180; ++i) {
    s = (s * 1103515245 + 12345) % 2147483648; u = 1 + s % 60
    s = (s * 1103515245 + 12345) % 2147483648; v = 1 + s % 60
    if (u == v) continue
    printf "%6d, %6d\n%6d, %6d\n", u, v, v, u
  }
}' >"${out}/parser/tudataset_A.txt"
printf '%s\n' '# comments, blank lines, tabs, CRLF, signs, timestamps, duplicates, self-loops' \
  '% another comment' '' '   # an indented comment' '10 20 5' '10	20	3' '+10,20,3' \
  '20 10 -7' '  -5 10' '10 -5 2' '7 7 1' '7 30' '30 7 0' '1000000000000 10' \
  '10 1000000000000 9' '30,40' $'40 30\r' $'40\t30\t11\r' '' >"${out}/parser/mixed.txt"
printf '%s\n' '%%MatrixMarket matrix coordinate pattern general' '% a comment line' '3 1 2' \
  '1 2' '2 3' >"${out}/parser/mm_general.mtx"
printf '%s\n' '%%MatrixMarket Matrix Coordinate Real Symmetric' '4 4 6' '2 1 0.5' '3 1 1.5' \
  '3 2 1' '4 3 2' '4 1 7' '2 2 1' >"${out}/parser/mm_symmetric.mtx"
printf '%s\n' '%%MatrixMarket matrix coordinate integer skew-symmetric' '3 3 2' '2 1 5' \
  '3 2 -1' >"${out}/parser/mm_skew.mtx"
printf '%s\n' '%%MatrixMarket matrix coordinate complex hermitian' '%' '' '% dims next' \
  '5 5 4' '2 1 1.0 2.0' '3 1 0 1' '5 4 1 1' '5 5 2 0' >"${out}/parser/mm_hermitian.mtx"
# Sparse ids (the range exceeds 4n + 1024): the sorted-id path of the compaction.
awk 'BEGIN {
  s = 11
  for (i = 0; i < 400; ++i) {
    s = (s * 1103515245 + 12345) % 2147483648; u = (s % 97) * 1000003 - 40000000
    s = (s * 1103515245 + 12345) % 2147483648; v = (s % 97) * 1000003 - 40000000
    s = (s * 1103515245 + 12345) % 2147483648; t = s % 50
    printf "%d %d %d\n", u, v, t
  }
}' >"${out}/parser/sparse_ids.txt"
for f in "${out}"/parser/*; do
  case "${f}" in *.parse | *.csr) continue ;; esac
  "${x}" parse "${f}" >"${f}.parse"
  "${x}" csr "${f}" >"${f}.csr"
done

# --- 2. Malformed inputs: the original's accept/reject decision and line ----------------------
bad() { # <name> <content lines...>
  local name="$1"
  shift
  printf '%s\n' "$@" >"${out}/errors/${name}.txt"
}
bad single_field '5'
bad four_fields '1 2 3' '1 2 3 4'
bad bad_timestamp '1 2' '# c' '1 2 notanumber'
bad glued_token '1 2' '3 4x'
bad plus_minus '+-1 2'
bad lone_plus '+ 1 2'
bad commas_only '1 2' ',,,'
bad overflow '1 9223372036854775808'
bad mm_array '%%MatrixMarket matrix array real general' '2 2' '1' '2' '3' '4'
bad mm_symmetry '%%MatrixMarket matrix coordinate pattern upper-triangular' '2 2 1' '1 2'
bad mm_short '%%MatrixMarket matrix coordinate' '2 2 1' '1 2'
bad mm_vector '%%MatrixMarket vector coordinate pattern general' '2 2 1' '1 2'
bad mm_one_field '%%MatrixMarket matrix coordinate pattern general' '2 2 2' '1 2' '2'
bad ok_commas '1,,2' '2,,,1,'
bad ok_mm_extra '%%MatrixMarket matrix coordinate complex general' '2 2 1' '1 2 3 4 5 6'
for f in "${out}"/errors/*.txt; do
  if msg="$("${x}" parse "${f}" 2>&1 >/dev/null)"; then
    echo "ok" >"${f%.txt}.expected"
  else
    # "export_cycle_enum: <path>:<line>: <reason>" -> "error <line>"
    line="$(sed -E 's/^export_cycle_enum: .*:([0-9]+): .*$/\1/' <<<"${msg}")"
    echo "error ${line}" >"${f%.txt}.expected"
  fi
done

# --- 3. Random cases: a small graph and an arbitrary batch each ----------------------------------
"${x}" random-cases "${out}/cases" 80 20260927
for f in "${out}"/cases/case_*.txt; do
  "${x}" apply "${f}" >"${f%.txt}.expected"
done

# --- 4. Generated batches ------------------------------------------------------------------------
awk 'BEGIN {
  s = 3
  for (u = 0; u < 300; ++u) {
    printf "%d %d\n", u, (u + 1) % 300
    for (j = 0; j < 3; ++j) {
      s = (s * 1103515245 + 12345) % 2147483648
      if (s % 4 == 0) { s = (s * 1103515245 + 12345) % 2147483648; printf "%d %d\n", u, s % 300 }
    }
  }
}' >"${out}/generator/graph300.txt"
awk 'BEGIN { for (v = 0; v < 40; ++v) printf "%d %d\n", v, (v + 1) % 40 }' \
  >"${out}/generator/ring40.txt"
cp "${out}/parser/reference_sample.txt" "${out}/generator/reference_sample.txt"
cp "${out}/parser/tudataset_A.txt" "${out}/generator/tudataset_A.txt"
# graph deletions insertions seed [window]
cat >"${out}/generator/cases.txt" <<'EOF'
reference_sample.txt 2 3 7
reference_sample.txt 0 5 1
reference_sample.txt 11 0 99
reference_sample.txt 12 0 99
reference_sample.txt 0 31 5
ring40.txt 3 4 42
ring40.txt 2 3 5 6
ring40.txt 2 3 5 0
ring40.txt 2 3 5 1
ring40.txt 1 1 5 39
ring40.txt 1 1 5 40
ring40.txt 100 0 1
ring40.txt 0 0 1
ring40.txt 8 20 18446744073709551615
tudataset_A.txt 40 40 1
tudataset_A.txt 100 100 2 30
tudataset_A.txt 7 9 3 2
graph300.txt 100 100 1
graph300.txt 390 0 4
graph300.txt 0 500 5 50
graph300.txt 25 25 6 17
graph300.txt 50 60 12345 299
EOF
i=0
while read -r g del ins seed window; do
  name="$(printf 'g%02d' "${i}")"
  if "${x}" generate "${out}/generator/${g}" "${del}" "${ins}" "${seed}" ${window:+"${window}"} \
    >"${out}/generator/${name}.expected" 2>/dev/null; then
    :
  else
    echo "error" >"${out}/generator/${name}.expected"
  fi
  i=$((i + 1))
done <"${out}/generator/cases.txt"

# --- 5. Dataset digests --------------------------------------------------------------------------
if [ "${datasets}" -eq 1 ]; then
  data="${scratch}/datasets/cycle"
  {
    echo "# Digests (FNV-1a 64 of the exporter's text) from CycleEnumeration-GPU@0a976ad on"
    echo "# \$DYNG_SCRATCH/datasets/cycle; regenerate with make_cycle_enum_fixtures.sh --datasets."
    echo "# <what> <dataset file> [<deletions> <insertions> <seed>] fnv1a64 <hex> bytes <n>"
    for d in DD/DD_A.txt github_stargazers/github_stargazers_A.txt \
      twitch_egos/twitch_egos_A.txt COLLAB/COLLAB_A.txt; do
      echo "parse ${d} $("${x}" parse "${data}/${d}" --digest)"
      echo "csr ${d} $("${x}" csr "${data}/${d}" --digest)"
    done
    for d in DD/DD_A.txt github_stargazers/github_stargazers_A.txt twitch_egos/twitch_egos_A.txt; do
      for n in 1000 25000 50000; do
        echo "generate ${d} ${n} ${n} 1 $("${x}" generate "${data}/${d}" "${n}" "${n}" 1 --digest)"
        echo "apply-generated ${d} ${n} ${n} 1 $("${x}" apply-generated "${data}/${d}" "${n}" \
          "${n}" 1 --digest)"
      done
    done
    for w in 100 1000 10000; do
      # A window of 100 vertices holds fewer than 1000 edges: the original's error.
      digest="$("${x}" generate "${data}/DD/DD_A.txt" 1000 1000 1 "${w}" --digest 2>/dev/null ||
        echo error)"
      echo "generate DD/DD_A.txt 1000 1000 1 ${w} ${digest}"
    done
  } >"${out}/datasets.txt"
elif [ -n "${keep_datasets}" ]; then
  printf '%s\n' "${keep_datasets}" >"${out}/datasets.txt"
fi
echo "fixtures written to ${out} ($(du -sh "${out}" | cut -f1))"
