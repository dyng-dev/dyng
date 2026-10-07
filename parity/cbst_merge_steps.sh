#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The merge-step replays of the ESCHER CBST store (M8a, acceptance criterion 10; PLAN 6.4.5):
# builds each listed commit in a temporary git worktree, re-runs its replays and prints one row of
# the merge-step table of parity/results/M8a.md per commit.
#
#   parity/cbst_merge_steps.sh <commit>...            # the small sets
#   parity/cbst_merge_steps.sh --large <commit>...    # the small and the large sets
#   DYNG_TEST_GPU=1 parity/cbst_merge_steps.sh ...     # the GPU of the replays (default 1)
#
# For every commit: `git worktree add --detach` into $DYNG_SCRATCH/m8a/merge-steps (removed
# afterwards), a Release CUDA build of the commit's replay targets only (dyng_cbst_replay and
# dyng_cbst_device_tests; DYNG_ALGORITHMS=sssp, minimal index types, warnings as errors), the
# commit's own replay tests on the committed traces (dyng_cbst_device_tests), and the replay of the
# golden sets through THIS tree's parity/compare.py cbst_trace --store device with the commit's
# tool: the sets of ESCHER-GPU must match in the exact field set (documented divergences listed),
# the sets of MOSP_ESCHER must diverge in every tree where the host model under the same preset
# does (until the store has MOSP_ESCHER's preset). The JSON records go to
# $DYNG_SCRATCH/m8a/merge-steps/<sha7>/; the row, with one column per set, to stdout and to
# $DYNG_SCRATCH/m8a/merge-steps/rows.md.
#
# Machine rules: every build and replay takes the SHARED perf lock and runs niced; the original
# repositories and the main worktree are not touched (only `git worktree add/remove`, which writes
# the repository's .git/worktrees).
set -euo pipefail

large=0
commits=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --large)
      large=1
      shift
      ;;
    -h | --help)
      sed -n '5,26p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      commits+=("$1")
      shift
      ;;
  esac
done
if [ "${#commits[@]}" -eq 0 ]; then
  echo "usage: parity/cbst_merge_steps.sh [--large] <commit>..." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
# shellcheck disable=SC1091
source scripts/dev_env.sh >/dev/null
gpu="${DYNG_TEST_GPU:-1}"
lock="${DYNG_PERF_LOCK-${DYNG_SCRATCH}/perf.lock}"
heavy() {
  if [ -n "${lock}" ] && [ -e "${lock}" ]; then
    flock -s "${lock}" nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}
work="${DYNG_SCRATCH}/m8a/merge-steps"
mkdir -p "${work}"
small_sets=(cbst_trace_escher_gpu_test cbst_trace_escher_gpu_targeted cbst_trace_mosp_escher_test
  cbst_trace_mosp_escher_targeted)
large_sets=(cbst_trace_escher_gpu_coauth50k cbst_trace_escher_gpu_syn1m
  cbst_trace_mosp_escher_dblp50k_hyperedge cbst_trace_mosp_escher_dblp50k_vertex)
sets=("${small_sets[@]}")
if [ "${large}" = 1 ]; then
  sets+=("${large_sets[@]}")
fi

status=0
for commit in "${commits[@]}"; do
  sha="$(git rev-parse --verify "${commit}^{commit}")"
  short="${sha:0:7}"
  subject="$(git log -1 --format=%s "${sha}")"
  wt="${work}/wt-${short}"
  out="${work}/${short}"
  rm -rf "${out}"
  mkdir -p "${out}"
  echo "==> ${short} ${subject}"
  if [ -e "${wt}" ]; then
    git worktree remove --force "${wt}" || rm -rf "${wt}"
  fi
  git worktree add --detach --quiet "${wt}" "${sha}"
  build="${wt}/build"
  row_status="ok"
  if ! heavy cmake -S "${wt}" -B "${build}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DDYNG_ENABLE_CUDA=ON -DDYNG_CUDA_ARCHITECTURES=native -DDYNG_BUILD_TESTS=ON \
    -DDYNG_ALGORITHMS=sssp -DDYNG_INDEX_TYPES=minimal -DDYNG_BUILD_TOOLS=OFF \
    -DDYNG_BUILD_EXAMPLES=OFF -DDYNG_BUILD_TUTORIALS=OFF -DDYNG_WARNINGS_AS_ERRORS=ON \
    >"${out}/configure.log" 2>&1 ||
    ! heavy cmake --build "${build}" --target dyng_cbst_replay dyng_cbst_device_tests \
      >"${out}/build.log" 2>&1; then
    echo "  build FAILED (${out})"
    row_status="build FAILED"
  fi
  cells=()
  if [ "${row_status}" = "ok" ]; then
    if CUDA_VISIBLE_DEVICES="${gpu}" heavy "${build}/cpp/tests/dyng_cbst_device_tests" \
      >"${out}/device_tests.log" 2>&1; then
      cells+=("committed traces: passed")
    else
      cells+=("committed traces: FAILED")
      row_status="FAILED"
    fi
    for set in "${sets[@]}"; do
      log="${out}/${set}.log"
      if CUDA_VISIBLE_DEVICES="${gpu}" heavy python3 parity/compare.py cbst_trace --store device \
        --fields exact --skip-verify --replay-exe "${build}/cpp/tests/dyng_cbst_replay" \
        --record "${out}" --set "${set}" >"${log}" 2>&1; then
        verdict="OK"
      else
        verdict="FAILED"
        row_status="FAILED"
      fi
      cells+=("${set#cbst_trace_}: $(python3 - "${out}" "${set}" "${verdict}" <<'PY'
import json, sys
from pathlib import Path
out, name, verdict = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
recs = list(out.glob(f"M8a-device-{name[len('cbst_trace_'):]}-exact*.json"))
if not recs:
    print(f"{verdict} (no record)")
    raise SystemExit
r = json.loads(recs[0].read_text())
t = r["totals"]
if r.get("against_model"):
    diverged = sum(1 for c in r["cases"] for _ in c.get("first_divergences", {}))
    print(f"{verdict}: {len(r['cases'])} traces, {t['compared']} ops compared, {diverged} trees "
          f"diverge, each where the host model under escher_gpu does")
else:
    print(f"{verdict}: {len(r['cases'])} traces, {t['compared']} ops, {t['mismatch_count']} "
          f"mismatches, {t['documented']} documented, {t['unreplayable']} unreplayable")
PY
)")
    done
  fi
  row="| \`${short}\` | ${subject} | ${row_status} | $(IFS=';'; echo "${cells[*]}" | sed 's/;/; /g') |"
  echo "${row}"
  echo "${row}" >>"${work}/rows.md"
  [ "${row_status}" = "ok" ] || status=1
  git worktree remove --force "${wt}"
done
exit "${status}"
