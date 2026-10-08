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
# does under escher_gpu() and, from the commit that adds it, under the staging preset
# mosp_escher_staged() (M8a step 4); from the commit that adds MOSP_ESCHER's preset mosp_escher()
# (M8a step 5) they must also match under it in the exact field set. The JSON records go to
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
      sed -n '5,27p' "${BASH_SOURCE[0]}"
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
    # MOSP_ESCHER's sets run under escher_gpu() (judged against the host model), under the
    # staging preset of its behaviours merged so far (store_policy::mosp_escher_staged(), M8a
    # step 4, ADR 0039) from the commit that adds it, and under its own preset
    # (store_policy::mosp_escher(), M8a step 5) from the commit that adds that.
    staged=0
    mosp=0
    policy_src=cpp/src/hypergraph/escher/store_policy.cpp
    if git grep -q mosp_escher_staged "${sha}" -- "${policy_src}"; then
      staged=1
    fi
    if git grep -q 'store_policy::mosp_escher(long long' "${sha}" -- "${policy_src}"; then
      mosp=1
    fi
    # dynG's default preset (store_policy::dyng_default(), M8a step 5): every set, field set
    # "policy" (criterion 10(c)), from the commit that adds it.
    dflt=0
    if git grep -q 'store_policy::dyng_default(long long' "${sha}" -- "${policy_src}"; then
      dflt=1
    fi
    for set in "${sets[@]}"; do
      policies=("")
      if [[ "${set}" == cbst_trace_mosp_escher_* ]]; then
        policies=("escher_gpu")
        [ "${staged}" = 1 ] && policies+=("mosp_escher_staged")
        [ "${mosp}" = 1 ] && policies+=("mosp_escher")
      fi
      [ "${dflt}" = 1 ] && policies+=("dyng_default")
      for policy in "${policies[@]}"; do
        suffix="${policy:+-${policy}}"
        log="${out}/${set}${suffix}.log"
        policy_args=()
        [ -n "${policy}" ] && policy_args=(--policy "${policy}")
        fields=exact
        [ "${policy}" = dyng_default ] && fields=policy
        if CUDA_VISIBLE_DEVICES="${gpu}" heavy python3 parity/compare.py cbst_trace --store device \
          --fields "${fields}" --skip-verify --replay-exe "${build}/cpp/tests/dyng_cbst_replay" \
          --record "${out}" --set "${set}" "${policy_args[@]}" >"${log}" 2>&1; then
          verdict="OK"
        else
          verdict="FAILED"
          row_status="FAILED"
        fi
        record="${out}/M8a-device-${set#cbst_trace_}-${fields}${suffix}.json"
        cells+=("${set#cbst_trace_}${policy:+ (${policy})}: $(python3 - "${record}" "${verdict}" <<'PY'
import json, sys
from pathlib import Path
record, verdict = Path(sys.argv[1]), sys.argv[2]
if not record.is_file():
    print(f"{verdict} (no record)")
    raise SystemExit
r = json.loads(record.read_text())
t = r["totals"]
policy = r.get("policy") or "escher_gpu"
if policy == "the original's preset":
    policy = "escher_gpu"
if r.get("against_model"):
    fields = {}
    for c in r["cases"]:
        for _op, field in c.get("first_divergences", {}).values():
            fields[field] = fields.get(field, 0) + 1
    diverged = sum(fields.values())
    classes = ", ".join(f"{f} {n}" for f, n in sorted(fields.items()))
    twins = f", {t['twin_constructs']} twin constructs" if "twin_constructs" in t else ""
    print(f"{verdict}: {len(r['cases'])} traces, {t['compared']} ops compared, {diverged} trees "
          f"diverge, each where the host model under {policy} does ({classes}){twins}")
else:
    twins = f", {t['twin_constructs']} twin constructs" if "twin_constructs" in t else ""
    rules = {}
    for c in r["cases"]:
        for rule, n in c.get("documented_rules", {}).items():
            rules[rule] = rules.get(rule, 0) + n
    named = ""
    if rules:
        named = " [" + ", ".join(f"{k} {n}" for k, n in sorted(rules.items())) + "]"
    print(f"{verdict}: {len(r['cases'])} traces, {t['compared']} ops, {t['mismatch_count']} "
          f"mismatches, {t['documented']} documented{named}, {t['unreplayable']} "
          f"unreplayable{twins}")
PY
)")
      done
    done
  fi
  row="| \`${short}\` | ${subject} | ${row_status} | $(IFS=';'; echo "${cells[*]}" | sed 's/;/; /g') |"
  echo "${row}"
  echo "${row}" >>"${work}/rows.md"
  [ "${row_status}" = "ok" ] || status=1
  git worktree remove --force "${wt}"
done
exit "${status}"
