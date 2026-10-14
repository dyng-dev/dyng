#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The local GPU gate (PLAN Section 8.8): the author's manual GPU run before merges that touch CUDA
# code, before milestone gates and before releases. Hosted CI has no GPU; cuda-build.yml only
# compiles.
#
#   ci/gpu_local.sh                        # dev-cuda: build, tests, parity, sanitizers, tidy
#   ci/gpu_local.sh --preset sanitize-cuda
#   ci/gpu_local.sh --comment 42           # also post the summary on pull request #42 (gh CLI)
#   DYNG_GPU_SKIP="cpu tidy" ci/gpu_local.sh
#
# Steps (each can be skipped by name in DYNG_GPU_SKIP):
#   build      configure and build the preset (default dev-cuda; any *-cuda preset works)
#   gpu        ctest -L gpu: the CUDA tests
#   cpu        ctest -L cpu in the same CUDA-enabled build (the default backend becomes cuda there,
#              so the CPU suites are checked against a CUDA build too)
#   parity     the sssp golden corpus on the cuda backend with both engines (parity/compare.py
#              --configs cuda,cuda-operators through the preset's dyng-compat-mosp, which runs the
#              whole MOSP update, combined graph included): byte parity with MOSP-CUDA e220ee2 on
#              every case, and conformance check C4 (fused == operators) on the corpus;
#              and the cycle_count corpus of the original's CUDA backend (compare.py cycle_count
#              --set cycle_count_cuda --configs cuda,cuda:resident through dyng-compat-cycle-enum):
#              byte parity with CycleEnumeration-GPU 0a976ad; each skipped when its goldens
#              ($DYNG_SCRATCH/goldens, parity/export_goldens.py) or its compat tool are missing
#   memcheck   compute-sanitizer --tool memcheck --leak-check full on every executable with gpu
#              tests (randomized suites with DYNG_TEST_SEEDS=2). Tests that make CUDA API calls
#              fail on purpose (suite CudaApiErrors) run in a second pass without API-error
#              reporting; every other test must be free of API errors as well as of memory errors.
#   synccheck  compute-sanitizer --tool synccheck on the CUDA sssp suite (dyng_sssp_cuda_tests) and
#              the CUDA mosp suite (dyng_mosp_cuda_tests, DYNG_TEST_SEEDS=2):
#              no barrier errors, and every test must pass under the sanitizer's scheduling, which
#              exposes data races in the counters the tests compare exactly (M1b review: MOSP's
#              `invalidated` read raced with the insertion-head appends; about 2 minutes); and on
#              the CUDA cycle_count suite (dyng_cycle_count_cuda_tests, DYNG_TEST_SEEDS=2)
#   racecheck  compute-sanitizer --tool racecheck on the CUDA cycle_count suite (shared-memory
#              hazards of its kernels and of the CUB scans; DYNG_TEST_SEEDS=2)
#   plugin     the CUDA plugin wheel from Python (ADR 0030/0031): ci/plugin_wheels.sh builds the
#              core wheel and the dyng-cu13 plugin wheel of this tree (DYNG_GPU_PLUGINS, default
#              cu13; the toolchain of ci/wheel-toolchain.yml; about 4 minutes) into
#              build/<preset>/plugin-wheels, installs them into a fresh venv (one Python,
#              DYNG_GPU_PLUGIN_PYTHONS, default python3.12) and runs the install smoke test, the
#              GPU tests of python/tests (pytest -m gpu: CUDA == sequential, goldens, device
#              arrays, fallback) and the CPU suite with DYNG_CPU_ONLY=1; with
#              DYNG_PLUGIN_INTEROP_PYTHON (a venv with PyTorch / CuPy) the GPU tests also run
#              there. Set DYNG_GPU_PLUGIN_WHEELS=<dir> to test wheels built before (that
#              directory's dist/ and core/dist/; no build). Skipped when the wheel toolchain is
#              missing.
#   cbst       the ESCHER CBST store (M8a, ADR 0038): compute-sanitizer racecheck, synccheck and
#              initcheck on the store's GPU test executables (dyng_cbst_store_tests,
#              dyng_cbst_device_tests: the committed traces of both originals,
#              dyng_cbst_coverage_tests: A's skewed and medium traces, dyng_cbst_compact_tests:
#              compact() after every op of the committed traces, and from M8a step 5
#              dyng_cbst_scenario_tests, dyng_cbst_policy_tests and dyng_cbst_graph_tests at their
#              reduced sizes, DYNG_CBST_REDUCED=1; memcheck covers them in the memcheck step), the
#              device replay of the small trace golden sets under each original's preset
#              (parity/compare.py cbst_trace --store device: escher_gpu_test, escher_gpu_targeted,
#              mosp_escher_test, mosp_escher_targeted; skipped without the goldens), the same sets
#              under dynG's default preset (field set "policy", M8a criterion 10(c)), each
#              writing its certificate record parity-<set>-device-<preset>.json to
#              $DYNG_GPU_RESULTS (default build/<preset>/cbst-records), MOSP_ESCHER's bin/main
#              recorded and replayed once (parity/cbst_trace_goldens.py once main), and the
#              lockstep with the host model over the small sets (dyng_cbst_compact_large_tests:
#              compact() after every op, and every preset's layout without compactions).
#              The CBST mutation tests cbst.mutation.* run in the ctest -L gpu step.
#   hypergraph the hypergraph container (M8b, ADRs 0041-0042): compute-sanitizer racecheck and
#              synccheck on its GPU test executables (dyng_hypergraph_cuda_tests: the container's
#              suites on cuda, the line graph, two containers on two non-blocking streams;
#              dyng_hypergraph_ids_cuda_tests: the committed batch records and A's skewed and
#              medium traces) at their reduced sizes (DYNG_CBST_REDUCED=1; memcheck covers them in
#              the memcheck step), and the container-level replays of the small golden sets on
#              cuda (parity/compare.py hypergraph_ids: escher_gpu_test and mosp_escher_test, the ids
#              and rows; escher_gpu_test again with --line-graph, A's H2H after every batch;
#              skipped without the goldens), their records in $DYNG_GPU_RESULTS (default
#              build/<preset>/hypergraph-records)
#   tidy       clang-tidy naming rules (as ci/check.sh) on the library sources with this build's
#              compile_commands.json, which covers the `#if DYNG_HAS_CUDA` branches that the CPU
#              gate does not compile (skipped if clang-tidy is missing)
#
# Machine rules (the development machine is shared): the GPU tests run on GPU ${DYNG_TEST_GPU}
# (default 1; GPU 0 is kept for performance runs), and every heavy step takes the SHARED lock
# ${DYNG_PERF_LOCK} and runs niced, so no build or test overlaps a timing run (which takes the lock
# exclusively). Set DYNG_PERF_LOCK= (empty) to run without the lock elsewhere.
set -euo pipefail

preset="dev-cuda"
comment_pr=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --preset)
      preset="$2"
      shift 2
      ;;
    --comment)
      comment_pr="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '5,79p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ci/gpu_local.sh: unknown argument '$1' (try --help)" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
# shellcheck disable=SC1091
source scripts/dev_env.sh

gpu="${DYNG_TEST_GPU:-1}"
lock="${DYNG_PERF_LOCK-${DYNG_SCRATCH}/perf.lock}"
skip=" ${DYNG_GPU_SKIP:-} "
build_dir="build/${preset}"
summary="${build_dir}/gpu_local_summary.md"
sanitizer="${DYNG_COMPUTE_SANITIZER:-$(command -v compute-sanitizer || true)}"
failed=()
results=()

step() { printf '\n==> %s\n' "$*"; }
skipped() { [[ "${skip}" == *" $1 "* ]]; }
# Heavy commands: shared lock (never during an exclusive timing run), niced.
heavy() {
  if [ -n "${lock}" ]; then
    flock -s "${lock}" nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}
record() {
  results+=("| $1 | $2 |")
  if [ "$2" != "passed" ] && [ "$2" != "skipped" ]; then
    failed+=("$1")
  fi
}

if ! skipped build; then
  step "preset ${preset}: configure and build"
  if heavy cmake --preset "${preset}" && heavy cmake --build --preset "${preset}"; then
    record build passed
  else
    record build FAILED
    echo "ci/gpu_local.sh: the build failed; nothing else can run" >&2
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="${gpu}"

# The CUDA tests skip (77) without a visible device; the gate must not pass by skipping.
if ! nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  echo "ci/gpu_local.sh: GPU ${gpu} is not visible (nvidia-smi -i ${gpu}); the GPU gate needs it" >&2
  exit 1
fi

if ! skipped gpu; then
  step "ctest -L gpu (GPU ${gpu})"
  if heavy ctest --preset "${preset}" -L gpu -j 8; then
    record "ctest -L gpu" passed
  else
    record "ctest -L gpu" FAILED
  fi
fi

if ! skipped cpu; then
  step "ctest -L cpu (CUDA-enabled build)"
  if heavy ctest --preset "${preset}" -L cpu -j 16; then
    record "ctest -L cpu" passed
  else
    record "ctest -L cpu" FAILED
  fi
fi

if ! skipped parity; then
  step "golden corpus on cuda (GPU ${gpu})"
  compat="${build_dir}/tools/compat/dyng-compat-mosp"
  if [ ! -f "${DYNG_SCRATCH}/goldens/sssp/MANIFEST.sha256" ] || [ ! -x "${compat}" ]; then
    echo "goldens or ${compat} missing; skipped"
    record "parity (cuda goldens)" skipped
  elif heavy python3 parity/compare.py --exe "${compat}" --configs cuda,cuda-operators --jobs 8; then
    record "parity (cuda goldens)" passed
  else
    record "parity (cuda goldens)" FAILED
  fi
  compat_cc="${build_dir}/tools/compat/dyng-compat-cycle-enum"
  if [ ! -f "${DYNG_SCRATCH}/goldens/cycle_count_cuda/MANIFEST.sha256" ] || [ ! -x "${compat_cc}" ]; then
    echo "cycle_count cuda goldens or ${compat_cc} missing; skipped"
    record "parity (cycle_count cuda goldens)" skipped
  elif heavy python3 parity/compare.py cycle_count --exe "${compat_cc}" \
    --set cycle_count_cuda --configs cuda,cuda:resident; then
    record "parity (cycle_count cuda goldens)" passed
  else
    record "parity (cycle_count cuda goldens)" FAILED
  fi
fi

if ! skipped memcheck; then
  step "compute-sanitizer memcheck (GPU ${gpu})"
  if [ -z "${sanitizer}" ]; then
    echo "compute-sanitizer not found"
    record memcheck FAILED
  else
    # The executables that contain gpu-labelled tests (written by dyng_add_test at configure).
    executables=()
    if [ -f "${build_dir}/gpu_test_executables.txt" ]; then
      mapfile -t executables <"${build_dir}/gpu_test_executables.txt"
    fi
    memcheck_ok=1
    for exe in "${executables[@]}"; do
      [ -n "${exe}" ] || continue
      echo "--- $(basename "${exe}")"
      if ! DYNG_TEST_SEEDS=2 DYNG_CBST_REDUCED=1 heavy "${sanitizer}" --tool memcheck \
        --leak-check full --error-exitcode 1 "${exe}" --gtest_filter='-CudaApiErrors.*' \
        --gtest_brief=1; then
        memcheck_ok=0
      fi
      # The second pass only where there are such tests: an executable that runs no test makes
      # no CUDA call, which compute-sanitizer reports as an error.
      if "${exe}" --gtest_list_tests --gtest_filter='CudaApiErrors.*' | grep -q '^  ' &&
        ! heavy "${sanitizer}" --tool memcheck --leak-check full --error-exitcode 1 \
          --report-api-errors no "${exe}" --gtest_filter='CudaApiErrors.*' --gtest_brief=1; then
        memcheck_ok=0
      fi
    done
    if [ "${memcheck_ok}" = 1 ] && [ "${#executables[@]}" -gt 0 ]; then
      record memcheck passed
    else
      record memcheck FAILED
    fi
  fi
fi

if ! skipped synccheck; then
  step "compute-sanitizer synccheck, CUDA sssp suite (GPU ${gpu})"
  sssp_tests="${build_dir}/cpp/tests/dyng_sssp_cuda_tests"
  if [ -z "${sanitizer}" ]; then
    echo "compute-sanitizer not found"
    record synccheck FAILED
  elif [ ! -x "${sssp_tests}" ]; then
    echo "${sssp_tests} missing"
    record synccheck FAILED
  elif heavy "${sanitizer}" --tool synccheck --error-exitcode 1 "${sssp_tests}" \
    --gtest_filter='-CudaApiErrors.*' --gtest_brief=1; then
    record synccheck passed
  else
    record synccheck FAILED
  fi
  mosp_tests="${build_dir}/cpp/tests/algorithms/mosp/dyng_mosp_cuda_tests"
  if [ -z "${sanitizer}" ] || [ ! -x "${mosp_tests}" ]; then
    echo "compute-sanitizer or ${mosp_tests} missing"
    record "synccheck (mosp)" FAILED
  elif DYNG_TEST_SEEDS=2 heavy "${sanitizer}" --tool synccheck --error-exitcode 1 "${mosp_tests}" \
    --gtest_brief=1; then
    record "synccheck (mosp)" passed
  else
    record "synccheck (mosp)" FAILED
  fi
  cc_tests="${build_dir}/cpp/tests/dyng_cycle_count_cuda_tests"
  if [ -z "${sanitizer}" ] || [ ! -x "${cc_tests}" ]; then
    echo "compute-sanitizer or ${cc_tests} missing"
    record "synccheck (cycle_count)" FAILED
  elif DYNG_TEST_SEEDS=2 heavy "${sanitizer}" --tool synccheck --error-exitcode 1 "${cc_tests}" \
    --gtest_brief=1; then
    record "synccheck (cycle_count)" passed
  else
    record "synccheck (cycle_count)" FAILED
  fi
fi

if ! skipped racecheck; then
  step "compute-sanitizer racecheck, CUDA cycle_count suite (GPU ${gpu})"
  cc_tests="${build_dir}/cpp/tests/dyng_cycle_count_cuda_tests"
  if [ -z "${sanitizer}" ] || [ ! -x "${cc_tests}" ]; then
    echo "compute-sanitizer or ${cc_tests} missing"
    record racecheck FAILED
  elif DYNG_TEST_SEEDS=2 heavy "${sanitizer}" --tool racecheck --racecheck-report all \
    --error-exitcode 1 "${cc_tests}" --gtest_brief=1; then
    record racecheck passed
  else
    record racecheck FAILED
  fi
fi

if ! skipped cbst; then
  step "the CBST store: sanitizers and the trace replays (GPU ${gpu})"
  cbst_ok=1
  cbst_exes=()
  for t in dyng_cbst_store_tests dyng_cbst_device_tests dyng_cbst_coverage_tests \
    dyng_cbst_compact_tests dyng_cbst_scenario_tests dyng_cbst_policy_tests \
    dyng_cbst_graph_tests; do
    if [ -x "${build_dir}/cpp/tests/${t}" ]; then
      cbst_exes+=("${build_dir}/cpp/tests/${t}")
    else
      echo "${build_dir}/cpp/tests/${t} missing"
      cbst_ok=0
    fi
  done
  if [ -z "${sanitizer}" ]; then
    echo "compute-sanitizer missing"
    cbst_ok=0
  else
    for tool in racecheck synccheck initcheck; do
      extra=()
      [ "${tool}" = racecheck ] && extra=(--racecheck-report all)
      for exe in "${cbst_exes[@]}"; do
        if ! DYNG_CBST_REDUCED=1 heavy "${sanitizer}" --tool "${tool}" "${extra[@]}" \
          --error-exitcode 1 "${exe}" --gtest_brief=1; then
          echo "compute-sanitizer --tool ${tool} ${exe}: FAILED"
          cbst_ok=0
        fi
      done
    done
  fi
  if [ "${cbst_ok}" = 1 ]; then
    record "cbst sanitizers (racecheck, synccheck, initcheck)" passed
  else
    record "cbst sanitizers (racecheck, synccheck, initcheck)" FAILED
  fi
  replay="${build_dir}/cpp/tests/dyng_cbst_replay"
  # The replay records of the release certificate (parity/certify.py requires every CBST set
  # under its original's preset and dyng_default; M8a review): parity-<set>-device-<preset>.json
  # in $DYNG_GPU_RESULTS (the release's benchmarks/results/<version>), else in the build tree.
  cbst_records="${DYNG_GPU_RESULTS:-${build_dir}/cbst-records}"
  if [ ! -f "${DYNG_SCRATCH}/goldens/cbst_trace/escher_gpu_test/MANIFEST.sha256" ]; then
    echo "the CBST trace goldens are missing (parity/export_goldens.py cbst_trace); skipped"
    record "cbst replays (device, small sets)" skipped
  elif heavy python3 parity/compare.py cbst_trace --store device --replay-exe "${replay}" \
    --gpu "${gpu}" --certificate "${cbst_records}" --set cbst_trace_escher_gpu_test \
    --set cbst_trace_escher_gpu_targeted --set cbst_trace_mosp_escher_test \
    --set cbst_trace_mosp_escher_targeted; then
    record "cbst replays (device, small sets)" passed
  else
    record "cbst replays (device, small sets)" FAILED
  fi
  # The small sets under dynG's default preset (M8a step 5, criterion 10(c); field set "policy":
  # rows, avail and subtree_avail identical, the policy differences documented).
  if [ ! -f "${DYNG_SCRATCH}/goldens/cbst_trace/mosp_escher_test/MANIFEST.sha256" ]; then
    record "cbst replays (device, default preset)" skipped
  elif heavy python3 parity/compare.py cbst_trace --store device --replay-exe "${replay}" \
    --gpu "${gpu}" --certificate "${cbst_records}" --policy dyng_default \
    --set cbst_trace_escher_gpu_test --set cbst_trace_escher_gpu_targeted \
    --set cbst_trace_mosp_escher_test --set cbst_trace_mosp_escher_targeted; then
    record "cbst replays (device, default preset)" passed
  else
    record "cbst replays (device, default preset)" FAILED
  fi
  # MOSP_ESCHER's bin/main (no golden set: its graph comes from std::random_device), recorded
  # once and replayed right away under mosp_escher() (M8a review).
  if [ ! -d "${DYNG_SCRATCH}/ref/MOSP_ESCHER@4b86159/patched" ]; then
    record "cbst bin/main recorded and replayed once" skipped
  elif heavy python3 parity/cbst_trace_goldens.py once main --replay-exe "${replay}" \
    --gpu "${gpu}"; then
    record "cbst bin/main recorded and replayed once" passed
  else
    record "cbst bin/main recorded and replayed once" FAILED
  fi
  # compact() after every op of the small sets, and every preset without compactions, in
  # lockstep with the host model (M8a step 4; the policy lockstep of the M8a review).
  if [ -x "${build_dir}/cpp/tests/dyng_cbst_compact_large_tests" ] &&
    heavy "${build_dir}/cpp/tests/dyng_cbst_compact_large_tests"; then
    record "cbst lockstep with the host model (compact after every op; every preset)" passed
  else
    record "cbst lockstep with the host model (compact after every op; every preset)" FAILED
  fi
fi

if ! skipped hypergraph; then
  step "the hypergraph container: sanitizers and the replays (GPU ${gpu})"
  hg_ok=1
  hg_exes=()
  for t in dyng_hypergraph_cuda_tests dyng_hypergraph_ids_cuda_tests; do
    if [ -x "${build_dir}/cpp/tests/${t}" ]; then
      hg_exes+=("${build_dir}/cpp/tests/${t}")
    else
      echo "${build_dir}/cpp/tests/${t} missing"
      hg_ok=0
    fi
  done
  if [ -z "${sanitizer}" ]; then
    echo "compute-sanitizer missing"
    hg_ok=0
  else
    for tool in racecheck synccheck; do
      extra=()
      [ "${tool}" = racecheck ] && extra=(--racecheck-report all)
      for exe in "${hg_exes[@]}"; do
        if ! DYNG_CBST_REDUCED=1 heavy "${sanitizer}" --tool "${tool}" "${extra[@]}" \
          --error-exitcode 1 "${exe}" --gtest_brief=1; then
          echo "compute-sanitizer --tool ${tool} ${exe}: FAILED"
          hg_ok=0
        fi
      done
    done
  fi
  if [ "${hg_ok}" = 1 ]; then
    record "hypergraph sanitizers (racecheck, synccheck)" passed
  else
    record "hypergraph sanitizers (racecheck, synccheck)" FAILED
  fi
  hg_tool="${build_dir}/cpp/tests/dyng_hypergraph_ids"
  hg_records="${DYNG_GPU_RESULTS:-${build_dir}/hypergraph-records}"
  if [ ! -f "${DYNG_SCRATCH}/goldens/cbst_trace/escher_gpu_test/MANIFEST.sha256" ] ||
    [ ! -f "${DYNG_SCRATCH}/goldens/cbst_trace/mosp_escher_test/MANIFEST.sha256" ]; then
    echo "the CBST trace goldens are missing (parity/export_goldens.py cbst_trace); skipped"
    record "hypergraph replays (cuda, small sets)" skipped
  elif heavy python3 parity/compare.py hypergraph_ids --backend cuda --gpu "${gpu}" --jobs 8 \
    --exe "${hg_tool}" --record "${hg_records}" --set cbst_trace_escher_gpu_test \
    --set cbst_trace_mosp_escher_test &&
    heavy python3 parity/compare.py hypergraph_ids --backend cuda --gpu "${gpu}" --jobs 8 \
      --exe "${hg_tool}" --record "${hg_records}" --line-graph --mode by_id \
      --set cbst_trace_escher_gpu_test; then
    record "hypergraph replays (cuda, small sets; ids, rows, line graph)" passed
  else
    record "hypergraph replays (cuda, small sets; ids, rows, line graph)" FAILED
  fi
fi

if ! skipped plugin; then
  step "the CUDA plugin wheel from Python (GPU ${gpu})"
  scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
  toolchain="${DYNG_WHEEL_TOOLCHAIN:-${scratch}/tools/manylinux228-tc}"
  tools="${DYNG_WHEEL_TOOLS:-${scratch}/tools/wheeltools}"
  wheels="${DYNG_GPU_PLUGIN_WHEELS:-}"
  if [ -z "${wheels}" ] && { [ ! -d "${toolchain}" ] || [ ! -x "${tools}/bin/auditwheel" ]; }; then
    echo "the wheel toolchain (${toolchain}, ${tools}) is missing (docs/developer/wheels.md); skipped"
    record "plugin wheel (pytest -m gpu)" skipped
  else
    pythons="${DYNG_GPU_PLUGIN_PYTHONS:-$(command -v python3.12 || command -v python3)}"
    # The step's own GPU selection: plugin_wheels.sh sets CUDA_VISIBLE_DEVICES itself.
    if [ -n "${wheels}" ]; then
      plugin_env=(DYNG_PLUGIN_TEST_ONLY=1 "DYNG_PLUGIN_OUT=${wheels}")
    else
      plugin_env=("DYNG_PLUGIN_OUT=${repo_root}/${build_dir}/plugin-wheels")
    fi
    if heavy env -u CUDA_VISIBLE_DEVICES "${plugin_env[@]}" "DYNG_PLUGINS=${DYNG_GPU_PLUGINS:-cu13}" \
      "DYNG_WHEEL_PYTHONS=${pythons}" "DYNG_TEST_GPU=${gpu}" ci/plugin_wheels.sh; then
      record "plugin wheel (pytest -m gpu)" passed
    else
      record "plugin wheel (pytest -m gpu)" FAILED
    fi
  fi
fi

if ! skipped tidy; then
  step "clang-tidy (naming rules, library sources, CUDA branches)"
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "clang-tidy not found; skipped"
    record clang-tidy skipped
  else
    gcc_include="$("${CXX:-g++}" -print-file-name=include)"
    # cpp/src/algorithms/_template is not compiled (scripts/new_algorithm.py instantiates it).
    if git ls-files 'cpp/src/*.cpp' ':!:cpp/src/algorithms/_template/*' |
      heavy xargs -r -n 1 -P "$(nproc)" clang-tidy \
      -p "${build_dir}" --quiet --checks='-*,readability-identifier-naming' \
      --warnings-as-errors='*' "--extra-arg=-isystem${gcc_include}" \
      --extra-arg=-Wno-deprecated-declarations; then
      record clang-tidy passed
    else
      record clang-tidy FAILED
    fi
  fi
fi

mkdir -p "${build_dir}"
{
  echo "### ci/gpu_local.sh: ${preset}"
  echo
  dirty="$(git diff --quiet HEAD -- || echo ' (dirty tree)')"
  gpu_name="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "${gpu}" \
    2>/dev/null || echo "GPU ${gpu}")"
  toolkit="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p')"
  echo "commit $(git rev-parse --short HEAD)${dirty}; GPU ${gpu}: ${gpu_name}; ${toolkit}"
  echo
  echo "| step | result |"
  echo "|---|---|"
  printf '%s\n' "${results[@]}"
} >"${summary}"
echo
cat "${summary}"

if [ -n "${comment_pr}" ]; then
  if command -v gh >/dev/null 2>&1; then
    gh pr comment "${comment_pr}" --body-file "${summary}"
  else
    echo "gh not found; the summary is in ${summary}" >&2
  fi
fi

if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/gpu_local.sh: FAILED: ${failed[*]}"
  exit 1
fi
echo "ci/gpu_local.sh: all steps passed"
