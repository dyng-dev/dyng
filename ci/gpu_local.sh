#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The local GPU gate (PLAN Section 8.8): the author's manual GPU run before merges that touch CUDA
# code, before milestone gates and before releases. Hosted CI has no GPU; cuda-build.yml only
# compiles.
#
#   ci/gpu_local.sh                        # dev-cuda: build, tests, parity, memcheck, synccheck, tidy
#   ci/gpu_local.sh --preset sanitize-cuda
#   ci/gpu_local.sh --comment 42           # also post the summary on pull request #42 (gh CLI)
#   DYNG_GPU_SKIP="cpu tidy" ci/gpu_local.sh
#
# Steps (each can be skipped by name in DYNG_GPU_SKIP):
#   build      configure and build the preset (default dev-cuda; any *-cuda preset works)
#   gpu        ctest -L gpu: the CUDA tests
#   cpu        ctest -L cpu in the same CUDA-enabled build (the default backend becomes cuda there,
#              so the CPU suites are checked against a CUDA build too)
#   parity     the sssp golden corpus on the cuda backend (parity/compare.py --configs cuda through
#              the preset's dyng-compat-mosp): byte parity with MOSP-CUDA e220ee2 on every case;
#              skipped when the goldens ($DYNG_SCRATCH/goldens, parity/export_goldens.py) or the
#              compat tool are missing
#   memcheck   compute-sanitizer --tool memcheck --leak-check full on every executable with gpu
#              tests (randomized suites with DYNG_TEST_SEEDS=2). Tests that make CUDA API calls
#              fail on purpose (suite CudaApiErrors) run in a second pass without API-error
#              reporting; every other test must be free of API errors as well as of memory errors.
#   synccheck  compute-sanitizer --tool synccheck on the CUDA sssp suite (dyng_sssp_cuda_tests):
#              no barrier errors, and every test must pass under the sanitizer's scheduling, which
#              exposes data races in the counters the tests compare exactly (M1b review: MOSP's
#              `invalidated` read raced with the insertion-head appends; about 2 minutes)
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
      sed -n '5,38p' "${BASH_SOURCE[0]}"
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
  elif heavy python3 parity/compare.py --exe "${compat}" --configs cuda --jobs 8; then
    record "parity (cuda goldens)" passed
  else
    record "parity (cuda goldens)" FAILED
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
      if ! DYNG_TEST_SEEDS=2 heavy "${sanitizer}" --tool memcheck --leak-check full \
        --error-exitcode 1 "${exe}" --gtest_filter='-CudaApiErrors.*' --gtest_brief=1; then
        memcheck_ok=0
      fi
      if ! heavy "${sanitizer}" --tool memcheck --leak-check full --error-exitcode 1 \
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
fi

if ! skipped tidy; then
  step "clang-tidy (naming rules, library sources, CUDA branches)"
  if ! command -v clang-tidy >/dev/null 2>&1; then
    echo "clang-tidy not found; skipped"
    record clang-tidy skipped
  else
    gcc_include="$("${CXX:-g++}" -print-file-name=include)"
    if git ls-files 'cpp/src/*.cpp' | heavy xargs -r -n 1 -P "$(nproc)" clang-tidy \
      -p "${build_dir}" --quiet --checks='-*,readability-identifier-naming' \
      --warnings-as-errors='*' "--extra-arg=-isystem${gcc_include}" \
      --extra-arg=-Wno-deprecated-declarations; then
      record clang-tidy passed
    else
      record clang-tidy FAILED
    fi
  fi
fi

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
