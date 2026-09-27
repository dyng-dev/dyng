#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The local CPU gate (PLAN Section 8.8): run before every commit. CI runs the same steps.
#
#   ci/check.sh                   # cpu-only and dev presets, clang-format, REUSE, Doxygen, pre-commit
#   DYNG_CHECK_PRESETS=dev ci/check.sh
#   DYNG_CHECK_SKIP="precommit docs" ci/check.sh
#
# Steps (each can be skipped by name in DYNG_CHECK_SKIP):
#   build      configure, build and `ctest -L cpu` for every preset in DYNG_CHECK_PRESETS
#   format     clang-format --dry-run --Werror on all tracked C++/CUDA sources
#   reuse      reuse lint (SPDX headers in every file)
#   docs       Doxygen on the public headers with warnings as errors (skipped if doxygen is missing)
#   precommit  pre-commit run --all-files (skipped if pre-commit is missing)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Tools: the dyng-dev conda env locally; in CI they are already on PATH.
if ! command -v cmake >/dev/null 2>&1 || [ -z "${CI:-}" ]; then
  # shellcheck disable=SC1091
  source scripts/dev_env.sh
fi

presets="${DYNG_CHECK_PRESETS:-cpu-only dev}"
skip=" ${DYNG_CHECK_SKIP:-} "
failed=()

step() { printf '\n==> %s\n' "$*"; }
skipped() { [[ "${skip}" == *" $1 "* ]]; }

if ! skipped build; then
  for preset in ${presets}; do
    step "preset ${preset}: configure, build, test (label cpu)"
    if cmake --preset "${preset}" && cmake --build --preset "${preset}" &&
      ctest --preset "${preset}" -L cpu; then
      echo "preset ${preset}: OK"
    else
      failed+=("build:${preset}")
    fi
  done
fi

if ! skipped format; then
  step "clang-format ($(clang-format --version | head -n1))"
  mapfile -t sources < <(git ls-files '*.hpp' '*.cpp' '*.cuh' '*.cu' '*.h')
  if [ "${#sources[@]}" -eq 0 ] || clang-format --dry-run --Werror "${sources[@]}"; then
    echo "clang-format: OK"
  else
    failed+=("format")
  fi
fi

if ! skipped reuse; then
  step "reuse lint"
  if command -v reuse >/dev/null 2>&1; then
    if reuse lint --quiet; then
      echo "reuse: OK"
    else
      reuse lint || true
      failed+=("reuse")
    fi
  else
    echo "reuse not found; skipped"
  fi
fi

if ! skipped docs; then
  step "doxygen (public headers, warnings as errors)"
  if command -v doxygen >/dev/null 2>&1; then
    if ci/docs.sh; then
      echo "doxygen: OK"
    else
      failed+=("docs")
    fi
  else
    echo "doxygen not found; skipped"
  fi
fi

if ! skipped precommit; then
  step "pre-commit run --all-files"
  if command -v pre-commit >/dev/null 2>&1; then
    if pre-commit run --all-files --show-diff-on-failure; then
      echo "pre-commit: OK"
    else
      failed+=("precommit")
    fi
  else
    echo "pre-commit not found; skipped"
  fi
fi

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/check.sh: FAILED: ${failed[*]}"
  exit 1
fi
echo "ci/check.sh: all checks passed"
