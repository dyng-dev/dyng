#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The local CPU gate (PLAN Section 8.8): run before every commit. CI runs the same steps, split
# over the hosted workflows (see the end of this comment).
#
#   ci/check.sh                   # cpu-only and dev presets, clang-format, REUSE, Doxygen, pre-commit
#   ci/check.sh --parity          # ... and the golden parity replay (parity preset, needs goldens)
#   DYNG_CHECK_PRESETS=dev ci/check.sh
#   DYNG_CHECK_SKIP="precommit docs" ci/check.sh
#   DYNG_CHECK_ONLY="tidy" ci/check.sh    # only the named steps (the hosted `tidy` job)
#
# Steps (each can be skipped by name in DYNG_CHECK_SKIP, or selected with DYNG_CHECK_ONLY):
#   format     clang-format --dry-run --Werror on all tracked C++/CUDA sources
#   build      configure, build and `ctest -L cpu` for every preset in DYNG_CHECK_PRESETS
#   tidy       clang-tidy with only the naming rules of ADR 0004 (readability-identifier-naming)
#              on the library sources and the public headers they include; needs the
#              compile_commands.json of a preset configured above (skipped if clang-tidy is
#              missing, except in CI)
#   reuse      reuse lint (SPDX headers in every file)
#   provenance ci/provenance_check.py: every file naming an original's symbol carries
#              '// Derived from <repo>@<sha>:<path>' (PLAN Sections 3.4, 6.3)
#   harness    python -m py_compile on parity/*.py and ci/*.py, and pytest parity/tests (the
#              smoke tests of the parity harness; pytest is in environment.yml)
#   docs       ci/docs.sh: Doxygen on the public headers with warnings as errors and the
#              convention check ci/doxygen_coverage.py, then the Sphinx site with warnings as
#              errors and the internal link check (skipped if doxygen is missing; fails if the
#              Sphinx packages of environment.yml are missing)
#   precommit  pre-commit run --all-files (skipped if pre-commit is missing)
#   parity     only with --parity (or DYNG_CHECK_PARITY=1): configure and build the `parity`
#              preset and run `ctest -L parity`, the byte-for-byte replay of the golden corpus
#              (PLAN Section 8.3). The goldens are not in the repository: create them first with
#              parity/build_reference.sh and parity/export_goldens.py (parity/README.md). With
#              --parity, missing goldens are an error, not a skip.
#
# The GitHub workflows mirror these steps: cpu.yml runs `build`; lint.yml runs `precommit`
# (which includes clang-format, REUSE and provenance), `harness`, `tidy` (on a configured
# cpu-only tree) and the name-reservation package check; docs.yml runs `docs`. Only `parity`
# (it needs the goldens, which are not in the repository) runs locally only.
set -euo pipefail

run_parity="${DYNG_CHECK_PARITY:-0}"
for arg in "$@"; do
  case "${arg}" in
    --parity) run_parity=1 ;;
    -h | --help)
      sed -n '5,40p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ci/check.sh: unknown argument '${arg}' (try --help)" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Tools: the dyng-dev conda env locally; in CI they are already on PATH.
if ! command -v cmake >/dev/null 2>&1 || [ -z "${CI:-}" ]; then
  # shellcheck disable=SC1091
  source scripts/dev_env.sh
fi

presets="${DYNG_CHECK_PRESETS:-cpu-only dev}"
skip=" ${DYNG_CHECK_SKIP:-} "
only=" ${DYNG_CHECK_ONLY:-} "
failed=()

step() { printf '\n==> %s\n' "$*"; }
skipped() { [[ "${skip}" == *" $1 "* ]] || { [ -n "${DYNG_CHECK_ONLY:-}" ] && [[ "${only}" != *" $1 "* ]]; }; }

if ! skipped format; then
  step "clang-format ($(clang-format --version | head -n1))"
  mapfile -t sources < <(git ls-files '*.hpp' '*.cpp' '*.cuh' '*.cu' '*.h')
  if [ "${#sources[@]}" -eq 0 ] || clang-format --dry-run --Werror "${sources[@]}"; then
    echo "clang-format: OK"
  else
    failed+=("format")
  fi
fi

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

if ! skipped tidy; then
  step "clang-tidy (naming rules, library sources)"
  tidy_db=""
  for preset in ${presets}; do
    if [ -f "build/${preset}/compile_commands.json" ]; then
      tidy_db="build/${preset}"
      break
    fi
  done
  if ! command -v clang-tidy >/dev/null 2>&1; then
    if [ -n "${CI:-}" ]; then
      echo "clang-tidy not found (in CI the step must run)"
      failed+=("tidy")
    else
      echo "clang-tidy not found; skipped"
    fi
  elif [ -z "${tidy_db}" ]; then
    echo "no compile_commands.json (build step skipped?)"
    failed+=("tidy")
  else
    # The conda clang-tidy ships without clang's resource headers (stddef.h, ...); GCC's builtin
    # include directory stands in for them. libstdc++ 12's get_temporary_buffer is deprecated
    # for clang (not for GCC); that diagnostic is not ours.
    gcc_include="$("${CXX:-g++}" -print-file-name=include)"
    if git ls-files 'cpp/src/*.cpp' | xargs -r -n 1 -P "$(nproc)" clang-tidy -p "${tidy_db}" \
      --quiet --checks='-*,readability-identifier-naming' --warnings-as-errors='*' \
      "--extra-arg=-isystem${gcc_include}" --extra-arg=-Wno-deprecated-declarations; then
      echo "clang-tidy: OK"
    else
      failed+=("tidy")
    fi
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

if ! skipped provenance; then
  step "provenance headers"
  if python3 ci/provenance_check.py; then
    :
  else
    failed+=("provenance")
  fi
fi

if ! skipped harness; then
  step "parity harness (Python): compile and smoke tests"
  if python3 -m py_compile parity/*.py ci/*.py && python3 -m pytest -q parity/tests; then
    echo "harness: OK"
  else
    failed+=("harness")
  fi
fi

if ! skipped docs; then
  step "docs (Doxygen and the Sphinx site, warnings as errors; internal links)"
  if command -v doxygen >/dev/null 2>&1; then
    if ci/docs.sh; then
      echo "docs: OK"
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

if [ "${run_parity}" = "1" ] && ! skipped parity; then
  step "parity preset: golden replay (ctest -L parity)"
  goldens="${DYNG_GOLDENS_DIR:-${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}/goldens}"
  if [ ! -f "${goldens}/sssp/MANIFEST.sha256" ]; then
    echo "no goldens in ${goldens}/sssp: run parity/build_reference.sh and parity/export_goldens.py"
    failed+=("parity:no-goldens")
  elif cmake --preset parity -DDYNG_GOLDENS_DIR="${goldens}" && cmake --build --preset parity &&
    ctest --preset parity -L parity; then
    echo "parity: OK"
  else
    failed+=("parity")
  fi
fi

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/check.sh: FAILED: ${failed[*]}"
  exit 1
fi
echo "ci/check.sh: all checks passed"
