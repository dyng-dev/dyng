#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The local CPU gate (PLAN Section 8.8): run before every commit. CI runs the same steps, split
# over the hosted workflows (see the end of this comment).
#
#   ci/check.sh                   # cpu-only and dev presets, clang-format, REUSE, docs, pre-commit
#   ci/check.sh --parity          # ... and the golden parity replay (parity preset, needs goldens)
#   ci/check.sh --wheels          # ... and the local build and install test of the distributions
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
#   harness    python -m py_compile on parity/*.py and ci/*.py, and pytest parity/tests ci/tests
#              (the smoke tests of the parity harness and the tests of the CI scripts; pytest is
#              in environment.yml)
#   docs       ci/docs.sh: Doxygen on the public headers with warnings as errors and the
#              convention check ci/doxygen_coverage.py, then the Sphinx site with warnings as
#              errors and ci/docs_links.py (repository URLs, the site's anchors) (skipped if
#              doxygen is missing; fails if the Sphinx packages of environment.yml are missing)
#   regen      scripts/regen.py --check: the algorithm tables, the CODEOWNERS block and the
#              registries are up to date with the manifests (PLAN Section 4.8)
#   python     ci/python.sh: the development install of the Python package (pip install -e .),
#              the committed stubs (scripts/regen.py --stubs --check), mypy --strict and the
#              pytest suite python/tests (skipped if scikit-build-core or nanobind is missing,
#              except in CI). The install goes into the active environment unless its dyng is
#              an editable install of another checkout (the shared dyng-dev and a fresh clone):
#              then into the throwaway venv build/py-venv of this checkout, so the shared
#              environment keeps pointing where it did (DYNG_PYTHON_INSTALL=force re-points it)
#   api        ci/api_check.sh: griffe compares the Python API with the base branch (origin/main,
#              else main) and fails on a breaking change (the api-change label accepts one in CI;
#              locally set DYNG_API_CHANGE=1); skipped if griffe is missing, except in CI
#   scaffold   ci/scaffold_check.sh: scripts/new_algorithm.py generates a throwaway algorithm of
#              each family, which builds and passes its conformance kit (a nested build in a
#              temporary copy of the tree)
#   precommit  pre-commit run --all-files (skipped if pre-commit is missing)
#   parity     only with --parity (or DYNG_CHECK_PARITY=1): configure and build the `parity`
#              preset and run `ctest -L parity`, the byte-for-byte replay of the golden corpus
#              (PLAN Section 8.3). The goldens are not in the repository: create them first with
#              parity/build_reference.sh and parity/export_goldens.py (parity/README.md). With
#              --parity, missing goldens are an error, not a skip.
#   wheels     only with --wheels (or DYNG_CHECK_WHEELS=1): ci/wheels.sh, the sdist and the
#              manylinux_2_28 CPU wheel built locally, auditwheel, twine check, ci/wheel_check.py
#              (the 90 MB budget) and the wheel alone in fresh venvs for Python 3.12 and 3.13 with
#              the pytest suite (needs the tools of docs/developer/wheels.md)
#
# The GitHub workflows mirror these steps: cpu.yml runs `build` and `scaffold`; lint.yml runs
# `precommit` (which includes clang-format, REUSE, provenance and regen), `harness`, `tidy` (on a
# configured cpu-only tree) and the name-reservation package check; docs.yml runs `docs`;
# python.yml runs `python` (and the suite against an installed sdist); api-check.yml runs `api`;
# wheels.yml builds the distributions with cibuildwheel (the hosted counterpart of `wheels`).
# Only `parity` (it needs the goldens, which are not in the repository) runs locally only.
# The CUDA tests
# are not part of this gate: ci/gpu_local.sh runs them on a GPU machine, and cuda-build.yml
# compiles the CUDA presets on hosted runners (no GPU).
#
# Machine rules (the development machine is shared): the heavy steps (build and tests, clang-tidy,
# the harness tests, the docs build, pre-commit, the parity replay) take the SHARED lock
# ${DYNG_PERF_LOCK} and run niced, so they never overlap a timing run (which takes the lock
# exclusively). The default lock is $DYNG_SCRATCH/perf.lock when that directory exists and CI is
# not set; DYNG_PERF_LOCK= (empty) runs without it.
set -euo pipefail

run_parity="${DYNG_CHECK_PARITY:-0}"
run_wheels="${DYNG_CHECK_WHEELS:-0}"
for arg in "$@"; do
  case "${arg}" in
    --parity) run_parity=1 ;;
    --wheels) run_wheels=1 ;;
    -h | --help)
      sed -n '5,68p' "${BASH_SOURCE[0]}"
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
if [ -n "${DYNG_PERF_LOCK+set}" ]; then
  lock="${DYNG_PERF_LOCK}"
elif [ -z "${CI:-}" ] && [ -n "${DYNG_SCRATCH:-}" ] && [ -d "${DYNG_SCRATCH}" ]; then
  lock="${DYNG_SCRATCH}/perf.lock"
else
  lock=""
fi

step() { printf '\n==> %s\n' "$*"; }
skipped() { [[ "${skip}" == *" $1 "* ]] || { [ -n "${DYNG_CHECK_ONLY:-}" ] && [[ "${only}" != *" $1 "* ]]; }; }
# Heavy commands: shared lock (never during an exclusive timing run), niced.
heavy() {
  if [ -n "${lock}" ]; then
    flock -s "${lock}" nice -n 10 "$@"
  else
    nice -n 10 "$@"
  fi
}

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
    if heavy cmake --preset "${preset}" && heavy cmake --build --preset "${preset}" &&
      heavy ctest --preset "${preset}" -L cpu; then
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
    # cpp/src/algorithms/_template is not compiled (scripts/new_algorithm.py instantiates it).
    if git ls-files 'cpp/src/*.cpp' ':!:cpp/src/algorithms/_template/*' | heavy xargs -r -n 1 -P "$(nproc)" clang-tidy \
      -p "${tidy_db}" --quiet --checks='-*,readability-identifier-naming' --warnings-as-errors='*' \
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

if ! skipped regen; then
  step "scripts/regen.py --check (the files generated from the manifests)"
  if python3 scripts/regen.py --check; then
    :
  else
    failed+=("regen")
  fi
fi

if ! skipped harness; then
  step "parity harness (Python): compile and smoke tests"
  if python3 -m py_compile parity/*.py ci/*.py scripts/*.py &&
    heavy python3 -m pytest -q parity/tests ci/tests; then
    echo "harness: OK"
  else
    failed+=("harness")
  fi
fi

if ! skipped python; then
  step "Python package: development install, stubs, pytest (ci/python.sh)"
  if ! python -c "import scikit_build_core, nanobind" >/dev/null 2>&1; then
    if [ -n "${CI:-}" ]; then
      echo "scikit-build-core / nanobind not found (in CI the step must run)"
      failed+=("python")
    else
      echo "scikit-build-core / nanobind not found (conda env update -f environment.yml); skipped"
    fi
  elif heavy ci/python.sh; then
    echo "python: OK"
  else
    failed+=("python")
  fi
fi

if ! skipped api; then
  step "Python API check: griffe against the base branch (ci/api_check.sh)"
  if ! command -v griffe >/dev/null 2>&1; then
    if [ -n "${CI:-}" ]; then
      echo "griffe not found (in CI the step must run)"
      failed+=("api")
    else
      echo "griffe not found (conda env update -f environment.yml); skipped"
    fi
  elif ci/api_check.sh; then
    :
  else
    failed+=("api")
  fi
fi

if ! skipped scaffold; then
  step "scaffold: scripts/new_algorithm.py builds and passes the conformance kit (ci/scaffold_check.sh)"
  if heavy ci/scaffold_check.sh; then
    :
  else
    failed+=("scaffold")
  fi
fi

if ! skipped docs; then
  step "docs (Doxygen and the Sphinx site, warnings as errors; repository links and anchors)"
  if command -v doxygen >/dev/null 2>&1; then
    if heavy ci/docs.sh; then
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
    if heavy pre-commit run --all-files --show-diff-on-failure; then
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
  elif heavy cmake --preset parity -DDYNG_GOLDENS_DIR="${goldens}" &&
    heavy cmake --build --preset parity && heavy ctest --preset parity -L parity; then
    echo "parity: OK"
  else
    failed+=("parity")
  fi
fi

if [ "${run_wheels}" = "1" ] && ! skipped wheels; then
  step "wheels: the distributions built and installed locally (ci/wheels.sh)"
  if heavy ci/wheels.sh; then
    echo "wheels: OK"
  else
    failed+=("wheels")
  fi
fi

echo
if [ "${#failed[@]}" -gt 0 ]; then
  echo "ci/check.sh: FAILED: ${failed[*]}"
  exit 1
fi
echo "ci/check.sh: all checks passed"
