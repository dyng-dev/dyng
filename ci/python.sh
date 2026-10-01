#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The Python checks (PLAN Sections 5.4 and 8.8; ADR 0011): build the package in development mode,
# check that the committed stubs are current, and run the pytest suite.
#
#   ci/python.sh                  # editable install (see below), stubs, mypy, pytest
#   DYNG_PYTHON_INSTALL=0 ci/python.sh      # use the dyng already installed (a wheel, a venv)
#   DYNG_PYTHON_INSTALL=venv ci/python.sh   # always the throwaway venv build/py-venv (below)
#   DYNG_PYTHON_INSTALL=force ci/python.sh  # re-point the active environment at this checkout
#
# Steps:
#   1. pip install -e . --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build
#      (the development install of PLAN 7.7; needs scikit-build-core and nanobind of
#      environment.yml / pyproject.toml in the environment, and CMake >= 3.30 and Ninja).
#      Into the active environment when it has no dyng yet or its editable dyng is this
#      checkout. When its dyng is an editable install of ANOTHER checkout (the shared dyng-dev
#      environment and a fresh clone, a review worktree), re-pointing it would leave that
#      environment broken once this checkout is deleted, and concurrent runs would race on it:
#      the script then installs into a throwaway venv, build/py-venv (--system-site-packages, so
#      the environment's NumPy, pytest, scikit-build-core, ... are used; the venv's own editable
#      finder shadows the environment's), and runs every step with it.
#   2. python scripts/regen.py --stubs --check: python/dyng/_core.pyi equals nanobind.stubgen's
#      output for the built module (regenerate with `python scripts/regen.py --stubs`)
#   3. mypy (strict, [tool.mypy] of pyproject.toml) over the typed layer python/dyng and its
#      stubs; skipped with a note when mypy is not installed, required in CI
#   4. python -m pytest python/tests (the Hypothesis profile: DYNG_HYPOTHESIS_PROFILE=dyng|ci|dev)
#
# Heavy steps take the shared perf lock and run niced when ci/check.sh calls this script (it
# wraps the whole script); run alone, wrap it the same way on the shared machine:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/python.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python="${PYTHON:-python}"

mode="${DYNG_PYTHON_INSTALL:-1}"
if [ "${mode}" = "1" ]; then
  # Where does the active environment's dyng point? (empty: not installed, or not editable)
  current="$("${python}" -m pip show dyng 2>/dev/null |
    sed -n 's/^Editable project location: //p')" || current=""
  if [ -n "${current}" ] && [ "$(realpath -m "${current}")" != "$(realpath "${repo_root}")" ]; then
    echo "==> the environment's dyng is an editable install of ${current}; not re-pointing it"
    echo "    (DYNG_PYTHON_INSTALL=force would): using the throwaway venv build/py-venv"
    mode=venv
  fi
fi
case "${mode}" in
  0) ;;
  venv)
    venv="${repo_root}/build/py-venv"
    if [ ! -x "${venv}/bin/python" ]; then
      "${python}" -m venv --system-site-packages "${venv}"
    fi
    python="${venv}/bin/python"
    echo "==> pip install -e . into ${venv} (build directory build/)"
    "${python}" -m pip install --no-deps --no-build-isolation -Ceditable.rebuild=true \
      -Cbuild-dir=build -e . -q
    ;;
  1 | force)
    echo "==> pip install -e . (development install, build directory build/)"
    "${python}" -m pip install --no-deps --no-build-isolation -Ceditable.rebuild=true \
      -Cbuild-dir=build -e . -q
    ;;
  *)
    echo "ci/python.sh: DYNG_PYTHON_INSTALL must be 0, 1, venv or force (got ${mode})" >&2
    exit 2
    ;;
esac

echo "==> installed: $("${python}" -c 'import dyng; print(dyng.__version__, dyng.__file__)')"

echo "==> scripts/regen.py --stubs --check"
"${python}" scripts/regen.py --stubs --check

echo "==> mypy (strict) over python/dyng"
if "${python}" -m mypy --version >/dev/null 2>&1; then
  "${python}" -m mypy
elif [ -n "${CI:-}" ]; then
  echo "mypy not found (in CI the step must run)" >&2
  exit 1
else
  echo "mypy not installed (conda env update -f environment.yml); skipped"
fi

echo "==> pytest python/tests"
"${python}" -m pytest -q python/tests
