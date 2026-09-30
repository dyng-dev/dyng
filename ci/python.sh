#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The Python checks (PLAN Sections 5.4 and 8.8; ADR 0011): build the package in development mode,
# check that the committed stubs are current, and run the pytest suite.
#
#   ci/python.sh                  # editable install into the active environment, stubs, pytest
#   DYNG_PYTHON_INSTALL=0 ci/python.sh   # use the dyng already installed (a wheel, a venv)
#
# Steps:
#   1. pip install -e . --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build
#      (the development install of PLAN 7.7; needs scikit-build-core and nanobind of
#      environment.yml / pyproject.toml in the environment, and CMake >= 3.30 and Ninja)
#   2. python scripts/regen.py --stubs --check: python/dyng/_core.pyi equals nanobind.stubgen's
#      output for the built module (regenerate with `python scripts/regen.py --stubs`)
#   3. python -m pytest python/tests (the Hypothesis profile: DYNG_HYPOTHESIS_PROFILE=dyng|ci|dev)
#
# Heavy steps take the shared perf lock and run niced when ci/check.sh calls this script (it
# wraps the whole script); run alone, wrap it the same way on the shared machine:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/python.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python="${PYTHON:-python}"

if [ "${DYNG_PYTHON_INSTALL:-1}" != "0" ]; then
  echo "==> pip install -e . (development install, build directory build/)"
  "${python}" -m pip install --no-deps --no-build-isolation -Ceditable.rebuild=true \
    -Cbuild-dir=build -e . -q
fi

echo "==> installed: $("${python}" -c 'import dyng; print(dyng.__version__, dyng.__file__)')"

echo "==> scripts/regen.py --stubs --check"
"${python}" scripts/regen.py --stubs --check

echo "==> pytest python/tests"
"${python}" -m pytest -q python/tests
