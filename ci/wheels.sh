#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The LOCAL build of the distributions of `dyng` (PLAN Section 7.7; ADR 0025): the sdist, a
# manylinux_2_28 x86_64 abi3 CPU wheel built FROM that sdist, auditwheel repair (libgomp bundled),
# `twine check`, ci/wheel_check.py (size budget of 90 MB, tags, contents) and an install test of
# the wheel alone in a fresh venv per Python version (import, the CLI, sssp and cycle_count on
# both host backends, then the pytest suite of python/tests against the installed wheel).
#
# CI builds the same distributions with cibuildwheel inside the manylinux_2_28 image
# (.github/workflows/wheels.yml). This machine has no container runtime, so this script uses
# `pip wheel` with a glibc 2.28 conda toolchain (ci/wheel-toolchain.yml) and auditwheel instead;
# the differences are listed in docs/developer/wheels.md.
#
#   ci/wheels.sh                                 # into $DYNG_SCRATCH/wheels/<version>
#   DYNG_WHEEL_OUT=/some/dir ci/wheels.sh
#   DYNG_WHEEL_PYTHONS="/path/python3.12 /path/python3.13" ci/wheels.sh
#   DYNG_WHEEL_SKIP_TESTS=1 ci/wheels.sh         # build and check only
#
# Environment (defaults under DYNG_SCRATCH, default $HOME/Projects/dyng-work):
#   DYNG_WHEEL_TOOLCHAIN  conda prefix of ci/wheel-toolchain.yml   ($DYNG_SCRATCH/tools/manylinux228-tc)
#   DYNG_WHEEL_TOOLS      venv with auditwheel, patchelf and twine ($DYNG_SCRATCH/tools/wheeltools)
#   DYNG_WHEEL_PLAT       auditwheel policy                        (manylinux_2_28_x86_64)
#   DYNG_WHEEL_PYTHONS    interpreters for the install test        (python3.12 on PATH, and
#                                                                   $DYNG_SCRATCH/tools/py313/bin/python)
# The build tools (scikit-build-core, nanobind, CMake, Ninja, build) come from the active
# environment (dyng-dev; the builds use --no-isolation). Run it under the shared perf lock:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/wheels.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
version="$(tr -d '[:space:]' < VERSION)"
out="${DYNG_WHEEL_OUT:-${scratch}/wheels/${version}}"
toolchain="${DYNG_WHEEL_TOOLCHAIN:-${scratch}/tools/manylinux228-tc}"
tools="${DYNG_WHEEL_TOOLS:-${scratch}/tools/wheeltools}"
plat="${DYNG_WHEEL_PLAT:-manylinux_2_28_x86_64}"
python="${PYTHON:-python}"

if [ -z "${DYNG_WHEEL_PYTHONS:-}" ]; then
  DYNG_WHEEL_PYTHONS="$(command -v python3.12 || true)"
  if [ -x "${scratch}/tools/py313/bin/python" ]; then
    DYNG_WHEEL_PYTHONS="${DYNG_WHEEL_PYTHONS} ${scratch}/tools/py313/bin/python"
  fi
fi

cxx="${toolchain}/bin/x86_64-conda-linux-gnu-g++"
cc="${toolchain}/bin/x86_64-conda-linux-gnu-gcc"
for f in "${cxx}" "${cc}" "${tools}/bin/auditwheel" "${tools}/bin/twine"; do
  if [ ! -x "${f}" ]; then
    echo "ci/wheels.sh: ${f} is missing; see the header of this script" >&2
    exit 1
  fi
done

rm -rf "${out}"
mkdir -p "${out}/raw" "${out}/dist"
build_dir="$(mktemp -d "${out}/build.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

echo "==> sdist"
"${python}" -m build --sdist --no-isolation --outdir "${out}/dist" . >"${out}/sdist.log" 2>&1 \
  || { tail -30 "${out}/sdist.log"; exit 1; }
sdist="${out}/dist/dyng-${version}.tar.gz"

echo "==> wheel from the sdist (GCC 12, glibc 2.28 sysroot, static and hidden libstdc++)"
CC="${cc}" CXX="${cxx}" "${python}" -m pip wheel --no-deps --no-build-isolation -w "${out}/raw" \
  -Ccmake.define.CMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL" \
  -Cbuild-dir="${build_dir}" "${sdist}" >"${out}/wheel.log" 2>&1 \
  || { tail -40 "${out}/wheel.log"; exit 1; }

echo "==> auditwheel repair --plat ${plat}"
PATH="${tools}/bin:${PATH}" LD_LIBRARY_PATH="${toolchain}/lib" "${tools}/bin/auditwheel" \
  repair --plat "${plat}" --only-plat -w "${out}/dist" "${out}/raw"/dyng-*.whl \
  >"${out}/repair.log" 2>&1 || { cat "${out}/repair.log"; exit 1; }
"${tools}/bin/auditwheel" show "${out}/dist"/dyng-*.whl | tail -n +2

echo "==> twine check, ci/wheel_check.py"
"${tools}/bin/twine" check --strict "${out}/dist"/*
"${python}" ci/wheel_check.py "${out}/dist"/* --platform "${plat}" --require-libgomp

if [ "${DYNG_WHEEL_SKIP_TESTS:-0}" = "1" ]; then
  echo "==> distributions in ${out}/dist (install tests skipped)"
  exit 0
fi

wheel="$(ls "${out}/dist"/dyng-*.whl)"
for py in ${DYNG_WHEEL_PYTHONS}; do
  pyver="$("${py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  venv="${out}/venv-${pyver}"
  echo "==> Python ${pyver}: a fresh venv with only the wheel (${venv})"
  "${py}" -m venv "${venv}"
  "${venv}/bin/python" -m pip install -q --upgrade pip
  "${venv}/bin/python" -m pip install -q "${wheel}"
  # From an empty directory, so nothing of the source tree can be imported.
  (
    cd "$(mktemp -d)"
    "${venv}/bin/python" - <<'PY'
import dyng
print("   ", dyng.__version__, dyng.__file__)
assert "site-packages" in dyng.__file__
g = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 0], [4, 1, 1, 5])
for backend in ("sequential", "openmp"):
    res = dyng.Resources(backend)
    t = dyng.sssp.compute(g, 0, resources=res)
    h = dyng.cycle_count.compute(g, max_length=3, resources=res)
    assert t.distances.tolist() == [0, 4, 1], t.distances.tolist()
    assert h.total == 2, h.total
print("    sssp and cycle_count on sequential and openmp: ok")
PY
    "${venv}/bin/dyng" --version
    "${venv}/bin/dyng" config | head -3
  )
  "${venv}/bin/python" -m pip install -q "pytest>=8" "hypothesis==6.167.1"
  echo "    pytest python/tests against the installed wheel"
  log="${out}/pytest-${pyver}.log"
  if ! OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "${venv}/bin/python" -m pytest -q \
    -p no:cacheprovider python/tests >"${log}" 2>&1; then
    tail -40 "${log}"
    exit 1
  fi
  tail -1 "${log}"
  # nanobind reports objects still alive when the module is finalized (a reference leak).
  if grep -q "nanobind: leaked" "${log}"; then
    grep -A5 "nanobind: leaked" "${log}"
    echo "ci/wheels.sh: nanobind reported leaked objects (${log})" >&2
    exit 1
  fi
done
echo "==> distributions in ${out}/dist"
ls -l "${out}/dist"
