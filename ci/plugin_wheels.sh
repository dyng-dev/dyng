#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The LOCAL build of the CUDA plugin wheels dyng-cu12 / dyng-cu13 (PLAN Sections 5.4, 7.7 and 7.8;
# ADR 0030), the counterpart of ci/wheels.sh for the CPU wheel. For each plugin:
#   1. the core sdist and wheel are built by ci/wheels.sh from this tree (or taken from
#      DYNG_CORE_DIST); the sdist is unpacked once per plugin and turned into
#      the plugin's source tree by ci/plugin_pyproject.py (its pyproject.toml, and the CUDA
#      toolkit's EULA copied in as a licence file);
#   2. `pip wheel` builds it with the glibc 2.28 toolchain of ci/wheel-toolchain.yml as host
#      compiler and the plugin's CUDA toolkit (CUDA 12.x for cu12, 13.x for cu13): every backend,
#      the static CUDA runtime, SASS for the release list of cmake/cuda_architectures.cmake and PTX
#      for its highest entry, libstdc++ / libgcc static and hidden as in the CPU wheel;
#   3. auditwheel repair --plat manylinux_2_28_x86_64 (libgomp bundled as dyng_cu<N>.libs/;
#      libcuda.so.1, the user's driver, excluded), `twine check --strict`, ci/wheel_check.py
#      (size budget 90 MB, contents, entry point, dependency, licence files, no libcuda /
#      libcudart needed or bundled);
#   4. unless DYNG_WHEEL_SKIP_TESTS=1, an install test per Python version: a fresh venv with the
#      core wheel and the plugin wheel only (no index), then
#        - selection: `dyng.show_config()`; the plugin's module is active and runs sssp,
#          cycle_count and mosp on the CUDA backend (GPU ${DYNG_TEST_GPU:-1}); the device arrays
#          equal the sequential backend's element by element; with no visible device
#          (CUDA_VISIBLE_DEVICES=) dynG falls back to dyng._core with a dyng.BackendWarning;
#        - the GPU tests of python/tests (pytest -m gpu, test_cuda.py: CUDA == sequential, a
#          subset of the goldens, device arrays, the fallback; DYNG_REQUIRE_CUDA=1, so they fail
#          instead of skipping without the plugin's CUDA backend);
#        - the pytest suite of python/tests with DYNG_CPU_ONLY=1 (the CPU path is unchanged by an
#          installed plugin);
#      and, with DYNG_PLUGIN_INTEROP_PYTHON (the python of a venv that has PyTorch and/or CuPy,
#      e.g. $DYNG_SCRATCH/venvs/m6a-interop-3.12), the core and plugin wheels are reinstalled
#      into that venv and the GPU tests run there too (the DLPack / CUDA array interface round
#      trips with PyTorch and CuPy).
#
#   ci/plugin_wheels.sh                                   # cu13 with /usr/local/cuda-13.1
#   DYNG_PLUGINS="cu12 cu13" DYNG_CUDA12_ROOT=$HOME/anaconda3/envs/cuda_12.8 ci/plugin_wheels.sh
#   DYNG_WHEEL_SKIP_TESTS=1 ci/plugin_wheels.sh            # build and check only
#   DYNG_PLUGIN_TEST_ONLY=1 ci/plugin_wheels.sh            # check and test the wheels built before
#
# Environment (defaults under DYNG_SCRATCH, default $HOME/Projects/dyng-work):
#   DYNG_PLUGINS          the plugins to build                     (cu13)
#   DYNG_CUDA12_ROOT      CUDA 12.x toolkit for cu12               (none: cu12 needs it)
#   DYNG_CUDA13_ROOT      CUDA 13.x toolkit for cu13               (/usr/local/cuda-13.1)
#   DYNG_CUDA12_EULA, DYNG_CUDA13_EULA   the toolkit's EULA, when it is not <root>/EULA.txt or a
#                         conda prefix's LICENSE (ci/plugin_pyproject.py --cuda-eula)
#   DYNG_CORE_DIST        a directory with the core sdist and wheel of this version (default:
#                         built by ci/wheels.sh into $DYNG_PLUGIN_OUT/core/dist)
#   DYNG_PLUGIN_OUT       output directory                         ($DYNG_SCRATCH/wheels/<version>-plugins)
#   DYNG_BUILD_JOBS       parallel compile jobs                    (16; the machine is shared)
#   DYNG_WHEEL_TOOLCHAIN, DYNG_WHEEL_TOOLS, DYNG_WHEEL_PLAT, DYNG_WHEEL_PYTHONS   as in ci/wheels.sh
#   DYNG_TEST_GPU         the GPU of the install test              (1; GPU 0 is for timing runs)
#   DYNG_PLUGIN_INTEROP_PYTHON  a venv's python with torch / cupy for the round trips (none)
# Run it under the shared perf lock:
#   flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/plugin_wheels.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
version="$(tr -d '[:space:]' < VERSION)"
plugins="${DYNG_PLUGINS:-cu13}"
core_dist="${DYNG_CORE_DIST:-}"
out="${DYNG_PLUGIN_OUT:-${scratch}/wheels/${version}-plugins}"
toolchain="${DYNG_WHEEL_TOOLCHAIN:-${scratch}/tools/manylinux228-tc}"
tools="${DYNG_WHEEL_TOOLS:-${scratch}/tools/wheeltools}"
plat="${DYNG_WHEEL_PLAT:-manylinux_2_28_x86_64}"
jobs="${DYNG_BUILD_JOBS:-16}"
gpu="${DYNG_TEST_GPU:-1}"
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
    echo "ci/plugin_wheels.sh: ${f} is missing; see docs/developer/wheels.md" >&2
    exit 1
  fi
done

test_only="${DYNG_PLUGIN_TEST_ONLY:-0}"
if [ "${test_only}" != "1" ]; then
  rm -rf "${out}"
  mkdir -p "${out}/raw" "${out}/dist"
fi
if [ -z "${core_dist}" ]; then
  core_dist="${out}/core/dist"
  if [ "${test_only}" != "1" ]; then
    # The plugins are built from the sdist of this tree, so the core is built first, here.
    echo "==> the core sdist and wheel of this tree: ci/wheels.sh (build and check only)"
    DYNG_WHEEL_OUT="${out}/core" DYNG_WHEEL_SKIP_TESTS=1 ci/wheels.sh | sed 's/^/    /'
  fi
fi
sdist="${core_dist}/dyng-${version}.tar.gz"
core_wheel="$(ls "${core_dist}"/dyng-"${version}"-*.whl)"
work="$(mktemp -d "${out}/work.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

for plugin in ${plugins}; do
  [ "${test_only}" = "1" ] && break
  case "${plugin}" in
    cu12) root="${DYNG_CUDA12_ROOT:-}"; eula="${DYNG_CUDA12_EULA:-}" ;;
    cu13) root="${DYNG_CUDA13_ROOT:-/usr/local/cuda-13.1}"; eula="${DYNG_CUDA13_EULA:-}" ;;
    *) echo "ci/plugin_wheels.sh: unknown plugin ${plugin} (cu12, cu13)" >&2; exit 2 ;;
  esac
  if [ -z "${root}" ] || [ ! -x "${root}/bin/nvcc" ]; then
    echo "ci/plugin_wheels.sh: ${plugin} needs its CUDA toolkit: set DYNG_CUDA${plugin#cu}_ROOT" \
      "(found '${root}')" >&2
    exit 1
  fi
  echo "==> ${plugin}: the source tree from dyng-${version}.tar.gz (CUDA $("${root}/bin/nvcc" \
    --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), ${root})"
  src="${work}/${plugin}"
  mkdir -p "${src}"
  tar -xzf "${sdist}" -C "${src}" --strip-components=1
  "${python}" ci/plugin_pyproject.py --plugin "${plugin}" --project-dir "${src}" \
    --cuda-root "${root}" ${eula:+--cuda-eula "${eula}"}

  echo "==> ${plugin}: wheel (GCC 12 / glibc 2.28 host compiler, static cudart, release architectures)"
  log="${out}/wheel-${plugin}.log"
  CC="${cc}" CXX="${cxx}" CUDACXX="${root}/bin/nvcc" CUDAHOSTCXX="${cxx}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${jobs}" \
    "${python}" -m pip wheel -v --no-deps --no-build-isolation -w "${out}/raw" \
    -Ccmake.define.CMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL" \
    -Ccmake.define.CUDAToolkit_ROOT="${root}" \
    -Cbuild-dir="${work}/build-${plugin}" "${src}" >"${log}" 2>&1 \
    || { tail -40 "${log}"; exit 1; }
  grep -E "dynG: CUDA .* architectures" "${log}" | sed 's/^ */    /' || true
  rm -rf "${work}/build-${plugin}" "${src}"

  echo "==> ${plugin}: auditwheel repair --plat ${plat} (libcuda excluded)"
  PATH="${tools}/bin:${PATH}" LD_LIBRARY_PATH="${toolchain}/lib" "${tools}/bin/auditwheel" \
    repair --plat "${plat}" --only-plat --exclude 'libcuda.so.1' --exclude 'libcuda.so' \
    -w "${out}/dist" "${out}/raw"/dyng_"${plugin}"-*.whl \
    >"${out}/repair-${plugin}.log" 2>&1 || { cat "${out}/repair-${plugin}.log"; exit 1; }
  "${tools}/bin/auditwheel" show "${out}/dist"/dyng_"${plugin}"-*.whl | tail -n +2
done

echo "==> twine check, ci/wheel_check.py"
"${tools}/bin/twine" check --strict "${out}/dist"/*
"${python}" ci/wheel_check.py "${out}/dist"/* "${core_wheel}" --platform "${plat}" --require-libgomp

if [ "${DYNG_WHEEL_SKIP_TESTS:-0}" = "1" ]; then
  echo "==> plugin wheels in ${out}/dist (install tests skipped)"
  ls -l "${out}/dist"
  exit 0
fi

for plugin in ${plugins}; do
  for py in ${DYNG_WHEEL_PYTHONS}; do
    pyver="$("${py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    venv="${out}/venv-${plugin}-${pyver}"
    echo "==> ${plugin}, Python ${pyver}: a fresh venv with the core and the plugin wheel (${venv})"
    "${py}" -m venv "${venv}"
    "${venv}/bin/python" -m pip install -q --upgrade pip
    # NumPy from the index; then no index: dyng==<version> must come from the core wheel.
    "${venv}/bin/python" -m pip install -q numpy
    "${venv}/bin/python" -m pip install -q --no-index --find-links "${core_dist}" \
      --find-links "${out}/dist" "dyng-${plugin}==${version}"
    (
      cd "$(mktemp -d)"
      CUDA_VISIBLE_DEVICES="${gpu}" DYNG_PLUGIN="${plugin}" "${venv}/bin/python" - <<'PY'
import os

import numpy as np

import dyng
import dyng._backend as backend

plugin = os.environ["DYNG_PLUGIN"]
dyng.show_config()
assert backend.active_module_name.startswith(f"dyng_{plugin}"), backend.active_module_name
c = dyng.config()
assert c["build"]["plugin"] == plugin and c["build"]["cuda_runtime"] == "static", c["build"]
assert c["backends"]["cuda"], c["backends"]
rng = np.random.default_rng(7)
n, m = 200, 1500
src, dst = rng.integers(0, n, m, dtype=np.int32), rng.integers(0, n, m, dtype=np.int32)
w = rng.integers(1, 20, m, dtype=np.int32)
keep = src != dst
src, dst, w = src[keep], dst[keep], w[keep]
seq, cuda = dyng.Resources.sequential(), dyng.Resources.cuda(0)
results = {}
for name, res in (("sequential", seq), ("cuda", cuda)):
    g = dyng.Graph.from_edges(src, dst, w, num_vertices=n, resources=res)
    t = dyng.sssp.compute(g, 0)
    h = dyng.cycle_count.compute(dyng.Graph.from_edges(src, dst, num_vertices=n, resources=res),
                                 max_length=4)
    gm = dyng.Graph.from_edges(src, dst, np.stack([w, (w * 7) % 13 + 1], axis=1),
                               num_vertices=n, properties="mosp_compatible", resources=res)
    mo = dyng.mosp.compute(gm, 0)
    results[name] = (t, h, mo)
t, h, mo = results["cuda"]
ts, hs, ms = results["sequential"]
assert t.distances.device == "cuda:0" and mo.distances(0).device == "cuda:0", t.distances.device
assert t.distances.__cuda_array_interface__["shape"] == (n,)
same = np.array_equal
assert same(t.distances.to_numpy(), ts.distances.to_numpy())
assert same(t.parents.to_numpy(), ts.parents.to_numpy())
assert same(h.counts.to_numpy(), hs.counts.to_numpy()) and h.total > 0
for k in range(2):
    assert same(mo.distances(k).to_numpy(), ms.distances(k).to_numpy())
    assert same(mo.parents(k).to_numpy(), ms.parents(k).to_numpy())
assert same(mo.combined_parents.to_numpy(), ms.combined_parents.to_numpy())
assert same(mo.path_costs.to_numpy(), ms.path_costs.to_numpy())
print(f"    sssp, cycle_count and mosp ran on cuda ({backend.active_module_name}); "
      "their arrays equal the sequential backend's")
PY
      out_cpu="$(CUDA_VISIBLE_DEVICES= "${venv}/bin/python" -c '
import warnings, dyng, dyng._backend as b
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dyng.__version__
assert [x.category for x in w] == [dyng.BackendWarning], w
print(b.active_module_name)')"
      [ "${out_cpu}" = "dyng._core" ] || { echo "no fallback without a device: ${out_cpu}" >&2; exit 1; }
      echo "    no visible device: falls back to ${out_cpu} with a dyng.BackendWarning"
    )
    log="${out}/pytest-gpu-${plugin}-${pyver}.log"
    "${venv}/bin/python" -m pip install -q "pytest>=8" "hypothesis==6.167.1"
    echo "    pytest -m gpu python/tests (GPU ${gpu})"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" DYNG_REQUIRE_CUDA=1 "${venv}/bin/python" -m pytest -q \
      -p no:cacheprovider -m gpu python/tests >"${log}" 2>&1; then
      tail -40 "${log}"
      exit 1
    fi
    tail -1 "${log}"
    log="${out}/pytest-${plugin}-${pyver}.log"
    echo "    pytest python/tests with DYNG_CPU_ONLY=1 (plugin installed)"
    if ! DYNG_CPU_ONLY=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "${venv}/bin/python" -m pytest -q \
      -p no:cacheprovider python/tests >"${log}" 2>&1; then
      tail -40 "${log}"
      exit 1
    fi
    tail -1 "${log}"
  done
done
interop="${DYNG_PLUGIN_INTEROP_PYTHON:-}"
if [ -n "${interop}" ]; then
  for plugin in ${plugins}; do
    echo "==> ${plugin}: the GPU tests with PyTorch / CuPy in ${interop}"
    "${interop}" -m pip install -q --force-reinstall --no-deps "${core_wheel}" \
      "${out}/dist"/dyng_"${plugin}"-*.whl
    log="${out}/pytest-gpu-interop-${plugin}.log"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" DYNG_REQUIRE_CUDA=1 "${interop}" -m pytest -q \
      -p no:cacheprovider -m gpu -rs python/tests >"${log}" 2>&1; then
      tail -40 "${log}"
      exit 1
    fi
    grep -E "SKIPPED|passed|failed" "${log}" | sed 's/^/    /'
    # The plugin stays installed only for this run's plugin list: uninstall, so the venv's
    # selection is not decided by a leftover plugin of another version.
    "${interop}" -m pip uninstall -q -y "dyng-${plugin}"
  done
fi
echo "==> plugin wheels in ${out}/dist"
ls -l "${out}/dist"
