#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# cibuildwheel's `repair-wheel-command` for the CUDA plugin wheels (ci/cibuildwheel-plugin.toml;
# ADRs 0030 and 0032), run INSIDE the manylinux_2_28 container after the build:
#
#   bash {project}/ci/cibw_plugin_repair.sh {project} {wheel} {dest_dir}
#
#   1. auditwheel repair --plat manylinux_2_28_x86_64 (libgomp bundled; libcuda, the user's
#      driver, excluded: ADR 0030 item 6);
#   2. the repaired wheel's module carries exactly the SASS and PTX of the release list of the
#      toolkit that built it (PLAN 7.7: cmake/cuda_architectures.cmake's list for that release),
#      read with that toolkit's cuobjdump (ci/wheel_check.py --code-objects): an architecture
#      lost on the way (CUDAARCHS or CMAKE_CUDA_ARCHITECTURES in the environment, the setting
#      lost when the tree was rendered, a toolkit on another branch of the list) fails the build
#      here instead of reaching PyPI.
set -euo pipefail

project="${1:?usage: ci/cibw_plugin_repair.sh <project> <wheel> <dest_dir>}"
wheel="${2:?usage: ci/cibw_plugin_repair.sh <project> <wheel> <dest_dir>}"
dest="${3:?usage: ci/cibw_plugin_repair.sh <project> <wheel> <dest_dir>}"
python=/opt/python/cp312-cp312/bin/python
cuda=/usr/local/cuda

auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude libcuda.so.1 \
  --exclude libcuda.so -w "${dest}" "${wheel}"

release="$("${cuda}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
echo "==> the code objects of the repaired wheel (CUDA ${release})"
"${python}" "${project}/ci/wheel_check.py" --code-objects --cuda-release "${release}" \
  --cuobjdump "${cuda}/bin/cuobjdump" "${dest}"/dyng_cu*.whl
