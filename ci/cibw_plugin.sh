#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# cibuildwheel's `before-all` for the CUDA plugin wheels (ci/cibuildwheel-plugin.toml; ADR 0032),
# run once INSIDE the manylinux_2_28 container before the build:
#
#   bash {project}/ci/cibw_plugin.sh {project} {package}      (DYNG_PLUGIN=cu12|cu13 passed in)
#
#   1. installs the plugin's CUDA toolkit from the RPM files that wheels.yml downloaded and
#      verified on the host (python3 ci/cuda_toolkit.py download --plugin <p> --dest
#      {project}/.cuda-rpms; every file pinned by its SHA-256 in ci/cuda_toolkits.toml and
#      signed by NVIDIA's repository key, whose fingerprint ci/cuda_toolkit.py pins; both checked
#      again here, the signatures with gpg and with rpm -K after importing that key), with dnf
#      (signatures checked once more: localpkg_gpgcheck; what they need from outside NVIDIA's
#      repository, the system gcc-c++, comes from the image's repositories) and links
#      /usr/local/cuda to it;
#   2. checks the toolkit: nvcc's release is the locked one, of the plugin's CUDA major, and it
#      compiles and links a kernel with the image's host compiler (gcc-toolset) and the static
#      CUDA runtime, so an unsupported host compiler fails here, in a line, not in the build;
#   3. turns {package} (the unpacked core sdist) into the plugin's source tree:
#      ci/plugin_pyproject.py --plugin <p> --project-dir {package} --cuda-root /usr/local/cuda
#      (its pyproject.toml, and the toolkit's EULA.txt copied in as a licence file).
set -euo pipefail

project="${1:?usage: ci/cibw_plugin.sh <project> <package>}"
package="${2:?usage: ci/cibw_plugin.sh <project> <package>}"
plugin="${DYNG_PLUGIN:?DYNG_PLUGIN (cu12 or cu13) must be passed into the container}"
python=/opt/python/cp312-cp312/bin/python
rpms="${project}/.cuda-rpms"

echo "==> ${plugin}: the CUDA toolkit from ${rpms} (ci/cuda_toolkits.toml)"
root="$("${python}" "${project}/ci/cuda_toolkit.py" root --plugin "${plugin}")"
(cd "${rpms}" && sha256sum --check --quiet SHA256SUMS)
# The digests and the signatures (gpg, with the key accepted only by its pinned fingerprint).
"${python}" "${project}/ci/cuda_toolkit.py" verify --signatures --plugin "${plugin}" \
  --dest "${rpms}"
# rpm's own check, with that key (and nothing else of NVIDIA's) imported into the rpm database.
key="${rpms}/$("${python}" -c 'import sys; sys.path.insert(0, sys.argv[1]); import cuda_toolkit as c; print(c.SIGNING_KEY)' "${project}/ci")"
rpm --import "${key}"
n=0
for f in "${rpms}"/*.rpm; do
  n=$((n + 1))
  checked="$(rpm -K "${f}")" || { echo "ci/cibw_plugin.sh: ${checked}" >&2; exit 1; }
  case "${checked}" in
    *"signatures OK") ;;
    *) echo "ci/cibw_plugin.sh: not signed by NVIDIA's key: ${checked}" >&2; exit 1 ;;
  esac
done
echo "    ${n} RPMs: SHA-256 pinned, signed by NVIDIA's key (gpg, rpm -K)"
dnf install -y --setopt=install_weak_deps=False --setopt=localpkg_gpgcheck=1 "${rpms}"/*.rpm
ln -sfn "${root}" /usr/local/cuda

echo "==> ${plugin}: check ${root}"
"${root}/bin/nvcc" --version
release="$("${root}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
expected="$(basename "${root}")"
if [ "cuda-${release}" != "${expected}" ] || [ "${release%%.*}" != "${plugin#cu}" ]; then
  echo "ci/cibw_plugin.sh: nvcc is CUDA ${release}; ${plugin} needs ${expected}" >&2
  exit 1
fi
test -f "${root}/EULA.txt"
check="$(mktemp -d)"
cat >"${check}/k.cu" <<'CU'
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(int* p) { *p = 1; }
int main() {
  int* p = nullptr;
  // No driver in the container: only that the program links and starts is checked.
  std::printf("%d\n", static_cast<int>(cudaMalloc(&p, sizeof(int))));
  return 0;
}
CU
"${root}/bin/nvcc" -ccbin "$(command -v g++)" -cudart static -arch=sm_75 \
  -o "${check}/k" "${check}/k.cu"
"${check}/k" >/dev/null || true
echo "    nvcc ${release} with $(g++ --version | head -1): a kernel compiles and links (static runtime)"
rm -rf "${check}"

echo "==> ${plugin}: the plugin's source tree from the core sdist in ${package}"
"${python}" "${project}/ci/plugin_pyproject.py" --plugin "${plugin}" --project-dir "${package}" \
  --cuda-root /usr/local/cuda
