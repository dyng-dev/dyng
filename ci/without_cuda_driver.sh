#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Run a command as on a machine WITHOUT the NVIDIA driver (the GitHub-hosted runners of
# wheels.yml), on a development machine that has one, without root: in a private user and mount
# namespace, the driver library libcuda.so.1 is hidden behind an empty file, so dlopen() fails as
# it does where no driver is installed. Nothing outside the command's namespace changes.
#
#   ci/without_cuda_driver.sh <command> [args...]
#   ci/without_cuda_driver.sh --check       # exit 0 if this machine can do it (unshare works)
#
# Needs util-linux's unshare and unprivileged user namespaces (Debian 12's default).
# ci/plugin_wheels.sh uses it for the CI smoke test of the plugin wheels (ci/plugin_smoke.py
# --expect fallback --reason "no CUDA driver").
set -euo pipefail

if [ "${1:-}" = "--inner" ]; then
  shift
  # Every name the loader could resolve libcuda.so.1 to: the soname's link targets.
  for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib64/libcuda.so* /usr/lib/libcuda.so*; do
    [ -e "${lib}" ] || continue
    target="$(readlink -f "${lib}")"
    mount --bind /dev/null "${target}"
  done
  exec "$@"
fi

if [ "${1:-}" = "--check" ]; then
  exec unshare --user --map-root-user --mount true
fi
if [ "$#" -eq 0 ]; then
  echo "usage: ci/without_cuda_driver.sh <command> [args...] | --check" >&2
  exit 2
fi
exec unshare --user --map-root-user --mount "$(readlink -f "${BASH_SOURCE[0]}")" --inner "$@"
