#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The weekly harmonic oracle of label_propagation (AD7; .github/workflows/oracle.yml, M10a step
# 5): delta 1e-4 at 50k on the sequential backend, both schedules, with the build's
# dyng-compat-dynlp and dyng_lp_replay (parity/compare.py label_propagation oracle-port).
#
#   ci/oracle_weekly.sh [<build dir>]     # default build/parity
#
# Exit 0 when every batch of both runs is within the oracle (mean |F - F*| <= 2000 delta, the PCG
# converged, no class flip where |F* - 0.5| > 3m). Writes the record to
# $DYNG_SCRATCH/runs/oracle-weekly.json. Runs on the host only (no GPU).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${1:-build/parity}"
compat="${build}/tools/compat/dyng-compat-dynlp"
replay="${build}/cpp/tests/dyng_lp_replay"
for exe in "${compat}" "${replay}"; do
  if [ ! -x "${exe}" ]; then
    echo "ci/oracle_weekly.sh: ${exe} missing (cmake --build --preset parity --target" \
      "dyng-compat-dynlp dyng_lp_replay)" >&2
    exit 2
  fi
done
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
mkdir -p "${scratch}/runs"
cd "${repo_root}"
DYNG_SCRATCH="${scratch}" python3 parity/compare.py label_propagation oracle-port \
  --exe "${compat}" --replay "${replay}" --backend sequential --threads "$(nproc)" \
  --cases 50k_d1e-4_s42,50k_d1e-4_s42_inplace --json "${scratch}/runs/oracle-weekly.json"
