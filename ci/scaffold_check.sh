#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The scaffold check (PLAN Section 4.8, M3 acceptance): scripts/new_algorithm.py produces an
# algorithm that builds and passes the conformance kit on the first build.
#
# In a throwaway copy of the working tree (the tracked and the untracked, not ignored files) it
# generates one algorithm of each family (scaffold_probe_fp: fixed_point on sequential and OpenMP;
# scaffold_probe_ad: aggregate_delta on sequential), checks that scripts/regen.py --check passes
# on the result, configures a Debug build of the probes with sssp (-DDYNG_ALGORITHMS: the subset
# build of the "add an algorithm" guide; sssp is the partner of the composition check C10) with
# warnings as errors and the budgets on, builds everything and runs the probes' tests (their
# conformance kits and hand cases), the registry test and the header self-containment build, then
# deletes the copy.
#
#   ci/scaffold_check.sh                     # uses a temporary directory
#   DYNG_SCAFFOLD_DIR=/path ci/scaffold_check.sh   # a work directory of your choice (kept on failure)
#   DYNG_SCAFFOLD_KEEP=1 ci/scaffold_check.sh      # keep the copy (for debugging)
#
# Heavy (a configure and a build): run it under the machine rules of ci/check.sh (its step
# `scaffold`, which takes the shared lock) or `ctest -L scaffold` in a configured build.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v cmake >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${repo_root}/scripts/dev_env.sh"
fi

work="${DYNG_SCAFFOLD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/dyng-scaffold.XXXXXX")}"
cleanup() {
  local status=$?
  if [ "${DYNG_SCAFFOLD_KEEP:-0}" = "1" ] || { [ "${status}" -ne 0 ] && [ -n "${DYNG_SCAFFOLD_DIR:-}" ]; }; then
    echo "scaffold_check: kept ${work}"
  else
    rm -rf "${work}"
  fi
}
trap cleanup EXIT

echo "==> copy of the working tree in ${work}/src"
mkdir -p "${work}/src"
(cd "${repo_root}" && git ls-files -z --cached --others --exclude-standard |
  while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done |
  xargs -0 cp --parents -t "${work}/src")

src="${work}/src"
echo "==> scripts/new_algorithm.py: one algorithm of each family"
python3 "${src}/scripts/new_algorithm.py" scaffold_probe_fp --family fixed_point --backends seq,omp \
  --title "Scaffold probe (fixed point)" --computes "a scaffold check" --root "${src}"
python3 "${src}/scripts/new_algorithm.py" scaffold_probe_ad --family aggregate_delta --backends seq \
  --title "Scaffold probe (aggregate delta)" --computes "a scaffold check" --root "${src}"
python3 "${src}/scripts/regen.py" --check --root "${src}"
# The generated CODEOWNERS lines (a long name once ran the path into the owner).
if python3 -c "import yaml" 2>/dev/null; then
  python3 "${src}/ci/github_meta_check.py" --root "${src}"
else
  echo "scaffold_check: PyYAML missing, github_meta_check skipped"
fi
if command -v clang-format >/dev/null 2>&1; then
  (cd "${src}" && clang-format --dry-run --Werror cpp/include/dyng/scaffold_probe_*.hpp \
    cpp/src/algorithms/scaffold_probe_*/*.[ch]pp cpp/tests/algorithms/scaffold_probe_*/*.[ch]pp)
fi

echo "==> configure and build (Debug, warnings as errors, budgets on; sssp and the probes)"
cmake -S "${src}" -B "${work}/build" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DDYNG_BUILD_TESTS=ON \
  -DDYNG_WARNINGS_AS_ERRORS=ON -DDYNG_DEBUG_BUDGETS=ON -DDYNG_ENABLE_CUDA=OFF \
  "-DDYNG_ALGORITHMS=sssp;scaffold_probe_fp;scaffold_probe_ad" -DDYNG_MUTATION_TESTS=OFF \
  -DDYNG_BUILD_EXAMPLES=OFF -DDYNG_BUILD_COMPAT_TOOLS=OFF >"${work}/configure.log" ||
  { cat "${work}/configure.log"; exit 1; }
cmake --build "${work}/build" --target dyng_scaffold_probe_fp_conformance_tests \
  dyng_scaffold_probe_ad_conformance_tests dyng_scaffold_probe_fp_tests dyng_scaffold_probe_ad_tests \
  dyng_conformance_registry_tests dyng_header_self_contained

echo "==> the probes' tests, the registry test"
ctest --test-dir "${work}/build" --output-on-failure -j "$(nproc)" \
  -L 'scaffold_probe_fp|scaffold_probe_ad' -R 'scaffold_probe|ScaffoldProbe'
ctest --test-dir "${work}/build" --output-on-failure -R '^Registry\.'
count="$(ctest --test-dir "${work}/build" -N -L 'scaffold_probe_fp|scaffold_probe_ad' | sed -n 's/^Total Tests: //p')"
if [ "${count:-0}" -lt 20 ]; then
  echo "scaffold_check: only ${count:-0} tests of the probes were found" >&2
  exit 1
fi

# The docs of the scaffolded algorithms: Doxygen and its coverage check, the API baseline (the
# documented step adds the probes' headers, and nothing else changes), then the Sphinx site with
# warnings as errors (the probes' API pages and algorithm pages).
if command -v doxygen >/dev/null 2>&1; then
  echo "==> docs: Doxygen, the API baseline, Sphinx"
  baseline="cpp/tests/api/api_snapshot/public_api.txt"
  cp "${src}/${baseline}" "${work}/public_api.before"
  (cd "${src}" && DYNG_BUILD_DIR="${work}/build" DYNG_DOXYGEN_OUTPUT="${work}/doxygen" \
    ci/docs.sh --update-api)
  python3 - "${work}/public_api.before" "${src}/${baseline}" <<'PY'
import sys

before = open(sys.argv[1]).read().splitlines()
after = open(sys.argv[2]).read().splitlines()
added = [line for line in after if line not in before]
removed = [line for line in before if line not in after]
bad = [line for line in added if line.startswith("[") and "scaffold_probe_" not in line]
if removed or bad:
    sys.exit(f"scaffold_check: the scaffold changed the API baseline beyond its headers: {removed + bad}")
if not any(line.startswith("[dyng/scaffold_probe_fp.hpp]") for line in added):
    sys.exit("scaffold_check: the probe's header is missing from the API baseline")
PY
  if python3 -c "import sphinx, myst_parser, breathe, pydata_sphinx_theme" 2>/dev/null; then
    (cd "${src}" && DYNG_BUILD_DIR="${work}/build" DYNG_DOXYGEN_OUTPUT="${work}/doxygen" \
      DYNG_DOCS_OUTPUT="${work}/docs" ci/docs.sh --no-linkcheck)
  else
    echo "scaffold_check: the Sphinx packages are missing, the Sphinx build of the probes skipped"
  fi
  cp "${work}/public_api.before" "${src}/${baseline}"
else
  echo "scaffold_check: doxygen not found, the docs steps skipped"
fi
echo "==> scripts/new_algorithm.py --remove"
python3 "${src}/scripts/new_algorithm.py" scaffold_probe_fp --remove --root "${src}"
python3 "${src}/scripts/new_algorithm.py" scaffold_probe_ad --remove --root "${src}"
python3 "${src}/scripts/regen.py" --check --root "${src}"
(cd "${repo_root}" && git ls-files -z --cached | while IFS= read -r -d '' f; do
  if [ -f "$f" ] && ! cmp -s "$f" "${src}/$f"; then
    echo "scaffold_check: --remove left ${f} changed" >&2
    exit 1
  fi
done)
echo "scaffold_check: OK (${count} tests of the probes passed)"
