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
# on the result, and follows the tutorial "Your first dynamic algorithm"
# (docs/tutorials/your_first_dynamic_algorithm.md): it scaffolds my_bfs with the tutorial's command
# and makes the tutorial's edits in the scaffold's files, each where the prose puts it
# (ci/tutorial_edits.py, which pastes the ranges of the reference solution
# examples/tutorial_algorithms/my_bfs that the tutorial quotes), so a step the prose gets wrong
# fails the build. It then configures a Debug build of
# the probes and my_bfs alone (-DDYNG_ALGORITHMS=<the probes>: the
# subset build of the "add an algorithm" guide, PLAN 9.4 step 3; the build adds sssp, which the
# library's generators and the composition check C10 need) with warnings as errors and the budgets
# on, builds every target (so a subset that does not link fails here) and runs the probes' tests
# (their conformance kits and hand cases) and the registry test, then deletes the copy.
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
echo "==> the tutorial: my_bfs, scaffolded, then the reader's edits made (ci/tutorial_edits.py)"
python3 "${src}/scripts/new_algorithm.py" my_bfs --family fixed_point --backends seq,omp \
  --title "My first dynamic BFS" --computes "BFS levels from a source" --root "${src}"
python3 "${src}/ci/tutorial_edits.py" --root "${src}"
python3 "${src}/scripts/regen.py" --check --root "${src}"
# The generated CODEOWNERS lines (a long name once ran the path into the owner).
if python3 -c "import yaml" 2>/dev/null; then
  python3 "${src}/ci/github_meta_check.py" --root "${src}"
else
  echo "scaffold_check: PyYAML missing, github_meta_check skipped"
fi
if command -v clang-format >/dev/null 2>&1; then
  (cd "${src}" && clang-format --dry-run --Werror cpp/include/dyng/scaffold_probe_*.hpp \
    cpp/src/algorithms/scaffold_probe_*/*.[ch]pp cpp/tests/algorithms/scaffold_probe_*/*.[ch]pp \
    cpp/include/dyng/my_bfs.hpp cpp/src/algorithms/my_bfs/*.[ch]pp \
    cpp/tests/algorithms/my_bfs/*.[ch]pp)
fi

echo "==> configure and build (Debug, warnings as errors, budgets on; the probes and my_bfs, sssp added)"
cmake -S "${src}" -B "${work}/build" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DDYNG_BUILD_TESTS=ON \
  -DDYNG_WARNINGS_AS_ERRORS=ON -DDYNG_DEBUG_BUDGETS=ON -DDYNG_ENABLE_CUDA=OFF \
  "-DDYNG_ALGORITHMS=scaffold_probe_fp;scaffold_probe_ad;my_bfs" -DDYNG_MUTATION_TESTS=OFF \
  -DDYNG_BUILD_EXAMPLES=OFF -DDYNG_BUILD_COMPAT_TOOLS=OFF >"${work}/configure.log" ||
  { cat "${work}/configure.log"; exit 1; }
grep -q "dynG: sssp added to DYNG_ALGORITHMS" "${work}/configure.log" ||
  { echo "scaffold_check: the subset build did not add sssp" >&2; exit 1; }
# Every target, as the guide's `cmake --build --preset dev` (the libraries, every suite of the
# subset and the header self-containment build).
cmake --build "${work}/build"

echo "==> the probes' tests, the registry test"
ctest --test-dir "${work}/build" --output-on-failure -j "$(nproc)" \
  -L 'scaffold_probe_fp|scaffold_probe_ad' -R 'scaffold_probe|ScaffoldProbe'
ctest --test-dir "${work}/build" --output-on-failure -R '^Registry\.'
count="$(ctest --test-dir "${work}/build" -N -L 'scaffold_probe_fp|scaffold_probe_ad' | sed -n 's/^Total Tests: //p')"
if [ "${count:-0}" -lt 20 ]; then
  echo "scaffold_check: only ${count:-0} tests of the probes were found" >&2
  exit 1
fi
echo "==> the tutorial's my_bfs: its conformance kit and hand cases (the tutorial's last step)"
ctest --test-dir "${work}/build" --output-on-failure -j "$(nproc)" -L my_bfs
tutorial_count="$(ctest --test-dir "${work}/build" -N -L my_bfs | sed -n 's/^Total Tests: //p')"
if [ "${tutorial_count:-0}" -lt 20 ]; then
  echo "scaffold_check: only ${tutorial_count:-0} tests of the tutorial's my_bfs were found" >&2
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
bad = [
    line
    for line in added
    if line.startswith("[") and "scaffold_probe_" not in line and "my_bfs" not in line
]
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
python3 "${src}/scripts/new_algorithm.py" my_bfs --remove --root "${src}"
python3 "${src}/scripts/regen.py" --check --root "${src}"
(cd "${repo_root}" && git ls-files -z --cached | while IFS= read -r -d '' f; do
  if [ -f "$f" ] && ! cmp -s "$f" "${src}/$f"; then
    echo "scaffold_check: --remove left ${f} changed" >&2
    exit 1
  fi
done)
echo "scaffold_check: OK (${count} tests of the probes and ${tutorial_count} of the tutorial's my_bfs passed)"
