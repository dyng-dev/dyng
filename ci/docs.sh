#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Build and check the documentation (PLAN Sections 9.1 and 9.2). The documented command:
#
#   ci/docs.sh                  # all three steps
#   ci/docs.sh --doxygen-only   # step 1 only (the CMake target docs-doxygen, and the pre-build
#                               # step of .readthedocs.yaml, which builds Sphinx itself)
#   ci/docs.sh --no-linkcheck   # steps 1 and 2
#
# 1. Doxygen on the public headers with warnings as errors (every public entity documented, every
#    parameter and return value described), then ci/doxygen_coverage.py on its XML (a @brief
#    everywhere, every namespace-scope entity in a group, @backends / @determinism / @paper /
#    @guarantee on compute() and update()), then ci/api_snapshot.py: the public declarations must
#    match the committed API baseline cpp/tests/api/api_snapshot/public_api.txt (the 0.1 freeze,
#    ADR 0023; after a reviewed API change: ci/api_snapshot.py --update).
# 2. The Sphinx site (MyST pages + Breathe over the Doxygen XML) with warnings as errors and
#    nitpicky references: every page in a toctree, every cross-reference and C++ name resolved.
#    This is also the check of the site's own links: a Markdown link or {doc} to a missing page,
#    heading or local file is a MyST warning, so an error. Output: $DYNG_DOCS_OUTPUT/html.
# 3. ci/docs_links.py, the links the Sphinx build cannot see: every GitHub URL of this repository
#    (github.com/dyng-dev/dyng/blob|tree/main/<path>, the way pages link files outside docs/) in
#    the tracked files must name an existing path, and every relative link of the built HTML an
#    existing file and #anchor. With DYNG_LINKCHECK_EXTERNAL=1 (the weekly job of docs.yml) the
#    Sphinx linkcheck builder also checks the external links; the gate never needs the network.
#
# Needs a configured build tree for the generated version.hpp / config.hpp (default:
# build/cpu-only, from `cmake --preset cpu-only`; override with DYNG_BUILD_DIR) and, for steps 2
# and 3, the Sphinx packages of environment.yml. Other overrides: DYNG_DOXYGEN_OUTPUT (default
# build/doxygen), DYNG_DOCS_OUTPUT (default build/docs).
set -euo pipefail

sphinx=1
linkcheck=1
for arg in "$@"; do
  case "${arg}" in
    --doxygen-only) sphinx=0 ;;
    --no-linkcheck) linkcheck=0 ;;
    -h | --help)
      sed -n '5,31p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "ci/docs.sh: unknown argument '${arg}' (try --help)" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

build_dir="${DYNG_BUILD_DIR:-build/cpu-only}"
if [ ! -f "${build_dir}/cpp/include/dyng/version.hpp" ]; then
  echo "ci/docs.sh: ${build_dir} is not configured; run: cmake --preset cpu-only" >&2
  exit 1
fi

export DYNG_GENERATED_INCLUDE="${build_dir}/cpp/include"
export DYNG_DOXYGEN_OUTPUT="${DYNG_DOXYGEN_OUTPUT:-build/doxygen}"
mkdir -p "${DYNG_DOXYGEN_OUTPUT}"
# Stale XML of a removed header would still be rendered; start from an empty directory.
rm -rf "${DYNG_DOXYGEN_OUTPUT}/xml"
doxygen docs/Doxyfile
python3 ci/doxygen_coverage.py "${DYNG_DOXYGEN_OUTPUT}/xml"
python3 ci/api_snapshot.py --xml "${DYNG_DOXYGEN_OUTPUT}/xml"
echo "Doxygen XML written to ${DYNG_DOXYGEN_OUTPUT}/xml"

if [ "${sphinx}" = "0" ]; then
  exit 0
fi

if ! python3 -c "import sphinx, myst_parser, breathe, pydata_sphinx_theme" 2>/dev/null; then
  echo "ci/docs.sh: the Sphinx packages are missing; run: conda env update -f environment.yml" >&2
  exit 1
fi

docs_output="${DYNG_DOCS_OUTPUT:-build/docs}"
# -W: warnings are errors; --keep-going: report all of them; -n: nitpicky references;
# -E: always read every page (a clean result, not one that depends on the previous build).
sphinx_opts=(-W --keep-going -n -E -q -d "${docs_output}/doctrees")

python3 -m sphinx -b html "${sphinx_opts[@]}" docs "${docs_output}/html"
echo "Documentation site written to ${docs_output}/html/index.html"

if [ "${linkcheck}" = "1" ]; then
  python3 ci/docs_links.py --html "${docs_output}/html"
  if [ "${DYNG_LINKCHECK_EXTERNAL:-0}" = "1" ]; then
    python3 -m sphinx -b linkcheck "${sphinx_opts[@]}" docs "${docs_output}/linkcheck"
    echo "External link check: OK (${docs_output}/linkcheck/output.txt)"
  fi
fi
