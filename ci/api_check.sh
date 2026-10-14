#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The API check (PLAN Section 5.9, "Enforcement") against a base revision, both halves:
#   - Python: griffe compares the public API of the typed layer python/dyng (and the committed
#     stubs of dyng._core) with the base and reports every breaking change: a removed or renamed
#     public name, a removed or reordered parameter, a changed default;
#   - C++: ci/api_snapshot.py --against compares the frozen sections of the committed API
#     baseline (cpp/tests/api/api_snapshot/public_api.txt) with the base's: a removed or changed
#     declaration of the frozen API is breaking. That the baseline matches the headers is checked
#     by ci/docs.sh (docs.yml); this half makes updating the baseline in the same pull request
#     need the label as well (R020).
# Either half's breaking change fails unless the pull request has the `api-change` label and a
# CHANGELOG.md entry.
#
#   ci/api_check.sh                    # against origin/main (or main without a remote)
#   ci/api_check.sh v0.1.0             # against a tag or any revision
#
# Environment:
#   DYNG_API_BASE    the base revision (default origin/main, else main); the first argument wins
#   DYNG_API_CHANGE  1 = the pull request carries the `api-change` label: breaking changes are
#                    accepted (before 1.0 a minor release may break the stable API), but they need
#                    an entry in CHANGELOG.md, which must differ from the base
#   DYNG_API_FORMAT  griffe's output format (default verbose; api-check.yml uses github)
#
# A base without the Python package (the branch that adds it) has nothing to compare: the check
# passes with a note. griffe reads the sources statically; nothing is built or imported.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

base="${1:-${DYNG_API_BASE:-}}"
if [ -z "${base}" ]; then
  # origin/main in CI and in clones with a remote; the local main branch otherwise.
  base=origin/main
  git rev-parse --verify --quiet "${base}^{commit}" >/dev/null || base=main
fi
format="${DYNG_API_FORMAT:-verbose}"

if ! command -v griffe >/dev/null 2>&1; then
  echo "ci/api_check.sh: griffe is missing (environment.yml, or: pip install griffe==2.3.0)" >&2
  exit 1
fi
if ! git rev-parse --verify --quiet "${base}^{commit}" >/dev/null; then
  echo "ci/api_check.sh: unknown base revision '${base}' (fetch it first)" >&2
  exit 1
fi

echo "==> the frozen C++ API against ${base} (ci/api_snapshot.py --against)"
cpp_rc=0
"${PYTHON:-python3}" ci/api_snapshot.py --against "${base}" || cpp_rc=$?

if ! git cat-file -e "${base}:python/dyng/__init__.py" 2>/dev/null; then
  echo "api-check: ${base} has no Python package (python/dyng): nothing to compare yet"
  exit "${cpp_rc}"
fi

echo "==> griffe check dyng against ${base} ($(griffe --version))"
set +e
griffe check dyng --search python --against "${base}" --format "${format}"
rc=$?
set -e
if [ "${rc}" -eq 0 ]; then
  echo "api-check: no breaking change of the Python API against ${base}"
  exit "${cpp_rc}"
fi

if [ "${DYNG_API_CHANGE:-0}" = "1" ]; then
  if git diff --quiet "${base}" -- CHANGELOG.md; then
    echo "api-check: breaking changes with the api-change label, but CHANGELOG.md has no entry" \
      "(PLAN 5.9: a 'Changed' or 'Removed' entry and a migration note)" >&2
    exit 1
  fi
  echo "api-check: breaking changes accepted (api-change label, CHANGELOG.md updated)"
  exit "${cpp_rc}"
fi
echo "api-check: the Python API has breaking changes against ${base}. If they are intended, add" \
  "the label api-change and a CHANGELOG 'Changed' or 'Removed' entry with a migration note" \
  "(PLAN Section 5.9); otherwise keep the old names and signatures." >&2
exit 1
