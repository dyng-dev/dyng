#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Build scratch copies of the pinned original repositories (PLAN Sections 6.3 step 1 and 8.3).
#
#   parity/build_reference.sh [options] [name ...]      (default: every reference)
#
# For every reference of parity/references.toml and every variant it
#   1. exports the pinned commit with `git archive` (the original repository is only READ: it is
#      never built in place and its git state never changes; this is checked before and after)
#      into $DYNG_SCRATCH/ref/<name>@<commit7>/<variant>;
#   2. checks that the copy still equals the archive (content, type and executable bit of every
#      tracked file), so the copy is exactly the pinned commit plus build products and, for
#      "patched", additive files;
#   3. builds it with the original build system (`build` in references.toml), log in
#      <name>@<commit7>/<variant>.build.log;
#   4. for "patched": runs parity/export_patches/<export_patch>/build.sh <copy> (additive only);
#   5. writes the archive-build check <name>@<commit7>/archive-check.txt: gitlinks that the archive
#      leaves out, files of the original's working tree that are not in the archive (local-only
#      inputs), absolute paths in tracked files, and the external inputs of references.toml.
#
# Variants: "unpatched" (performance baselines) and "patched" (goldens). The script is idempotent:
# a second run re-verifies the copies and lets make find that nothing is out of date.
#
# Options:
#   --variant unpatched|patched|both   (default both)
#   --fresh        delete the copies first and export again
#   --test         also run the original's own test suite (`smoke_test`) in the unpatched copy
#   --jobs N       parallel build jobs (default 16)
#   --scratch DIR  work area (default $DYNG_SCRATCH, else $HOME/Projects/dyng-work)
#   --print-dir    only print the copy directory of each name/variant (no export, no build)
#
# Environment: DYNG_SCRATCH; CUDA_VISIBLE_DEVICES for --test of CUDA references (default 1, the
# development GPU of the author's machine).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
references="${repo_root}/parity/references.toml"
variant=both
fresh=0
run_tests=0
jobs=16
print_dir=0
scratch="${DYNG_SCRATCH:-${HOME}/Projects/dyng-work}"
names=()

usage() {
  sed -n '5,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) variant="${2:?}"; shift 2 ;;
    --fresh) fresh=1; shift ;;
    --test) run_tests=1; shift ;;
    --jobs) jobs="${2:?}"; shift 2 ;;
    --scratch) scratch="${2:?}"; shift 2 ;;
    --print-dir) print_dir=1; shift ;;
    -h | --help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) names+=("$1"); shift ;;
  esac
done
case "${variant}" in
  unpatched) variants=(unpatched) ;;
  patched) variants=(patched) ;;
  both) variants=(unpatched patched) ;;
  *) echo "--variant: expected unpatched, patched or both" >&2; exit 2 ;;
esac
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs: expected a positive integer" >&2; exit 2; }

# One reference field (Python's tomllib; lists are printed one item per line).
field() { # <name> <key>
  python3 - "${references}" "$1" "$2" <<'PY'
import os, sys, tomllib
path, name, key = sys.argv[1:4]
with open(path, "rb") as f:
    refs = {r["name"]: r for r in tomllib.load(f)["reference"]}
if name not in refs:
    sys.exit(f"unknown reference: {name} (known: {', '.join(refs)})")
value = refs[name].get(key, "")
if isinstance(value, list):
    print("\n".join(os.path.expandvars(str(v)) for v in value))
else:
    print(os.path.expandvars(str(value)))
PY
}

if [ "${#names[@]}" -eq 0 ]; then
  mapfile -t names < <(python3 -c 'import sys, tomllib
print("\n".join(r["name"] for r in tomllib.load(open(sys.argv[1], "rb"))["reference"]))' \
    "${references}")
fi

# Compare a copy with the archive of the pinned commit: every tracked file must exist with the
# same content, type and executable bit (owner and umask-dependent modes are ignored). Files that
# are not in the archive (build products, additive export files) are allowed.
verify_copy() { # <origin> <commit> <dir>
  git -C "$1" archive --format=tar "$2" | python3 -c '
import os, sys, tarfile
root = sys.argv[1]
bad = []
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
    for m in tar:
        path = os.path.join(root, m.name)
        if m.isdir():
            ok = os.path.isdir(path)
        elif m.issym():
            ok = os.path.islink(path) and os.readlink(path) == m.linkname
        elif m.isfile():
            data = tar.extractfile(m).read()
            ok = (os.path.isfile(path) and not os.path.islink(path)
                  and open(path, "rb").read() == data
                  and bool(os.stat(path).st_mode & 0o100) == bool(m.mode & 0o100))
        else:
            ok = True
        if not ok:
            bad.append(m.name)
for name in bad[:20]:
    print("  differs: " + name)
sys.exit(1 if bad else 0)
' "$3"
}

origin_state() { # <origin>: HEAD + status, to prove the original was not touched
  git -C "$1" rev-parse HEAD
  git -C "$1" status --porcelain=v1 --untracked-files=all
}

build_copy() { # <name> <variant>
  local name="$1" var="$2"
  local origin commit short base dir log patch
  origin="$(field "${name}" origin)"
  commit="$(field "${name}" commit)"
  short="${commit:0:7}"
  base="${scratch}/ref/${name}@${short}"
  dir="${base}/${var}"
  if [ "${print_dir}" -eq 1 ]; then
    echo "${dir}"
    return
  fi
  log="${base}/${var}.build.log"
  echo "==> ${name}@${short} (${var}) -> ${dir}"

  [ -d "${origin}/.git" ] || { echo "original repository not found: ${origin}" >&2; return 1; }
  git -C "${origin}" cat-file -e "${commit}^{commit}" ||
    { echo "${origin} has no commit ${commit}" >&2; return 1; }
  local before after
  before="$(origin_state "${origin}")"

  # 1. Export.
  local archive_sha
  archive_sha="$(git -C "${origin}" archive --format=tar "${commit}" | sha256sum | cut -d' ' -f1)"
  if [ "${fresh}" -eq 1 ]; then
    rm -rf "${dir}"
  fi
  if [ -f "${dir}/.dyng-reference" ] &&
    ! grep -qx "archive_sha256=${archive_sha}" "${dir}/.dyng-reference"; then
    echo "${dir} was exported from another archive; use --fresh" >&2
    return 1
  fi
  if [ ! -f "${dir}/.dyng-reference" ]; then
    rm -rf "${dir}"
    mkdir -p "${dir}"
    git -C "${origin}" archive --format=tar "${commit}" | tar -x -C "${dir}"
    {
      echo "name=${name}"
      echo "commit=${commit}"
      echo "variant=${var}"
      echo "archive_sha256=${archive_sha}"
      echo "exported=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${dir}/.dyng-reference"
  fi

  # 2. The copy must still be the archive.
  if ! verify_copy "${origin}" "${commit}" "${dir}" >&2; then
    echo "${dir} differs from ${name}@${short}" >&2
    return 1
  fi

  # 3. Build with the original build system.
  local build env_args=()
  build="$(field "${name}" build)"
  build="${build//\{jobs\}/${jobs}}"
  mapfile -t env_args < <(field "${name}" build_env | sed "/^$/d")
  mkdir -p "${base}"
  echo "    build: ${build}"
  if ! (cd "${dir}" && env ${env_args[@]+"${env_args[@]}"} bash -c "${build}") >"${log}" 2>&1; then
    tail -n 30 "${log}" >&2
    echo "build failed: ${log}" >&2
    return 1
  fi

  # 4. Additive export patch (patched copy only).
  patch="$(field "${name}" export_patch)"
  if [ "${var}" = patched ] && [ -n "${patch}" ]; then
    if ! "${repo_root}/parity/export_patches/${patch}/build.sh" "${dir}" >>"${log}" 2>&1; then
      tail -n 30 "${log}" >&2
      echo "export patch failed: ${log}" >&2
      return 1
    fi
    # Additive: the tracked files are still those of the archive.
    verify_copy "${origin}" "${commit}" "${dir}" >&2 ||
      { echo "the export patch changed a tracked file of ${dir}" >&2; return 1; }
  fi

  # 5. Archive-build check report.
  local report="${base}/archive-check.txt"
  {
    echo "# archive-build check of ${name}@${commit} ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "archive sha256: ${archive_sha}"
    echo "copy: ${base}/{unpatched,patched}; builds with: ${build}"
    echo
    echo "## gitlinks / submodules left out by git archive"
    git -C "${origin}" ls-tree -r "${commit}" | awk '$2 == "commit" { print "  " $4 " -> " $3 }'
    git -C "${origin}" cat-file -e "${commit}:.gitmodules" 2>/dev/null &&
      echo "  .gitmodules present" || true
    echo "  (end)"
    echo
    echo "## present in the original's working tree but not in the archive (local only)"
    git -C "${origin}" ls-files --others --ignored --exclude-standard --directory |
      sed 's/^/  /'
    echo "  (end; build products such as bin/ build/ test-output/ are expected here)"
    echo
    echo "## absolute paths in tracked files"
    (cd "${dir}" && git -C "${origin}" ls-tree -r --name-only "${commit}" |
      xargs -d '\n' grep -nIE '(/home/|/Users/|/usr/local/cuda[^ "]*)' 2>/dev/null |
      sed 's/^/  /' | head -n 40) || true
    echo "  (end)"
    echo
    echo "## external inputs (references.toml)"
    field "${name}" external_inputs | sed 's/^/  - /'
  } >"${report}"

  after="$(origin_state "${origin}")"
  if [ "${before}" != "${after}" ]; then
    echo "the original repository ${origin} changed during the build" >&2
    return 1
  fi
  echo "    ok (log ${log}; archive check ${report})"
}

run_smoke_test() { # <name>
  local name="$1" commit short dir log cmd env_args=()
  commit="$(field "${name}" commit)"
  short="${commit:0:7}"
  dir="${scratch}/ref/${name}@${short}/unpatched"
  log="${scratch}/ref/${name}@${short}/unpatched.test.log"
  cmd="$(field "${name}" smoke_test)"
  mapfile -t env_args < <(field "${name}" run_env | sed "/^$/d")
  echo "==> ${name}@${short}: ${cmd} (log ${log})"
  if (cd "${dir}" && env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    ${env_args[@]+"${env_args[@]}"} bash -c "${cmd}") >"${log}" 2>&1; then
    grep -E '^==|passed|Summary' "${log}" | sed 's/^/    /'
  else
    tail -n 30 "${log}" >&2
    echo "smoke test failed: ${log}" >&2
    return 1
  fi
}

for name in "${names[@]}"; do
  for var in "${variants[@]}"; do
    build_copy "${name}" "${var}"
  done
  if [ "${run_tests}" -eq 1 ] && [ "${print_dir}" -eq 0 ]; then
    run_smoke_test "${name}"
  fi
done
