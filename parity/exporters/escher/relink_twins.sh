#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Relinks the executables of a PATCHED scratch copy of ESCHER-GPU or MOSP_ESCHER into CBST
# recording twins (PLAN 8.3 export patches; ADR 0013's additive rule; ADR 0036). Called by
# parity/export_patches/{ESCHER-GPU,MOSP_ESCHER}/build.sh:
#
#   relink_twins.sh <copy> <adapter> <wrap list> <hooks adapter> <hook wrap list> <hook object>
#                   <include dirs> -- <target>...
#
# <adapter> and <hooks adapter> are sources of parity/exporters/escher; the hooks adapter (the
# batch dumps) and its wrap list are linked only into twins whose link command names
# <hook object> (the object that defines the hooked functions).
# For every target (a make target of the copy, e.g. build/main or bin/test_cbst_ops):
#   1. the original's own link command is taken from `make -n -B <target>` (with the make
#      variables of the reference's build command, $DYNG_REFERENCE_BUILD) and kept in
#      parity_export/link/<name>.cmd;
#   2. that command is run again with `-o parity_export/bin/<name>`, the recorder objects first
#      (so that an archive later on the line still supplies what they call, e.g. checkTreeRows)
#      and -Xlinker --wrap=<symbol> for every symbol of the wrap list: the UNCHANGED objects and
#      archives of the original, relinked;
#   3. the original executable is kept as parity_export/orig/<name> and the twin is installed in
#      its place (a fresh mtime, so make does not relink it: `make -q <target>` must succeed),
#      so the original's test scripts run the twins.
# The recorder sources are copied into parity_export/src (the copy records what was built) and
# compiled with the host compiler $CXX. No tracked file of the copy is changed.
set -euo pipefail

copy="${1:?usage: relink_twins.sh <copy> <adapter> <wraps> <hooks> <hook wraps> <hook object> <includes> -- <target>...}"
adapter="${2:?}"
wrap_list="${3:?}"
hooks="${4:?}"
hook_list="${5:?}"
hook_object="${6:?}"
includes="${7:?}"
shift 7
[ "${1:-}" = "--" ] && shift
targets=("$@")
[ "${#targets[@]}" -gt 0 ] || { echo "relink_twins.sh: no targets" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst="${copy}/parity_export"
mkdir -p "${dst}/src" "${dst}/obj" "${dst}/bin" "${dst}/orig" "${dst}/link"
for f in cbst_recorder.hpp cbst_recorder.cpp "${adapter}" "${hooks}"; do
  cmp -s "${here}/${f}" "${dst}/src/${f}" || cp -p "${here}/${f}" "${dst}/src/${f}"
done
cmp -s "${wrap_list}" "${dst}/src/wrap_symbols.txt" || cp -p "${wrap_list}" "${dst}/src/wrap_symbols.txt"
cmp -s "${hook_list}" "${dst}/src/wrap_hooks.txt" || cp -p "${hook_list}" "${dst}/src/wrap_hooks.txt"

cxx="${CXX:-g++}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.1}"
inc=(-I"${dst}/src" -I"${cuda_home}/include")
for d in ${includes}; do inc+=(-I"${copy}/${d}"); done
compile() { # <source> <object>
  if [ ! -f "$2" ] || [ "$1" -nt "$2" ] || [ "${dst}/src/cbst_recorder.hpp" -nt "$2" ]; then
    "${cxx}" -std=c++17 -O2 -g -Wall -Wextra -c "$1" "${inc[@]}" -o "$2"
  fi
}
compile "${dst}/src/cbst_recorder.cpp" "${dst}/obj/cbst_recorder.o"
compile "${dst}/src/${adapter}" "${dst}/obj/${adapter%.cpp}.o"
compile "${dst}/src/${hooks}" "${dst}/obj/${hooks%.cpp}.o"

wrap_flags() { # <list>
  local sym out=""
  while IFS= read -r sym; do
    sym="${sym%%#*}"
    sym="${sym//[[:space:]]/}"
    [ -n "${sym}" ] && out+=" -Xlinker --wrap=${sym}"
  done <"$1"
  printf '%s' "${out}"
}
cbst_wraps="$(wrap_flags "${wrap_list}")"
hook_wraps="$(wrap_flags "${hook_list}")"

# The make variables of the reference's build command (NVCC=..., ARCH=... / CUDA_ARCH=...).
make_vars=()
for tok in ${DYNG_REFERENCE_BUILD:-}; do
  [[ "${tok}" =~ ^[A-Z_][A-Z0-9_]*= ]] && make_vars+=("${tok}")
done

is_twin() { [ -f "$1" ] && grep -qa 'dyng-cbst-recorder-twin format 1' "$1"; }

for target in "${targets[@]}"; do
  name="$(basename "${target}")"
  line="$(cd "${copy}" && make -n -B ${make_vars[@]+"${make_vars[@]}"} "${target}" 2>/dev/null |
    grep -E -- "-o ${target}( |\$)" | tail -n 1)"
  [ -n "${line}" ] || { echo "relink_twins.sh: no link command for ${target}" >&2; exit 1; }
  printf '%s\n' "${line}" >"${dst}/link/${name}.cmd"
  objs="parity_export/obj/cbst_recorder.o parity_export/obj/${adapter%.cpp}.o"
  wraps="${cbst_wraps}"
  if [[ " ${line} " == *"${hook_object} "* ]]; then
    objs+=" parity_export/obj/${hooks%.cpp}.o"
    wraps+="${hook_wraps}"
  fi
  pat=" -o ${target}"
  twin_cmd="${line/"${pat}"/ -o parity_export/bin/${name} ${objs}}${wraps}"
  printf '%s\n' "${twin_cmd}" >"${dst}/link/${name}.twin.cmd"
  newest="$(ls -t "${dst}/obj/cbst_recorder.o" "${dst}/obj/${adapter%.cpp}.o" \
    "${dst}/obj/${hooks%.cpp}.o" "${dst}/link/${name}.cmd" | head -n 1)"
  if [ ! -x "${dst}/bin/${name}" ] || [ "${newest}" -nt "${dst}/bin/${name}" ] ||
    { [ -f "${copy}/${target}" ] && ! is_twin "${copy}/${target}"; }; then
    (cd "${copy}" && bash -c "${twin_cmd}")
  fi
  is_twin "${dst}/bin/${name}" || { echo "relink_twins.sh: ${name}: no recorder" >&2; exit 1; }
  if [ -f "${copy}/${target}" ] && ! is_twin "${copy}/${target}"; then
    cp -p "${copy}/${target}" "${dst}/orig/${name}"
  fi
  [ -f "${dst}/orig/${name}" ] || { echo "relink_twins.sh: ${target} was not built" >&2; exit 1; }
  if ! cmp -s "${dst}/bin/${name}" "${copy}/${target}"; then
    cp "${dst}/bin/${name}" "${copy}/${target}"
  fi
  touch "${copy}/${target}"
  if ! (cd "${copy}" && make -q ${make_vars[@]+"${make_vars[@]}"} "${target}"); then
    echo "relink_twins.sh: make would relink ${target} over the twin" >&2
    exit 1
  fi
  echo "twin: ${target} (original kept as parity_export/orig/${name})"
done
