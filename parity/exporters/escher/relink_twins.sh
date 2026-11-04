#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Relinks the executables of a PATCHED scratch copy of ESCHER-GPU or MOSP_ESCHER into CBST
# recording twins (PLAN 8.3 export patches; ADR 0013's additive rule; ADR 0036). Called by
# parity/export_patches/{ESCHER-GPU,MOSP_ESCHER}/build.sh:
#
#   relink_twins.sh <copy> <adapter> <wrap list> <hooks adapter> <hook wrap list> <hook object>
#                   <include dirs> [--extra <sources> <wrap list> <object>]... -- <target>...
#
# <adapter> and <hooks adapter> are sources of parity/exporters/escher; the hooks adapter (the
# batch dumps) and its wrap list are linked only into twins whose link command names
# <hook object> (the object that defines the hooked functions). Each --extra group (M11a, ADR
# 0081: the H-SOSP dump of MOSP_ESCHER) is a comma-separated list of files under
# parity/exporters (its .cpp files are compiled, the others copied next to them), a wrap list
# and an object: the group's objects and wraps are linked only into twins whose link command
# names that object.
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
extra_sources=()
extra_lists=()
extra_objects=()
while [ "${1:-}" = "--extra" ]; do
  extra_sources+=("${2:?--extra <sources> <wrap list> <object>}")
  extra_lists+=("${3:?--extra <sources> <wrap list> <object>}")
  extra_objects+=("${4:?--extra <sources> <wrap list> <object>}")
  shift 4
done
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
compile() { # <source> <object>  (again when any copied source or header is newer)
  if [ ! -f "$2" ] || [ -n "$(find "${dst}/src" -newer "$2" -print -quit)" ]; then
    "${cxx}" -std=c++17 -O2 -g -Wall -Wextra -c "$1" "${inc[@]}" -o "$2"
  fi
}
# The --extra groups' files (copied first: their headers may be included by their sources).
extra_objs=()   # per group: its objects (space-separated, relative to the copy)
for g in "${!extra_sources[@]}"; do
  IFS=',' read -r -a files <<<"${extra_sources[$g]}"
  for f in "${files[@]}"; do
    cmp -s "${here}/../${f}" "${dst}/src/$(basename "${f}")" ||
      cp -p "${here}/../${f}" "${dst}/src/$(basename "${f}")"
  done
  cmp -s "${extra_lists[$g]}" "${dst}/src/$(basename "${extra_lists[$g]}")" ||
    cp -p "${extra_lists[$g]}" "${dst}/src/$(basename "${extra_lists[$g]}")"
done
compile "${dst}/src/cbst_recorder.cpp" "${dst}/obj/cbst_recorder.o"
compile "${dst}/src/${adapter}" "${dst}/obj/${adapter%.cpp}.o"
compile "${dst}/src/${hooks}" "${dst}/obj/${hooks%.cpp}.o"
for g in "${!extra_sources[@]}"; do
  IFS=',' read -r -a files <<<"${extra_sources[$g]}"
  objs_g=""
  for f in "${files[@]}"; do
    [[ "${f}" == *.cpp ]] || continue
    b="$(basename "${f}" .cpp)"
    compile "${dst}/src/${b}.cpp" "${dst}/obj/${b}.o"
    objs_g+=" parity_export/obj/${b}.o"
  done
  extra_objs+=("${objs_g}")
done

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
extra_wraps=()
for g in "${!extra_sources[@]}"; do extra_wraps+=("$(wrap_flags "${extra_lists[$g]}")"); done

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
  extra_newest=()
  for g in "${!extra_sources[@]}"; do
    if [[ " ${line} " == *"${extra_objects[$g]} "* ]]; then
      objs+="${extra_objs[$g]}"
      wraps+="${extra_wraps[$g]}"
      for o in ${extra_objs[$g]}; do extra_newest+=("${copy}/${o}"); done
    fi
  done
  pat=" -o ${target}"
  twin_cmd="${line/"${pat}"/ -o parity_export/bin/${name} ${objs}}${wraps}"
  if [ -f "${dst}/link/${name}.twin.cmd" ] && ! cmp -s <(printf '%s\n' "${twin_cmd}") \
    "${dst}/link/${name}.twin.cmd"; then
    rm -f "${dst}/bin/${name}"  # the twin's link command changed: relink it
  fi
  printf '%s\n' "${twin_cmd}" >"${dst}/link/${name}.twin.cmd"
  newest="$(ls -t "${dst}/obj/cbst_recorder.o" "${dst}/obj/${adapter%.cpp}.o" \
    "${dst}/obj/${hooks%.cpp}.o" ${extra_newest[@]+"${extra_newest[@]}"} \
    "${dst}/link/${name}.cmd" | head -n 1)"
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
