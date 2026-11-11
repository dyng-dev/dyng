#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Additive export patch of LabelPropagation-CUDA@a276a3a (PLAN Sections 6.3 step 1 and 8.3; ADRs
# 0013, 0036 and 0065). Called by parity/build_reference.sh with the PATCHED scratch copy as its
# only argument, after the copy was built with the original Makefile (make all bin/propagation_test
# bin/input_contract_test, the Makefile defaults). It changes no tracked file and patches no
# source: for bin/dynLP and bin/propagation_test it
#   1. takes the original's own link command from `make -n -B <target>` (kept in
#      parity_export/link/<name>.cmd);
#   2. runs it again with -o parity_export/bin/<name>, the recorder objects first
#      (parity/exporters/dynlp: dynlp_trace.cpp and recorder_dynlp.cpp, or recorder_dynlp_hub.cpp
#      for the test, compiled with the host compiler $CXX) and -Xlinker --wrap=<symbol> for every
#      symbol of wrap_dynlp.txt / wrap_hub.txt: O's UNCHANGED obj/*.o relinked;
#   3. keeps the original executable as parity_export/orig/<name> and installs the twin in its
#      place (a fresh mtime, so make does not relink it).
# bin/input_contract_test stays as built. Recording is on only when DYNG_DYNLP_TRACE_DIR is set;
# otherwise a twin behaves as the original (parity/exporters/dynlp/check_twins.py).
set -euo pipefail
copy="${1:?usage: build.sh <patched copy>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="${here}/../../exporters/dynlp"
dst="${copy}/parity_export"
mkdir -p "${dst}/src" "${dst}/obj" "${dst}/bin" "${dst}/orig" "${dst}/link"
for f in dynlp_trace.hpp dynlp_trace.cpp recorder_dynlp.cpp recorder_dynlp_hub.cpp; do
  cmp -s "${src}/${f}" "${dst}/src/${f}" || cp -p "${src}/${f}" "${dst}/src/${f}"
done
for f in wrap_dynlp.txt wrap_hub.txt; do
  cmp -s "${here}/${f}" "${dst}/src/${f}" || cp -p "${here}/${f}" "${dst}/src/${f}"
done

cxx="${CXX:-g++}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
inc=(-I"${dst}/src" -I"${copy}/headers" -I"${cuda_home}/include")
compile() { # <source> <object>
  if [ ! -f "$2" ] || [ "$1" -nt "$2" ] || [ "${dst}/src/dynlp_trace.hpp" -nt "$2" ]; then
    "${cxx}" -std=c++17 -O2 -g -Wall -Wextra -Werror -c "$1" "${inc[@]}" -o "$2"
  fi
}
for f in dynlp_trace recorder_dynlp recorder_dynlp_hub; do
  compile "${dst}/src/${f}.cpp" "${dst}/obj/${f}.o"
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

is_twin() { [ -f "$1" ] && grep -qa 'dyng-dynlp-recorder-twin format 1' "$1"; }

relink() { # <target> <recorder object> <wrap list>
  local target="$1" adapter="$2" list="$3" name line pat twin_cmd
  name="$(basename "${target}")"
  line="$(cd "${copy}" && make -n -B "${target}" 2>/dev/null | grep -E -- "-o ${target}( |\$)" |
    tail -n 1)"
  [ -n "${line}" ] || { echo "build.sh: no link command for ${target}" >&2; exit 1; }
  printf '%s\n' "${line}" >"${dst}/link/${name}.cmd"
  pat=" -o ${target}"
  twin_cmd="${line/"${pat}"/ -o parity_export/bin/${name} parity_export/obj/dynlp_trace.o parity_export/obj/${adapter}.o}$(wrap_flags "${list}")"
  printf '%s\n' "${twin_cmd}" >"${dst}/link/${name}.twin.cmd"
  (cd "${copy}" && bash -c "${twin_cmd}")
  is_twin "${dst}/bin/${name}" || { echo "build.sh: ${name}: no recorder" >&2; exit 1; }
  if [ -f "${copy}/${target}" ] && ! is_twin "${copy}/${target}"; then
    cp -p "${copy}/${target}" "${dst}/orig/${name}"
  fi
  [ -f "${dst}/orig/${name}" ] || { echo "build.sh: ${target} was not built" >&2; exit 1; }
  cp "${dst}/bin/${name}" "${copy}/${target}"
  touch "${copy}/${target}"
  if (cd "${copy}" && make -n "${target}" 2>/dev/null | grep -qE -- "-o ${target}( |\$)"); then
    echo "build.sh: make would relink ${target} over the twin" >&2
    exit 1
  fi
  echo "twin: ${target} (original kept as parity_export/orig/${name})"
}
relink bin/dynLP recorder_dynlp "${here}/wrap_dynlp.txt"
relink bin/propagation_test recorder_dynlp_hub "${here}/wrap_hub.txt"
