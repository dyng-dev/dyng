#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Builds parity_export/bin/cbst_crossreplay, parity_export/bin/cbst_targeted (the targeted
# scenarios) and parity_export/bin/cbst_trace_tool (the trace verifier) in a PATCHED copy
# (ADR 0036). The cross-check driver
# (cbst_crossreplay.cpp) and the host-only trace reader (cpp/tests/parity/escher/cbst_trace.cpp),
# linked like the twin <template> (the copy's unchanged objects, the recorder, -Xlinker --wrap)
# with the template's main object <main object> replaced by the driver. Called by the export
# patches after relink_twins.sh:
#
#   build_crossreplay.sh <copy> <ESCHER_GPU|MOSP_ESCHER> <template twin> <main object> <includes>
set -euo pipefail
copy="${1:?}"
store="${2:?}"
template="${3:?}"
main_object="${4:?}"
includes="${5:?}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
dst="${copy}/parity_export"
mkdir -p "${dst}/src/parity/escher"
for f in cbst_crossreplay.cpp cbst_trace_tool.cpp cbst_targeted.cpp; do
  cmp -s "${here}/${f}" "${dst}/src/${f}" || cp -p "${here}/${f}" "${dst}/src/${f}"
done
for f in cbst_trace.hpp cbst_trace.cpp; do
  src="${repo}/cpp/tests/parity/escher/${f}"
  cmp -s "${src}" "${dst}/src/parity/escher/${f}" || cp -p "${src}" "${dst}/src/parity/escher/${f}"
done
cxx="${CXX:-g++}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.1}"
inc=(-I"${dst}/src" -I"${cuda_home}/include")
for d in ${includes}; do inc+=(-I"${copy}/${d}"); done
compile() { # <source> <object>
  if [ ! -f "$2" ] || [ "$1" -nt "$2" ]; then
    "${cxx}" -std=c++17 -O2 -g -Wall -Wextra "$3" -c "$1" "${inc[@]}" -o "$2"
  fi
}
compile "${dst}/src/cbst_crossreplay.cpp" "${dst}/obj/cbst_crossreplay.o" "-DDYNG_REPLAY_STORE_${store}"
compile "${dst}/src/parity/escher/cbst_trace.cpp" "${dst}/obj/cbst_trace.o" "-DDYNG_CBST_TRACE_READER"
twin_cmd="$(cat "${dst}/link/${template}.twin.cmd")"
pat=" ${main_object} "
[[ "${twin_cmd} " == *"${pat}"* ]] || { echo "build_crossreplay.sh: ${main_object} not in the ${template} link" >&2; exit 1; }
cmd="${twin_cmd} "
cmd="${cmd/"${pat}"/ parity_export/obj/cbst_crossreplay.o parity_export/obj/cbst_trace.o }"
pat=" -o parity_export/bin/${template} "
cmd="${cmd/"${pat}"/ -o parity_export/bin/cbst_crossreplay }"
newest="$(ls -t "${dst}/obj/cbst_crossreplay.o" "${dst}/obj/cbst_trace.o" "${dst}/bin/${template}" | head -n 1)"
if [ ! -x "${dst}/bin/cbst_crossreplay" ] || [ "${newest}" -nt "${dst}/bin/cbst_crossreplay" ]; then
  (cd "${copy}" && bash -c "${cmd}")
fi
echo "cross-check driver: parity_export/bin/cbst_crossreplay (store ${store})"
# The targeted scenarios (cbst_targeted.cpp), linked the same way.
compile "${dst}/src/cbst_targeted.cpp" "${dst}/obj/cbst_targeted.o" "-DDYNG_REPLAY_STORE_${store}"
cmd="${twin_cmd} "
pat=" ${main_object} "
cmd="${cmd/"${pat}"/ parity_export/obj/cbst_targeted.o }"
pat=" -o parity_export/bin/${template} "
cmd="${cmd/"${pat}"/ -o parity_export/bin/cbst_targeted }"
if [ ! -x "${dst}/bin/cbst_targeted" ] || [ "${dst}/obj/cbst_targeted.o" -nt "${dst}/bin/cbst_targeted" ] ||
  [ "${dst}/bin/${template}" -nt "${dst}/bin/cbst_targeted" ]; then
  (cd "${copy}" && bash -c "${cmd}")
fi
echo "targeted scenarios: parity_export/bin/cbst_targeted (store ${store})"
# The trace verifier (host code only).
compile "${dst}/src/cbst_trace_tool.cpp" "${dst}/obj/cbst_trace_tool.o" "-DDYNG_CBST_TRACE_TOOL"
if [ ! -x "${dst}/bin/cbst_trace_tool" ] || [ "${dst}/obj/cbst_trace_tool.o" -nt "${dst}/bin/cbst_trace_tool" ] ||
  [ "${dst}/obj/cbst_trace.o" -nt "${dst}/bin/cbst_trace_tool" ]; then
  "${cxx}" "${dst}/obj/cbst_trace_tool.o" "${dst}/obj/cbst_trace.o" -o "${dst}/bin/cbst_trace_tool"
fi
echo "trace verifier: parity_export/bin/cbst_trace_tool"
