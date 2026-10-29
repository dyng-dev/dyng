# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Converts an H-SOSP dump of MOSP_ESCHER@4b86159 twice with `dyng-compat-hsosp to-dgb` (the two
# conversions must give the same bytes), then replays every converted session with
# `dyng-compat-hsosp replay --line-graph` on one backend (the expected ids, B's skipped counts and
# the pairs of every batch against testing::line_graph_delta). Arguments:
# -DEXE= -DDUMP= -DOUT= -DBACKEND=.
file(REMOVE_RECURSE "${OUT}")
file(MAKE_DIRECTORY "${OUT}/a" "${OUT}/b")
get_filename_component(_name "${DUMP}" NAME_WE)
foreach(_run a b)
  execute_process(
    COMMAND "${EXE}" to-dgb "${DUMP}" "${OUT}/${_run}/${_name}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  message("${_out}")
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "to-dgb failed (${_rc})")
  endif()
endforeach()
file(GLOB _files RELATIVE "${OUT}/a" "${OUT}/a/*")
list(LENGTH _files _count)
if(_count EQUAL 0)
  message(FATAL_ERROR "to-dgb wrote nothing")
endif()
foreach(_f IN LISTS _files)
  file(SHA256 "${OUT}/a/${_f}" _sa)
  file(SHA256 "${OUT}/b/${_f}" _sb)
  if(NOT _sa STREQUAL _sb)
    message(FATAL_ERROR "${_f}: the two conversions differ")
  endif()
endforeach()
file(GLOB _sessions "${OUT}/a/*.dgb")
foreach(_dgb IN LISTS _sessions)
  string(REGEX REPLACE "\\.dgb$" "" _prefix "${_dgb}")
  execute_process(
    COMMAND "${EXE}" replay "${_prefix}" --backend "${BACKEND}" --threads 4 --line-graph
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _out
  )
  message("${_out}")
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "replay of ${_prefix} failed (${_rc})")
  endif()
endforeach()
