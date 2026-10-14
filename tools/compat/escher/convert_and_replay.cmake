# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Converts a CBST trace of ESCHER-GPU@abcf9b2 with `dyng-compat-escher to-dgb` and replays the
# files with `dyng-compat-escher replay` on one backend (by id against the expected ids, by
# content against the by-id member sets). Arguments: -DEXE= -DTRACE= -DOUT= -DBACKEND=. Exits
# with 77 (skip) when BACKEND is cuda and no CUDA device is visible.
file(MAKE_DIRECTORY "${OUT}")
get_filename_component(_name "${TRACE}" NAME_WE)
set(_prefix "${OUT}/${_name}")
execute_process(
  COMMAND "${EXE}" to-dgb "${TRACE}" "${_prefix}" --tmp "${OUT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
message("${_out}")
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "to-dgb failed (${_rc})")
endif()
execute_process(
  COMMAND "${EXE}" replay "${_prefix}" --backend "${BACKEND}" --threads 4
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _out
)
message("${_out}")
if(NOT _rc EQUAL 0 AND BACKEND STREQUAL "cuda" AND _out MATCHES "no CUDA device")
  message(STATUS "dyng-compat-escher: no CUDA device visible; skipped")
  cmake_language(EXIT 77)
endif()
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "replay failed (${_rc})")
endif()
