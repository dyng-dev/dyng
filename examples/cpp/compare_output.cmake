# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# CTest driver: run a program and compare its standard output with a file (the README quickstart
# test, example.readme_quickstart; the triad_count examples, example.triad_count_update.*).
#
#   cmake -DEXE=<program> -DEXPECTED=<file> [-DARGS=<arg|arg|...>] -P compare_output.cmake
#
# ARGS separates the program's arguments with `|` (a `;` would split the -D argument itself).
#
# A program that exits with 77 (no CUDA device visible) is a skipped test (SKIP_RETURN_CODE 77).

string(REPLACE "|" ";" _args "${ARGS}")
execute_process(COMMAND "${EXE}" ${_args} OUTPUT_VARIABLE _out RESULT_VARIABLE _rc)
if(_rc EQUAL 77)
  message(STATUS "${EXE}: skipped (exit code 77)")
  cmake_language(EXIT 77)
elseif(NOT _rc EQUAL 0)
  message(FATAL_ERROR "${EXE} failed with exit code ${_rc}")
endif()
file(READ "${EXPECTED}" _expected)
if(NOT _out STREQUAL _expected)
  message(FATAL_ERROR "${EXE} printed\n${_out}instead of\n${_expected}")
endif()
