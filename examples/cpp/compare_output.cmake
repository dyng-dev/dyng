# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# CTest driver: run a program and compare its standard output with a file (the README quickstart
# test, example.readme_quickstart).
#
#   cmake -DEXE=<program> -DEXPECTED=<file> -P compare_output.cmake

execute_process(COMMAND "${EXE}" OUTPUT_VARIABLE _out RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "${EXE} failed with exit code ${_rc}")
endif()
file(READ "${EXPECTED}" _expected)
if(NOT _out STREQUAL _expected)
  message(FATAL_ERROR "${EXE} printed\n${_out}instead of\n${_expected}")
endif()
