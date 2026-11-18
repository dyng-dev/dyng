# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# CTest driver: run a program that writes a file, and compare the file's SHA-256 with the line of
# a digest file for a key (the label_propagation examples, example.label_propagation_stream.*).
#
#   cmake -DEXE=<program> -DARGS=<arg|arg|...> -DFILE=<written file> -DDIGESTS=<file> -DKEY=<key>
#         -P compare_digest.cmake
#
# ARGS separates the program's arguments with `|`. DIGESTS holds '<sha256>  <key>' lines ('#'
# comments). A program that exits with 77 (no CUDA device visible) is a skipped test.

string(REPLACE "|" ";" _args "${ARGS}")
file(REMOVE "${FILE}")
execute_process(COMMAND "${EXE}" ${_args} RESULT_VARIABLE _rc)
if(_rc EQUAL 77)
  message(STATUS "${EXE}: skipped (exit code 77)")
  cmake_language(EXIT 77)
elseif(NOT _rc EQUAL 0)
  message(FATAL_ERROR "${EXE} failed with exit code ${_rc}")
endif()
file(STRINGS "${DIGESTS}" _lines REGEX "^[0-9a-f]+  ${KEY}$")
list(LENGTH _lines _n)
if(NOT _n EQUAL 1)
  message(FATAL_ERROR "${DIGESTS}: no single line for '${KEY}'")
endif()
string(REGEX REPLACE "^([0-9a-f]+)  .*$" "\\1" _expected "${_lines}")
file(SHA256 "${FILE}" _got)
if(NOT _got STREQUAL _expected)
  message(FATAL_ERROR "${FILE}: SHA-256 ${_got}, expected ${_expected} (${KEY})")
endif()
