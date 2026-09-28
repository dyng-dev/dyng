# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs dyng-compat-cycle-enum on every line of cpp/tests/data/cycle_enum/cli/cases.txt (the
# arguments of one run of the original `cycle-enum`, @DATA@ = the fixture directory) and requires
# the original's standard output (cNN.out) byte for byte and its exit status (cNN.status).
# Variables: EXE, DATA (cpp/tests/data/cycle_enum), OPENMP (ON if the OpenMP backend is built;
# otherwise the OpenMP lines are skipped).

file(STRINGS "${DATA}/cli/cases.txt" cases)
set(index 0)
set(failures 0)
foreach(line IN LISTS cases)
  string(REPLACE "@DATA@" "${DATA}" line "${line}")
  separate_arguments(args UNIX_COMMAND "${line}")
  math(EXPR n "${index}")
  if(n LESS 10)
    set(name "c0${n}")
  else()
    set(name "c${n}")
  endif()
  math(EXPR index "${index} + 1")
  if(NOT OPENMP AND (line MATCHES "--backend (openmp|omp)"))
    message(STATUS "${name}: skipped (no OpenMP)")
    continue()
  endif()
  execute_process(
    COMMAND "${EXE}" ${args}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
  )
  file(READ "${DATA}/cli/${name}.out" expected)
  file(STRINGS "${DATA}/cli/${name}.status" expected_status)
  if(NOT out STREQUAL expected)
    message(SEND_ERROR "${name} (${line}): standard output differs from the original's\n"
                       "--- dynG ---\n${out}--- original ---\n${expected}--- stderr ---\n${err}")
    math(EXPR failures "${failures} + 1")
  elseif(NOT rc STREQUAL expected_status)
    message(SEND_ERROR "${name} (${line}): exit status ${rc}, the original's ${expected_status}\n"
                       "${err}")
    math(EXPR failures "${failures} + 1")
  else()
    message(STATUS "${name}: ok (exit ${rc})")
  endif()
endforeach()
if(failures GREATER 0)
  message(FATAL_ERROR "${failures} case(s) differ from the original")
endif()
