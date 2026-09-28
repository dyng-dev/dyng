# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one mutation-test executable (cpp/tests/CMakeLists.txt, DYNG_MUTATION_TESTS) and checks
# the verdict of the suite. Variables: EXE (the suite built against a copy of libdyng whose
# cycle_count module has one recorded mutation compiled in), EXPECT ("fail" for a recorded
# mutation, "pass" for the control copy without one), NAME.
#
# A mutation is caught only if the suite ran to its end (the GoogleTest summary line is printed, so
# the failure is not a crash or a missing binary) and reported at least one failed test. The
# control copy is built the same way without a mutation and must pass: it shows that a failure of
# a mutant comes from the mutation, not from the way the copies are built.

execute_process(
  COMMAND "${EXE}" --gtest_brief=1
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
)
string(REGEX MATCH "\\[==========\\] [0-9]+ tests? from [0-9]+ test suites? ran" summary "${out}")
if(NOT summary)
  message(FATAL_ERROR "mutation ${NAME}: the suite did not run to its end (exit ${rc})\n${out}\n${err}")
endif()
string(REGEX MATCHALL "\\[  FAILED  \\] [A-Za-z0-9_/.]+" failed "${out}")
list(LENGTH failed num_failed)
if(EXPECT STREQUAL "fail")
  if(rc EQUAL 0 OR num_failed EQUAL 0)
    message(FATAL_ERROR "mutation ${NAME} survived: the suite passed (exit ${rc})")
  endif()
  message(STATUS "mutation ${NAME} killed by ${num_failed} failure line(s):")
  list(REMOVE_DUPLICATES failed)
  foreach(line IN LISTS failed)
    message(STATUS "  ${line}")
  endforeach()
else()
  if(NOT rc EQUAL 0 OR NOT num_failed EQUAL 0)
    message(FATAL_ERROR "control copy ${NAME}: the suite failed (exit ${rc})\n${out}")
  endif()
  message(STATUS "control copy ${NAME}: the suite passed")
endif()
