# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng_add_algorithm(NAME <algo> SOURCES <files...>)
#
# Adds the algorithm's private OBJECT library dyng_<algo> to libdyng (PLAN Section 4.8, file 3).
# Honours DYNG_ALGORITHMS ("all" or a semicolon list). CUDA_SOURCES and FAST_MATH arrive with the
# CUDA backend (M1b).

function(dyng_add_algorithm)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME" "SOURCES")
  if(NOT arg_NAME OR NOT arg_SOURCES)
    message(FATAL_ERROR "dyng_add_algorithm: NAME and SOURCES are required")
  endif()
  if(NOT DYNG_ALGORITHMS STREQUAL "all" AND NOT arg_NAME IN_LIST DYNG_ALGORITHMS)
    message(STATUS "dynG: algorithm ${arg_NAME} skipped (DYNG_ALGORITHMS=${DYNG_ALGORITHMS})")
    return()
  endif()
  set(_sources)
  foreach(_source IN LISTS arg_SOURCES)
    list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
  endforeach()
  dyng_add_module(NAME ${arg_NAME} SOURCES ${_sources})
endfunction()
