# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng_add_algorithm(NAME <algo> SOURCES <files...> [CUDA_SOURCES <files...>] [FAST_MATH])
#
# Adds the algorithm's private OBJECT library dyng_<algo> to libdyng (PLAN Section 4.8, file 3).
# Honours DYNG_ALGORITHMS ("all" or a semicolon list). CUDA_SOURCES (.cu files) are compiled only
# with DYNG_ENABLE_CUDA=ON. FAST_MATH compiles the algorithm's CUDA sources with --use_fast_math
# (label_propagation only, as its original; there is no project-wide fast math, PLAN Section 7.3).

function(dyng_add_algorithm)
  cmake_parse_arguments(PARSE_ARGV 0 arg "FAST_MATH" "NAME" "SOURCES;CUDA_SOURCES")
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
  set(_cuda_sources)
  foreach(_source IN LISTS arg_CUDA_SOURCES)
    list(APPEND _cuda_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
  endforeach()
  dyng_add_module(NAME ${arg_NAME} SOURCES ${_sources} CUDA_SOURCES ${_cuda_sources})
  if(arg_FAST_MATH AND DYNG_ENABLE_CUDA)
    target_compile_options(dyng_${arg_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endfunction()
