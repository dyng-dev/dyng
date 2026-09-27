# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# CUDA target architectures (PLAN Sections 7.3 and 7.8). Included by the top-level CMakeLists.txt
# BEFORE enable_language(CUDA) and before any target is created (a lesson of CycleEnumeration-GPU:
# an architecture set after the first target silently leaves that target on the compiler default).
#
#   DYNG_CUDA_ARCHITECTURES  (cache)
#     native   the GPUs of the build machine (development presets; needs a visible GPU)
#     release  the release list of the toolkit in use (release, wheel and CI presets):
#                CUDA >= 12.8 and 13.x   75-real;80-real;86-real;89-real;90-real;100-real;120
#                CUDA 12.4 - 12.7        75-real;80-real;86-real;89-real;90
#              The last entry also embeds PTX, for forward compatibility. CUDA 13 dropped Volta
#              and older, so sm_75 is the floor on every toolkit; sm_100 / sm_120 need 12.8.
#     <list>   any explicit CMAKE_CUDA_ARCHITECTURES value, e.g. "86" (the parity preset: the
#              originals' -arch=sm_86, i.e. sm_86 SASS plus compute_86 PTX)
#   An explicit -DCMAKE_CUDA_ARCHITECTURES=... always wins over DYNG_CUDA_ARCHITECTURES.
#
#   dyng_set_cuda_architectures()      sets CMAKE_CUDA_ARCHITECTURES (needs CMAKE_CUDA_COMPILER)
#   dyng_release_cuda_architectures(<version> <out-var>)   the release list of a toolkit version

set(DYNG_CUDA_ARCHITECTURES
    "native"
    CACHE STRING "CUDA architectures: native, release, or an explicit CMAKE_CUDA_ARCHITECTURES list"
)

# The oldest toolkit dynG builds with (PLAN Section 7.1; ADR 0003).
set(DYNG_CUDA_MINIMUM_VERSION 12.4)

function(dyng_release_cuda_architectures version out_var)
  if(version VERSION_LESS DYNG_CUDA_MINIMUM_VERSION)
    message(FATAL_ERROR "dynG needs CUDA >= ${DYNG_CUDA_MINIMUM_VERSION}, found ${version}")
  elseif(version VERSION_LESS 12.8)
    set(_archs "75-real;80-real;86-real;89-real;90")
  else()
    set(_archs "75-real;80-real;86-real;89-real;90-real;100-real;120")
  endif()
  set(${out_var} "${_archs}" PARENT_SCOPE)
endfunction()

# The release (X.Y) of the CUDA compiler, read from `nvcc --version` (enable_language(CUDA), which
# would set CMAKE_CUDA_COMPILER_VERSION, needs the architectures first).
function(dyng_query_cuda_compiler_version out_var)
  execute_process(
    COMMAND "${CMAKE_CUDA_COMPILER}" --version
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0 OR NOT _out MATCHES "release ([0-9]+\\.[0-9]+)")
    message(FATAL_ERROR "cannot read the version of ${CMAKE_CUDA_COMPILER}: ${_out}${_err}")
  endif()
  set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

macro(dyng_set_cuda_architectures)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "dyng_set_cuda_architectures(): no CUDA compiler (CMAKE_CUDA_COMPILER)")
  endif()
  dyng_query_cuda_compiler_version(DYNG_CUDA_COMPILER_RELEASE)
  if(DYNG_CUDA_COMPILER_RELEASE VERSION_LESS DYNG_CUDA_MINIMUM_VERSION)
    message(FATAL_ERROR "dynG needs CUDA >= ${DYNG_CUDA_MINIMUM_VERSION}; ${CMAKE_CUDA_COMPILER} "
                        "is ${DYNG_CUDA_COMPILER_RELEASE}")
  endif()
  if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES} AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(_dyng_cuda_arch_source "CMAKE_CUDA_ARCHITECTURES (explicit)")
  elseif(DYNG_CUDA_ARCHITECTURES STREQUAL "release")
    dyng_release_cuda_architectures("${DYNG_CUDA_COMPILER_RELEASE}" CMAKE_CUDA_ARCHITECTURES)
    set(_dyng_cuda_arch_source "release list of CUDA ${DYNG_CUDA_COMPILER_RELEASE}")
  elseif(DYNG_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "DYNG_CUDA_ARCHITECTURES is empty (use native, release or a list)")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "${DYNG_CUDA_ARCHITECTURES}")
    set(_dyng_cuda_arch_source "DYNG_CUDA_ARCHITECTURES")
  endif()
  set(DYNG_CUDA_ARCHITECTURES_RESOLVED
      "${CMAKE_CUDA_ARCHITECTURES}"
      CACHE INTERNAL "The CUDA architectures of this build (reports)"
  )
  message(STATUS "dynG: CUDA ${DYNG_CUDA_COMPILER_RELEASE} architectures ${CMAKE_CUDA_ARCHITECTURES} "
                 "(${_dyng_cuda_arch_source})")
  unset(_dyng_cuda_arch_source)
endmacro()
