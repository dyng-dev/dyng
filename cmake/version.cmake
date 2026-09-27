# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The single version source is the VERSION file at the repository root (PLAN Section 7.6).
# It holds a PEP 440 version string, e.g. "0.1.0" or "0.1.0.dev0". CMake's project() only
# accepts the numeric part, so this module splits it:
#   DYNG_VERSION_STRING  the full string (as in VERSION)
#   DYNG_VERSION_NUMERIC MAJOR.MINOR.PATCH, passed to project(VERSION ...)
# dyng_generate_version_headers() writes version.hpp and config.hpp into the build tree.

file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _dyng_version_raw)
string(STRIP "${_dyng_version_raw}" DYNG_VERSION_STRING)
if(NOT DYNG_VERSION_STRING MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)$")
  message(FATAL_ERROR "VERSION must start with MAJOR.MINOR.PATCH, got '${DYNG_VERSION_STRING}'")
endif()
set(DYNG_VERSION_NUMERIC "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
set(DYNG_VERSION_SUFFIX "${CMAKE_MATCH_4}")
unset(_dyng_version_raw)

function(dyng_generate_version_headers out_dir)
  set(DYNG_HAS_OPENMP_VALUE 0)
  set(DYNG_HAS_CUDA_VALUE 0)
  if(DYNG_ENABLE_OPENMP)
    set(DYNG_HAS_OPENMP_VALUE 1)
  endif()
  if(DYNG_ENABLE_CUDA)
    set(DYNG_HAS_CUDA_VALUE 1)
  endif()
  configure_file(
    "${PROJECT_SOURCE_DIR}/cpp/include/dyng/version.hpp.in"
    "${out_dir}/dyng/version.hpp"
    @ONLY
  )
  configure_file(
    "${PROJECT_SOURCE_DIR}/cpp/include/dyng/config.hpp.in"
    "${out_dir}/dyng/config.hpp"
    @ONLY
  )
endfunction()
