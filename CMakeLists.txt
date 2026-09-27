# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dynG: dynamic graph and hypergraph updates on GPUs.
# Top-level build (PLAN Section 7). Contributor loop:
#   source scripts/dev_env.sh
#   cmake --preset dev && cmake --build --preset dev && ctest --preset dev

cmake_minimum_required(VERSION 3.30)

include(cmake/version.cmake)

project(
  dyng
  VERSION ${DYNG_VERSION_NUMERIC}
  DESCRIPTION "dynG: dynamic graph and hypergraph updates on GPUs"
  HOMEPAGE_URL "https://github.com/dyng-dev/dyng"
  LANGUAGES CXX
)

# ---------------------------------------------------------------------------------------------
# Options (PLAN Section 7.3)
# ---------------------------------------------------------------------------------------------
find_package(OpenMP QUIET COMPONENTS CXX)
set(_dyng_openmp_default OFF)
if(OpenMP_CXX_FOUND)
  set(_dyng_openmp_default ON)
endif()

option(DYNG_ENABLE_OPENMP "Build the OpenMP backends" ${_dyng_openmp_default})
# The CUDA backends arrive in milestone M1b; until then the option exists (so presets and
# scripts keep their shape) but defaults to OFF and cannot be enabled.
option(DYNG_ENABLE_CUDA "Build the CUDA backends (available from milestone M1b)" OFF)
option(BUILD_SHARED_LIBS "Build libdyng as a shared library" ON)
option(DYNG_BUILD_TESTS "Build the GoogleTest suites" ${PROJECT_IS_TOP_LEVEL})
option(DYNG_BUILD_EXAMPLES "Build the C++ examples" ${PROJECT_IS_TOP_LEVEL})
option(DYNG_BUILD_TOOLS "Build the dyng CLI and tools" ${PROJECT_IS_TOP_LEVEL})
option(DYNG_BUILD_TUTORIALS "Build and test the tutorial algorithms" ${PROJECT_IS_TOP_LEVEL})
option(DYNG_BUILD_BENCHMARKS "Build nvbench / Google Benchmark / dyng-bench" OFF)
option(DYNG_BUILD_PARITY_TESTS "Build the golden parity tests (needs goldens)" OFF)
option(DYNG_BUILD_COMPAT_TOOLS "Build drop-in clones of the originals' CLIs" OFF)
option(DYNG_BUILD_PYTHON "Build the Python extension module" OFF)
option(DYNG_BUILD_DOCS "Build the Doxygen + Sphinx documentation" OFF)
option(DYNG_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DYNG_INSTALL "Generate the install and export rules" ${PROJECT_IS_TOP_LEVEL})
set(DYNG_SANITIZE "" CACHE STRING "Host sanitizers: empty, 'address;undefined' or 'thread'")
set(DYNG_INDEX_TYPES "default" CACHE STRING "Instantiated index types: default or minimal")
set_property(CACHE DYNG_INDEX_TYPES PROPERTY STRINGS default minimal)
set(DYNG_ALGORITHMS "all" CACHE STRING "Semicolon list of algorithms to build, or 'all'")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_dyng_budgets_default ON)
else()
  set(_dyng_budgets_default OFF)
endif()
option(DYNG_DEBUG_BUDGETS "Allocation and synchronization counters (invariant I9)" ${_dyng_budgets_default})

if(DYNG_ENABLE_CUDA)
  message(FATAL_ERROR "DYNG_ENABLE_CUDA=ON: the CUDA backends are not implemented yet (milestone M1b)")
endif()
foreach(_dyng_unavailable DYNG_BUILD_PYTHON DYNG_BUILD_DOCS DYNG_BUILD_BENCHMARKS)
  if(${_dyng_unavailable})
    message(WARNING "${_dyng_unavailable}=ON has no effect yet (added in a later milestone)")
  endif()
endforeach()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(cmake/warnings.cmake)
include(cmake/sanitizers.cmake)
include(cmake/dependencies.cmake)

message(STATUS "dynG ${DYNG_VERSION_STRING}: OpenMP=${DYNG_ENABLE_OPENMP} CUDA=${DYNG_ENABLE_CUDA} "
               "shared=${BUILD_SHARED_LIBS} tests=${DYNG_BUILD_TESTS} Werror=${DYNG_WARNINGS_AS_ERRORS}")

set(DYNG_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/cpp/include")
add_subdirectory(cpp)

if(DYNG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()

if(DYNG_BUILD_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()

if(DYNG_BUILD_COMPAT_TOOLS)
  add_subdirectory(tools/compat)
endif()

# `cmake --build --preset <p> --target docs-doxygen`: the Doxygen check of ci/docs.sh on this
# build tree's generated headers (the Sphinx site and DYNG_BUILD_DOCS arrive in M5).
find_program(DYNG_DOXYGEN_EXECUTABLE doxygen)
if(DYNG_DOXYGEN_EXECUTABLE AND PROJECT_IS_TOP_LEVEL)
  add_custom_target(
    docs-doxygen
    COMMAND ${CMAKE_COMMAND} -E env "DYNG_BUILD_DIR=${PROJECT_BINARY_DIR}"
            "DYNG_DOXYGEN_OUTPUT=${PROJECT_BINARY_DIR}/doxygen" "${PROJECT_SOURCE_DIR}/ci/docs.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Doxygen on the public headers (warnings as errors) and the convention check"
    VERBATIM USES_TERMINAL
  )
endif()

if(DYNG_INSTALL)
  include(cmake/install.cmake)
endif()
