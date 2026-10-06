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

# CUDA: ON by default when a CUDA compiler is found (CUDACXX, or nvcc on PATH; scripts/dev_env.sh
# sets CUDACXX). The presets pin it: the CPU presets OFF, the *-cuda presets ON.
if(NOT DEFINED DYNG_ENABLE_CUDA OR DYNG_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
endif()
set(_dyng_cuda_default OFF)
if(CMAKE_CUDA_COMPILER)
  set(_dyng_cuda_default ON)
endif()
option(DYNG_ENABLE_CUDA "Build the CUDA backends" ${_dyng_cuda_default})
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
option(DYNG_BUILD_FUZZERS "Build the libFuzzer targets of the readers (Clang only; preset fuzz)" OFF)
option(DYNG_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DYNG_INSTALL "Generate the install and export rules" ${PROJECT_IS_TOP_LEVEL})
set(DYNG_SANITIZE "" CACHE STRING
    "Host sanitizers: empty, 'address;undefined' or 'thread' (+ 'fuzzer-no-link' with Clang)")
set(DYNG_INDEX_TYPES "default" CACHE STRING "Instantiated index types: default or minimal")
set_property(CACHE DYNG_INDEX_TYPES PROPERTY STRINGS default minimal)
set(DYNG_ALGORITHMS "all" CACHE STRING "Semicolon list of algorithms to build, or 'all'")
# sssp is part of every build: the library itself calls it (generators::legacy::mosp_changes
# builds its SOSP trees with sssp::compute) and the shared suites use it as the partner of the
# composition checks. A subset that leaves it out gets it added (PLAN 9.4's
# -DDYNG_ALGORITHMS=<name> builds and links; ADR 0022).
if(NOT DYNG_ALGORITHMS STREQUAL "all" AND NOT "sssp" IN_LIST DYNG_ALGORITHMS)
  message(STATUS "dynG: sssp added to DYNG_ALGORITHMS (the library's generators and the shared "
                 "test suites need it)")
  list(APPEND DYNG_ALGORITHMS sssp)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_dyng_budgets_default ON)
else()
  set(_dyng_budgets_default OFF)
endif()
option(DYNG_DEBUG_BUDGETS "Allocation and synchronization counters (invariant I9)" ${_dyng_budgets_default})
option(DYNG_STDLIB_ASSERTIONS "Bounds-checked standard containers (_GLIBCXX_ASSERTIONS) in host code"
       ${_dyng_budgets_default}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_dyng_debug_sync_default ON)
else()
  set(_dyng_debug_sync_default OFF)
endif()
option(DYNG_CUDA_DEBUG_SYNC "DYNG_CHECK_KERNEL also synchronizes the stream (Debug, sanitize-cuda)"
       ${_dyng_debug_sync_default}
)
option(DYNG_CUDA_LINEINFO "Compile CUDA code with -lineinfo (always on in RelWithDebInfo)" OFF)

# CUDA: the architectures must be known before enable_language(CUDA) and before any target exists.
if(DYNG_ENABLE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "DYNG_ENABLE_CUDA=ON but no CUDA compiler was found: set CUDACXX or "
                        "CMAKE_CUDA_COMPILER (source scripts/dev_env.sh on the development machine)")
  endif()
  include(cmake/cuda_architectures.cmake)
  dyng_set_cuda_architectures()
  # nvcc compiles the host part of .cu files with the same compiler as the .cpp files.
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  # The cudart must match the compiler's toolkit (PLAN Section 7.3): Shared except for wheels.
  if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  endif()
  enable_language(CUDA)
  # native without a visible GPU would silently compile for nvcc's default architecture only.
  dyng_check_native_cuda_architectures()
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
  endif()
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()
# NVTX ranges for the profiler's stages (PLAN Section 7.3; profiler_options::nvtx): the header-only
# NVTX 3 of the CUDA toolkit, ON when it is found. FindCUDAToolkit defines CUDA::nvtx3 from the
# toolkit's include directories without checking for the headers (an incomplete toolkit, e.g. some
# conda packages, has none), so the header itself is looked for.
set(_dyng_nvtx_found OFF)
if(DYNG_ENABLE_CUDA AND TARGET CUDA::nvtx3)
  get_target_property(_dyng_nvtx_dirs CUDA::nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _dyng_nvtx_dirs)
    set(_dyng_nvtx_dirs "")
  endif()
  find_path(
    DYNG_NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h
    PATHS ${_dyng_nvtx_dirs} ${CUDAToolkit_INCLUDE_DIRS}
    NO_DEFAULT_PATH
  )
  if(DYNG_NVTX3_INCLUDE_DIR)
    set(_dyng_nvtx_found ON)
  endif()
endif()
option(DYNG_WITH_NVTX "NVTX ranges for profiler stages (needs the toolkit's NVTX 3 headers)"
       ${_dyng_nvtx_found}
)
if(DYNG_WITH_NVTX AND NOT _dyng_nvtx_found)
  message(FATAL_ERROR "DYNG_WITH_NVTX=ON needs a CUDA build whose toolkit provides CUDA::nvtx3 "
                      "with its headers (nvtx3/nvToolsExt.h)"
  )
endif()
# The benchmark suites (nvbench, Google Benchmark, dyng-bench) come in 0.2 (PLAN Section 11.2).
if(DYNG_BUILD_BENCHMARKS)
  message(WARNING "DYNG_BUILD_BENCHMARKS=ON has no effect yet (the benchmark suites of 0.2)")
endif()

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

if(DYNG_BUILD_TESTS OR DYNG_BUILD_FUZZERS)
  enable_testing()
endif()
if(DYNG_BUILD_TESTS)
  add_subdirectory(cpp/tests)
endif()
# The reader fuzzers (PLAN Sections 5.7 and 8.1): the libFuzzer targets with DYNG_BUILD_FUZZERS,
# and in every test build their replay over the committed seed corpus (cpp/fuzz/README.md).
if(DYNG_BUILD_TESTS OR DYNG_BUILD_FUZZERS)
  add_subdirectory(cpp/fuzz)
endif()

# The examples and the compat CLIs use sssp and cycle_count: a build of a subset of the algorithms
# (DYNG_ALGORITHMS) leaves them out.
if(TARGET dyng_sssp AND TARGET dyng_cycle_count)
  set(_dyng_both_algorithms ON)
else()
  set(_dyng_both_algorithms OFF)
endif()
if(DYNG_BUILD_EXAMPLES AND _dyng_both_algorithms)
  add_subdirectory(examples/cpp)
endif()

# dyng-compat-mosp --mosp also needs mosp (M7).
if(DYNG_BUILD_COMPAT_TOOLS AND _dyng_both_algorithms AND TARGET dyng_mosp)
  add_subdirectory(tools/compat)
endif()

# The Python extension module dyng._core (PLAN Section 5.4; built by the root pyproject.toml
# through scikit-build-core, or in any CMake build with -DDYNG_BUILD_PYTHON=ON).
if(DYNG_BUILD_PYTHON)
  if(NOT _dyng_both_algorithms OR NOT TARGET dyng_mosp)
    message(FATAL_ERROR "DYNG_BUILD_PYTHON=ON needs every algorithm (DYNG_ALGORITHMS=all)")
  endif()
  add_subdirectory(python)
endif()

# `cmake --build --preset <p> --target docs-doxygen`: the Doxygen check of ci/docs.sh on this
# build tree's generated headers. With DYNG_BUILD_DOCS=ON the target `docs` (part of `all`) builds
# the whole site as ci/docs.sh does (Doxygen, the API baseline, Sphinx with -W, the link check)
# into <build>/docs/html.
find_program(DYNG_DOXYGEN_EXECUTABLE doxygen)
if(DYNG_BUILD_DOCS)
  find_program(DYNG_SPHINX_BUILD_EXECUTABLE sphinx-build)
  if(NOT DYNG_DOXYGEN_EXECUTABLE OR NOT DYNG_SPHINX_BUILD_EXECUTABLE)
    message(FATAL_ERROR "DYNG_BUILD_DOCS=ON needs doxygen and sphinx-build with the packages of "
                        "environment.yml (conda env update -f environment.yml)"
    )
  endif()
  add_custom_target(
    docs ALL
    COMMAND ${CMAKE_COMMAND} -E env "DYNG_BUILD_DIR=${PROJECT_BINARY_DIR}"
            "DYNG_DOXYGEN_OUTPUT=${PROJECT_BINARY_DIR}/doxygen"
            "DYNG_DOCS_OUTPUT=${PROJECT_BINARY_DIR}/docs" "${PROJECT_SOURCE_DIR}/ci/docs.sh"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "The documentation site (ci/docs.sh) into ${PROJECT_BINARY_DIR}/docs/html"
    VERBATIM USES_TERMINAL
  )
endif()
if(DYNG_DOXYGEN_EXECUTABLE AND PROJECT_IS_TOP_LEVEL)
  add_custom_target(
    docs-doxygen
    COMMAND ${CMAKE_COMMAND} -E env "DYNG_BUILD_DIR=${PROJECT_BINARY_DIR}"
            "DYNG_DOXYGEN_OUTPUT=${PROJECT_BINARY_DIR}/doxygen" "${PROJECT_SOURCE_DIR}/ci/docs.sh"
            --doxygen-only
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Doxygen on the public headers (warnings as errors) and the convention check"
    VERBATIM USES_TERMINAL
  )
endif()

if(DYNG_INSTALL)
  include(cmake/install.cmake)
endif()
