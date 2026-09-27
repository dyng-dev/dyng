# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# GoogleTest suites (PLAN Section 8). One executable per module; every test case carries CTest
# labels (cpu, gpu, slow, parity, sanitize, <algorithm>).

include("${PROJECT_SOURCE_DIR}/cmake/dyng_add_test.cmake")

# Shared test helpers (cpp/tests/support), header-only.
add_library(dyng_test_support INTERFACE)
target_include_directories(dyng_test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

dyng_add_test(
  NAME dyng_core_tests
  SOURCES
    core/array_view_test.cpp
    core/backend_test.cpp
    core/buffer_test.cpp
    core/citation_test.cpp
    core/error_test.cpp
    core/logging_test.cpp
    core/memory_test.cpp
    core/profiler_test.cpp
    core/resources_test.cpp
    core/types_test.cpp
    core/version_test.cpp
  LABELS cpu core
)
if(DYNG_ENABLE_OPENMP)
  target_link_libraries(dyng_core_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(api)
