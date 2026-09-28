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

# Graph and I/O tests read fixtures from cpp/tests/data and use private headers of cpp/src.
set(DYNG_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

dyng_add_test(
  NAME dyng_graph_tests
  SOURCES graph/apply_test.cpp graph/graph_test.cpp graph/mosp_apply_fixture_test.cpp
  LABELS cpu graph
)
dyng_add_test(NAME dyng_io_tests SOURCES io/io_test.cpp io/util_test.cpp LABELS cpu io)
dyng_add_test(
  NAME dyng_framework_tests
  SOURCES framework/workspace_test.cpp framework/thread_list_test.cpp
          framework/list_gather_test.cpp
  LABELS cpu framework
)
if(DYNG_ENABLE_OPENMP)
  target_link_libraries(dyng_framework_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
dyng_add_test(NAME dyng_generators_tests SOURCES generators/legacy_test.cpp LABELS cpu generators)
foreach(_dyng_test dyng_graph_tests dyng_io_tests dyng_framework_tests dyng_generators_tests)
  target_include_directories(${_dyng_test} PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  target_compile_definitions(${_dyng_test} PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
endforeach()

dyng_add_test(
  NAME dyng_sssp_tests
  SOURCES algorithms/sssp/sssp_test.cpp algorithms/sssp/sssp_random_test.cpp
          algorithms/sssp/sssp_fixture_test.cpp algorithms/sssp/sssp_workspace_test.cpp
  LABELS cpu sssp
  LIBRARIES dyng::testing
)
target_compile_definitions(dyng_sssp_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
target_include_directories(dyng_sssp_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

dyng_add_test(NAME dyng_testing_tests SOURCES testing/oracle_test.cpp LABELS cpu testing LIBRARIES dyng::testing)

# CUDA core (label gpu; each test skips when no device is visible). ci/gpu_local.sh runs them on
# GPU 1 and under compute-sanitizer memcheck.
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_cuda_tests
    SOURCES
      cuda/cuda_error_test.cpp
      cuda/cuda_memory_test.cpp
      cuda/cuda_resources_test.cpp
      cuda/cuda_test_kernels.cu
      cuda/cuda_workspace_test.cpp
    LABELS gpu core
  )
  target_include_directories(dyng_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_cuda_tests)

  # sssp on the cuda backend: the shared sssp suites compiled with DYNG_TEST_CUDA=1 (their cases
  # run on cuda, the randomized ones against the host backends) plus the CUDA-only cases.
  dyng_add_test(
    NAME dyng_sssp_cuda_tests
    SOURCES algorithms/sssp/sssp_test.cpp algorithms/sssp/sssp_random_test.cpp
            algorithms/sssp/sssp_fixture_test.cpp algorithms/sssp/sssp_cuda_test.cpp
    LABELS gpu sssp
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(
    dyng_sssp_cuda_tests PRIVATE DYNG_TEST_CUDA=1 DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
  )
  target_include_directories(dyng_sssp_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_sssp_cuda_tests)  # user streams in sssp_cuda_test.cpp
  if(DYNG_ENABLE_OPENMP)
    target_link_libraries(dyng_sssp_cuda_tests PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

add_subdirectory(api)

dyng_write_gpu_test_list()
