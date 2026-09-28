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
          graph/set_semantics_test.cpp graph/structural_change_test.cpp
  LABELS cpu graph
  LIBRARIES dyng::testing
)
dyng_add_test(
  NAME dyng_io_tests SOURCES io/io_test.cpp io/util_test.cpp io/edge_list_io_test.cpp LABELS cpu io
)
dyng_add_test(
  NAME dyng_framework_tests
  SOURCES framework/workspace_test.cpp framework/thread_list_test.cpp
          framework/list_gather_test.cpp
  LABELS cpu framework
)
if(DYNG_ENABLE_OPENMP)
  target_link_libraries(dyng_framework_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
dyng_add_test(
  NAME dyng_generators_tests
  SOURCES generators/legacy_test.cpp generators/cycle_enum_batch_test.cpp
  LABELS cpu generators
)
# Parity with CycleEnumeration-GPU@0a976ad on the TUDataset graphs (outside the repository; the
# test skips when they are missing). Label parity, not cpu: run with `ctest -L parity`.
dyng_add_test(
  NAME dyng_cycle_enum_parity_tests
  SOURCES parity/cycle_enum_datasets_test.cpp parity/cycle_count_datasets_test.cpp
  LABELS parity cycle_count
)
foreach(_dyng_test dyng_graph_tests dyng_io_tests dyng_framework_tests dyng_generators_tests
                   dyng_cycle_enum_parity_tests
)
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

# cycle_count (CycleEnumeration-GPU@0a976ad): API, the original's suites, randomized differential
# tests against the oracles and parity with the original's histograms on the committed fixtures.
set(DYNG_CYCLE_COUNT_TEST_SOURCES
    algorithms/cycle_count/cycle_count_test.cpp algorithms/cycle_count/cycle_count_random_test.cpp
    algorithms/cycle_count/cycle_count_fixture_test.cpp
    algorithms/cycle_count/cycle_count_kernels_test.cpp
)
dyng_add_test(
  NAME dyng_cycle_count_tests
  SOURCES ${DYNG_CYCLE_COUNT_TEST_SOURCES}
  LABELS cpu cycle_count
  LIBRARIES dyng::testing
)
target_compile_definitions(
  dyng_cycle_count_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
)
target_include_directories(dyng_cycle_count_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

# Compile-failure tests: cycle_count on an unsupported graph type (int64_t vertex ids) must stop
# at the static_assert that names the supported types (PLAN 4.5: plain-English static_asserts),
# not at a link error. The test runs the compiler with -fsyntax-only and passes on the message.
foreach(_case "compute;" "update;-DDYNG_COMPILE_FAIL_UPDATE")
  list(GET _case 0 _name)
  list(GET _case 1 _define)
  add_test(NAME cycle_count.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/cycle_count_int64_vertices.cpp
  )
  set_tests_properties(
    cycle_count.static_assert.${_name}
    PROPERTIES LABELS "cpu;cycle_count" PASS_REGULAR_EXPRESSION
               "dyng::cycle_count supports graph<int32_t, int32_t or int64_t"
  )
endforeach()

# Mutation tests (PLAN Section 8.4): the recorded mutations of CycleEnumeration-GPU (double counting
# 5-cycles; a weakened ownership rule) must make the randomized cycle_count suite fail. Each is
# compiled into a copy of libdyng whose cycle_count module is rebuilt with DYNG_MUTATION_<NAME>
# (a static library of every other module's objects plus the mutated ones), against which the
# suite cycle_count_random_test.cpp is linked; the CTest passes when the suite fails. A control
# copy built the same way without a mutation must pass. Label "mutation" (and cpu).
option(DYNG_MUTATION_TESTS "Build the mutation tests of cycle_count" ON)
if(DYNG_MUTATION_TESTS AND TARGET dyng_cycle_count)
  get_property(_dyng_modules GLOBAL PROPERTY DYNG_MODULE_TARGETS)
  get_property(_dyng_cc_sources GLOBAL PROPERTY DYNG_MODULE_SOURCES_dyng_cycle_count)
  set(_dyng_other_objects)
  foreach(_module IN LISTS _dyng_modules)
    if(NOT _module STREQUAL "dyng_cycle_count")
      list(APPEND _dyng_other_objects $<TARGET_OBJECTS:${_module}>)
    endif()
  endforeach()
  # skip_workspace_resize is dynG's own (review of M2a): the insert phase keeps the per-thread
  # marks of the graph before the batch, so a batch that adds vertices must make the suite fail
  # (the phases check their workspace) instead of writing out of bounds unseen.
  foreach(_mutant "control;;pass" "double_count_5;DYNG_MUTATION_DOUBLE_COUNT_5;fail"
                  "weak_ownership;DYNG_MUTATION_WEAK_OWNERSHIP;fail"
                  "skip_workspace_resize;DYNG_MUTATION_SKIP_WORKSPACE_RESIZE;fail"
  )
    list(GET _mutant 0 _name)
    list(GET _mutant 1 _define)
    list(GET _mutant 2 _expect)
    set(_lib dyng_mutant_${_name})
    add_library(${_lib}_cycle_count OBJECT ${_dyng_cc_sources})
    add_library(
      ${_lib} STATIC ${_dyng_other_objects} $<TARGET_OBJECTS:${_lib}_cycle_count>
                     "${PROJECT_SOURCE_DIR}/cpp/src/testing/cycle_oracle.cpp"
    )
    foreach(_target ${_lib}_cycle_count ${_lib})
      target_compile_features(${_target} PRIVATE cxx_std_17)
      target_include_directories(
        ${_target} PRIVATE "${PROJECT_SOURCE_DIR}/cpp/include" "${DYNG_GENERATED_INCLUDE_DIR}"
                           "${PROJECT_SOURCE_DIR}/cpp/src"
      )
      if(_define)
        target_compile_definitions(${_target} PRIVATE ${_define}=1)
      endif()
      if(DYNG_ENABLE_OPENMP)
        target_link_libraries(${_target} PRIVATE OpenMP::OpenMP_CXX)
      endif()
      dyng_set_warnings(${_target})
      dyng_enable_sanitizers(${_target})
    endforeach()
    set_target_properties(${_lib}_cycle_count PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
      ${_lib} INTERFACE "${PROJECT_SOURCE_DIR}/cpp/include" "${DYNG_GENERATED_INCLUDE_DIR}"
    )
    target_link_libraries(${_lib} INTERFACE Threads::Threads)
    if(DYNG_ENABLE_OPENMP)
      target_link_libraries(${_lib} INTERFACE OpenMP::OpenMP_CXX)
    endif()
    if(DYNG_ENABLE_CUDA)
      target_link_libraries(${_lib} INTERFACE CUDA::cudart)
    endif()
    set(_exe dyng_cycle_count_mutant_${_name})
    add_executable(${_exe} algorithms/cycle_count/cycle_count_random_test.cpp)
    target_link_libraries(
      ${_exe} PRIVATE ${_lib} dyng_test_support GTest::gtest GTest::gmock GTest::gtest_main
    )
    target_compile_features(${_exe} PRIVATE cxx_std_17)
    dyng_set_warnings(${_exe})
    dyng_enable_sanitizers(${_exe})
    add_test(NAME cycle_count.mutation.${_name}
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                     -DNAME=${_name} -P "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
    )
    set_tests_properties(
      cycle_count.mutation.${_name} PROPERTIES LABELS "cpu;mutation;cycle_count" ENVIRONMENT
                                               "${DYNG_TEST_ENVIRONMENT}"
    )
  endforeach()
endif()

dyng_add_test(
  NAME dyng_testing_tests
  SOURCES testing/oracle_test.cpp testing/cycle_oracle_test.cpp
  LABELS cpu testing
  LIBRARIES dyng::testing
)
target_compile_definitions(dyng_testing_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")

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
