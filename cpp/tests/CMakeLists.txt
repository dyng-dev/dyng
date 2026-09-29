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
          framework/list_gather_test.cpp framework/enactor_test.cpp
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
set(_dyng_private_tests dyng_graph_tests dyng_io_tests dyng_framework_tests dyng_generators_tests)
if(TARGET dyng_cycle_count)
  dyng_add_test(
    NAME dyng_cycle_enum_parity_tests
    SOURCES parity/cycle_enum_datasets_test.cpp parity/cycle_count_datasets_test.cpp
    LABELS parity cycle_count
  )
  list(APPEND _dyng_private_tests dyng_cycle_enum_parity_tests)
endif()
foreach(_dyng_test IN LISTS _dyng_private_tests)
  target_include_directories(${_dyng_test} PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  target_compile_definitions(${_dyng_test} PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
endforeach()

# The suites of one algorithm are built when the algorithm is (DYNG_ALGORITHMS).
if(TARGET dyng_sssp)
dyng_add_test(
  NAME dyng_sssp_tests
  SOURCES algorithms/sssp/sssp_test.cpp algorithms/sssp/sssp_random_test.cpp
          algorithms/sssp/sssp_fixture_test.cpp algorithms/sssp/sssp_workspace_test.cpp
  LABELS cpu sssp
  LIBRARIES dyng::testing
)
target_compile_definitions(dyng_sssp_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
target_include_directories(dyng_sssp_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
endif()

# cycle_count (CycleEnumeration-GPU@0a976ad): API, the original's suites, randomized differential
# tests against the oracles and parity with the original's histograms on the committed fixtures.
# The shared suites run on the host backends here and on cuda in dyng_cycle_count_cuda_tests;
# cycle_count_kernels_test.cpp tests host-only pieces (the host searches, the ownership table,
# the workspace, the host planning of the CUDA work items) and runs here only.
if(TARGET dyng_cycle_count)
set(DYNG_CYCLE_COUNT_TEST_SOURCES
    algorithms/cycle_count/cycle_count_test.cpp algorithms/cycle_count/cycle_count_random_test.cpp
    algorithms/cycle_count/cycle_count_fixture_test.cpp
)
dyng_add_test(
  NAME dyng_cycle_count_tests
  SOURCES ${DYNG_CYCLE_COUNT_TEST_SOURCES} algorithms/cycle_count/cycle_count_kernels_test.cpp
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

endif()

# Compile-failure tests of the framework's conformance checks (framework/conformance.hpp, PLAN
# Section 8.2): a problem missing a required declaration or hook stops at a static_assert with a
# plain-English message. The control case (no define) must compile.
set(_dyng_framework_cf_includes
    -I${PROJECT_SOURCE_DIR}/cpp/src -I${PROJECT_SOURCE_DIR}/cpp/include
    -I${DYNG_GENERATED_INCLUDE_DIR}
)
# name|define|expected message (a regular expression)
foreach(
  _case
  "control||"
  "no_base|-DDYNG_CF_NO_BASE|a problem must derive from"
  "fixed_point_without_loop|-DDYNG_CF_FP_NO_LOOP|a fixed_point problem must provide `loop"
  "aggregate_delta_without_ownership|-DDYNG_CF_AG_NO_OWNER|must declare `using ownership_type"
  "aggregate_delta_without_count_minus|-DDYNG_CF_AG_NO_MINUS|count.context&, old_view"
  "participant_without_target|-DDYNG_CF_NO_TARGET|must provide `const void. target"
)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME framework.conformance.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   ${_dyng_framework_cf_includes}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/framework_conformance.cpp
  )
  if(_message STREQUAL "")
    set_tests_properties(framework.conformance.${_name} PROPERTIES LABELS "cpu;framework")
  else()
    set_tests_properties(
      framework.conformance.${_name} PROPERTIES LABELS "cpu;framework" PASS_REGULAR_EXPRESSION
                                                "${_message}"
    )
  endif()
endforeach()

# Mutation tests (PLAN Section 8.4): the recorded mutations of CycleEnumeration-GPU (double counting
# 5-cycles; a weakened ownership rule) must make the randomized cycle_count suite fail. Each is
# compiled into a copy of libdyng whose cycle_count module is rebuilt with DYNG_MUTATION_<NAME>
# (a static library of every other module's objects plus the mutated ones), against which the
# suite cycle_count_random_test.cpp is linked; the CTest passes when the suite fails. A control
# copy built the same way without a mutation must pass. Label "mutation" (and cpu).
#
# The original recorded both mutations in its CUDA kernels (the static work-queue kernel and the
# update kernel). DYNG_MUTATION_<NAME> mutates the host searches; DYNG_MUTATION_CUDA_<NAME> mutates
# the same rule in the device kernels only (dfs.cuh, cuda.cu). In CUDA builds the cuda mutants are
# linked with the gpu suite (the shared suites with DYNG_TEST_CUDA=1 plus cycle_count_cuda_test.cpp,
# which compare the cuda backend with the unmutated host backends and the oracles): CTests
# cycle_count.mutation.cuda.<name>, labels gpu;mutation;cycle_count, with the control copy.
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
  # name|define|expected verdict
  set(_dyng_mutants "control||pass" "double_count_5|DYNG_MUTATION_DOUBLE_COUNT_5|fail"
                    "weak_ownership|DYNG_MUTATION_WEAK_OWNERSHIP|fail"
                    "skip_workspace_resize|DYNG_MUTATION_SKIP_WORKSPACE_RESIZE|fail"
  )
  if(DYNG_ENABLE_CUDA)
    list(APPEND _dyng_mutants "cuda_double_count_5|DYNG_MUTATION_CUDA_DOUBLE_COUNT_5|fail"
         "cuda_weak_ownership|DYNG_MUTATION_CUDA_WEAK_OWNERSHIP|fail"
    )
  endif()
  foreach(_entry IN LISTS _dyng_mutants)
    string(REPLACE "|" ";" _mutant "${_entry}")
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
      dyng_set_cuda_options(${_target})  # the module's .cu files (cuda builds)
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
    if(NOT _name MATCHES "^cuda_")
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
    endif()
    if(DYNG_ENABLE_CUDA AND (_name STREQUAL "control" OR _name MATCHES "^cuda_"))
      # The gpu suite against the mutant (the device mutations, and the control).
      set(_exe dyng_cycle_count_cuda_mutant_${_name})
      add_executable(
        ${_exe} algorithms/cycle_count/cycle_count_random_test.cpp
                algorithms/cycle_count/cycle_count_cuda_test.cpp
      )
      target_link_libraries(
        ${_exe} PRIVATE ${_lib} dyng_test_support GTest::gtest GTest::gmock GTest::gtest_main
      )
      target_compile_definitions(${_exe} PRIVATE DYNG_TEST_CUDA=1)
      target_include_directories(${_exe} PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
      target_compile_features(${_exe} PRIVATE cxx_std_17)
      dyng_set_warnings(${_exe})
      dyng_enable_sanitizers(${_exe})
      dyng_set_cuda_options(${_exe})
      add_test(NAME cycle_count.mutation.cuda.${_name}
               COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                       -DNAME=cuda.${_name} -P
                       "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
      )
      set_tests_properties(
        cycle_count.mutation.cuda.${_name}
        PROPERTIES LABELS "gpu;mutation;cycle_count" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}"
      )
    endif()
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
  if(TARGET dyng_sssp)
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

  # cycle_count on the cuda backend (M2b): the shared cycle_count suites compiled with
  # DYNG_TEST_CUDA=1 (their cases run on cuda; the fixtures against the original's CUDA backend)
  # plus the CUDA-only cases (bounds, schedulers, cross-backend chains, the resident graph).
  if(TARGET dyng_cycle_count)
  dyng_add_test(
    NAME dyng_cycle_count_cuda_tests
    SOURCES ${DYNG_CYCLE_COUNT_TEST_SOURCES} algorithms/cycle_count/cycle_count_cuda_test.cpp
    LABELS gpu cycle_count
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(
    dyng_cycle_count_cuda_tests PRIVATE DYNG_TEST_CUDA=1 DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
  )
  target_include_directories(dyng_cycle_count_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_cycle_count_cuda_tests)  # cudaMemcpy in cycle_count_cuda_test.cpp
  if(DYNG_ENABLE_OPENMP)
    target_link_libraries(dyng_cycle_count_cuda_tests PRIVATE OpenMP::OpenMP_CXX)
  endif()
  endif()
endif()

# The conformance kit (PLAN Section 8.2; cpp/tests/conformance): one executable per built
# algorithm (labels cpu, conformance and the algorithm's name), and in CUDA builds a second one
# compiled with DYNG_TEST_CUDA=1 (labels gpu, conformance, name). The suite is the algorithm's
# one-line algorithms/<name>/<name>_conformance_test.cpp; conformance/registry.hpp (generated by
# scripts/regen.py) includes the test_traits of every algorithm built here (DYNG_ALGORITHM_<NAME>),
# for the composition check C10. conformance/allocation_counter.cpp counts host heap allocations
# for the budget check C8.
get_property(_dyng_algorithms GLOBAL PROPERTY DYNG_BUILT_ALGORITHMS)
set(_dyng_algorithm_defines)
foreach(_dyng_algorithm IN LISTS _dyng_algorithms)
  string(TOUPPER "${_dyng_algorithm}" _dyng_upper)
  list(APPEND _dyng_algorithm_defines DYNG_ALGORITHM_${_dyng_upper}=1)
endforeach()
foreach(_dyng_algorithm IN LISTS _dyng_algorithms)
  set(_suite "algorithms/${_dyng_algorithm}/${_dyng_algorithm}_conformance_test.cpp")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_suite}")
    message(FATAL_ERROR "dynG: ${_dyng_algorithm} has no conformance suite (cpp/tests/${_suite}); "
                        "an algorithm cannot be registered without one (PLAN 8.2, invariant I8)")
  endif()
  set(_variants host)
  if(DYNG_ENABLE_CUDA)
    list(APPEND _variants cuda)
  endif()
  foreach(_variant IN LISTS _variants)
    if(_variant STREQUAL "host")
      set(_target dyng_${_dyng_algorithm}_conformance_tests)
      dyng_add_test(
        NAME ${_target}
        SOURCES ${_suite} conformance/allocation_counter.cpp
        LABELS cpu conformance ${_dyng_algorithm}
        LIBRARIES dyng::testing
      )
    else()
      set(_target dyng_${_dyng_algorithm}_conformance_cuda_tests)
      dyng_add_test(
        NAME ${_target}
        SOURCES ${_suite} conformance/allocation_counter.cpp
        LABELS gpu conformance ${_dyng_algorithm}
        LIBRARIES dyng::testing
        TEST_PREFIX "cuda."
      )
      target_compile_definitions(${_target} PRIVATE DYNG_TEST_CUDA=1)
      dyng_set_cuda_options(${_target})  # the streams of C12
    endif()
    target_compile_definitions(${_target} PRIVATE ${_dyng_algorithm_defines})
    target_include_directories(${_target} PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  endforeach()
endforeach()

# The registry and the kit agree: every algorithm of dyng::algorithms() has test_traits here.
dyng_add_test(
  NAME dyng_conformance_registry_tests
  SOURCES conformance/registry_test.cpp
  LABELS cpu conformance
)
target_compile_definitions(dyng_conformance_registry_tests PRIVATE ${_dyng_algorithm_defines})
target_include_directories(dyng_conformance_registry_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

# The scaffold check (PLAN Section 4.8): scripts/new_algorithm.py generates a throwaway algorithm of
# each family in a copy of the source tree, which is configured, built and must pass its
# conformance kit (ci/scaffold_check.sh; a nested build of a few minutes). Label scaffold only (not
# cpu): run it with `ctest -L scaffold` or through ci/check.sh (step scaffold) and cpu.yml.
find_program(DYNG_BASH bash)
find_package(Git QUIET)
if(DYNG_BASH AND GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  add_test(NAME scaffold.new_algorithm COMMAND "${DYNG_BASH}"
                                               "${PROJECT_SOURCE_DIR}/ci/scaffold_check.sh")
  set_tests_properties(scaffold.new_algorithm PROPERTIES LABELS "scaffold" TIMEOUT 3600)
endif()

# Algorithms may add their own test folder (scripts/new_algorithm.py writes one for its
# <name>_test.cpp); sssp and cycle_count are listed above.
foreach(_dyng_algorithm IN LISTS _dyng_algorithms)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/${_dyng_algorithm}/CMakeLists.txt")
    add_subdirectory(algorithms/${_dyng_algorithm})
  endif()
endforeach()

add_subdirectory(api)

dyng_write_gpu_test_list()
