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
    core/update_dispatch_test.cpp
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
# Host allocation failures in header-inline code (its own executable: it replaces operator new).
dyng_add_test(
  NAME dyng_allocation_failure_tests
  SOURCES graph/allocation_failure_test.cpp
  LABELS cpu graph
)
dyng_add_test(
  NAME dyng_io_tests
  SOURCES io/io_test.cpp io/util_test.cpp io/edge_list_io_test.cpp io/hypergraph_io_test.cpp
          io/batch_binary_test.cpp
  LABELS cpu io
)
if(DYNG_ENABLE_CUDA)
  # batch_reader with CUDA resources: the pinned staging and the device views (M8b).
  dyng_add_test(
    NAME dyng_io_cuda_tests
    SOURCES io/batch_binary_cuda_test.cpp
    LABELS gpu io
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_io_cuda_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
endif()
# The reader fuzzers' findings (its own executable: it replaces operator new to bound the
# allocations; cpp/fuzz/README.md).
dyng_add_test(
  NAME dyng_io_fuzz_regression_tests
  SOURCES io/fuzz_regression_test.cpp
  LABELS cpu io fuzz
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
          generators/hypergraph_generators_test.cpp
  LABELS cpu generators
)
# Parity with CycleEnumeration-GPU@0a976ad on the TUDataset graphs (outside the repository; the
# test skips when they are missing). Label parity, not cpu: run with `ctest -L parity`.
set(_dyng_private_tests dyng_graph_tests dyng_io_tests dyng_io_fuzz_regression_tests dyng_framework_tests
                         dyng_generators_tests
)
if(TARGET dyng_cycle_count)
  dyng_add_test(
    NAME dyng_cycle_enum_parity_tests
    SOURCES parity/cycle_enum_datasets_test.cpp parity/cycle_count_datasets_test.cpp
    LABELS parity cycle_count
  )
  list(APPEND _dyng_private_tests dyng_cycle_enum_parity_tests)
endif()
# The CBST traces of ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159 (M8a, ADR 0036): the host-only
# trace reader and state tracker that the store replays build on, and its test on the committed
# traces (cpp/tests/data/goldens/cbst_trace).
add_library(dyng_cbst_trace STATIC parity/escher/cbst_trace.cpp parity/escher/escher_dump.cpp)
target_include_directories(dyng_cbst_trace PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(dyng_cbst_trace PUBLIC cxx_std_17)
dyng_set_warnings(dyng_cbst_trace)
dyng_enable_sanitizers(dyng_cbst_trace)
dyng_add_test(
  NAME dyng_cbst_trace_tests
  SOURCES parity/escher/cbst_trace_test.cpp parity/escher/escher_dump_test.cpp
  LABELS cpu parity cbst
  LIBRARIES dyng_cbst_trace
)
list(APPEND _dyng_private_tests dyng_cbst_trace_tests)
# The CBST host model (dyng::testing::cbst_host_model) and the trace replay comparator (M8a step
# 2, ADR 0037): the comparator library, the replay tool that parity/compare.py cbst_trace drives
# over the golden sets, the model's tests (the cbst_test layout sizes, the perturbed traces, the
# replay of the committed subset), the replay of the large sets in $DYNG_SCRATCH (skipped when
# they are absent) and the include-graph / link check of the model.
add_library(dyng_cbst_compare STATIC parity/escher/cbst_compare.cpp)
target_link_libraries(dyng_cbst_compare PUBLIC dyng_cbst_trace dyng::testing)
target_compile_features(dyng_cbst_compare PUBLIC cxx_std_17)
dyng_set_warnings(dyng_cbst_compare)
dyng_enable_sanitizers(dyng_cbst_compare)
add_executable(dyng_cbst_replay parity/escher/cbst_replay_main.cpp)
target_link_libraries(dyng_cbst_replay PRIVATE dyng_cbst_compare)
dyng_set_warnings(dyng_cbst_replay)
dyng_enable_sanitizers(dyng_cbst_replay)
dyng_add_test(
  NAME dyng_cbst_model_tests
  SOURCES parity/escher/cbst_host_model_test.cpp parity/escher/cbst_replay_test.cpp
  LABELS cpu parity cbst
  LIBRARIES dyng_cbst_compare
)
list(APPEND _dyng_private_tests dyng_cbst_model_tests)
dyng_add_test(
  NAME dyng_cbst_model_large_tests
  SOURCES parity/escher/cbst_replay_large_test.cpp
  LABELS parity cbst slow
  LIBRARIES dyng_cbst_compare
)
list(APPEND _dyng_private_tests dyng_cbst_model_large_tests)
# The source rules of the CBST store (M8a criterion 9): no process exit, print, verbose(),
# cudaDeviceSynchronize, raw cudaMalloc, Thrust without the resources' policy or default-stream
# launch in cpp/src/hypergraph, and no public header that exposes the store.
add_test(NAME cbst_source_rules
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}/cpp -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/hypergraph/escher/check_source_rules.cmake"
)
set_tests_properties(cbst_source_rules PROPERTIES LABELS "cpu;cbst")
# The same rules over the triad_count port (M9a, ADR 0053): ESCHER-GPU's wedge engine after its
# mechanical classes (errors, memory, streams).
add_test(NAME triad_count_source_rules
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}/cpp
                 -DSOURCE_SUBDIRS=algorithms/triad_count -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/hypergraph/escher/check_source_rules.cmake"
)
set_tests_properties(triad_count_source_rules PROPERTIES LABELS "cpu;triad_count")
# The launch rule itself: a launch on a literal default stream is reported (M8a review).
set(_cbst_rules_fixture "${CMAKE_CURRENT_BINARY_DIR}/cbst_source_rules_fixture")
file(MAKE_DIRECTORY "${_cbst_rules_fixture}/src/hypergraph" "${_cbst_rules_fixture}/include/dyng")
file(WRITE "${_cbst_rules_fixture}/src/hypergraph/launch.cu"
     "void f(cudaStream_t s) {\n"
     "  k<<<1, 32, 0, 0>>>();\n"
     "  k<<<1, 32, 0, nullptr>>>();\n"
     "  k<<<1, 32, 0, cudaStreamLegacy>>>();\n"
     "  k<<<1, 32>>>();\n"
     "  k<<<1, 32, 0, s>>>();\n"
     "}\n"
)
add_test(NAME cbst_source_rules_default_stream
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${_cbst_rules_fixture} -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/hypergraph/escher/check_source_rules.cmake"
)
set_tests_properties(
  cbst_source_rules_default_stream
  PROPERTIES LABELS "cpu;cbst" PASS_REGULAR_EXPRESSION
             "launch.cu:2: launch on the default stream: 1, 32, 0, 0.*launch.cu:3: .*launch.cu:4: .*launch.cu:5: "
             FAIL_REGULAR_EXPRESSION "launch.cu:6: "
)
# The host parts of the store (its payload policy, flatten_rows), on every build.
dyng_add_test(
  NAME dyng_cbst_host_tests
  SOURCES hypergraph/escher/cbst_host_test.cpp
  LABELS cpu cbst
)
list(APPEND _dyng_private_tests dyng_cbst_host_tests)
# The dynG CBST store (cpp/src/hypergraph/escher, M8a step 3, ADR 0038) as a store of the replay:
# the device adapter, the replays of the committed traces on the device (labels gpu, parity, cbst)
# and dyng_cbst_replay --store device.
if(DYNG_ENABLE_CUDA)
  add_library(dyng_cbst_device STATIC parity/escher/cbst_device_adapter.cpp)
  target_link_libraries(dyng_cbst_device PUBLIC dyng_cbst_compare)
  target_include_directories(dyng_cbst_device PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_warnings(dyng_cbst_device)
  dyng_enable_sanitizers(dyng_cbst_device)
  dyng_set_cuda_options(dyng_cbst_device)
  target_link_libraries(dyng_cbst_replay PRIVATE dyng_cbst_device)
  target_compile_definitions(dyng_cbst_replay PRIVATE DYNG_CBST_DEVICE_STORE=1)
  dyng_add_test(
    NAME dyng_cbst_device_tests
    SOURCES parity/escher/cbst_device_replay_test.cpp
    LABELS gpu parity cbst
    LIBRARIES dyng_cbst_device
  )
  list(APPEND _dyng_private_tests dyng_cbst_device_tests)
  # The store's own tests: ESCHER-GPU's cbst_test layout sizes on the construction kernels, the
  # dynG additions, the error policy, the streams, MOSP_ESCHER's merged behaviours (M8a step 4).
  dyng_add_test(
    NAME dyng_cbst_store_tests
    SOURCES hypergraph/escher/cbst_store_test.cpp hypergraph/escher/cbst_layout_test.cu
            hypergraph/escher/cbst_merge_test.cpp hypergraph/escher/cbst_contract_test.cpp
    LABELS gpu cbst
    LIBRARIES dyng::testing
  )
  dyng_set_cuda_options(dyng_cbst_store_tests)
  list(APPEND _dyng_private_tests dyng_cbst_store_tests)
  # The explicit compact() after every operation of both originals' traces, in lockstep with the
  # host model (M8a step 4): the committed traces, and the small golden sets of $DYNG_SCRATCH
  # (slow; skipped without them).
  dyng_add_test(
    NAME dyng_cbst_compact_tests
    SOURCES parity/escher/cbst_compact_test.cpp
    LABELS gpu parity cbst
    LIBRARIES dyng_cbst_device
  )
  list(APPEND _dyng_private_tests dyng_cbst_compact_tests)
  dyng_add_test(
    NAME dyng_cbst_compact_large_tests
    SOURCES parity/escher/cbst_compact_test.cpp
    LABELS gpu parity cbst slow
    LIBRARIES dyng_cbst_device
  )
  target_compile_definitions(dyng_cbst_compact_large_tests PRIVATE DYNG_CBST_COMPACT_LARGE=1)
  list(APPEND _dyng_private_tests dyng_cbst_compact_large_tests)
  # MOSP_ESCHER's 9 CBST scenarios and its smoke test under every preset, in lockstep with the
  # host model (M8a step 5).
  dyng_add_test(
    NAME dyng_cbst_scenario_tests
    SOURCES parity/escher/cbst_scenarios_test.cpp
    LABELS gpu parity cbst
    LIBRARIES dyng_cbst_device
  )
  dyng_set_cuda_options(dyng_cbst_scenario_tests)
  list(APPEND _dyng_private_tests dyng_cbst_scenario_tests)
  # The internal graph-on-CBST test (PLAN 6.1 D8, M8a step 5): a graph on two stores against
  # dyng::graph's apply, under every preset, and MOSP_ESCHER's test_dynamicgraph_roundtrip trace.
  dyng_add_test(
    NAME dyng_cbst_graph_tests
    SOURCES parity/escher/cbst_graph_test.cpp
    LABELS gpu parity cbst
    LIBRARIES dyng_cbst_device
  )
  dyng_set_cuda_options(dyng_cbst_graph_tests)
  list(APPEND _dyng_private_tests dyng_cbst_graph_tests)
  # The presets of M8a step 5 (ADR 0039): random operation sequences, requested keys and best-fit
  # ties under every preset, in lockstep with the host model.
  dyng_add_test(
    NAME dyng_cbst_policy_tests
    SOURCES parity/escher/cbst_policy_test.cpp
    LABELS gpu parity cbst
    LIBRARIES dyng_cbst_device
  )
  dyng_set_cuda_options(dyng_cbst_policy_tests)
  list(APPEND _dyng_private_tests dyng_cbst_policy_tests)
  # Kernel-path coverage on ESCHER-GPU's skewed and medium traces (in $DYNG_SCRATCH; skipped
  # without them).
  dyng_add_test(
    NAME dyng_cbst_coverage_tests
    SOURCES parity/escher/cbst_device_coverage_test.cpp
    LABELS gpu parity cbst slow
    LIBRARIES dyng_cbst_device
  )
  list(APPEND _dyng_private_tests dyng_cbst_coverage_tests)
  # The early store-memory reading on escher_gpu_coauth50k (M8a criterion 15; in $DYNG_SCRATCH).
  dyng_add_test(
    NAME dyng_cbst_memory_tests
    SOURCES parity/escher/cbst_device_memory_test.cpp
    LABELS gpu parity cbst slow
    LIBRARIES dyng_cbst_device
  )
  dyng_set_cuda_options(dyng_cbst_memory_tests)
  list(APPEND _dyng_private_tests dyng_cbst_memory_tests)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_NM)
  add_test(
    NAME cbst_model_isolation
    COMMAND
      "${CMAKE_COMMAND}" -DCXX=${CMAKE_CXX_COMPILER} -DNM=${CMAKE_NM} -DREPO=${PROJECT_SOURCE_DIR}
      -DSOURCE=${PROJECT_SOURCE_DIR}/cpp/src/testing/cbst_host_model.cpp
      "-DOBJECT=$<TARGET_OBJECTS:dyng_testing_cbst_model>" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/parity/escher/check_model_isolation.cmake"
  )
  set_tests_properties(cbst_model_isolation PROPERTIES LABELS "cpu;parity;cbst")
  # The line-graph oracle (M8b, ADR 0042): MOSP_ESCHER's algorithm, sharing no code with the
  # container's line graph (criterion 6).
  add_test(
    NAME line_graph_oracle_isolation
    COMMAND
      "${CMAKE_COMMAND}" -DCXX=${CMAKE_CXX_COMPILER} -DNM=${CMAKE_NM} -DREPO=${PROJECT_SOURCE_DIR}
      -DSOURCE=${PROJECT_SOURCE_DIR}/cpp/src/testing/line_graph_oracle.cpp
      -DHEADER=${PROJECT_SOURCE_DIR}/cpp/include/dyng/testing/line_graph_oracle.hpp
      "-DOBJECT=$<TARGET_OBJECTS:dyng_testing_line_graph_oracle>" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/parity/escher/check_model_isolation.cmake"
  )
  set_tests_properties(line_graph_oracle_isolation PROPERTIES LABELS "cpu;hypergraph")
  # The triad oracle (M9a criterion 4, ADR 0049): ESCHER-GPU's CPU reference algorithm with its own
  # class map, sharing no code with triad_count, its engines or the container. The preprocessor
  # runs with the flags the build compiles it with (OpenMP, the target's definitions).
  set(_triad_reference_flags "")
  if(DYNG_ENABLE_OPENMP)
    separate_arguments(_triad_reference_flags UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
  endif()
  list(JOIN _triad_reference_flags "|" _triad_reference_flags)
  set(_triad_reference_defs
      "$<TARGET_PROPERTY:dyng_testing_triad_reference,COMPILE_DEFINITIONS>"
  )
  add_test(
    NAME triad_reference_isolation
    COMMAND
      "${CMAKE_COMMAND}" -DCXX=${CMAKE_CXX_COMPILER} -DNM=${CMAKE_NM} -DREPO=${PROJECT_SOURCE_DIR}
      -DSOURCE=${PROJECT_SOURCE_DIR}/cpp/src/testing/triad_reference.cpp
      -DHEADER=${PROJECT_SOURCE_DIR}/cpp/include/dyng/testing/triad_reference.hpp
      "-DFLAGS=${_triad_reference_flags}|-D$<JOIN:${_triad_reference_defs},|-D>"
      "-DOBJECT=$<TARGET_OBJECTS:dyng_testing_triad_reference>" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/parity/escher/check_model_isolation.cmake"
  )
  set_tests_properties(triad_reference_isolation PROPERTIES LABELS "cpu;testing;triad_count")
endif()
# The hypergraph container (M8b, ADR 0041): construction, apply semantics, guarantees,
# placement, the host id allocator against the CBST host model, random streams in lockstep with
# the model on every backend. The shared suites run on the host backends here and on cuda in
# dyng_hypergraph_cuda_tests.
dyng_add_test(
  NAME dyng_hypergraph_tests
  SOURCES hypergraph/hypergraph_test.cpp hypergraph/hypergraph_random_test.cpp
          hypergraph/hypergraph_store_test.cpp hypergraph/hypergraph_line_graph_test.cpp
          hypergraph/hypergraph_update_test.cpp
  LABELS cpu hypergraph
  LIBRARIES dyng_cbst_compare
)
list(APPEND _dyng_private_tests dyng_hypergraph_tests)
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_hypergraph_cuda_tests
    SOURCES hypergraph/hypergraph_test.cpp hypergraph/hypergraph_random_test.cpp
            hypergraph/hypergraph_line_graph_test.cpp hypergraph/hypergraph_cuda_test.cpp
            hypergraph/hypergraph_update_test.cpp framework/fake_hyper_kernels.cu
            hypergraph/stream_probe_kernels.cu
    LABELS gpu hypergraph
    LIBRARIES dyng_cbst_compare
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_hypergraph_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  dyng_set_cuda_options(dyng_hypergraph_cuda_tests)  # the non-blocking streams of the C12 test
  list(APPEND _dyng_private_tests dyng_hypergraph_cuda_tests)
endif()
# Container-level id parity against both originals' batch records (M8b, criterion 4; ADR 0041):
# the replay library, the tool parity/compare.py hypergraph_ids drives over the golden sets, and
# its test on the committed traces (hosted CI: the CPU backends).
add_library(dyng_hypergraph_ids STATIC parity/escher/hypergraph_ids.cpp)
target_link_libraries(dyng_hypergraph_ids PUBLIC dyng_cbst_compare dyng::dyng)
target_include_directories(dyng_hypergraph_ids PUBLIC "${PROJECT_SOURCE_DIR}/cpp/src")
target_compile_features(dyng_hypergraph_ids PUBLIC cxx_std_17)
dyng_set_warnings(dyng_hypergraph_ids)
dyng_enable_sanitizers(dyng_hypergraph_ids)
add_executable(dyng_hypergraph_ids_tool parity/escher/hypergraph_ids_main.cpp)
set_target_properties(dyng_hypergraph_ids_tool PROPERTIES OUTPUT_NAME dyng_hypergraph_ids)
target_link_libraries(dyng_hypergraph_ids_tool PRIVATE dyng_hypergraph_ids)
dyng_set_warnings(dyng_hypergraph_ids_tool)
dyng_enable_sanitizers(dyng_hypergraph_ids_tool)
dyng_add_test(
  NAME dyng_hypergraph_ids_tests
  SOURCES parity/escher/hypergraph_ids_test.cpp
  LABELS cpu parity hypergraph
  LIBRARIES dyng_hypergraph_ids
)
list(APPEND _dyng_private_tests dyng_hypergraph_ids_tests)
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_hypergraph_ids_cuda_tests
    SOURCES parity/escher/hypergraph_ids_test.cpp
    LABELS gpu parity hypergraph
    LIBRARIES dyng_hypergraph_ids
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_hypergraph_ids_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  list(APPEND _dyng_private_tests dyng_hypergraph_ids_cuda_tests)
endif()
# The triad_count replay of ESCHER-GPU's golden set (M9a step 2, ADR 0050): the replay library
# (a case's .hg and batch files through dyng::hypergraph, by key and by content, a counter after the
# build and every batch) and the tool parity/compare.py triad_count drives over the set.
add_library(dyng_triad_replay_lib STATIC parity/escher/triad_replay.cpp)
target_link_libraries(dyng_triad_replay_lib PUBLIC dyng_cbst_trace dyng::testing dyng::dyng)
target_compile_features(dyng_triad_replay_lib PUBLIC cxx_std_17)
dyng_set_warnings(dyng_triad_replay_lib)
dyng_enable_sanitizers(dyng_triad_replay_lib)
# --counter algorithm needs triad_count in the build (DYNG_ALGORITHMS).
if(TARGET dyng_triad_count)
  # The algorithm counter reads the result's launch counters (internal: cpp/src).
  target_compile_definitions(dyng_triad_replay_lib PRIVATE DYNG_TRIAD_REPLAY_ALGORITHM=1)
  target_include_directories(dyng_triad_replay_lib PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
else()
  target_compile_definitions(dyng_triad_replay_lib PRIVATE DYNG_TRIAD_REPLAY_ALGORITHM=0)
endif()
add_executable(dyng_triad_replay parity/escher/triad_replay_main.cpp)
target_link_libraries(dyng_triad_replay PRIVATE dyng_triad_replay_lib)
dyng_set_warnings(dyng_triad_replay)
dyng_enable_sanitizers(dyng_triad_replay)
dyng_add_test(
  NAME dyng_triad_replay_tests
  SOURCES parity/escher/triad_replay_test.cpp
  LABELS cpu parity triad_count
  LIBRARIES dyng_triad_replay_lib
)
if(TARGET dyng_triad_count)
  target_compile_definitions(dyng_triad_replay_tests PRIVATE DYNG_TRIAD_REPLAY_ALGORITHM=1)
else()
  target_compile_definitions(dyng_triad_replay_tests PRIVATE DYNG_TRIAD_REPLAY_ALGORITHM=0)
endif()
list(APPEND _dyng_private_tests dyng_triad_replay_tests)
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
  "mixed_tiers_without_engine_choice|-DDYNG_CF_MIXED_TIERS|must say where its fused engine runs"
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

# Compile-failure tests of invariant I2 (PLAN 4.5.5: count_delta requires an ownership rule, with no
# default; ADR 0055): operators::count_delta and count_owned without a rule, with a value that is
# not one, and the triad host engine's phase without one. The control case must compile.
foreach(
  _case
  "control||"
  "count_delta_without_ownership|-DDYNG_CF_DELTA_NO_OWNERSHIP|count_delta needs an ownership rule"
  "count_owned_without_ownership|-DDYNG_CF_OWNED_NO_OWNERSHIP|count_delta needs an ownership rule"
  "not_an_ownership_rule|-DDYNG_CF_NOT_AN_OWNERSHIP|must be an ownership rule"
  "triad_phase_without_ownership|-DDYNG_CF_TRIAD_NO_OWNERSHIP|count_phase needs an ownership rule"
)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME operators.count_delta.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   ${_dyng_framework_cf_includes}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/count_delta_ownership.cpp
  )
  if(_message STREQUAL "")
    set_tests_properties(operators.count_delta.static_assert.${_name} PROPERTIES LABELS
                                                                         "cpu;operators")
  else()
    set_tests_properties(
      operators.count_delta.static_assert.${_name}
      PROPERTIES LABELS "cpu;operators" PASS_REGULAR_EXPRESSION "${_message}"
    )
  endif()
endforeach()

# The operators (cpp/src/operators): count_delta on the host executors and, in CUDA builds, on
# cuda_exec (ADR 0055).
dyng_add_test(
  NAME dyng_operators_tests
  SOURCES operators/count_delta_test.cpp
  LABELS cpu operators
)
target_include_directories(dyng_operators_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
if(DYNG_ENABLE_OPENMP)
  target_link_libraries(dyng_operators_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_operators_cuda_tests
    SOURCES operators/count_delta_cuda_test.cu
    LABELS gpu operators
    TEST_PREFIX cuda.
  )
  target_include_directories(dyng_operators_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_operators_cuda_tests)
endif()

# Compile-failure tests: sssp on an unsupported graph type stops at the static_assert that names the
# supported types (as cycle_count's), not at a link error; so does another distance type.
foreach(_case "compute;;dyng::sssp supports graph<" "update;-DDYNG_COMPILE_FAIL_UPDATE;dyng::sssp supports graph<"
        "multi_update;-DDYNG_COMPILE_FAIL_MULTI;dyng::sssp supports graph<"
        "distance_type;-DDYNG_COMPILE_FAIL_DISTANCE;distance_t is std::int64_t in 0.1"
)
  list(GET _case 0 _name)
  list(GET _case 1 _define)
  list(GET _case 2 _message)
  add_test(NAME sssp.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/sssp_unsupported_types.cpp
  )
  set_tests_properties(
    sssp.static_assert.${_name} PROPERTIES LABELS "cpu;sssp" PASS_REGULAR_EXPRESSION "${_message}"
  )
endforeach()

# Compile-failure tests of dyng::update()'s argument checks (plain-English static_asserts instead
# of an error inside the header): the owning batch instead of its view, and a type that is not a
# container of dyng::update(). The control case must compile.
foreach(
  _case
  "control||"
  "owning_batch|-DDYNG_CF_OWNING_BATCH|the batch must be the container's batch view"
  "not_a_container|-DDYNG_CF_NOT_A_CONTAINER|the container is not one dyng::update.. drives"
)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME update.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/update_arguments.cpp
  )
  if(_message STREQUAL "")
    set_tests_properties(update.static_assert.${_name} PROPERTIES LABELS "cpu;core")
  else()
    set_tests_properties(
      update.static_assert.${_name} PROPERTIES LABELS "cpu;core" PASS_REGULAR_EXPRESSION
                                               "${_message}"
    )
  endif()
endforeach()

# Compile-failure tests of dyng::update() over hypergraphs (ADR 0043): the owning batch, a batch
# view of the other container, a hypergraph algorithm's result with a graph (the plain-English
# static_assert of its update_traits, detail::is_hypergraph_v) and a graph algorithm's result with
# a hypergraph (sssp's update_traits needs the graph's edge type). The control case must compile.
foreach(
  _case
  "control||"
  "owning_batch|-DDYNG_CF_HYPER_OWNING_BATCH|the batch must be the container's batch view"
  "edge_batch_with_hypergraph|-DDYNG_CF_HYPER_EDGE_BATCH|the batch must be the container's batch view"
  "hyperedge_batch_with_graph|-DDYNG_CF_GRAPH_HYPER_BATCH|the batch must be the container's batch view"
  "hypergraph_result_with_graph|-DDYNG_CF_HYPER_RESULT_ON_GRAPH|a hypergraph algorithm.*takes a dyng::hypergraph"
  "graph_result_with_hypergraph|-DDYNG_CF_GRAPH_RESULT_ON_HYPER|edge_type"
)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME hypergraph.update.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/hypergraph_update_arguments.cpp
  )
  if(_message STREQUAL "")
    set_tests_properties(hypergraph.update.static_assert.${_name} PROPERTIES LABELS
                                                                     "cpu;hypergraph")
  else()
    set_tests_properties(
      hypergraph.update.static_assert.${_name} PROPERTIES LABELS "cpu;hypergraph"
                                                          PASS_REGULAR_EXPRESSION "${_message}"
    )
  endif()
endforeach()

# Compile-failure tests of the conformance kit's registration rules (invariant I8, PLAN Section
# 8.2): test_traits that break a rule stop DYNG_CONFORMANCE_SUITE at a static_assert with a
# plain-English message (conformance.hpp, registration_check). The control case must compile.
set(_dyng_kit_cf_includes
    -I${PROJECT_SOURCE_DIR}/cpp/tests -I${PROJECT_SOURCE_DIR}/cpp/src
    -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR}
    "-I$<JOIN:$<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:GTest::gmock,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
)
# name|define|expected message (a regular expression)
foreach(
  _case
  "control||"
  "no_traits|-DDYNG_CF_NO_TRAITS|the algorithm has no test_traits specialisation"
  "no_compute|-DDYNG_CF_NO_COMPUTE|cannot be registered without compute"
  "no_update|-DDYNG_CF_NO_UPDATE|test_traits::update<graph_t>.res, g, batch, result. must"
  "no_oracle_kind|-DDYNG_CF_NO_ORACLE|must declare its oracle kind"
  "no_determinism_level|-DDYNG_CF_NO_LEVEL|must declare its determinism level"
  "stats_without_batch|-DDYNG_CF_STATS_NO_BATCH|contain `apply_summary batch`"
  "tolerance_without_compare|-DDYNG_CF_TOLERANCE_NO_CMP|determinism::tolerance must provide"
  "reference_without_near_reference|-DDYNG_CF_REFERENCE_NO_NEAR|oracle_kind::reference needs"
)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME conformance.registration.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define}
                   ${_dyng_kit_cf_includes}
                   ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/conformance_registration.cpp
           COMMAND_EXPAND_LISTS
  )
  if(_message STREQUAL "")
    set_tests_properties(conformance.registration.${_name} PROPERTIES LABELS "cpu;conformance")
  else()
    set_tests_properties(
      conformance.registration.${_name} PROPERTIES LABELS "cpu;conformance"
                                                   PASS_REGULAR_EXPRESSION "${_message}"
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
      target_link_libraries(${_lib} INTERFACE ${DYNG_CUDART_TARGET})
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

# The mutants' copies of the hypergraph module are compiled for one architecture (the build's
# first, real and virtual, so the PTX also runs on newer GPUs; `native` stays native): 12 copies
# for the 7 release architectures took 49.5 of the 174 CPU-minutes of the compile-only ci-cuda12
# preset (M8b review), and a mutant only needs to run on the machine that runs its CTest.
function(dyng_mutant_cuda_architectures target)
  if(NOT DYNG_ENABLE_CUDA OR NOT CMAKE_CUDA_ARCHITECTURES)
    return()
  endif()
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _first)
  string(REGEX MATCH "^[0-9]+" _number "${_first}")
  if(_number)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${_number}")
  endif()
endfunction()

# CBST mutation tests (PLAN Section 8.4; M8a acceptance criterion 17): the 7 recorded mutations of
# the ESCHER CBST store (MOSP_ESCHER@4b86159's #4 and #5 and the five rules of the store's
# README) must make the store's suites fail. As for cycle_count, each mutant is a copy of libdyng
# whose hypergraph module is rebuilt with DYNG_MUTATION_CBST_<NAME> (the hooks are in
# cpp/src/hypergraph/escher; the graph-on-CBST hook is in its test), linked with the replays of
# the committed traces of both originals, the lockstep tests of the presets against the host
# model and the graph-on-CBST test; the CTest cbst.mutation.<name> passes when that suite fails,
# and the control copy must pass. Labels gpu;mutation;cbst (CUDA builds).
option(DYNG_CBST_MUTATION_TESTS "Build the mutation tests of the CBST store" ${DYNG_MUTATION_TESTS})
if(DYNG_CBST_MUTATION_TESTS AND DYNG_ENABLE_CUDA AND TARGET dyng_hypergraph)
  get_property(_dyng_modules GLOBAL PROPERTY DYNG_MODULE_TARGETS)
  get_property(_dyng_hg_relative GLOBAL PROPERTY DYNG_MODULE_SOURCES_dyng_hypergraph)
  set(_dyng_hg_sources)
  foreach(_source IN LISTS _dyng_hg_relative)  # dyng_add_module's paths are relative to cpp/
    if(NOT IS_ABSOLUTE "${_source}")
      set(_source "${PROJECT_SOURCE_DIR}/cpp/${_source}")
    endif()
    list(APPEND _dyng_hg_sources "${_source}")
  endforeach()
  set(_dyng_hg_other_objects)
  foreach(_module IN LISTS _dyng_modules)
    if(NOT _module STREQUAL "dyng_hypergraph")
      list(APPEND _dyng_hg_other_objects $<TARGET_OBJECTS:${_module}>)
    endif()
  endforeach()
  # name|define|expected verdict
  set(_dyng_cbst_mutants
      "control||pass"
      "unfill_drops_last|DYNG_MUTATION_CBST_UNFILL_DROPS_LAST|fail"
      "graph_own_ids|DYNG_MUTATION_CBST_GRAPH_OWN_IDS|fail"
      "reversed_ties|DYNG_MUTATION_CBST_REVERSED_TIES|fail"
      "surplus_rule1|DYNG_MUTATION_CBST_SURPLUS_RULE1|fail"
      "keep_dead_slots|DYNG_MUTATION_CBST_KEEP_DEAD_SLOTS|fail"
      "compaction_padding|DYNG_MUTATION_CBST_COMPACTION_PADDING|fail"
      "requested_keys|DYNG_MUTATION_CBST_REQUESTED_KEYS|fail"
  )
  foreach(_entry IN LISTS _dyng_cbst_mutants)
    string(REPLACE "|" ";" _mutant "${_entry}")
    list(GET _mutant 0 _name)
    list(GET _mutant 1 _define)
    list(GET _mutant 2 _expect)
    set(_lib dyng_cbst_mutant_${_name})
    add_library(${_lib}_hypergraph OBJECT ${_dyng_hg_sources})
    add_library(${_lib} STATIC ${_dyng_hg_other_objects} $<TARGET_OBJECTS:${_lib}_hypergraph>)
    set(_exe dyng_cbst_mutant_${_name}_tests)
    add_executable(
      ${_exe}
      parity/escher/cbst_device_replay_test.cpp
      parity/escher/cbst_policy_test.cpp
      parity/escher/cbst_graph_test.cpp
      parity/escher/cbst_trace.cpp
      parity/escher/escher_dump.cpp
      parity/escher/cbst_compare.cpp
      parity/escher/cbst_device_adapter.cpp
      $<TARGET_OBJECTS:dyng_testing_cbst_model>
    )
    foreach(_target ${_lib}_hypergraph ${_lib} ${_exe})
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
      dyng_set_cuda_options(${_target})
    endforeach()
    set_target_properties(${_lib}_hypergraph PROPERTIES POSITION_INDEPENDENT_CODE ON)
    dyng_mutant_cuda_architectures(${_lib}_hypergraph)
    target_link_libraries(${_lib} INTERFACE Threads::Threads ${DYNG_CUDART_TARGET})
    if(DYNG_ENABLE_OPENMP)
      target_link_libraries(${_lib} INTERFACE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(${_exe} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${_exe} PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
    target_link_libraries(${_exe} PRIVATE ${_lib} dyng_test_support GTest::gtest GTest::gtest_main)
    add_test(NAME cbst.mutation.${_name}
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                     -DNAME=cbst.${_name} -P "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
    )
    set_tests_properties(
      cbst.mutation.${_name} PROPERTIES LABELS "gpu;mutation;cbst" ENVIRONMENT
                                        "${DYNG_TEST_ENVIRONMENT}"
    )
  endforeach()
endif()

# The hypergraph container's mutations (M8b criterion 20; PLAN 8.4): a copy of libdyng whose
# hypergraph module is rebuilt with DYNG_MUTATION_HYPERGRAPH_<NAME> (MOSP_ESCHER's mutation #6,
# the h2v insert payload not deduplicated; the host allocator's best-fit tie rule reversed; the
# line graph's mirrored additions dropped, M8b step 2),
# linked with the container's suites on the host backends (the lockstep with the CBST host model,
# the random streams, the allocator against the model, the replays of the committed batch
# extracts); the CTest hypergraph.mutation.<name> passes when that suite fails, the control
# copy must pass. Labels cpu;mutation;hypergraph (every build: the host backends only).
option(DYNG_HYPERGRAPH_MUTATION_TESTS "Build the mutation tests of the hypergraph container"
       ${DYNG_MUTATION_TESTS}
)
if(DYNG_HYPERGRAPH_MUTATION_TESTS AND TARGET dyng_hypergraph)
  get_property(_dyng_modules GLOBAL PROPERTY DYNG_MODULE_TARGETS)
  get_property(_dyng_hg_relative GLOBAL PROPERTY DYNG_MODULE_SOURCES_dyng_hypergraph)
  set(_dyng_hg_sources)
  foreach(_source IN LISTS _dyng_hg_relative)
    if(NOT IS_ABSOLUTE "${_source}")
      set(_source "${PROJECT_SOURCE_DIR}/cpp/${_source}")
    endif()
    list(APPEND _dyng_hg_sources "${_source}")
  endforeach()
  set(_dyng_hg_other_objects)
  foreach(_module IN LISTS _dyng_modules)
    if(NOT _module STREQUAL "dyng_hypergraph")
      list(APPEND _dyng_hg_other_objects $<TARGET_OBJECTS:${_module}>)
    endif()
  endforeach()
  # name|define|expected verdict
  set(_dyng_hypergraph_mutants
      "control||pass"
      "h2v_not_deduplicated|DYNG_MUTATION_HYPERGRAPH_H2V_NOT_DEDUPLICATED|fail"
      "allocator_ties_reversed|DYNG_MUTATION_HYPERGRAPH_ALLOCATOR_TIES|fail"
      "mirror_additions_dropped|DYNG_MUTATION_HYPERGRAPH_MIRROR_ADDITIONS_DROPPED|fail"
  )
  foreach(_entry IN LISTS _dyng_hypergraph_mutants)
    string(REPLACE "|" ";" _mutant "${_entry}")
    list(GET _mutant 0 _name)
    list(GET _mutant 1 _define)
    list(GET _mutant 2 _expect)
    set(_lib dyng_hypergraph_mutant_${_name})
    add_library(${_lib}_hypergraph OBJECT ${_dyng_hg_sources})
    add_library(${_lib} STATIC ${_dyng_hg_other_objects} $<TARGET_OBJECTS:${_lib}_hypergraph>)
    set(_exe dyng_hypergraph_mutant_${_name}_tests)
    add_executable(
      ${_exe}
      hypergraph/hypergraph_test.cpp
      hypergraph/hypergraph_random_test.cpp
      hypergraph/hypergraph_store_test.cpp
      hypergraph/hypergraph_line_graph_test.cpp
      parity/escher/hypergraph_ids_test.cpp
      parity/escher/hypergraph_ids.cpp
      parity/escher/cbst_trace.cpp
      parity/escher/escher_dump.cpp
      parity/escher/cbst_compare.cpp
      $<TARGET_OBJECTS:dyng_testing_cbst_model>
      $<TARGET_OBJECTS:dyng_testing_line_graph_oracle>
    )
    foreach(_target ${_lib}_hypergraph ${_lib} ${_exe})
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
      dyng_set_cuda_options(${_target})
    endforeach()
    set_target_properties(${_lib}_hypergraph PROPERTIES POSITION_INDEPENDENT_CODE ON)
    dyng_mutant_cuda_architectures(${_lib}_hypergraph)
    target_link_libraries(${_lib} INTERFACE Threads::Threads)
    if(DYNG_ENABLE_CUDA)
      target_link_libraries(${_lib} INTERFACE ${DYNG_CUDART_TARGET})
    endif()
    if(DYNG_ENABLE_OPENMP)
      target_link_libraries(${_lib} INTERFACE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(${_exe} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${_exe} PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
    target_link_libraries(
      ${_exe} PRIVATE ${_lib} dyng_test_support GTest::gtest GTest::gmock GTest::gtest_main
    )
    add_test(NAME hypergraph.mutation.${_name}
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                     -DNAME=hypergraph.${_name} -P
                     "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
    )
    set_tests_properties(
      hypergraph.mutation.${_name} PROPERTIES LABELS "cpu;mutation;hypergraph" ENVIRONMENT
                                              "${DYNG_TEST_ENVIRONMENT}"
    )
  endforeach()
endif()

# triad_count's mutations (M9a criterion 11; PLAN 8.4: a wrong bin sign, a skipped frontier
# filter, an unfill that drops its last value): a copy of libdyng whose triad_count module is
# rebuilt with DYNG_MUTATION_TRIAD_<NAME> (the hooks are in the code both engines share:
# engine/phases.hpp, engine/ownership.hpp and triad_count.cpp's triad_inserted_ids), linked with
# the triad tests on the host backends (triad_count.mutation.<name>, labels
# cpu;mutation;triad_count, every build) and, in CUDA builds, with the CUDA triad suite
# (triad_count.mutation.cuda.<name>, labels gpu;mutation;triad_count). The CTest passes when the
# suite fails; the control copy must pass. cbst_unfill_drops_last (M8a's CBST mutation, MOSP_ESCHER
# #4: the store's unfill ignores the last value of each removal list) rebuilds the hypergraph
# module instead and runs against the CUDA triad suite only. The mutants' CUDA code is compiled for
# one architecture (dyng_mutant_cuda_architectures).
option(DYNG_TRIAD_MUTATION_TESTS "Build the mutation tests of triad_count" ${DYNG_MUTATION_TESTS})
if(DYNG_TRIAD_MUTATION_TESTS AND TARGET dyng_triad_count AND TARGET dyng_hypergraph)
  get_property(_dyng_modules GLOBAL PROPERTY DYNG_MODULE_TARGETS)
  # name|define|module|expected verdict|host suite too
  set(_dyng_triad_mutants
      "control||dyng_triad_count|pass|host"
      "wrong_bin_sign|DYNG_MUTATION_TRIAD_WRONG_BIN_SIGN|dyng_triad_count|fail|host"
      "skip_frontier_filter_b|DYNG_MUTATION_TRIAD_SKIP_FRONTIER_FILTER_B|dyng_triad_count|fail|host"
      "skip_frontier_filter_k|DYNG_MUTATION_TRIAD_SKIP_FRONTIER_FILTER_K|dyng_triad_count|fail|host"
      "inserted_requested_ids|DYNG_MUTATION_TRIAD_INSERTED_REQUESTED_IDS|dyng_triad_count|fail|host"
      "type1_unbounded_keys|DYNG_MUTATION_TRIAD_TYPE1_UNBOUNDED_KEYS|dyng_triad_count|fail|host"
  )
  if(DYNG_ENABLE_CUDA)
    list(APPEND _dyng_triad_mutants
         "cbst_unfill_drops_last|DYNG_MUTATION_CBST_UNFILL_DROPS_LAST|dyng_hypergraph|fail|cuda"
    )
  endif()
  foreach(_entry IN LISTS _dyng_triad_mutants)
    string(REPLACE "|" ";" _mutant "${_entry}")
    list(GET _mutant 0 _name)
    list(GET _mutant 1 _define)
    list(GET _mutant 2 _module)
    list(GET _mutant 3 _expect)
    list(GET _mutant 4 _suites)
    get_property(_dyng_mutated_relative GLOBAL PROPERTY DYNG_MODULE_SOURCES_${_module})
    set(_dyng_mutated_sources)
    foreach(_source IN LISTS _dyng_mutated_relative)  # dyng_add_module's core paths are relative
      if(NOT IS_ABSOLUTE "${_source}")
        set(_source "${PROJECT_SOURCE_DIR}/cpp/${_source}")
      endif()
      list(APPEND _dyng_mutated_sources "${_source}")
    endforeach()
    set(_dyng_unmutated_objects)
    foreach(_other IN LISTS _dyng_modules)
      if(NOT _other STREQUAL _module)
        list(APPEND _dyng_unmutated_objects $<TARGET_OBJECTS:${_other}>)
      endif()
    endforeach()
    set(_lib dyng_triad_mutant_${_name})
    add_library(${_lib}_module OBJECT ${_dyng_mutated_sources})
    add_library(
      ${_lib} STATIC ${_dyng_unmutated_objects} $<TARGET_OBJECTS:${_lib}_module>
                     $<TARGET_OBJECTS:dyng_testing_triad_reference>
    )
    set(_exes)
    if(_suites STREQUAL "host")
      add_executable(dyng_triad_count_mutant_${_name} algorithms/triad_count/triad_count_test.cpp)
      list(APPEND _exes dyng_triad_count_mutant_${_name})
    endif()
    if(DYNG_ENABLE_CUDA)
      add_executable(
        dyng_triad_count_cuda_mutant_${_name}
        algorithms/triad_count/triad_count_test.cpp
        algorithms/triad_count/triad_count_kernels_test.cu framework/fake_hyper_kernels.cu
      )
      target_compile_definitions(dyng_triad_count_cuda_mutant_${_name} PRIVATE DYNG_TEST_CUDA=1)
      list(APPEND _exes dyng_triad_count_cuda_mutant_${_name})
    endif()
    foreach(_target ${_lib}_module ${_lib} ${_exes})
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
      dyng_set_cuda_options(${_target})
    endforeach()
    set_target_properties(${_lib}_module PROPERTIES POSITION_INDEPENDENT_CODE ON)
    dyng_mutant_cuda_architectures(${_lib}_module)
    target_link_libraries(${_lib} INTERFACE Threads::Threads)
    if(DYNG_ENABLE_CUDA)
      target_link_libraries(${_lib} INTERFACE ${DYNG_CUDART_TARGET})
    endif()
    if(DYNG_ENABLE_OPENMP)
      target_link_libraries(${_lib} INTERFACE OpenMP::OpenMP_CXX)
    endif()
    foreach(_exe ${_exes})
      target_link_libraries(
        ${_exe} PRIVATE ${_lib} dyng_test_support GTest::gtest GTest::gmock GTest::gtest_main
      )
      if(_exe MATCHES "_cuda_mutant_")
        dyng_mutant_cuda_architectures(${_exe})
        add_test(NAME triad_count.mutation.cuda.${_name}
                 COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                         -DNAME=triad_count.cuda.${_name} -P
                         "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
        )
        set_tests_properties(
          triad_count.mutation.cuda.${_name}
          PROPERTIES LABELS "gpu;mutation;triad_count" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}"
        )
      else()
        add_test(NAME triad_count.mutation.${_name}
                 COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:${_exe}> -DEXPECT=${_expect}
                         -DNAME=triad_count.${_name} -P
                         "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
        )
        set_tests_properties(
          triad_count.mutation.${_name} PROPERTIES LABELS "cpu;mutation;triad_count" ENVIRONMENT
                                                   "${DYNG_TEST_ENVIRONMENT}"
        )
      endif()
    endforeach()
  endforeach()
endif()

dyng_add_test(
  NAME dyng_testing_tests
  SOURCES testing/oracle_test.cpp testing/cycle_oracle_test.cpp
  LABELS cpu testing
  LIBRARIES dyng::testing
)
target_compile_definitions(dyng_testing_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}")
# The triad oracle (M9a criterion 4): its class map against ESCHER-GPU's table (a test-only
# fixture), one triple per reachable bin, hand cases, a brute force over all triples.
dyng_add_test(
  NAME dyng_triad_reference_tests
  SOURCES testing/triad_reference_test.cpp
  LABELS cpu testing triad_count
  LIBRARIES dyng::testing
)
# The oracle's planted mutation (parity/mutate.py, triad_reference_class_map): two classes of its
# representative list swapped; its tests must fail (testing.mutation.triad_reference_class_map
# passes when the suite fails; the control is dyng_triad_reference_tests). Labels
# cpu;mutation;testing;triad_count.
if(DYNG_MUTATION_TESTS)
  add_executable(
    dyng_triad_reference_mutant_tests testing/triad_reference_test.cpp
                                      "${PROJECT_SOURCE_DIR}/cpp/src/testing/triad_reference.cpp"
  )
  target_link_libraries(
    dyng_triad_reference_mutant_tests PRIVATE dyng_test_support GTest::gtest GTest::gmock
                                              GTest::gtest_main
  )
  target_include_directories(dyng_triad_reference_mutant_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/include")
  target_compile_features(dyng_triad_reference_mutant_tests PRIVATE cxx_std_17)
  target_compile_definitions(
    dyng_triad_reference_mutant_tests PRIVATE DYNG_MUTATION_TRIAD_REFERENCE_CLASS_MAP=1
  )
  if(DYNG_ENABLE_OPENMP)
    target_link_libraries(dyng_triad_reference_mutant_tests PRIVATE OpenMP::OpenMP_CXX)
  endif()
  dyng_set_warnings(dyng_triad_reference_mutant_tests)
  dyng_enable_sanitizers(dyng_triad_reference_mutant_tests)
  add_test(NAME testing.mutation.triad_reference_class_map
           COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:dyng_triad_reference_mutant_tests>
                   -DEXPECT=fail -DNAME=testing.triad_reference_class_map -P
                   "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
  )
  set_tests_properties(
    testing.mutation.triad_reference_class_map
    PROPERTIES LABELS "cpu;mutation;testing;triad_count" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}"
  )
endif()

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

# triad_count's kit: the suites of StatHyper's types beside the h_motif suite, and in C10 the
# kit's hypergraph fake as a partner, whose CUDA counts are kernels.
if(TARGET dyng_triad_count_conformance_tests)
  target_sources(dyng_triad_count_conformance_tests
                 PRIVATE algorithms/triad_count/triad_count_types_conformance_test.cpp)
endif()
if(TARGET dyng_triad_count_conformance_cuda_tests)
  target_sources(
    dyng_triad_count_conformance_cuda_tests
    PRIVATE algorithms/triad_count/triad_count_types_conformance_test.cpp framework/fake_hyper_kernels.cu
  )
endif()

# The registry and the kit agree: every algorithm of dyng::algorithms() has test_traits here.
dyng_add_test(
  NAME dyng_conformance_registry_tests
  SOURCES conformance/registry_test.cpp conformance/engines_agree_test.cpp
  LABELS cpu conformance
)
target_compile_definitions(dyng_conformance_registry_tests PRIVATE ${_dyng_algorithm_defines})
target_include_directories(dyng_conformance_registry_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

# The kit's hypergraph axis (ADR 0043; M8b criterion 11): the kit C0-C12 over the test-only
# aggregate-delta fake test_hyper_pairs (cpp/tests/framework/fake_hyper_problems.hpp; its partner
# of C10 is test_hyper_squares) on the host backends (labels cpu, conformance, hypergraph) and, in
# CUDA builds, on cuda (gpu, ...; the fakes count on the device: fake_hyper_kernels.cu).
dyng_add_test(
  NAME dyng_hypergraph_kit_tests
  SOURCES conformance/hypergraph_kit_test.cpp conformance/allocation_counter.cpp
  LABELS cpu conformance hypergraph
  LIBRARIES dyng::testing
)
target_include_directories(dyng_hypergraph_kit_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_hypergraph_kit_cuda_tests
    SOURCES conformance/hypergraph_kit_test.cpp conformance/allocation_counter.cpp
            framework/fake_hyper_kernels.cu
    LABELS gpu conformance hypergraph
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_hypergraph_kit_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  target_include_directories(dyng_hypergraph_kit_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_hypergraph_kit_cuda_tests)  # the streams of C12
endif()
# The planted bug of the kit (M8b criterion 11; parity/mutate.py, suite `kit`): the fake's count(+)
# reads the ids the request would get without reuse instead of inserted_ids; the kit must fail
# (conformance.mutation.frontier_requested_ids passes when the suite fails; the control is
# dyng_hypergraph_kit_tests). Labels cpu;mutation;conformance;hypergraph.
if(DYNG_MUTATION_TESTS)
  add_executable(dyng_hypergraph_kit_mutant_tests conformance/hypergraph_kit_test.cpp
                                                  conformance/allocation_counter.cpp)
  target_link_libraries(
    dyng_hypergraph_kit_mutant_tests PRIVATE dyng::dyng dyng::testing dyng_test_support GTest::gtest
                                             GTest::gmock GTest::gtest_main
  )
  target_compile_features(dyng_hypergraph_kit_mutant_tests PRIVATE cxx_std_17)
  target_compile_definitions(
    dyng_hypergraph_kit_mutant_tests PRIVATE DYNG_MUTATION_KIT_FRONTIER_REQUESTED_IDS=1
  )
  target_include_directories(
    dyng_hypergraph_kit_mutant_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src"
                                             "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  dyng_set_warnings(dyng_hypergraph_kit_mutant_tests)
  dyng_enable_sanitizers(dyng_hypergraph_kit_mutant_tests)
  add_test(NAME conformance.mutation.frontier_requested_ids
           COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:dyng_hypergraph_kit_mutant_tests>
                   -DEXPECT=fail -DNAME=kit.frontier_requested_ids -P
                   "${CMAKE_CURRENT_SOURCE_DIR}/mutation/run_mutant.cmake"
  )
  set_tests_properties(
    conformance.mutation.frontier_requested_ids
    PROPERTIES LABELS "cpu;mutation;conformance;hypergraph" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}"
  )
endif()

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
