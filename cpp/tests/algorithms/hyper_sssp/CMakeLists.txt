# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of hyper_sssp (PLAN Section 4.8, file 12; M11a step 4): hyper_sssp_test.cpp on the
# host backends (dyng_hyper_sssp_tests: sequential, and the refusal of openmp) and, in CUDA builds,
# on cuda (dyng_hyper_sssp_cuda_tests). Its conformance suite is built by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_hyper_sssp_tests
  SOURCES hyper_sssp_test.cpp
  LABELS cpu hyper_sssp
  LIBRARIES dyng::testing
)
target_include_directories(dyng_hyper_sssp_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_hyper_sssp_cuda_tests
    SOURCES hyper_sssp_test.cpp
    LABELS gpu hyper_sssp
    LIBRARIES dyng::testing
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_hyper_sssp_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  target_include_directories(dyng_hyper_sssp_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
endif()
# Host allocation failures in compute(), update(), clone() and vertex_distances() are
# out_of_memory_error with the documented guarantee (M11a step 4; ADR 0061, D13): its own operator
# new, so its own executable.
dyng_add_test(
  NAME dyng_hyper_sssp_allocation_failure_tests
  SOURCES hyper_sssp_allocation_failure_test.cpp
  LABELS cpu hyper_sssp
  LIBRARIES dyng::testing
)
target_include_directories(dyng_hyper_sssp_allocation_failure_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
# The C++ code of the "Shortest hyperpaths" tutorial (docs/tutorials/shortest_hyperpaths.md) beyond
# its example program, on the example's data (M11b; quoted by marked ranges, ci/tutorial_quotes.py).
dyng_add_test(
  NAME dyng_hyper_sssp_tutorial_tests
  SOURCES hyper_sssp_tutorial_test.cpp
  LABELS cpu hyper_sssp examples
  LIBRARIES dyng::testing
)
target_compile_definitions(dyng_hyper_sssp_tutorial_tests
                           PRIVATE DYNG_EXAMPLES_DATA="${PROJECT_SOURCE_DIR}/examples/data")
