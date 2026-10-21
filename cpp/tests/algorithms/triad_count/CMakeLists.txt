# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases and property tests of triad_count (PLAN Section 4.8, file 12; M9a criteria 5, 6,
# 8, 9, 10; the 64-bit limits of ADR 0057): triad_count_test.cpp on the host backends (dyng_triad_count_tests) and, in CUDA builds,
# on cuda with the device tests of the engine (dyng_triad_count_cuda_tests; the kit's fake partner
# of the composition case counts with kernels). Its conformance suites are built by
# cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_triad_count_tests
  SOURCES triad_count_test.cpp
  LABELS cpu triad_count
  LIBRARIES dyng::testing
)
target_include_directories(dyng_triad_count_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
# The triad-bins JSON of a result (M9b, ADR 0059; the io-only tests are in dyng_io_tests).
dyng_add_test(
  NAME dyng_triad_count_bins_tests
  SOURCES triad_count_bins_test.cpp
  LABELS cpu triad_count io
)
target_compile_definitions(
  dyng_triad_count_bins_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
)
# Host allocation failures in compute(), update() and clone() are out_of_memory_error (the API
# review, ADR 0060): its own operator new, so its own executable.
dyng_add_test(
  NAME dyng_triad_count_allocation_failure_tests
  SOURCES triad_count_allocation_failure_test.cpp
  LABELS cpu triad_count
)
# ESCHER-GPU's make test on the host backends in hosted CI (M9a step 4, decision D5): the
# synthetic groups against A's committed counts, the file-input shapes against the oracle.
dyng_add_test(
  NAME dyng_triad_count_make_test_tests
  SOURCES triad_count_make_test_test.cpp
  LABELS cpu triad_count
  LIBRARIES dyng::testing
)
target_compile_definitions(
  dyng_triad_count_make_test_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
)
# The triangle patterns on the heavy groups (syn100k, skewed, medium): labels parity and slow
# (ci/check.sh's parity step runs them in the Release parity preset).
dyng_add_test(
  NAME dyng_triad_count_make_test_large_tests
  SOURCES triad_count_make_test_test.cpp
  LABELS parity triad_count slow
  LIBRARIES dyng::testing
)
target_compile_definitions(
  dyng_triad_count_make_test_large_tests PRIVATE DYNG_TEST_DATA_DIR="${DYNG_TEST_DATA_DIR}"
                                                 DYNG_TRIAD_MAKE_TEST_LARGE=1
)
# The 64-bit limits (ADR 0057): hyperedges of millions of members, so parity and slow on the host
# (ci/check.sh's parity step, a Release build) and gpu on cuda.
dyng_add_test(
  NAME dyng_triad_count_limits_tests
  SOURCES triad_count_limits_test.cpp
  LABELS parity triad_count slow
  LIBRARIES dyng::testing
)
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_triad_count_limits_cuda_tests
    SOURCES triad_count_limits_test.cpp
    LABELS gpu triad_count
    LIBRARIES dyng::testing
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_triad_count_limits_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  dyng_add_test(
    NAME dyng_triad_count_cuda_tests
    SOURCES triad_count_test.cpp triad_count_kernels_test.cu ../../framework/fake_hyper_kernels.cu
    LABELS gpu triad_count
    LIBRARIES dyng::testing
    TEST_PREFIX cuda.
  )
  target_compile_definitions(dyng_triad_count_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  target_include_directories(dyng_triad_count_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_triad_count_cuda_tests)
endif()
