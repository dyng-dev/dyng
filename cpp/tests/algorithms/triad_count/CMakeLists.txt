# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases and property tests of triad_count (PLAN Section 4.8, file 12; M9a criteria 5, 6,
# 9, 10): triad_count_test.cpp on the host backends (dyng_triad_count_tests) and, in CUDA builds,
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
if(DYNG_ENABLE_CUDA)
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
