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
