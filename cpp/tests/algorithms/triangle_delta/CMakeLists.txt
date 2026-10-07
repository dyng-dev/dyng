# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of triangle_delta (PLAN Section 4.8, file 12) on the host backends (label cpu) and,
# compiled again with DYNG_TEST_CUDA=1, on cuda (label gpu); its conformance suite is built by
# cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_triangle_delta_tests
  SOURCES triangle_delta_test.cpp
  LABELS cpu triangle_delta
)

if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_triangle_delta_cuda_tests
    SOURCES triangle_delta_test.cpp
    LABELS gpu triangle_delta
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_triangle_delta_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
endif()
