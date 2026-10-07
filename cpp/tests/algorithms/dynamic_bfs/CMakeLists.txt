# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of dynamic_bfs (PLAN Section 4.8, file 12) on the host backends (label cpu) and,
# compiled again with DYNG_TEST_CUDA=1, on cuda (label gpu); its conformance suite is built by
# cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_dynamic_bfs_tests
  SOURCES dynamic_bfs_test.cpp
  LABELS cpu dynamic_bfs
)

if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_dynamic_bfs_cuda_tests
    SOURCES dynamic_bfs_test.cpp
    LABELS gpu dynamic_bfs
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_dynamic_bfs_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
endif()
