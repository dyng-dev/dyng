# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The suites of mosp (its conformance suite is built by cpp/tests/CMakeLists.txt): the hand cases
# and the randomized differential tests, on the host backends (label cpu) and, compiled again with
# DYNG_TEST_CUDA=1, on cuda against the host backends (label gpu).

set(_mosp_sources mosp_test.cpp mosp_random_test.cpp)
dyng_add_test(
  NAME dyng_mosp_tests
  SOURCES ${_mosp_sources}
  LABELS cpu mosp
  LIBRARIES dyng::testing
)
target_include_directories(dyng_mosp_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

# The parallel path costs against the sequential ones (internal functions, host only).
dyng_add_test(
  NAME dyng_mosp_path_costs_tests
  SOURCES mosp_path_costs_test.cpp
  LABELS cpu mosp
)
target_include_directories(dyng_mosp_path_costs_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")

# set_options() under host allocation failures (its own executable: it replaces operator new).
dyng_add_test(
  NAME dyng_mosp_allocation_failure_tests
  SOURCES mosp_allocation_failure_test.cpp
  LABELS cpu mosp
)

if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_mosp_cuda_tests
    SOURCES ${_mosp_sources}
    LABELS gpu mosp
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_mosp_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  target_include_directories(dyng_mosp_cuda_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
endif()
