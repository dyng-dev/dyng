# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of label_propagation (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt. The allocation counter of the conformance kit counts the host heap
# allocations of the algorithm phase (invariant I9).

dyng_add_test(
  NAME dyng_label_propagation_tests
  SOURCES label_propagation_test.cpp "${PROJECT_SOURCE_DIR}/cpp/tests/conformance/allocation_counter.cpp"
  LABELS cpu label_propagation
  LIBRARIES dyng::testing
)
target_include_directories(dyng_label_propagation_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/tests"
                                                                 "${PROJECT_SOURCE_DIR}/cpp/src")

# The same hand cases on the cuda backend (DYNG_TEST_CUDA=1: the labels copied to the host), and
# the cases only the fused CUDA engine has (M10a step 4, ADR 0075; private headers of cpp/src).
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_label_propagation_cuda_tests
    SOURCES label_propagation_test.cpp label_propagation_cuda_test.cpp
            "${PROJECT_SOURCE_DIR}/cpp/tests/conformance/allocation_counter.cpp"
    LABELS gpu label_propagation
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_label_propagation_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
  target_include_directories(dyng_label_propagation_cuda_tests
                             PRIVATE "${PROJECT_SOURCE_DIR}/cpp/tests" "${PROJECT_SOURCE_DIR}/cpp/src")
  dyng_set_cuda_options(dyng_label_propagation_cuda_tests)
endif()
