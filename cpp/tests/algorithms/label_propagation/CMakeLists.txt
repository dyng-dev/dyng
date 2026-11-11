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

# Host allocation failures inside compute(), update(), reserve() and clone() are
# out_of_memory_error with the documented guarantee (M10a review; the ADR 0060 precedent): its own
# operator new, so its own executable.
dyng_add_test(
  NAME dyng_label_propagation_allocation_failure_tests
  SOURCES label_propagation_allocation_failure_test.cpp
  LABELS cpu label_propagation
)

# The original's checker oracles as tests (M10a step 5, criterion 14): the published flow at 5k5
# and 50k, delta 1e-6, both schedules, every batch against union_find, lp_host_model and the
# harmonic solution; the host executable carries the label slow (the 50k cases), the CUDA one runs
# in ctest -L gpu and in ci/gpu_local.sh's label_propagation step.
dyng_add_test(
  NAME dyng_label_propagation_checker_tests
  SOURCES label_propagation_checker_test.cpp
  LABELS cpu label_propagation slow
  LIBRARIES dyng::testing
)
if(DYNG_ENABLE_CUDA)
  dyng_add_test(
    NAME dyng_label_propagation_checker_cuda_tests
    SOURCES label_propagation_checker_test.cpp
    LABELS gpu label_propagation
    LIBRARIES dyng::testing
    TEST_PREFIX "cuda."
  )
  target_compile_definitions(dyng_label_propagation_checker_cuda_tests PRIVATE DYNG_TEST_CUDA=1)
endif()
