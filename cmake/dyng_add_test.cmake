# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng_add_test(NAME <name> SOURCES <files...> [LABELS <labels...>] [LIBRARIES <targets...>]
#               [TEST_PREFIX <prefix>])
#
# Builds one GoogleTest executable <name> linked against dyng::dyng and GTest::gtest_main and
# registers its test cases with CTest (gtest_discover_tests), each carrying LABELS (lower case:
# cpu, gpu, slow, parity, sanitize, and the algorithm name; PLAN Section 4.4.1). The default
# label is "cpu". TEST_PREFIX is prepended to the CTest names; an executable that compiles the
# same suites as another one (the CUDA builds of the shared algorithm suites) needs it, because
# CTest test names must be unique.
#
# Every test runs with DYNG_TEST_ENVIRONMENT (default OMP_WAIT_POLICY=PASSIVE): OpenMP worker
# threads otherwise spin between parallel regions, and `ctest -j` runs many OpenMP tests at once,
# each with one thread per core; on a 4-core runner that made `ctest -j4` about 40x slower than a
# serial run. Passive waiting leaves the results unchanged (tests never measure time).
#
# Executables with a "gpu" label are listed (one path per line) in
# <build>/gpu_test_executables.txt by dyng_write_gpu_test_list(), for ci/gpu_local.sh's
# compute-sanitizer pass.

include(GoogleTest)

set(DYNG_TEST_ENVIRONMENT
    "OMP_WAIT_POLICY=PASSIVE"
    CACHE STRING "Environment (VAR=value;...) of every CTest test"
)

function(dyng_add_test)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME;TEST_PREFIX" "SOURCES;LABELS;LIBRARIES")
  if(NOT arg_NAME OR NOT arg_SOURCES)
    message(FATAL_ERROR "dyng_add_test: NAME and SOURCES are required")
  endif()
  if(NOT arg_LABELS)
    set(arg_LABELS cpu)
  endif()
  add_executable(${arg_NAME} ${arg_SOURCES})
  target_link_libraries(
    ${arg_NAME}
    PRIVATE dyng::dyng dyng_test_support GTest::gtest GTest::gmock GTest::gtest_main ${arg_LIBRARIES}
  )
  target_compile_features(${arg_NAME} PRIVATE cxx_std_17)
  dyng_set_warnings(${arg_NAME})
  dyng_enable_sanitizers(${arg_NAME})
  if("gpu" IN_LIST arg_LABELS)
    set_property(GLOBAL APPEND PROPERTY DYNG_GPU_TEST_TARGETS ${arg_NAME})
  endif()
  gtest_discover_tests(
    ${arg_NAME}
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "${arg_TEST_PREFIX}"
    PROPERTIES LABELS "${arg_LABELS}" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}"
  )
endfunction()

# Write <build>/gpu_test_executables.txt: the executables of every test added with LABELS gpu.
function(dyng_write_gpu_test_list)
  get_property(_targets GLOBAL PROPERTY DYNG_GPU_TEST_TARGETS)
  set(_content "")
  foreach(_target IN LISTS _targets)
    string(APPEND _content "$<TARGET_FILE:${_target}>\n")
  endforeach()
  file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/gpu_test_executables.txt" CONTENT "${_content}")
endfunction()
