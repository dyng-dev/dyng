# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng_add_test(NAME <name> SOURCES <files...> [LABELS <labels...>] [LIBRARIES <targets...>])
#
# Builds one GoogleTest executable <name> linked against dyng::dyng and GTest::gtest_main and
# registers its test cases with CTest (gtest_discover_tests), each carrying LABELS (lower case:
# cpu, gpu, slow, parity, sanitize, and the algorithm name; PLAN Section 4.4.1). The default
# label is "cpu".

include(GoogleTest)

function(dyng_add_test)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME" "SOURCES;LABELS;LIBRARIES")
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
  gtest_discover_tests(
    ${arg_NAME}
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "${arg_LABELS}"
  )
endfunction()
