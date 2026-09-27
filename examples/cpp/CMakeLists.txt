# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# C++ examples (PLAN Section 9.6): compiled and run by the test suite.

add_executable(dyng_example_sssp_update sssp_update.cpp)
target_link_libraries(dyng_example_sssp_update PRIVATE dyng::dyng)
target_compile_features(dyng_example_sssp_update PRIVATE cxx_std_17)
set_target_properties(dyng_example_sssp_update PROPERTIES OUTPUT_NAME sssp_update)
dyng_set_warnings(dyng_example_sssp_update)
dyng_enable_sanitizers(dyng_example_sssp_update)

if(DYNG_BUILD_TESTS)
  set(_case "${PROJECT_SOURCE_DIR}/cpp/tests/data/mosp_graph_io/testCase0")
  set(_expected "${PROJECT_SOURCE_DIR}/cpp/tests/data/mosp_sssp/testCase0/updated/obj0")
  set(_backends sequential)
  if(DYNG_ENABLE_OPENMP)
    list(APPEND _backends openmp)
  endif()
  foreach(_backend IN LISTS _backends)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/${_backend}")
    file(MAKE_DIRECTORY "${_out}")
    add_test(NAME example.sssp_update.${_backend}
             COMMAND dyng_example_sssp_update "${_case}/graphCsr" "${_case}/insert.txt"
                     "${_case}/delete.txt" "${_out}" ${_backend})
    add_test(NAME example.sssp_update.${_backend}.matches_original
             COMMAND ${CMAKE_COMMAND} -E compare_files "${_out}/SSSPTreeUpdated.txt"
                     "${_expected}/SSSPTreeUpdated.txt")
    set_tests_properties(example.sssp_update.${_backend}.matches_original
                         PROPERTIES DEPENDS example.sssp_update.${_backend})
    set_tests_properties(example.sssp_update.${_backend} example.sssp_update.${_backend}.matches_original
                         PROPERTIES LABELS "cpu;sssp;examples")
  endforeach()
endif()
