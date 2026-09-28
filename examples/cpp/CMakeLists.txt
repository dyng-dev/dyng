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
                         PROPERTIES LABELS "cpu;sssp;examples" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
  endforeach()
endif()

add_executable(dyng_example_cycle_count_update cycle_count_update.cpp)
target_link_libraries(dyng_example_cycle_count_update PRIVATE dyng::dyng)
target_compile_features(dyng_example_cycle_count_update PRIVATE cxx_std_17)
set_target_properties(dyng_example_cycle_count_update PROPERTIES OUTPUT_NAME cycle_count_update)
dyng_set_warnings(dyng_example_cycle_count_update)
dyng_enable_sanitizers(dyng_example_cycle_count_update)

if(DYNG_BUILD_TESTS)
  # The updated histogram equals the output of the original `cycle-enum --task update` on the same
  # graph and batch (cpp/tests/data/cycle_enum/cli/c06).
  set(_data "${PROJECT_SOURCE_DIR}/cpp/tests/data/cycle_enum")
  set(_backends sequential)
  if(DYNG_ENABLE_OPENMP)
    list(APPEND _backends openmp)
  endif()
  foreach(_backend IN LISTS _backends)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/cycle_count_update.${_backend}.txt")
    add_test(
      NAME example.cycle_count_update.${_backend}
      COMMAND
        ${CMAKE_COMMAND} -E env ${DYNG_TEST_ENVIRONMENT} sh -c
        "\"$0\" \"$1\" 4 40 40 1 \"$2\" > \"$3\" && cmp \"$3\" \"$4\""
        $<TARGET_FILE:dyng_example_cycle_count_update> "${_data}/parser/tudataset_A.txt"
        ${_backend} "${_out}" "${_data}/cli/c06.out"
    )
    set_tests_properties(
      example.cycle_count_update.${_backend} PROPERTIES LABELS "cpu;cycle_count;examples"
    )
  endforeach()
endif()
