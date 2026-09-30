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

# The program quoted by docs/getting_started/first_update_cpp.md and the landing page.
add_executable(dyng_example_first_update first_update.cpp)
target_link_libraries(dyng_example_first_update PRIVATE dyng::dyng)
target_compile_features(dyng_example_first_update PRIVATE cxx_std_17)
set_target_properties(dyng_example_first_update PROPERTIES OUTPUT_NAME first_update)
dyng_set_warnings(dyng_example_first_update)
dyng_enable_sanitizers(dyng_example_first_update)

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
    # The tutorial compares both output files with the original's (docs/tutorials).
    add_test(NAME example.sssp_update.${_backend}.distances_match_original
             COMMAND ${CMAKE_COMMAND} -E compare_files "${_out}/distancesUpdated.txt"
                     "${_expected}/distancesUpdated.txt")
    set_tests_properties(example.sssp_update.${_backend}.matches_original
                         example.sssp_update.${_backend}.distances_match_original
                         PROPERTIES DEPENDS example.sssp_update.${_backend})
    set_tests_properties(example.sssp_update.${_backend} example.sssp_update.${_backend}.matches_original
                         example.sssp_update.${_backend}.distances_match_original
                         PROPERTIES LABELS "cpu;sssp;examples" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
    # The first-update program prints exactly what its page says (one regular expression: CTest
    # passes a test if any one of several expressions matches).
    add_test(NAME example.first_update.${_backend} COMMAND dyng_example_first_update ${_backend})
    set_tests_properties(example.first_update.${_backend} PROPERTIES
                         PASS_REGULAR_EXPRESSION
                         "invalidated 2, affected 2\ndistances 0 4 1 2\nparents -1 0 0 2\n"
                         LABELS "cpu;sssp;examples" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
  endforeach()

  # The same example on the GPU (label gpu): run and compare in one test, skipped (77) when no
  # device is visible.
  if(DYNG_ENABLE_CUDA)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/cuda")
    file(MAKE_DIRECTORY "${_out}")
    add_test(
      NAME example.sssp_update.cuda
      COMMAND
        ${CMAKE_COMMAND} "-DEXE=$<TARGET_FILE:dyng_example_sssp_update>" "-DCASE=${_case}"
        "-DOUT=${_out}" "-DEXPECTED=${_expected}/SSSPTreeUpdated.txt" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/run_cuda_example.cmake"
    )
    set_tests_properties(example.sssp_update.cuda PROPERTIES LABELS "gpu;sssp;examples"
                         SKIP_RETURN_CODE 77 ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
    add_test(NAME example.first_update.cuda COMMAND dyng_example_first_update cuda)
    set_tests_properties(example.first_update.cuda PROPERTIES
                         PASS_REGULAR_EXPRESSION
                         "invalidated 2, affected 2\ndistances 0 4 1 2\nparents -1 0 0 2\n"
                         SKIP_RETURN_CODE 77 SKIP_REGULAR_EXPRESSION "no CUDA device visible"
                         LABELS "gpu;sssp;examples" ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
  endif()
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

# The C++ quickstart of README.md (PLAN Section 9.6): extracted at configure time, compiled with
# the examples' warnings and run by CTest, which compares its output with the README's output
# block, so the README cannot drift from the library. ci/doc_snippets.py checks its size.
include("${PROJECT_SOURCE_DIR}/cmake/dyng_doc_snippet.cmake")
dyng_extract_doc_snippet(
  "${PROJECT_SOURCE_DIR}/README.md" quickstart-cpp
  "${CMAKE_CURRENT_BINARY_DIR}/readme_quickstart.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/readme_quickstart.expected"
)
add_executable(dyng_example_readme_quickstart "${CMAKE_CURRENT_BINARY_DIR}/readme_quickstart.cpp")
target_link_libraries(dyng_example_readme_quickstart PRIVATE dyng::dyng)
target_compile_features(dyng_example_readme_quickstart PRIVATE cxx_std_17)
set_target_properties(dyng_example_readme_quickstart PROPERTIES OUTPUT_NAME readme_quickstart)
dyng_set_warnings(dyng_example_readme_quickstart)
dyng_enable_sanitizers(dyng_example_readme_quickstart)
if(DYNG_BUILD_TESTS)
  add_test(
    NAME example.readme_quickstart
    COMMAND
      ${CMAKE_COMMAND} "-DEXE=$<TARGET_FILE:dyng_example_readme_quickstart>"
      "-DEXPECTED=${CMAKE_CURRENT_BINARY_DIR}/readme_quickstart.expected" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/compare_output.cmake"
  )
  set_tests_properties(example.readme_quickstart PROPERTIES LABELS "cpu;sssp;examples"
                       ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
endif()
