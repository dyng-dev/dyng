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
  endif()
endif()
