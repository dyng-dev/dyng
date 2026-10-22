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
# mosp (0.2): left out of a subset build without it (DYNG_ALGORITHMS).
if(TARGET dyng_mosp)
  add_executable(dyng_example_mosp_update mosp_update.cpp)
  target_link_libraries(dyng_example_mosp_update PRIVATE dyng::dyng)
  target_compile_features(dyng_example_mosp_update PRIVATE cxx_std_17)
  set_target_properties(dyng_example_mosp_update PROPERTIES OUTPUT_NAME mosp_update)
  dyng_set_warnings(dyng_example_mosp_update)
  dyng_enable_sanitizers(dyng_example_mosp_update)
  if(DYNG_BUILD_TESTS)
    # The combined graph's outputs of the thesis example (Pref 4,1,4) equal the originals'
    # (cpp/tests/data/mosp_combined/thesis_414: MOSP-OpenMP's and MOSP-CUDA's `mosp --pref`).
    set(_data "${PROJECT_SOURCE_DIR}/cpp/tests/data/mosp_combined")
    set(_backends sequential)
    if(DYNG_ENABLE_OPENMP)
      list(APPEND _backends openmp)
    endif()
    foreach(_backend IN LISTS _backends)
      set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/mosp_update.${_backend}")
      file(MAKE_DIRECTORY "${_out}")
      add_test(
        NAME example.mosp_update.${_backend}
        COMMAND
          ${CMAKE_COMMAND} -E env ${DYNG_TEST_ENVIRONMENT} sh -c
          "\"$0\" \"$1/graphCsr\" \"$1/insert.txt\" \"$1/delete.txt\" \"$2\" 4,1,4 \"$3\" && cmp \"$2/distancesCsr.txt\" \"$4/distancesCsr.txt\" && cmp \"$2/SSSPTreeCsr.txt\" \"$4/SSSPTreeCsr.txt\" && cmp \"$2/mospCosts.txt\" \"$4/mospCosts.txt\""
          $<TARGET_FILE:dyng_example_mosp_update> "${_data}/thesis/input" "${_out}" ${_backend}
          "${_data}/thesis_414/combined"
      )
      set_tests_properties(example.mosp_update.${_backend} PROPERTIES LABELS "cpu;mosp;examples")
    endforeach()
    if(DYNG_ENABLE_CUDA)
      # The same on the GPU (label gpu), skipped (77) when no device is visible.
      set(_out "${CMAKE_CURRENT_BINARY_DIR}/out/mosp_update.cuda")
      file(MAKE_DIRECTORY "${_out}")
      add_test(
        NAME example.mosp_update.cuda
        COMMAND
          ${CMAKE_COMMAND} -E env ${DYNG_TEST_ENVIRONMENT} sh -c
          "\"$0\" \"$1/graphCsr\" \"$1/insert.txt\" \"$1/delete.txt\" \"$2\" 4,1,4 cuda && cmp \"$2/distancesCsr.txt\" \"$3/distancesCsr.txt\" && cmp \"$2/SSSPTreeCsr.txt\" \"$3/SSSPTreeCsr.txt\" && cmp \"$2/mospCosts.txt\" \"$3/mospCosts.txt\""
          $<TARGET_FILE:dyng_example_mosp_update> "${_data}/thesis/input" "${_out}"
          "${_data}/thesis_414/combined"
      )
      set_tests_properties(example.mosp_update.cuda PROPERTIES LABELS "gpu;mosp;examples"
                                                               SKIP_RETURN_CODE 77)
    endif()
  endif()
endif()

# triad_count (0.3): left out of a subset build without it (DYNG_ALGORITHMS). The program the
# "Hypergraph triads" tutorial quotes; its output on examples/data/triad_count.{hg,dgt} is
# examples/data/triad_count_update.expected.txt (equal to a brute force over every triple of
# hyperedges, and to examples/python/triad_count.py's output) on every backend.
if(TARGET dyng_triad_count)
  add_executable(dyng_example_triad_count_update triad_count_update.cpp)
  target_link_libraries(dyng_example_triad_count_update PRIVATE dyng::dyng)
  target_compile_features(dyng_example_triad_count_update PRIVATE cxx_std_17)
  set_target_properties(dyng_example_triad_count_update PROPERTIES OUTPUT_NAME triad_count_update)
  dyng_set_warnings(dyng_example_triad_count_update)
  dyng_enable_sanitizers(dyng_example_triad_count_update)
  if(DYNG_BUILD_TESTS)
    set(_data "${PROJECT_SOURCE_DIR}/examples/data")
    set(_backends sequential)
    if(DYNG_ENABLE_OPENMP)
      list(APPEND _backends openmp)
    endif()
    if(DYNG_ENABLE_CUDA)
      list(APPEND _backends cuda)
    endif()
    foreach(_backend IN LISTS _backends)
      add_test(
        NAME example.triad_count_update.${_backend}
        COMMAND
          ${CMAKE_COMMAND} "-DEXE=$<TARGET_FILE:dyng_example_triad_count_update>"
          "-DARGS=${_data}/triad_count.hg|${_data}/triad_count.dgt|${_backend}"
          "-DEXPECTED=${_data}/triad_count_update.expected.txt" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/compare_output.cmake"
      )
      if(_backend STREQUAL "cuda")
        set(_labels "gpu;triad_count;examples")
      else()
        set(_labels "cpu;triad_count;examples")
      endif()
      set_tests_properties(example.triad_count_update.${_backend} PROPERTIES
                           LABELS "${_labels}" SKIP_RETURN_CODE 77
                           ENVIRONMENT "${DYNG_TEST_ENVIRONMENT}")
    endforeach()
  endif()
endif()

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
