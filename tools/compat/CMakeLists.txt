# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Drop-in clones of the originals' command-line tools, for the parity harness only (PLAN Section
# 5.6). Built with DYNG_BUILD_COMPAT_TOOLS=ON (the dev and parity presets); not installed.

add_executable(dyng-compat-mosp mosp/mosp_compat.cpp)
target_link_libraries(dyng-compat-mosp PRIVATE dyng::dyng)
target_compile_features(dyng-compat-mosp PRIVATE cxx_std_17)
dyng_set_warnings(dyng-compat-mosp)
dyng_enable_sanitizers(dyng-compat-mosp)

if(DYNG_BUILD_TESTS)
  # The driver reproduces the original's output files byte for byte on committed fixtures.
  set(_data "${PROJECT_SOURCE_DIR}/cpp/tests/data")
  set(_backends sequential)
  if(DYNG_ENABLE_OPENMP)
    list(APPEND _backends openmp)
  endif()
  foreach(_case "testCase9;mosp_graph_io/testCase9;3" "c2i_0;mosp_sssp/c2i_0/input;2"
                "h0;mosp_graph_io/h0;2")
    list(GET _case 0 _name)
    list(GET _case 1 _input)
    list(GET _case 2 _k)
    foreach(_backend IN LISTS _backends)
      set(_test compat_mosp.${_name}.${_backend})
      add_test(
        NAME ${_test}
        COMMAND
          ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:dyng-compat-mosp> -DINPUT=${_data}/${_input}
          -DEXPECTED=${_data}/mosp_sssp/${_name} -DK=${_k} -DBACKEND=${_backend}
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/out/${_name}.${_backend} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/mosp/run_and_compare.cmake
      )
      set_tests_properties(${_test} PROPERTIES LABELS "cpu;sssp;compat")
    endforeach()
  endforeach()
endif()
