# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Drop-in clones of the originals' command-line tools, for the parity harness only (PLAN Section
# 5.6). Built with DYNG_BUILD_COMPAT_TOOLS=ON (the dev and parity presets); not installed.

add_executable(dyng-compat-mosp mosp/mosp_compat.cpp)
target_link_libraries(dyng-compat-mosp PRIVATE dyng::dyng)
if(DYNG_ENABLE_OPENMP)
  # The driver reads its input files concurrently in an OpenMP region, like MOSP's `mosp`.
  target_link_libraries(dyng-compat-mosp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_features(dyng-compat-mosp PRIVATE cxx_std_17)
dyng_set_warnings(dyng-compat-mosp)
dyng_enable_sanitizers(dyng-compat-mosp)

add_executable(dyng-compat-cycle-enum cycle_enum/cycle_enum_compat.cpp)
target_link_libraries(dyng-compat-cycle-enum PRIVATE dyng::dyng)
target_compile_features(dyng-compat-cycle-enum PRIVATE cxx_std_17)
dyng_set_warnings(dyng-compat-cycle-enum)
dyng_enable_sanitizers(dyng-compat-cycle-enum)

if(DYNG_BUILD_TESTS)
  # The cycle-enum clone reproduces the original CLI's standard output and exit status on the
  # committed cases (cpp/tests/data/cycle_enum/cli, from CycleEnumeration-GPU@0a976ad).
  add_test(
    NAME compat_cycle_enum.cli
    COMMAND
      ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:dyng-compat-cycle-enum>
      -DDATA=${PROJECT_SOURCE_DIR}/cpp/tests/data/cycle_enum -DOPENMP=${DYNG_ENABLE_OPENMP} -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cycle_enum/run_and_compare.cmake
  )
  set_tests_properties(
    compat_cycle_enum.cli PROPERTIES LABELS "cpu;cycle_count;compat" ENVIRONMENT
                                     "${DYNG_TEST_ENVIRONMENT}"
  )
endif()

if(DYNG_BUILD_TESTS)
  # The driver reproduces the original's output files byte for byte on committed fixtures.
  set(_data "${PROJECT_SOURCE_DIR}/cpp/tests/data")
  set(_backends sequential)
  if(DYNG_ENABLE_OPENMP)
    list(APPEND _backends openmp)
  endif()
  if(DYNG_ENABLE_CUDA)
    list(APPEND _backends cuda)
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
      if(_backend STREQUAL "cuda")
        set(_labels "gpu;sssp;compat")
      else()
        set(_labels "cpu;sssp;compat")
      endif()
      set_tests_properties(${_test} PROPERTIES LABELS "${_labels}" ENVIRONMENT
                                               "${DYNG_TEST_ENVIRONMENT}")
    endforeach()
  endforeach()
endif()

if(DYNG_BUILD_TESTS AND DYNG_BUILD_PARITY_TESTS)
  # Golden replay against the pinned original (PLAN Section 8.3; CTest label "parity"): every case
  # of the golden corpus, sequential and OpenMP with 1, 4 and 16 threads, byte for byte. The
  # goldens live outside the repository ($DYNG_SCRATCH/goldens, parity/export_goldens.py); without
  # them the test is skipped (exit code 77).
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  set(DYNG_GOLDENS_DIR
      "$ENV{DYNG_SCRATCH}/goldens"
      CACHE PATH "Directory of the golden sets (default: \$DYNG_SCRATCH/goldens)"
  )
  if(DYNG_GOLDENS_DIR STREQUAL "/goldens")
    set(DYNG_GOLDENS_DIR "$ENV{HOME}/Projects/dyng-work/goldens")
  endif()
  add_test(
    NAME parity.sssp.mosp_openmp_c352151
    COMMAND
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/parity/compare.py --exe
      $<TARGET_FILE:dyng-compat-mosp> --goldens ${DYNG_GOLDENS_DIR}/sssp --jobs 8 --json
      ${CMAKE_CURRENT_BINARY_DIR}/parity-sssp.json
  )
  set_tests_properties(
    parity.sssp.mosp_openmp_c352151 PROPERTIES LABELS "parity;sssp" SKIP_RETURN_CODE 77 TIMEOUT
                                               1800
  )
  # The cycle_count corpus of CycleEnumeration-GPU 0a976ad (parity/cycle_count_goldens.py):
  # histograms and generated batches byte for byte, sequential and OpenMP with 4 and 56 threads
  # (COLLAB only on OpenMP; `compare.py cycle_count --full` adds its sequential replay).
  add_test(
    NAME parity.cycle_count.cycle_enum_0a976ad
    COMMAND
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/parity/compare.py cycle_count --exe
      $<TARGET_FILE:dyng-compat-cycle-enum> --goldens ${DYNG_GOLDENS_DIR}/cycle_count --jobs 8
      --json ${CMAKE_CURRENT_BINARY_DIR}/parity-cycle_count.json
  )
  set_tests_properties(
    parity.cycle_count.cycle_enum_0a976ad PROPERTIES LABELS "parity;cycle_count" SKIP_RETURN_CODE
                                                     77 TIMEOUT 3600
  )
  if(DYNG_ENABLE_CUDA)
    # The same corpus on the cuda backend: byte parity with MOSP-CUDA e220ee2 (the goldens are
    # MOSP-OpenMP's; MOSP-CUDA == MOSP-OpenMP on every case was checked once, PLAN 6.3 step 2).
    add_test(
      NAME parity.sssp.mosp_cuda_e220ee2
      COMMAND
        ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/parity/compare.py --exe
        $<TARGET_FILE:dyng-compat-mosp> --goldens ${DYNG_GOLDENS_DIR}/sssp --configs cuda --jobs 8
        --json ${CMAKE_CURRENT_BINARY_DIR}/parity-sssp-cuda.json
    )
    set_tests_properties(
      parity.sssp.mosp_cuda_e220ee2 PROPERTIES LABELS "parity;gpu;sssp" SKIP_RETURN_CODE 77
                                               TIMEOUT 3600
    )
  endif()
endif()
