# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs dyng-compat-mosp (init and update) on one fixture case and requires byte-identical output
# files. Variables: EXE, INPUT (graphCsr*, insert.txt, delete.txt), EXPECTED (init/ and updated/
# from the original), K, BACKEND, OUT, EDGE (edge-offset type, int32 or int64; default int32).
# Exits with 77 (skip) when BACKEND is cuda and the tool reports that no CUDA device is visible
# (ci/gpu_local.sh checks for a visible device first, so the GPU gate cannot pass by skipping).

if(NOT EDGE)
  set(EDGE int32)
endif()
file(REMOVE_RECURSE "${OUT}")
execute_process(
  COMMAND "${EXE}" init "${INPUT}/graphCsr" "${OUT}/init" -k ${K} --backend ${BACKEND}
          --edge-type ${EDGE}
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE out
  ERROR_VARIABLE out
)
message("${out}")
if(NOT rc EQUAL 0)
  if(BACKEND STREQUAL "cuda" AND out MATCHES "no CUDA device is visible")
    message(STATUS "dyng-compat-mosp: no CUDA device visible; skipped")
    cmake_language(EXIT 77)
  endif()
  message(FATAL_ERROR "dyng-compat-mosp init failed (${rc})")
endif()
execute_process(
  COMMAND
    "${EXE}" --graph "${INPUT}/graphCsr" --changes "${INPUT}" --init "${EXPECTED}/init" -k ${K}
    --out "${OUT}/updated" --backend ${BACKEND} --threads 2 --timing "${OUT}/timing.csv"
    --edge-type ${EDGE}
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "dyng-compat-mosp update failed (${rc})")
endif()
math(EXPR last "${K} - 1")
foreach(k RANGE 0 ${last})
  foreach(pair "init;distancesOriginal.txt" "init;SSSPTreeOriginal.txt"
               "updated;distancesUpdated.txt" "updated;SSSPTreeUpdated.txt")
    list(GET pair 0 dir)
    list(GET pair 1 file)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/${dir}/obj${k}/${file}"
              "${EXPECTED}/${dir}/obj${k}/${file}"
      RESULT_VARIABLE differ
    )
    if(NOT differ EQUAL 0)
      message(FATAL_ERROR "${dir}/obj${k}/${file} differs from the original's")
    endif()
  endforeach()
endforeach()
