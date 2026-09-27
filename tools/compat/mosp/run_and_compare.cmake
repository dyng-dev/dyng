# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs dyng-compat-mosp (init and update) on one fixture case and requires byte-identical output
# files. Variables: EXE, INPUT (graphCsr*, insert.txt, delete.txt), EXPECTED (init/ and updated/
# from the original), K, BACKEND, OUT.

file(REMOVE_RECURSE "${OUT}")
execute_process(
  COMMAND "${EXE}" init "${INPUT}/graphCsr" "${OUT}/init" -k ${K} --backend ${BACKEND}
  RESULT_VARIABLE rc
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "dyng-compat-mosp init failed (${rc})")
endif()
execute_process(
  COMMAND
    "${EXE}" --graph "${INPUT}/graphCsr" --changes "${INPUT}" --init "${EXPECTED}/init" -k ${K}
    --out "${OUT}/updated" --backend ${BACKEND} --threads 2 --timing "${OUT}/timing.csv"
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
