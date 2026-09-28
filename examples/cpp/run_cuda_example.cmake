# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# CTest driver of example.sssp_update.cuda: run the sssp_update example on the cuda backend and
# compare its tree file with MOSP's output. Exits with 77 (skip) when the example reports that no
# CUDA device is visible.
#
#   cmake -DEXE=<sssp_update> -DCASE=<case dir> -DOUT=<dir> -DEXPECTED=<SSSPTreeUpdated.txt> -P ...

execute_process(
  COMMAND "${EXE}" "${CASE}/graphCsr" "${CASE}/insert.txt" "${CASE}/delete.txt" "${OUT}" cuda
  RESULT_VARIABLE _rc
)
if(_rc EQUAL 77)
  message(STATUS "sssp_update: no CUDA device visible; skipped")
  cmake_language(EXIT 77)
elseif(NOT _rc EQUAL 0)
  message(FATAL_ERROR "sssp_update cuda failed with exit code ${_rc}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/SSSPTreeUpdated.txt" "${EXPECTED}"
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "${OUT}/SSSPTreeUpdated.txt differs from ${EXPECTED}")
endif()
