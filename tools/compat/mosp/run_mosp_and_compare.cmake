# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Runs dyng-compat-mosp --mosp (the whole MOSP update) on one case of cpp/tests/data/mosp_combined
# and requires the combinedGraph/ outputs to be byte-identical to the originals' (MOSP-OpenMP and
# MOSP-CUDA `mosp --pref`, parity/fixtures/mosp/make_mosp_fixtures.sh). Variables: EXE, INPUT
# (graphCsr*, insert.txt, delete.txt), INIT (obj<k>/ initial trees), EXPECTED (combined/), K, PREF,
# BACKEND, OUT. Exits with 77 (skip) when BACKEND is cuda and no CUDA device is visible.

file(REMOVE_RECURSE "${OUT}")
execute_process(
  COMMAND
    "${EXE}" --graph "${INPUT}/graphCsr" --changes "${INPUT}" --init "${INIT}" -k ${K} --mosp
    --pref ${PREF} --out "${OUT}" --backend ${BACKEND} --threads 2
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
  message(FATAL_ERROR "dyng-compat-mosp --mosp failed (${rc})")
endif()
foreach(file distancesCsr.txt SSSPTreeCsr.txt mospCosts.txt)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/combinedGraph/${file}"
            "${EXPECTED}/combined/${file}"
    RESULT_VARIABLE differ
  )
  if(NOT differ EQUAL 0)
    message(FATAL_ERROR "combinedGraph/${file} differs from the original's")
  endif()
endforeach()
