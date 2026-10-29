# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# label_propagation (PLAN Section 4.8, file 3; scaffolded by scripts/new_algorithm.py). FAST_MATH
# scopes --use_fast_math to this target's CUDA sources (the original's nvcc line; -fmad=true is the
# parity-cuda preset's global flag, PLAN 7.4). The host sources are compiled without floating-point
# contraction, so the sequential backend gives the same bits on every host build (ADR 0072).

dyng_add_algorithm(
  NAME
  label_propagation
  SOURCES
  label_propagation.cpp
  sequential.cpp
  trace.cpp
  CUDA_SOURCES
  sparsify.cu
  components.cu
  label_init.cu
  propagation.cu
  irlp.cu
  cuda.cu
  FAST_MATH
)
if(TARGET dyng_label_propagation)
  set_source_files_properties(label_propagation.cpp sequential.cpp PROPERTIES COMPILE_OPTIONS
                                                                              "-ffp-contract=off")
endif()
