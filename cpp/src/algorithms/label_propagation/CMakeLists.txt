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
  operators_engine.cu
  FAST_MATH
)
if(TARGET dyng_label_propagation)
  set_source_files_properties(label_propagation.cpp sequential.cpp PROPERTIES COMPILE_OPTIONS
                                                                              "-ffp-contract=off")
endif()

# AD6's calibration of the refactored engines' bound (parity/label_propagation_engines.py
# calibrate; ADR 0101): extra CUDA flags of this target's sources only, after its fast-math flag
# (an explicit -fmad=false wins over --use_fast_math's -fmad=true), and the calibration's
# DYNG_CALIBRATION_LP_* defines. Empty in every preset and release build.
set(DYNG_LP_CALIBRATION_CUDA_FLAGS
    ""
    CACHE STRING "AD6's calibration only: extra CUDA flags of label_propagation's sources"
)
mark_as_advanced(DYNG_LP_CALIBRATION_CUDA_FLAGS)
if(TARGET dyng_label_propagation
   AND DYNG_ENABLE_CUDA
   AND DYNG_LP_CALIBRATION_CUDA_FLAGS
)
  separate_arguments(_dyng_lp_calibration UNIX_COMMAND "${DYNG_LP_CALIBRATION_CUDA_FLAGS}")
  # Quoted: the flags are a list, and a list inside an unquoted generator expression splits it.
  target_compile_options(dyng_label_propagation
                         PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_dyng_lp_calibration}>")
endif()
