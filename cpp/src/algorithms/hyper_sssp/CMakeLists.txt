# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# hyper_sssp (PLAN Section 4.8, file 3; scaffolded by scripts/new_algorithm.py, M11a step 4): the
# host engine (sequential.cpp) and the CUDA engine (cuda.cu, around MOSP_ESCHER's transplanted
# engine.cu, kernels.cuh and state.cuh).

dyng_add_algorithm(
  NAME
  hyper_sssp
  SOURCES
  hyper_sssp.cpp
  sequential.cpp
)
