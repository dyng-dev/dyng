# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# triad_count (PLAN Section 4.8, file 3): the host backends and, with CUDA, ESCHER-GPU's wedge
# engine (engine/, ADR 0053); since M9b also the driver and row accessor of
# experimental::count_local_patterns (ADR 0058).

dyng_add_algorithm(
  NAME
  triad_count
  SOURCES
  triad_count.cpp
  builtin_ops.cpp
  local_patterns.cpp
  sequential.cpp
  openmp.cpp
  CUDA_SOURCES
  cuda.cu
  engine/counters.cu
)
