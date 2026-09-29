# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0

dyng_add_algorithm(
  NAME
  cycle_count
  SOURCES
  cycle_count.cpp
  static_sequential.cpp
  static_openmp.cpp
  sequential.cpp
  openmp.cpp
  CUDA_SOURCES
  static_cuda.cu
  cuda.cu
)
