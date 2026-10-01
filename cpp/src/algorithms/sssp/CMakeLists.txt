# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0

dyng_add_algorithm(NAME sssp SOURCES sssp.cpp sequential.cpp openmp.cpp CUDA_SOURCES cuda.cu operators.cu)
