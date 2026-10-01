# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# mosp (0.2): K x sssp and the combined graph. It uses sssp's internal engines (sssp is always
# built, CMakeLists.txt at the root).

dyng_add_algorithm(NAME mosp SOURCES mosp.cpp sequential.cpp openmp.cpp CUDA_SOURCES cuda.cu)
