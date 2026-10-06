// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda.cu
 * @brief triangle_delta: the CUDA backend (PLAN Section
 *        4.8, file 8): the counts of engine.hpp run by the CUDA executor (one grid-stride
 *        kernel per count on the stream of the resources handle).
 */
#include "algorithms/triangle_delta/engine.hpp"
#include "algorithms/triangle_delta/problem.hpp"
#include "operators/cuda_execution.cuh"

#include <cstdint>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
const triangle_delta_engine<vertex_t, edge_t>& triangle_delta_cuda_engine() {
  static const triangle_delta_engine_impl<operators::cuda_exec, vertex_t, edge_t> engine;
  return engine;
}

template const triangle_delta_engine<std::int32_t, std::int32_t>&
triangle_delta_cuda_engine<std::int32_t, std::int32_t>();
template const triangle_delta_engine<std::int32_t, std::int64_t>&
triangle_delta_cuda_engine<std::int32_t, std::int64_t>();
template const triangle_delta_engine<std::int64_t, std::int64_t>&
triangle_delta_cuda_engine<std::int64_t, std::int64_t>();

}  // namespace dyng::detail
