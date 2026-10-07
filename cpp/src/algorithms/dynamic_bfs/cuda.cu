// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda.cu
 * @brief dynamic_bfs: the CUDA backend (PLAN Section
 *        4.8, file 8): the passes of engine.hpp run by the CUDA executor (one grid-stride
 *        kernel per pass on the stream of the resources handle).
 */
#include "algorithms/dynamic_bfs/engine.hpp"
#include "algorithms/dynamic_bfs/problem.hpp"
#include "operators/cuda_execution.cuh"

#include <cstdint>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_cuda_engine() {
  static const dynamic_bfs_engine_impl<operators::cuda_exec, vertex_t, edge_t> engine;
  return engine;
}

template const dynamic_bfs_engine<std::int32_t, std::int32_t>&
dynamic_bfs_cuda_engine<std::int32_t, std::int32_t>();
template const dynamic_bfs_engine<std::int32_t, std::int64_t>&
dynamic_bfs_cuda_engine<std::int32_t, std::int64_t>();
template const dynamic_bfs_engine<std::int64_t, std::int64_t>&
dynamic_bfs_cuda_engine<std::int64_t, std::int64_t>();

}  // namespace dyng::detail
