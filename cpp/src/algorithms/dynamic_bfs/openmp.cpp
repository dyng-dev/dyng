// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file openmp.cpp
 * @brief dynamic_bfs: the OpenMP backend (PLAN Section
 *        4.8, file 7): the passes of engine.hpp run by the OpenMP executor (the threads of the
 *        resources handle; the atomics make the passes safe to run in parallel).
 */
#include "algorithms/dynamic_bfs/engine.hpp"
#include "algorithms/dynamic_bfs/problem.hpp"
#include "operators/execution.hpp"

#include <cstdint>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_openmp_engine() {
  static const dynamic_bfs_engine_impl<operators::openmp_exec, vertex_t, edge_t> engine;
  return engine;
}

template const dynamic_bfs_engine<std::int32_t, std::int32_t>&
dynamic_bfs_openmp_engine<std::int32_t, std::int32_t>();
template const dynamic_bfs_engine<std::int32_t, std::int64_t>&
dynamic_bfs_openmp_engine<std::int32_t, std::int64_t>();
template const dynamic_bfs_engine<std::int64_t, std::int64_t>&
dynamic_bfs_openmp_engine<std::int64_t, std::int64_t>();

}  // namespace dyng::detail
