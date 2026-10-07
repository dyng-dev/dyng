// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sequential.cpp
 * @brief dynamic_bfs: the sequential backend, the reference of the other backends (PLAN Section
 *        4.8, file 6): the passes of engine.hpp run by the sequential executor (one thread, in
 *        index order).
 */
#include "algorithms/dynamic_bfs/engine.hpp"
#include "algorithms/dynamic_bfs/problem.hpp"
#include "operators/execution.hpp"

#include <cstdint>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_sequential_engine() {
  static const dynamic_bfs_engine_impl<operators::sequential_exec, vertex_t, edge_t> engine;
  return engine;
}

template const dynamic_bfs_engine<std::int32_t, std::int32_t>&
dynamic_bfs_sequential_engine<std::int32_t, std::int32_t>();
template const dynamic_bfs_engine<std::int32_t, std::int64_t>&
dynamic_bfs_sequential_engine<std::int32_t, std::int64_t>();
template const dynamic_bfs_engine<std::int64_t, std::int64_t>&
dynamic_bfs_sequential_engine<std::int64_t, std::int64_t>();

}  // namespace dyng::detail
