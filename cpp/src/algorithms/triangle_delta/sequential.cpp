// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sequential.cpp
 * @brief triangle_delta: the sequential backend, the reference of the other backends (PLAN Section
 *        4.8, file 6): the counts of engine.hpp run by the sequential executor (one thread, in
 *        index order).
 */
#include "algorithms/triangle_delta/engine.hpp"
#include "algorithms/triangle_delta/problem.hpp"
#include "operators/execution.hpp"

#include <cstdint>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
const triangle_delta_engine<vertex_t, edge_t>& triangle_delta_sequential_engine() {
  static const triangle_delta_engine_impl<operators::sequential_exec, vertex_t, edge_t> engine;
  return engine;
}

template const triangle_delta_engine<std::int32_t, std::int32_t>&
triangle_delta_sequential_engine<std::int32_t, std::int32_t>();
template const triangle_delta_engine<std::int32_t, std::int64_t>&
triangle_delta_sequential_engine<std::int32_t, std::int64_t>();
template const triangle_delta_engine<std::int64_t, std::int64_t>&
triangle_delta_sequential_engine<std::int64_t, std::int64_t>();

}  // namespace dyng::detail
