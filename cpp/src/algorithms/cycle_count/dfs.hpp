// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/cuda/cuda_dfs.cuh (extend_prefix,
// find_edge, lower_bound_u32, dispatch_capacity, kMaxDeviceCycleLength)
/**
 * @file dfs.hpp
 * @brief The exact pruned depth-first search with the lower_bound closure test: the search of
 *        CycleEnumeration-GPU's CUDA static counters, ported for the host.
 *
 * Rows are sorted by neighbour id and hold distinct neighbours, so one lower_bound of the root in
 * the row of each pushed vertex gives both the closing test (the root is present) and the first
 * neighbour greater than the root: neighbours at or below the root are never read and the last
 * level never scans a row. The search keeps its path and cursors in arrays of a compile-time
 * capacity, as the device code does with thread-local arrays.
 *
 * The CPU backends of the original (and of dynG) use the Johnson searches of static_sequential.cpp
 * and static_openmp.cpp; this is the building block of the CUDA static kernels (the root, edge and
 * two-hop work items of the work queue), ported straight in M2a and checked on the host against
 * the oracles so that M2b (the CUDA backend) builds its kernels on a tested search. The mechanical
 * changes: templates on the index types, std::lower_bound for the device's lower_bound_u32 and
 * plain loads for __ldg, a plain counts array for ThreadHistogram (whose warp reduction is device
 * code).
 */
#pragma once

#include "algorithms/cycle_count/problem.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dyng::detail {

/// Largest cycle length of the fixed-capacity search (kMaxDeviceCycleLength).
inline constexpr int cycle_count_max_capacity = 64;

/**
 * @brief Call `body(std::integral_constant<int, cap>{})` with the smallest supported capacity
 *        (4, 8, 16, 32, 64) that holds cycles of length `max_cycle_length` (dispatch_capacity).
 * @tparam body_t Callable.
 * @param[in] max_cycle_length Longest counted length.
 * @param[in] body             The callable.
 * @throws invalid_argument_error if `max_cycle_length` exceeds cycle_count_max_capacity.
 */
template <typename body_t>
void cycle_count_dispatch_capacity(std::size_t max_cycle_length, body_t&& body) {
  if (max_cycle_length <= 4) {
    body(std::integral_constant<int, 4>{});
  } else if (max_cycle_length <= 8) {
    body(std::integral_constant<int, 8>{});
  } else if (max_cycle_length <= 16) {
    body(std::integral_constant<int, 16>{});
  } else if (max_cycle_length <= 32) {
    body(std::integral_constant<int, 32>{});
  } else if (max_cycle_length <= static_cast<std::size_t>(cycle_count_max_capacity)) {
    body(std::integral_constant<int, cycle_count_max_capacity>{});
  } else {
    DYNG_EXPECTS(false, "cycle_count: the fixed-capacity search supports max_length up to ",
                 cycle_count_max_capacity, ", got ", max_cycle_length);
  }
}

/**
 * @brief Whether the edge from -> to exists (find_edge); `position` receives the lower_bound of
 *        `to` in the row of `from`, `row_end` the end of that row.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]  graph    The graph (sorted rows without repeats).
 * @param[in]  from     Tail.
 * @param[in]  to       Head.
 * @param[out] position lower_bound of `to` in the row of `from`.
 * @param[out] row_end  End of the row of `from`.
 * @return true if the edge exists.
 */
template <typename vertex_t, typename edge_t>
bool cycle_count_find_edge(const cycle_graph<vertex_t, edge_t>& graph, vertex_t from, vertex_t to,
                           std::size_t& position, std::size_t& row_end) {
  const auto begin = static_cast<std::size_t>(graph.offsets[from]);
  row_end = static_cast<std::size_t>(graph.offsets[from + 1]);
  position = static_cast<std::size_t>(
      std::lower_bound(graph.neighbors + begin, graph.neighbors + row_end, to) - graph.neighbors);
  return position < row_end && graph.neighbors[position] == to;
}

/**
 * @brief Exact pruned depth-first search below a path prefix (extend_prefix).
 *
 * `path[0]` is the root and `path[1..prefix-1]` are distinct vertices greater than the root; the
 * caller has already counted the cycle that closes the prefix itself. The search counts, in
 * `counts`, every simple cycle of length at most `max_length` whose minimum vertex is the root and
 * that begins with the prefix, each exactly once.
 * @tparam cap      Capacity (longest path; >= max_length).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph      The graph (sorted rows without repeats).
 * @param[in,out] path       The prefix (entries from `prefix` on are overwritten).
 * @param[in]     prefix     Length of the prefix (>= 1).
 * @param[in]     max_length Longest counted length (<= cap).
 * @param[in,out] counts     counts[len] += cycles of length len (cap + 1 entries).
 */
template <int cap, typename vertex_t, typename edge_t>
void cycle_count_extend_prefix(const cycle_graph<vertex_t, edge_t>& graph, vertex_t (&path)[cap],
                               const int prefix, const int max_length,
                               std::uint64_t (&counts)[cap + 1]) {
  if (prefix >= max_length) {
    return;
  }
  const vertex_t root = path[0];
  std::size_t cursor[cap];
  std::size_t end[cap];

  int depth = prefix;
  {
    std::size_t position = 0;
    std::size_t row_end = 0;
    const bool closes = cycle_count_find_edge(graph, path[depth - 1], root, position, row_end);
    cursor[depth - 1] = position + (closes ? 1U : 0U);
    end[depth - 1] = row_end;
  }

  while (depth >= prefix) {
    if (cursor[depth - 1] >= end[depth - 1]) {
      --depth;
      continue;
    }
    const vertex_t next = graph.neighbors[cursor[depth - 1]];
    ++cursor[depth - 1];

    bool on_path = false;
    for (int index = 1; index < cap; ++index) {
      if (index < depth && path[index] == next) {
        on_path = true;
      }
    }
    if (on_path) {
      continue;
    }

    std::size_t position = 0;
    std::size_t row_end = 0;
    const bool closes = cycle_count_find_edge(graph, next, root, position, row_end);
    if (closes) {
      ++counts[depth + 1];  // cycle root, path[1..depth-1], next
    }
    if (depth + 1 < max_length) {
      path[depth] = next;
      cursor[depth] = position + (closes ? 1U : 0U);
      end[depth] = row_end;
      ++depth;
    }
  }
}

}  // namespace dyng::detail
