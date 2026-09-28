// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/cycles_through_edge.cpp
// (count_cycles_through_edge, search)
/**
 * @file cycles_through_edge.hpp
 * @brief Edge-anchored simple-cycle enumeration with edge-id ownership: the count of one change
 *        edge in the count(-) and count(+) hooks of update().
 *
 * Straight port of count_cycles_through_edge(): the mechanical changes are the templates on the
 * index types, the graph as a cycle_graph, and the visited buffer passed in by the caller (the
 * original keeps a `static thread_local` buffer; dynG keeps one per thread in the pooled
 * workspace, no globals).
 */
#pragma once

#include "algorithms/cycle_count/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace cycle_count_kernels {

/// Depth-first search for simple paths from `current` back to the anchored source `u`.
/// `path_size` is the number of vertices on the path starting at the anchored target `v`; closing
/// at `u` yields a cycle of length `path_size + 1`.
template <typename vertex_t, typename edge_t>
void search(const cycle_graph<vertex_t, edge_t>& graph, const vertex_t u, const vertex_t v,
            const std::size_t owner_id, const changed_edge_index& phase_changes,
            const std::size_t max_length, const vertex_t current, const std::size_t path_size,
            std::vector<char>& visited, std::uint64_t* counts) {
  const auto begin = static_cast<std::size_t>(graph.offsets[current]);
  const auto end = static_cast<std::size_t>(graph.offsets[current + 1]);
  for (std::size_t offset = begin; offset < end; ++offset) {
    const vertex_t next = graph.neighbors[offset];

    if (phase_changes.forbidden_before(current, next, owner_id)) {
      continue;  // owned by a smaller-id changed edge
    }

    if (next == u) {
      const std::size_t length = path_size + 1;
      if (length >= 2 && length <= max_length) {
        counts[length] += 1;
      }
      continue;
    }

    if (next == v || visited[static_cast<std::size_t>(next)] != 0) {
      continue;
    }

    if (path_size + 1 < max_length) {
      visited[static_cast<std::size_t>(next)] = 1;
      search(graph, u, v, owner_id, phase_changes, max_length, next, path_size + 1, visited,
             counts);
      visited[static_cast<std::size_t>(next)] = 0;
    }
  }
}

}  // namespace cycle_count_kernels

/**
 * @brief Count the simple cycles through the directed edge (source -> target) that it owns,
 *        bucketed by length (CycleEnumeration-GPU's count_cycles_through_edge).
 *
 * The cycle is source -> target -> ... -> source. It is owned by this edge iff it contains no
 * same-phase changed edge with a smaller id, which the search enforces by refusing to cross such
 * edges. Each found cycle of length len (2 <= len <= max_length) adds one to counts[len].
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     graph         The phase's frozen graph.
 * @param[in]     source        Tail of the anchored edge.
 * @param[in]     target        Head of the anchored edge.
 * @param[in]     owner_id      Ownership id of the anchored edge in its phase.
 * @param[in]     phase_changes Ownership index of the phase.
 * @param[in]     max_length    Longest counted length (>= 2).
 * @param[in,out] visited       Path marks, >= graph.vertex_count entries, all zero (and zero again
 *                              on return).
 * @param[out]    counts        Bucket array of max_length + 1 entries.
 */
template <typename vertex_t, typename edge_t>
void count_cycles_through_edge(const cycle_graph<vertex_t, edge_t>& graph, const vertex_t source,
                               const vertex_t target, const std::size_t owner_id,
                               const changed_edge_index& phase_changes,
                               const std::size_t max_length, std::vector<char>& visited,
                               std::uint64_t* counts) {
  if (max_length < 2 || static_cast<std::size_t>(target) >= graph.vertex_count ||
      static_cast<std::size_t>(source) >= graph.vertex_count) {
    return;
  }
  visited[static_cast<std::size_t>(source)] = 1;
  visited[static_cast<std::size_t>(target)] = 1;
  cycle_count_kernels::search(graph, source, target, owner_id, phase_changes, max_length, target, 1,
                              visited, counts);
  visited[static_cast<std::size_t>(source)] = 0;
  visited[static_cast<std::size_t>(target)] = 0;
}

}  // namespace dyng::detail
