// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_sequential.cpp (accumulate_phase)
/**
 * @file sequential.cpp
 * @brief update() on the sequential backend: one phase of CycleEnumeration-GPU's
 *        update_static_histogram() (the count(-) and count(+) hooks).
 *
 * Straight port of accumulate_phase(): for every change edge in id order, the cycles through it
 * that it owns (count_cycles_through_edge) are added to the phase's histogram. The sign is applied
 * by the finalize hook (the original accumulates with the sign into one delta; the sums are
 * identical). The ownership index of the phase is built by the caller (the hook that owns the
 * phase), from the same normalized change list.
 */
#include "algorithms/cycle_count/cycles_through_edge.hpp"
#include "algorithms/cycle_count/problem.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
void cycle_count_sequential_phase(const cycle_graph<vertex_t, edge_t>& graph,
                                  const std::vector<edge_change<vertex_t>>& changes,
                                  std::size_t max_length, cycle_count_workspace<vertex_t>& ws,
                                  std::vector<std::uint64_t>& phase) {
  std::uint64_t* counts = ws.thread_counts(0);
  std::vector<char>& visited = ws.visited[0];
  for (std::size_t owner_id = 0; owner_id < changes.size(); ++owner_id) {
    std::fill(counts, counts + max_length + 1, 0);
    count_cycles_through_edge(graph, changes[owner_id].source, changes[owner_id].target, owner_id,
                              ws.index, max_length, visited, counts);
    for (std::size_t length = 2; length <= max_length; ++length) {
      phase[length] += counts[length];
    }
  }
}

template void cycle_count_sequential_phase<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);
template void cycle_count_sequential_phase<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
