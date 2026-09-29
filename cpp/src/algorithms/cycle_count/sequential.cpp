// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_sequential.cpp (accumulate_phase)
/**
 * @file sequential.cpp
 * @brief update() on the sequential backend: one phase of CycleEnumeration-GPU's
 *        update_static_histogram() (the count(-) and count(+) hooks).
 *
 * Port of accumulate_phase(): for every change edge in id order, the cycles through it that it
 * owns (count_cycles_through_edge) are added to the phase's histogram. The sign is applied by the
 * finalize hook (the original accumulates with the sign into one delta; the sums are identical).
 * The ownership index of the phase is built by the caller (the hook that owns the phase), from the
 * same normalized change list. The per-edge counts go straight into the thread's counters (see
 * cycles_through_edge.hpp), and only the lengths a search reached are summed and cleared.
 */
#include "algorithms/cycle_count/cycles_through_edge.hpp"
#include "algorithms/cycle_count/problem.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

template <typename vertex_t>
void cycle_count_drain(cycle_count_workspace<vertex_t>& ws, std::size_t thread_count,
                       std::vector<std::uint64_t>& phase) {
  try {
    std::size_t reached = 0;
    for (std::size_t t = 0; t < thread_count; ++t) {
      reached = std::max(reached, ws.threads[t].reached);
    }
    if (reached >= 2 && phase.size() < reached + 1) {
      phase.resize(reached + 1, 0);
    }
    for (std::size_t t = 0; t < thread_count; ++t) {
      cycle_count_thread<vertex_t>& mine = ws.threads[t];
      for (std::size_t length = 2; length <= mine.reached; ++length) {
        const std::uint64_t value = mine.partial[length];
        mine.partial[length] = 0;
        cycle_count_checked_add(phase[length], value);
      }
      mine.reached = 0;
    }
  } catch (...) {
    ws.reset_threads();
    throw;
  }
}

template void cycle_count_drain<std::int32_t>(cycle_count_workspace<std::int32_t>&, std::size_t,
                                              std::vector<std::uint64_t>&);

template <typename vertex_t, typename edge_t>
void cycle_count_sequential_phase(const cycle_graph<vertex_t, edge_t>& graph,
                                  const std::vector<edge_change<vertex_t>>& changes,
                                  std::size_t max_length, cycle_count_workspace<vertex_t>& ws,
                                  std::vector<std::uint64_t>& phase) {
  if (changes.empty()) {
    return;
  }
  if (ws.threads.empty() || ws.marks_needed < graph.vertex_count) {
    DYNG_FAIL("cycle_count: the workspace is not reserved for the phase's graph (",
              graph.vertex_count, " vertices, marks for ", ws.marks_needed, ")");
  }
  cycle_count_thread<vertex_t>& mine = ws.threads[0];
  try {
    mine.prepare(ws.marks_needed, std::min<std::size_t>(max_length, 64) + 1);
    std::size_t reached = 0;
    for (std::size_t owner_id = 0; owner_id < changes.size(); ++owner_id) {
      reached = std::max(reached, count_cycles_through_edge(graph, changes[owner_id].source,
                                                            changes[owner_id].target, owner_id,
                                                            ws.index, max_length, mine));
    }
    mine.reached = reached;
  } catch (...) {
    mine.reset();
    throw;
  }
  cycle_count_drain(ws, 1, phase);
}

template void cycle_count_sequential_phase<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);
template void cycle_count_sequential_phase<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
