// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_openmp.cpp
// (accumulate_phase_parallel)
/**
 * @file openmp.cpp
 * @brief update() on the OpenMP backend: one phase of CycleEnumeration-GPU's
 *        update_static_histogram_openmp() (the count(-) and count(+) hooks).
 *
 * Straight port of accumulate_phase_parallel(): the change edges of a phase in parallel
 * (schedule(dynamic)); within a phase the graph snapshot is read only, and edge-id ownership makes
 * each affected cycle the responsibility of exactly one edge, so the per-edge work needs no
 * coordination beyond a final reduction of the per-thread histograms. The mechanical changes: the
 * templates, the thread count from the resources, and the per-thread arrays (visited marks,
 * per-edge counts, per-thread sums) in the pooled workspace instead of `static thread_local` and
 * per-call vectors.
 */
#include "algorithms/cycle_count/cycles_through_edge.hpp"
#include "algorithms/cycle_count/problem.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail {

template <typename vertex_t, typename edge_t>
void cycle_count_openmp_phase(const cycle_graph<vertex_t, edge_t>& graph,
                              const std::vector<edge_change<vertex_t>>& changes,
                              std::size_t max_length, int threads,
                              cycle_count_workspace<vertex_t>& ws,
                              std::vector<std::uint64_t>& phase) {
  if (changes.empty()) {
    return;
  }
#if DYNG_HAS_OPENMP
  for (int t = 0; t < threads; ++t) {
    std::uint64_t* mine = ws.thread_partial(static_cast<std::size_t>(t));
    std::fill(mine, mine + max_length + 1, 0);
  }

#pragma omp parallel num_threads(threads)
  {
    const auto thread_id = static_cast<std::size_t>(omp_get_thread_num());
    std::uint64_t* mine = ws.thread_partial(thread_id);
    std::uint64_t* counts = ws.thread_counts(thread_id);
    std::vector<char>& visited = ws.visited[thread_id];

#pragma omp for schedule(dynamic)
    for (std::ptrdiff_t owner = 0; owner < static_cast<std::ptrdiff_t>(changes.size()); ++owner) {
      std::fill(counts, counts + max_length + 1, 0);
      const edge_change<vertex_t>& change = changes[static_cast<std::size_t>(owner)];
      count_cycles_through_edge(graph, change.source, change.target,
                                static_cast<std::size_t>(owner), ws.index, max_length, visited,
                                counts);
      for (std::size_t length = 2; length <= max_length; ++length) {
        mine[length] += counts[length];
      }
    }
  }

  for (int t = 0; t < threads; ++t) {
    const std::uint64_t* partial = ws.thread_partial(static_cast<std::size_t>(t));
    for (std::size_t length = 2; length <= max_length; ++length) {
      cycle_count_checked_add(phase[length], partial[length]);
    }
  }
#else
  (void)graph;
  (void)max_length;
  (void)threads;
  (void)ws;
  (void)phase;
  DYNG_FAIL("cycle_count: the OpenMP phase in a build without OpenMP");
#endif
}

template void cycle_count_openmp_phase<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, int, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);
template void cycle_count_openmp_phase<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, const std::vector<edge_change<std::int32_t>>&,
    std::size_t, int, cycle_count_workspace<std::int32_t>&, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
