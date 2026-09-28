// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_openmp.cpp
// (accumulate_phase_parallel)
/**
 * @file openmp.cpp
 * @brief update() on the OpenMP backend: one phase of CycleEnumeration-GPU's
 *        update_static_histogram_openmp() (the count(-) and count(+) hooks).
 *
 * Port of accumulate_phase_parallel(): the change edges of a phase in parallel
 * (schedule(dynamic)); within a phase the graph snapshot is read only, and edge-id ownership makes
 * each affected cycle the responsibility of exactly one edge, so the per-edge work needs no
 * coordination beyond a final reduction of the per-thread histograms. The mechanical changes: the
 * templates, the thread count from the resources, and the per-thread arrays (visited marks, search
 * stack, per-thread sums) in the pooled workspace instead of `static thread_local` and per-call
 * vectors. Each thread sizes its own scratch at the start of the one parallel region that uses it
 * (so the pages are first touched in parallel, and a team smaller or larger than an earlier one
 * cannot meet unsized marks). The per-edge counts go straight into the thread's counters (see
 * cycles_through_edge.hpp), and only the lengths a search reached are summed and cleared.
 */
#include "algorithms/cycle_count/cycles_through_edge.hpp"
#include "algorithms/cycle_count/problem.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
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
  const auto thread_count = static_cast<std::size_t>(std::max(threads, 1));
  if (ws.threads.size() < thread_count || ws.marks_needed < graph.vertex_count) {
    DYNG_FAIL("cycle_count: the workspace is not reserved for the phase (", thread_count,
              " threads, ", graph.vertex_count, " vertices; it has ", ws.threads.size(),
              " threads, marks for ", ws.marks_needed, ")");
  }
  for (std::size_t t = 0; t < thread_count; ++t) {
    ws.threads[t].reached = 0;
  }
  const std::size_t lengths = std::min<std::size_t>(max_length, 64) + 1;
  const std::size_t marks = ws.marks_needed;

  // An allocation failure inside the region is reported after it (an exception must not leave a
  // parallel region); the thread that failed skips its remaining change edges.
  bool failed = false;
#pragma omp parallel num_threads(threads) reduction(|| : failed)
  {
    cycle_count_thread<vertex_t>& mine = ws.threads[static_cast<std::size_t>(omp_get_thread_num())];
    bool ready = true;
    try {
      mine.prepare(marks, lengths);
    } catch (...) {
      ready = false;
    }
    std::size_t reached = 0;

#pragma omp for schedule(dynamic)
    for (std::ptrdiff_t owner = 0; owner < static_cast<std::ptrdiff_t>(changes.size()); ++owner) {
      if (!ready) {
        continue;
      }
      const edge_change<vertex_t>& change = changes[static_cast<std::size_t>(owner)];
      try {
        reached = std::max(reached, count_cycles_through_edge(graph, change.source, change.target,
                                                              static_cast<std::size_t>(owner),
                                                              ws.index, max_length, mine));
      } catch (...) {
        ready = false;
      }
    }
    mine.reached = reached;
    failed = failed || !ready;
  }
  if (failed) {
    ws.reset_threads();
    throw std::bad_alloc();
  }
  cycle_count_drain(ws, thread_count, phase);
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
