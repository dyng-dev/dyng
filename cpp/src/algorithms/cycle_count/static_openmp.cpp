// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/openmp/openmp_johnson.cpp (count_root,
// count_single_thread, count_simple_cycles_johnson)
/**
 * @file static_openmp.cpp
 * @brief compute() on the OpenMP backend: CycleEnumeration-GPU's coarse-grained OpenMP counter.
 *
 * Port of openmp::count_simple_cycles_johnson(): one path search per root, roots distributed with
 * schedule(dynamic), one histogram per thread merged at the end; one thread runs the single-thread
 * loop. The duplicate-avoidance rule is that of the sequential Johnson search (each cycle counted
 * from its smallest vertex). The mechanical changes: the templates on the index types, the graph
 * as a cycle_graph, the thread count from the resources (never the global OpenMP setting), dense
 * per-thread counts arrays instead of CycleHistogram maps (grown on demand, so an unbounded count
 * does not allocate threads x n counters), and the search of count_root() with an explicit stack
 * (root_search.hpp).
 *
 * Without a length bound the search enumerates simple paths (the original has no blocked lists
 * here either): its cost is exponential in the number of simple paths even on graphs with few
 * cycles, where the sequential backend's Johnson search is linear per cycle.
 */
#include "algorithms/cycle_count/problem.hpp"
#include "algorithms/cycle_count/root_search.hpp"

#include <dyng/config.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail {

namespace {

template <typename vertex_t, typename edge_t>
void count_single_thread(const cycle_graph<vertex_t, edge_t>& graph, const std::size_t max_length,
                         std::vector<std::uint64_t>& counts) {
  cycle_root_scratch<vertex_t> scratch;
  for (std::size_t root = 0; root < graph.vertex_count; ++root) {
    cycle_count_search_root(graph, static_cast<vertex_t>(root), max_length, counts, scratch);
  }
}

}  // namespace

template <typename vertex_t, typename edge_t>
void cycle_count_openmp_compute(const cycle_graph<vertex_t, edge_t>& graph, std::int64_t max_length,
                                int threads, std::vector<std::uint64_t>& counts) {
  // counts has min(max_length, max(n, 2)) + 1 entries: no simple cycle is longer than n, so the
  // cap of the search is its last index (without a bound: a path never reaches it).
  (void)max_length;
  const std::size_t cap = counts.size() - 1;
#if DYNG_HAS_OPENMP
  if (threads > 1) {
    const std::size_t initial = std::min<std::size_t>(cap, 64) + 1;
    std::vector<std::vector<std::uint64_t>> local_histograms(static_cast<std::size_t>(threads));

    bool failed = false;
#pragma omp parallel num_threads(threads) reduction(|| : failed)
    {
      const int thread_id = omp_get_thread_num();
      std::vector<std::uint64_t>& local = local_histograms[static_cast<std::size_t>(thread_id)];
      cycle_root_scratch<vertex_t> scratch;
      bool ready = true;
      try {
        local.assign(initial, 0);
      } catch (...) {
        ready = false;
      }

#pragma omp for schedule(dynamic)
      for (std::ptrdiff_t root = 0; root < static_cast<std::ptrdiff_t>(graph.vertex_count);
           ++root) {
        if (!ready) {
          continue;
        }
        try {
          cycle_count_search_root(graph, static_cast<vertex_t>(root), cap, local, scratch);
        } catch (...) {
          ready = false;
        }
      }
      failed = failed || !ready;
    }
    if (failed) {
      throw std::bad_alloc();
    }

    for (const std::vector<std::uint64_t>& local : local_histograms) {
      for (std::size_t len = 0; len < local.size(); ++len) {
        cycle_count_checked_add(counts[len], local[len]);
      }
    }
    return;
  }
#endif

  (void)threads;
  count_single_thread(graph, cap, counts);
}

template void cycle_count_openmp_compute<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, std::int64_t, int, std::vector<std::uint64_t>&);
template void cycle_count_openmp_compute<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, std::int64_t, int, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
