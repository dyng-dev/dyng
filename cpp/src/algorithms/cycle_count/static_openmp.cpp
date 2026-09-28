// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/openmp/openmp_johnson.cpp (count_root,
// count_single_thread, count_simple_cycles_johnson)
/**
 * @file static_openmp.cpp
 * @brief compute() on the OpenMP backend: CycleEnumeration-GPU's coarse-grained OpenMP counter.
 *
 * Straight port of openmp::count_simple_cycles_johnson(): one path search per root, roots
 * distributed with schedule(dynamic), one histogram per thread merged at the end; one thread runs
 * the single-thread loop. The duplicate-avoidance rule is that of the sequential Johnson search
 * (each cycle counted from its smallest vertex). The mechanical changes: the templates on the
 * index types, the graph as a cycle_graph, the thread count from the resources (never the global
 * OpenMP setting), and dense per-thread counts arrays instead of CycleHistogram maps.
 */
#include "algorithms/cycle_count/problem.hpp"

#include <dyng/config.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail {

namespace {

template <typename vertex_t, typename edge_t>
void count_root(const cycle_graph<vertex_t, edge_t>& graph, const vertex_t root,
                const std::size_t max_cycle_length, std::vector<std::uint64_t>& histogram,
                std::vector<unsigned char>& visited, std::vector<vertex_t>& path) {
  // The search clears every mark it sets, so the buffer is all-zero between roots: size it once
  // instead of clearing all V entries per root, which made the whole count O(V^2).
  if (visited.size() != graph.vertex_count) {
    visited.assign(graph.vertex_count, 0);
  }
  path.clear();
  path.push_back(root);
  visited[static_cast<std::size_t>(root)] = 1;

  auto dfs = [&](auto& self, const vertex_t current) -> void {
    const auto begin = static_cast<std::size_t>(graph.offsets[current]);
    const auto end = static_cast<std::size_t>(graph.offsets[current + 1]);

    for (std::size_t offset = begin; offset < end; ++offset) {
      const vertex_t next = graph.neighbors[offset];

      if (next == root && path.size() >= 2) {
        cycle_count_record(histogram, path.size());
        continue;
      }

      if (next <= root || visited[static_cast<std::size_t>(next)] != 0) {
        continue;
      }

      if (path.size() >= max_cycle_length) {
        continue;
      }

      visited[static_cast<std::size_t>(next)] = 1;
      path.push_back(next);
      self(self, next);
      path.pop_back();
      visited[static_cast<std::size_t>(next)] = 0;
    }
  };

  dfs(dfs, root);
  visited[static_cast<std::size_t>(root)] = 0;
}

template <typename vertex_t, typename edge_t>
void count_single_thread(const cycle_graph<vertex_t, edge_t>& graph,
                         const std::size_t max_cycle_length, std::vector<std::uint64_t>& counts) {
  std::vector<unsigned char> visited;
  std::vector<vertex_t> path;
  path.reserve(graph.vertex_count);

  for (std::size_t root = 0; root < graph.vertex_count; ++root) {
    count_root(graph, static_cast<vertex_t>(root), max_cycle_length, counts, visited, path);
  }
}

}  // namespace

template <typename vertex_t, typename edge_t>
void cycle_count_openmp_compute(const cycle_graph<vertex_t, edge_t>& graph, std::int64_t max_length,
                                int threads, std::vector<std::uint64_t>& counts) {
  // No bound: a path never reaches the cap (the original's empty std::optional).
  const std::size_t max_cycle_length = max_length < 0 ? std::numeric_limits<std::size_t>::max()
                                                      : static_cast<std::size_t>(max_length);
#if DYNG_HAS_OPENMP
  if (threads > 1) {
    const std::size_t length = counts.size();
    std::vector<std::vector<std::uint64_t>> local_histograms(static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
      const int thread_id = omp_get_thread_num();
      std::vector<std::uint64_t>& local = local_histograms[static_cast<std::size_t>(thread_id)];
      local.assign(length, 0);
      std::vector<unsigned char> visited;
      std::vector<vertex_t> path;
      path.reserve(graph.vertex_count);

#pragma omp for schedule(dynamic)
      for (std::ptrdiff_t root = 0; root < static_cast<std::ptrdiff_t>(graph.vertex_count);
           ++root) {
        count_root(graph, static_cast<vertex_t>(root), max_cycle_length, local, visited, path);
      }
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
  count_single_thread(graph, max_cycle_length, counts);
}

template void cycle_count_openmp_compute<std::int32_t, std::int32_t>(
    const cycle_graph<std::int32_t, std::int32_t>&, std::int64_t, int, std::vector<std::uint64_t>&);
template void cycle_count_openmp_compute<std::int32_t, std::int64_t>(
    const cycle_graph<std::int32_t, std::int64_t>&, std::int64_t, int, std::vector<std::uint64_t>&);

}  // namespace dyng::detail
