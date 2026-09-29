// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:tests/support/cycle_oracles.hpp (oracle_simple_cycles),
// src/sequential/bruteforce.cpp (count_simple_cycles_bruteforce) and
// tests/integration/randomized_update_parity_test.cpp (reference_final_view)
/**
 * @file cycle_oracle.cpp
 * @brief dyng::testing cycle oracles: subset DP, brute force, edge-set recount.
 *
 * Mechanical changes only: names, templates on the index types, a csr_view instead of a GraphView,
 * a dense histogram vector instead of the CycleHistogram map, invalid_argument_error instead of
 * std::invalid_argument, and an explicit DFS stack frame instead of the recursive std::function.
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace dyng::testing {

namespace {

template <typename value_t>
void expect_host(const array_view<value_t>& view, const char* what) {
  DYNG_EXPECTS(view.empty() || is_host_accessible(view.space()), what,
               " must be in host-accessible memory");
}

/// The size of the result histogram (max_length + 1, or n + 1 without a bound, at least 3).
std::size_t histogram_size(std::int64_t n, int max_length) {
  DYNG_EXPECTS(max_length == -1 || max_length >= 2,
               "cycle oracle: max_length must be -1 (no bound) or >= 2, got ", max_length);
  return static_cast<std::size_t>(max_length == -1 ? std::max<std::int64_t>(n, 2) : max_length) + 1;
}

int popcount(std::uint32_t mask) {
  int count = 0;
  for (std::uint32_t m = mask; m != 0; m &= m - 1) {
    ++count;
  }
  return count;
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
cycle_histogram oracle_simple_cycles(const csr_view<vertex_t, edge_t, weight_t>& graph,
                                     int max_length) try {
  expect_host(graph.row_ptr, "oracle_simple_cycles: the row offsets");
  expect_host(graph.col_ind, "oracle_simple_cycles: the neighbours");
  const auto n = static_cast<std::size_t>(graph.num_vertices());
  DYNG_EXPECTS(n <= 16, "oracle graphs are limited to 16 vertices, got ", n);
  cycle_histogram histogram(histogram_size(static_cast<std::int64_t>(n), max_length), 0);

  // Distinct out-neighbours per vertex, ignoring self-loops.
  std::vector<std::vector<std::size_t>> out(n);
  std::vector<std::vector<bool>> edge(n, std::vector<bool>(n, false));
  for (std::size_t u = 0; u < n; ++u) {
    for (auto e = static_cast<std::size_t>(graph.row_ptr[u]);
         e < static_cast<std::size_t>(graph.row_ptr[u + 1]); ++e) {
      const auto v = static_cast<std::size_t>(graph.col_ind[e]);
      if (v != u) {
        out[u].push_back(v);
      }
    }
    std::sort(out[u].begin(), out[u].end());
    out[u].erase(std::unique(out[u].begin(), out[u].end()), out[u].end());
    for (const std::size_t v : out[u]) {
      edge[u][v] = true;
    }
  }

  std::vector<std::uint64_t> counts(n + 1, 0);
  const std::uint32_t full = n == 0 ? 0U : ((1U << n) - 1U);
  std::vector<std::uint64_t> paths((static_cast<std::size_t>(full) + 1) * n);
  for (std::size_t s = 0; s < n; ++s) {
    std::fill(paths.begin(), paths.end(), 0);
    const std::uint32_t start = 1U << s;
    paths[start * n + s] = 1;
    // Masks only grow, so increasing numeric order is a topological order.
    for (std::uint32_t mask = start; mask <= full; ++mask) {
      if ((mask & start) == 0 || (mask & ((1U << s) - 1U)) != 0) {
        continue;
      }
      for (std::size_t v = 0; v < n; ++v) {
        const std::uint64_t ways = paths[mask * n + v];
        if (ways == 0) {
          continue;
        }
        if (v != s && edge[v][s]) {
          counts[static_cast<std::size_t>(popcount(mask))] += ways;
        }
        for (const std::size_t w : out[v]) {
          if (w > s && (mask & (1U << w)) == 0) {
            paths[(mask | (1U << w)) * n + w] += ways;
          }
        }
      }
    }
  }
  for (std::size_t length = 2; length <= n && length < histogram.size(); ++length) {
    histogram[length] = counts[length];
  }
  return histogram;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::oracle_simple_cycles")

template <typename vertex_t, typename edge_t, typename weight_t>
cycle_histogram brute_force_simple_cycles(const csr_view<vertex_t, edge_t, weight_t>& graph,
                                          int max_length) try {
  expect_host(graph.row_ptr, "brute_force_simple_cycles: the row offsets");
  expect_host(graph.col_ind, "brute_force_simple_cycles: the neighbours");
  const auto n = static_cast<std::size_t>(graph.num_vertices());
  cycle_histogram histogram(histogram_size(static_cast<std::int64_t>(n), max_length), 0);
  const std::size_t bound = max_length == -1 ? n + 1 : static_cast<std::size_t>(max_length);
  std::vector<unsigned char> visited(n, 0);
  std::vector<std::size_t> path;         // the vertices of the current path, the root first
  std::vector<std::size_t> next_offset;  // per path vertex: the next row entry to follow
  path.reserve(n);
  next_offset.reserve(n);

  for (std::size_t root = 0; root < n; ++root) {
    path.assign(1, root);
    next_offset.assign(1, static_cast<std::size_t>(graph.row_ptr[root]));
    visited[root] = 1;
    while (!path.empty()) {
      const std::size_t current = path.back();
      std::size_t& offset = next_offset.back();
      if (offset == static_cast<std::size_t>(graph.row_ptr[current + 1])) {
        visited[current] = 0;
        path.pop_back();
        next_offset.pop_back();
        continue;
      }
      const auto next = static_cast<std::size_t>(graph.col_ind[offset++]);
      if (next == root && path.size() >= 2) {
        if (path.size() < histogram.size()) {
          ++histogram[path.size()];
        }
        continue;
      }
      if (next <= root || visited[next] != 0) {
        continue;
      }
      if (path.size() >= bound) {
        continue;
      }
      visited[next] = 1;
      path.push_back(next);
      next_offset.push_back(static_cast<std::size_t>(graph.row_ptr[next]));
    }
  }
  return histogram;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::brute_force_simple_cycles")

template <typename vertex_t, typename edge_t, typename weight_t>
csr<vertex_t, edge_t, weight_t> edge_set_after_batch(
    const csr_view<vertex_t, edge_t, weight_t>& graph,
    const edge_batch_view<vertex_t, weight_t>& batch) try {
  expect_host(graph.row_ptr, "edge_set_after_batch: the row offsets");
  expect_host(graph.col_ind, "edge_set_after_batch: the neighbours");
  expect_host(batch.insert_src, "edge_set_after_batch: the insertions");
  expect_host(batch.insert_dst, "edge_set_after_batch: the insertions");
  expect_host(batch.delete_src, "edge_set_after_batch: the deletions");
  expect_host(batch.delete_dst, "edge_set_after_batch: the deletions");
  const std::int64_t n = graph.num_vertices();
  std::set<std::pair<vertex_t, vertex_t>> edges;
  for (std::int64_t u = 0; u < n; ++u) {
    for (auto e = static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(u)]);
         e < static_cast<std::size_t>(graph.row_ptr[static_cast<std::size_t>(u) + 1]); ++e) {
      edges.emplace(static_cast<vertex_t>(u), graph.col_ind[e]);
    }
  }
  std::int64_t vertex_count = n;
  for (std::size_t j = 0; j < batch.delete_src.size(); ++j) {
    DYNG_EXPECTS(batch.delete_src[j] >= 0 && batch.delete_dst[j] >= 0, "deletion ", j,
                 " has a negative id");
    edges.erase({batch.delete_src[j], batch.delete_dst[j]});
  }
  for (std::size_t i = 0; i < batch.insert_src.size(); ++i) {
    const vertex_t u = batch.insert_src[i];
    const vertex_t v = batch.insert_dst[i];
    DYNG_EXPECTS(u >= 0 && v >= 0, "insertion ", i, " has a negative id");
    if (u != v) {
      edges.emplace(u, v);
      vertex_count = std::max<std::int64_t>(vertex_count, std::max<std::int64_t>(u, v) + 1);
    }
  }
  csr<vertex_t, edge_t, weight_t> out;
  out.num_weights = 0;
  out.row_ptr.assign(static_cast<std::size_t>(vertex_count) + 1, edge_t{0});
  out.col_ind.reserve(edges.size());
  for (const auto& [u, v] : edges) {
    ++out.row_ptr[static_cast<std::size_t>(u) + 1];
    out.col_ind.push_back(v);
  }
  for (std::size_t u = 0; u < static_cast<std::size_t>(vertex_count); ++u) {
    out.row_ptr[u + 1] += out.row_ptr[u];
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::edge_set_after_batch")

#define DYNG_INSTANTIATE_CYCLE_ORACLE(V, E, W)                                                \
  template cycle_histogram oracle_simple_cycles<V, E, W>(const csr_view<V, E, W>&, int);      \
  template cycle_histogram brute_force_simple_cycles<V, E, W>(const csr_view<V, E, W>&, int); \
  template csr<V, E, W> edge_set_after_batch<V, E, W>(const csr_view<V, E, W>&,               \
                                                      const edge_batch_view<V, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_CYCLE_ORACLE)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_CYCLE_ORACLE)
#undef DYNG_INSTANTIATE_CYCLE_ORACLE

}  // namespace dyng::testing
