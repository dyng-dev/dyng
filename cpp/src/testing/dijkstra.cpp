// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/Dijkstra.cpp (dijkstraCsrGraph)
/**
 * @file dijkstra.cpp
 * @brief dyng::testing::dijkstra(): binary-heap Dijkstra with lowest-id parent ties.
 *
 * Mechanical changes: templates on the index types, objective-major weight columns, exceptions
 * for invalid arguments.
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/testing/dijkstra.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace dyng::testing {

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_tree<vertex_t> dijkstra(const csr_view<vertex_t, edge_t, weight_t>& out, vertex_t source,
                             int objective) try {
  const vertex_t n = out.num_vertices();
  DYNG_EXPECTS(out.row_ptr.empty() || is_host_accessible(out.row_ptr.space()),
               "testing::dijkstra: the graph must be in host memory");
  const array_view<const weight_t> weights = out.weight_column(objective);
  sssp_tree<vertex_t> tree;
  tree.distances.assign(static_cast<std::size_t>(n), infinite_distance<std::int64_t>());
  tree.parents.assign(static_cast<std::size_t>(n), vertex_t{-1});
  if (source < 0 || source >= n) {
    return tree;
  }
  std::vector<std::int64_t>& distances = tree.distances;
  std::vector<vertex_t>& parent = tree.parents;

  using node = std::pair<std::int64_t, vertex_t>;
  std::priority_queue<node, std::vector<node>, std::greater<node>> pq;
  distances[static_cast<std::size_t>(source)] = 0;
  pq.push({0, source});
  while (!pq.empty()) {
    const auto [d, u] = pq.top();
    pq.pop();
    if (d != distances[static_cast<std::size_t>(u)]) {
      continue;
    }
    for (edge_t e = out.row_ptr[static_cast<std::size_t>(u)];
         e < out.row_ptr[static_cast<std::size_t>(u) + 1]; ++e) {
      const vertex_t v = out.col_ind[static_cast<std::size_t>(e)];
      const std::int64_t candidate =
          d + static_cast<std::int64_t>(weights[static_cast<std::size_t>(e)]);
      auto& dv = distances[static_cast<std::size_t>(v)];
      auto& pv = parent[static_cast<std::size_t>(v)];
      if (candidate < dv) {
        dv = candidate;
        pv = u;
        pq.push({candidate, v});
      } else if (candidate == dv && v != source && u < pv) {
        pv = u;  // ties go to the lowest parent id
      }
    }
  }
  return tree;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::dijkstra (", out.num_vertices(), " vertices)")

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_tree<vertex_t> dijkstra(const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                             int objective) {
  return dijkstra(g.view().out, source, objective);
}

#define DYNG_INSTANTIATE_DIJKSTRA(V, E, W)                                   \
  template sssp_tree<V> dijkstra<V, E, W>(const csr_view<V, E, W>&, V, int); \
  template sssp_tree<V> dijkstra<V, E, W>(const graph<V, E, W>&, V, int);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_DIJKSTRA)
#undef DYNG_INSTANTIATE_DIJKSTRA

}  // namespace dyng::testing
