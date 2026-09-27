// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/validation.cpp (checkSospTree, TreeCheck::summary)
/**
 * @file check_sssp.cpp
 * @brief dyng::testing::check_sssp_tree(): distances, parent consistency and the canonical
 *        (lowest-id) parent rule, against dyng::testing::dijkstra().
 *
 * Mechanical changes: templates on the index types, the in-edges rebuilt locally from the
 * out-edges (MOSP's caller passed transposeCsrGraph()), the Dijkstra reference computed inside,
 * exceptions for invalid arguments.
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/dijkstra.hpp>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::testing {

std::string sssp_tree_check::summary() const {
  std::ostringstream out;
  out << "distance mismatches=" << distance_mismatches
      << " inconsistent parents=" << inconsistent_parents
      << " non-canonical parents=" << non_canonical_parents
      << " parent mismatches=" << parent_mismatches;
  return out.str();
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_tree_check check_sssp_tree(const csr_view<vertex_t, edge_t, weight_t>& out, vertex_t source,
                                array_view<const std::int64_t> distances,
                                array_view<const vertex_t> parents,
                                const check_sssp_options& opt) try {
  const auto n = static_cast<std::size_t>(out.num_vertices());
  DYNG_EXPECTS(distances.size() == n && parents.size() == n,
               "testing::check_sssp_tree: ", distances.size(), " distances and ", parents.size(),
               " parents for ", n, " vertices");
  DYNG_EXPECTS(
      n == 0 || (is_host_accessible(distances.space()) && is_host_accessible(parents.space())),
      "testing::check_sssp_tree: the arrays must be in host memory");
  const array_view<const weight_t> weights = out.weight_column(opt.objective);
  const sssp_tree<vertex_t> reference = dijkstra(out, source, opt.objective);

  // Reverse (in-edge) CSR of the graph.
  std::vector<std::size_t> in_start(n + 1, 0);
  const std::size_t m = out.col_ind.size();
  for (std::size_t e = 0; e < m; ++e) {
    ++in_start[static_cast<std::size_t>(out.col_ind[e]) + 1];
  }
  for (std::size_t v = 0; v < n; ++v) {
    in_start[v + 1] += in_start[v];
  }
  std::vector<vertex_t> in_from(m);
  std::vector<std::int64_t> in_weight(m);
  {
    std::vector<std::size_t> cursor(in_start.begin(), in_start.end() - 1);
    for (std::size_t u = 0; u < n; ++u) {
      for (auto e = static_cast<std::size_t>(out.row_ptr[u]);
           e < static_cast<std::size_t>(out.row_ptr[u + 1]); ++e) {
        const auto v = static_cast<std::size_t>(out.col_ind[e]);
        in_from[cursor[v]] = static_cast<vertex_t>(u);
        in_weight[cursor[v]] = static_cast<std::int64_t>(weights[e]);
        ++cursor[v];
      }
    }
  }

  sssp_tree_check check;
  check.require_canonical = opt.require_canonical;
  const std::int64_t infinity = infinite_distance<std::int64_t>();
  auto finite = [&](std::int64_t d) { return d < infinity / 2; };
  for (std::size_t v = 0; v < n; ++v) {
    const std::int64_t dv = distances[v];
    const bool reachable = finite(dv);
    if (reachable != finite(reference.distances[v]) ||
        (reachable && dv != reference.distances[v])) {
      ++check.distance_mismatches;
    }
    if (opt.require_canonical && parents[v] != reference.parents[v]) {
      ++check.parent_mismatches;
    }
    if (static_cast<std::int64_t>(v) == static_cast<std::int64_t>(source)) {
      if (parents[v] != -1 || dv != 0) {
        ++check.inconsistent_parents;
      }
      continue;
    }
    if (!reachable) {
      if (parents[v] != -1) {
        ++check.inconsistent_parents;
      }
      continue;
    }
    const vertex_t p = parents[v];
    bool consistent = false;
    vertex_t lowest = -1;
    for (std::size_t e = in_start[v]; e < in_start[v + 1]; ++e) {
      const vertex_t u = in_from[e];
      if (!finite(distances[static_cast<std::size_t>(u)])) {
        continue;
      }
      // A tight edge must have a positive weight: parents then have strictly smaller distances,
      // so consistent parents form no cycle.
      const std::int64_t w = in_weight[e];
      if (w <= 0 || distances[static_cast<std::size_t>(u)] + w != dv) {
        continue;
      }
      if (u == p) {
        consistent = true;
      }
      if (lowest < 0 || u < lowest) {
        lowest = u;
      }
    }
    if (!consistent) {
      ++check.inconsistent_parents;
    } else if (p != lowest) {
      ++check.non_canonical_parents;
    }
  }
  return check;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::check_sssp_tree (", out.num_vertices(), " vertices)")

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_tree_check check_sssp_tree(const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                                array_view<const std::int64_t> distances,
                                array_view<const vertex_t> parents, const check_sssp_options& opt) {
  return check_sssp_tree(g.view().out, source, distances, parents, opt);
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_tree_check check_sssp_tree(const graph<vertex_t, edge_t, weight_t>& g,
                                const sssp::result<vertex_t>& r, bool require_canonical) {
  check_sssp_options opt;
  opt.objective = r.get_options().objective;
  opt.require_canonical = require_canonical;
  return check_sssp_tree(g.view().out, r.source(), r.distances(), r.parents(), opt);
}

#define DYNG_INSTANTIATE_CHECK_SSSP(V, E, W)                                                       \
  template sssp_tree_check check_sssp_tree<V, E, W>(                                               \
      const csr_view<V, E, W>&, V, array_view<const std::int64_t>, array_view<const V>,            \
      const check_sssp_options&);                                                                  \
  template sssp_tree_check check_sssp_tree<V, E, W>(                                               \
      const graph<V, E, W>&, V, array_view<const std::int64_t>, array_view<const V>,               \
      const check_sssp_options&);                                                                  \
  template sssp_tree_check check_sssp_tree<V, E, W>(const graph<V, E, W>&, const sssp::result<V>&, \
                                                    bool);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_CHECK_SSSP)
#undef DYNG_INSTANTIATE_CHECK_SSSP

}  // namespace dyng::testing
