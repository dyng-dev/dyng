// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/validation.cu (combinedGraphReference) and
// src/combinedGraphGpu.cu (preferenceScale, combinedEdgeWeight, mospPathCosts)
/**
 * @file mosp_oracle.cpp
 * @brief dyng::testing::combined_graph_reference() and mosp_path_costs_reference(): the host
 *        references of mosp, written without the algorithm's code (per-vertex rows of (head,
 *        weight) pairs and a tree-membership mask, as MOSP's reference; costs by memoized walks
 *        up the tree instead of the traversal of the children lists).
 */
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/testing/mosp_oracle.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace dyng::testing {

namespace {

/// MOSP's preferenceScale() with the reference's own checks.
std::int64_t scale_of(const std::vector<std::int32_t>& preferences, std::size_t k) {
  if (preferences.empty()) {
    return 1;
  }
  DYNG_EXPECTS(preferences.size() == k, "testing::combined_graph_reference: ", preferences.size(),
               " preferences for ", k, " trees");
  std::int64_t scale = 1;
  for (const std::int32_t p : preferences) {
    DYNG_EXPECTS(p >= 1, "testing::combined_graph_reference: a preference is ", p);
    scale = std::lcm(scale, static_cast<std::int64_t>(p));
    DYNG_EXPECTS(scale <= (std::int64_t{1} << 20),
                 "testing::combined_graph_reference: lcm(preferences) exceeds 2^20");
  }
  return scale;
}

}  // namespace

template <typename vertex_t>
csr<vertex_t, std::int64_t, std::int32_t> combined_graph_reference(
    const std::vector<std::vector<vertex_t>>& parents, vertex_t source,
    const std::vector<std::int32_t>& preferences) try {
  const std::size_t num_k = parents.size();
  DYNG_EXPECTS(num_k >= 1 && num_k <= 64, "testing::combined_graph_reference: ", num_k,
               " trees (1 to 64 are supported)");
  const std::size_t n = parents.front().size();
  for (const auto& tree : parents) {
    DYNG_EXPECTS(tree.size() == n, "testing::combined_graph_reference: trees of ", n, " and ",
                 tree.size(), " vertices");
  }
  const std::int64_t scale = scale_of(preferences, num_k);
  // combinedEdgeWeight(): L * (K + 1) - sum over the trees in the mask of L / Pref_i.
  const auto weight_of = [&](std::uint64_t mask) {
    std::int64_t w = scale * static_cast<std::int64_t>(num_k + 1);
    for (std::size_t i = 0; i < num_k; ++i) {
      if ((mask >> i) & 1U) {
        w -= preferences.empty() ? scale : scale / preferences[i];
      }
    }
    return static_cast<std::int32_t>(w);
  };
  std::vector<std::vector<std::pair<vertex_t, std::int32_t>>> rows(n);
  for (std::size_t v = 0; v < n; ++v) {
    if (static_cast<vertex_t>(v) == source) {
      continue;
    }
    for (std::size_t k = 0; k < num_k; ++k) {
      const vertex_t p = parents[k][v];
      bool seen = p < 0;
      for (std::size_t j = 0; j < k && !seen; ++j) {
        seen = parents[j][v] == p;
      }
      if (seen) {
        continue;
      }
      DYNG_EXPECTS(static_cast<std::size_t>(p) < n, "testing::combined_graph_reference: parent ", p,
                   " of vertex ", v, " is out of range");
      std::uint64_t mask = 0;
      for (std::size_t j = 0; j < num_k; ++j) {
        mask |= parents[j][v] == p ? (std::uint64_t{1} << j) : 0U;
      }
      rows[static_cast<std::size_t>(p)].emplace_back(static_cast<vertex_t>(v), weight_of(mask));
    }
  }
  csr<vertex_t, std::int64_t, std::int32_t> out;
  out.num_weights = 1;
  out.row_ptr.assign(n + 1, 0);
  for (std::size_t u = 0; u < n; ++u) {
    out.row_ptr[u + 1] = out.row_ptr[u] + static_cast<std::int64_t>(rows[u].size());
    for (const auto& [v, w] : rows[u]) {
      out.col_ind.push_back(v);
      out.weights.push_back(w);
    }
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::combined_graph_reference (", parents.size(), " trees)")

template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<std::int64_t> mosp_path_costs_reference(const csr_view<vertex_t, edge_t, weight_t>& out,
                                                    const std::vector<vertex_t>& parent,
                                                    vertex_t source, int num_objectives) try {
  DYNG_EXPECTS(out.row_ptr.empty() || is_host_accessible(out.row_ptr.space()),
               "testing::mosp_path_costs_reference: the graph must be in host memory");
  const int num_k = num_objectives == 0 ? out.num_weights : num_objectives;
  DYNG_EXPECTS(num_k >= 1 && num_k <= out.num_weights,
               "testing::mosp_path_costs_reference: ", num_k, " objectives for a graph with ",
               out.num_weights, " weight column(s)");
  const auto n = static_cast<std::size_t>(out.num_vertices());
  DYNG_EXPECTS(parent.size() == n, "testing::mosp_path_costs_reference: ", parent.size(),
               " parents for ", n, " vertices");
  const auto k = static_cast<std::size_t>(num_k);
  const std::int64_t inf = infinite_distance<std::int64_t>();
  std::vector<std::int64_t> costs(n * k, inf);
  // 0 unknown, 1 on the current walk, 2 done.
  std::vector<char> state(n, 0);
  std::vector<vertex_t> path;
  if (static_cast<std::size_t>(source) < n) {
    for (std::size_t j = 0; j < k; ++j) {
      costs[static_cast<std::size_t>(source) * k + j] = 0;
    }
    state[static_cast<std::size_t>(source)] = 2;
  }
  for (std::size_t start = 0; start < n; ++start) {
    if (state[start] != 0) {
      continue;
    }
    // Walk up until a known vertex (or a root that is not the source), then fill downwards.
    path.clear();
    auto v = static_cast<vertex_t>(start);
    while (v >= 0 && state[static_cast<std::size_t>(v)] == 0) {
      state[static_cast<std::size_t>(v)] = 1;
      path.push_back(v);
      v = parent[static_cast<std::size_t>(v)];
    }
    DYNG_EXPECTS(v < 0 || state[static_cast<std::size_t>(v)] == 2,
                 "testing::mosp_path_costs_reference: the tree has a cycle through vertex ", v);
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      const vertex_t x = *it;
      const vertex_t p = parent[static_cast<std::size_t>(x)];
      state[static_cast<std::size_t>(x)] = 2;
      if (p < 0 || costs[static_cast<std::size_t>(p) * k] >= inf / 2) {
        continue;  // not connected to the source: stays infinite
      }
      // The first edge p -> x in row order.
      const auto row_begin = static_cast<std::size_t>(out.row_ptr[static_cast<std::size_t>(p)]);
      const auto row_end = static_cast<std::size_t>(out.row_ptr[static_cast<std::size_t>(p) + 1]);
      std::size_t edge = row_end;
      for (std::size_t e = row_begin; e < row_end; ++e) {
        if (out.col_ind[e] == x) {
          edge = e;
          break;
        }
      }
      DYNG_EXPECTS(edge < row_end, "testing::mosp_path_costs_reference: the tree edge (", p, ", ",
                   x, ") is not an edge of the graph");
      for (std::size_t j = 0; j < k; ++j) {
        costs[static_cast<std::size_t>(x) * k + j] =
            costs[static_cast<std::size_t>(p) * k + j] +
            static_cast<std::int64_t>(out.weight_column(static_cast<int>(j))[edge]);
      }
    }
  }
  return costs;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("testing::mosp_path_costs_reference (", parent.size(),
                                  " vertices)")

template csr<std::int32_t, std::int64_t, std::int32_t> combined_graph_reference<std::int32_t>(
    const std::vector<std::vector<std::int32_t>>&, std::int32_t, const std::vector<std::int32_t>&);
template csr<std::int64_t, std::int64_t, std::int32_t> combined_graph_reference<std::int64_t>(
    const std::vector<std::vector<std::int64_t>>&, std::int64_t, const std::vector<std::int32_t>&);

#define DYNG_INSTANTIATE_MOSP_ORACLE(V, E, W)                            \
  template std::vector<std::int64_t> mosp_path_costs_reference<V, E, W>( \
      const csr_view<V, E, W>&, const std::vector<V>&, V, int);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_ORACLE)
#undef DYNG_INSTANTIATE_MOSP_ORACLE

}  // namespace dyng::testing
