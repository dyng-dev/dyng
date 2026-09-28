// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_support.hpp
 * @brief Helpers of the cycle_count suites: resources per backend, graphs from edge pairs, random
 *        graphs of the shapes of CycleEnumeration-GPU's randomized tests, batches, histograms as
 *        vectors and in the exporter's text form.
 */
#pragma once

#include "support/gtest_helpers.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::test {

/// A directed edge.
using cc_edge = std::pair<std::int32_t, std::int32_t>;

/// Resources of a host backend for the cycle_count suites (openmp: 4 threads, like the original's
/// randomized tests).
inline resources cc_resources(backend b, int threads = 4) {
  return b == backend::openmp ? resources::openmp(threads) : resources::sequential();
}

/// The name of a backend for traces.
inline std::string cc_name(backend b) {
  return b == backend::openmp ? "openmp" : "sequential";
}

/// A graph on n vertices from edge pairs (weights 1 for a weighted graph type).
template <typename graph_t>
graph_t cc_graph(const resources& res, std::int64_t n, const std::vector<cc_edge>& edges,
                 graph_properties props = graph_properties::cycle_enum_compatible()) {
  using weight_t = typename graph_t::weight_type;
  edge_list<std::int32_t, weight_t> list;
  list.num_vertices = static_cast<std::int32_t>(n);
  if constexpr (!is_unweighted_v<weight_t>) {
    list.num_weights = 1;
  }
  for (const auto& [u, v] : edges) {
    if constexpr (is_unweighted_v<weight_t>) {
      list.add_edge(u, v);
    } else {
      list.add_edge(u, v, {weight_t{1}});
    }
  }
  if constexpr (!is_unweighted_v<weight_t>) {
    props.num_weights = 1;
  }
  return graph_t::from_edges(res, list.view(), props);
}

/// A batch of deletions and insertions (weights 1 for a weighted type).
template <typename weight_t>
edge_batch<std::int32_t, weight_t> cc_batch(const std::vector<cc_edge>& deletions,
                                            const std::vector<cc_edge>& insertions) {
  edge_batch<std::int32_t, weight_t> b;
  for (const auto& [u, v] : deletions) {
    b.delete_edge(u, v);
  }
  for (const auto& [u, v] : insertions) {
    if constexpr (is_unweighted_v<weight_t>) {
      b.insert_edge(u, v);
    } else {
      b.insert_edge(u, v, {weight_t{1}});
    }
  }
  return b;
}

/// The histogram of a result as a vector of lengths 0..k for a bound k (up to 4096; the result
/// stores min(k, max(n, 2)) + 1 entries, the lengths past n are 0), else as stored.
inline std::vector<std::uint64_t> cc_counts(const cycle_count::result& r) {
  const auto c = r.counts();
  std::vector<std::uint64_t> h(c.begin(), c.end());
  const int k = r.get_options().max_length;
  if (k >= 0 && k <= 4096 && h.size() < static_cast<std::size_t>(k) + 1) {
    h.resize(static_cast<std::size_t>(k) + 1, 0);
  }
  return h;
}

/// A histogram resized to max_length + 1 entries (for comparisons with an oracle of another size;
/// entries beyond are required to be 0).
inline std::vector<std::uint64_t> cc_resize(std::vector<std::uint64_t> h, std::size_t size) {
  for (std::size_t i = size; i < h.size(); ++i) {
    EXPECT_EQ(h[i], 0U) << "length " << i;
  }
  h.resize(size, 0);
  return h;
}

/// The exporter's histogram text "<len>:<count> ..." (non-zero lengths >= 2, space separated).
inline std::string cc_text(const std::vector<std::uint64_t>& counts) {
  std::ostringstream out;
  bool first = true;
  for (std::size_t len = 2; len < counts.size(); ++len) {
    if (counts[len] != 0) {
      out << (first ? "" : " ") << len << ':' << counts[len];
      first = false;
    }
  }
  return out.str();
}

/// The shape of a random graph (CycleEnumeration-GPU's RandomGraphSpec, without timestamps).
struct cc_spec {
  std::int64_t vertex_count = 6;       ///< n
  double edge_probability = 0.3;       ///< probability of each directed pair
  std::int64_t hubs = 0;               ///< vertices with an edge to and from every vertex
  double self_loop_probability = 0.0;  ///< probability of a self-loop per vertex
};

/// Random distinct directed pairs of a spec (in (u, v) order), as random_edges() draws them.
inline std::vector<cc_edge> cc_random_edges(const cc_spec& spec, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::vector<cc_edge> edges;
  for (std::int64_t u = 0; u < spec.vertex_count; ++u) {
    for (std::int64_t v = 0; v < spec.vertex_count; ++v) {
      bool present = false;
      if (u == v) {
        present = coin(rng) < spec.self_loop_probability;
      } else {
        present = coin(rng) < spec.edge_probability || u < spec.hubs || v < spec.hubs;
      }
      if (present) {
        edges.emplace_back(static_cast<std::int32_t>(u), static_cast<std::int32_t>(v));
      }
    }
  }
  return edges;
}

/// A random spec covering sparse, medium, dense and hub-heavy shapes (random_spec()).
inline cc_spec cc_random_spec(std::mt19937_64& rng, std::int64_t min_vertices,
                              std::int64_t max_vertices) {
  std::uniform_int_distribution<std::int64_t> vertices(min_vertices, max_vertices);
  std::uniform_int_distribution<int> shape(0, 3);
  cc_spec spec;
  spec.vertex_count = vertices(rng);
  switch (shape(rng)) {
    case 0:
      spec.edge_probability = 0.15;
      break;
    case 1:
      spec.edge_probability = 0.35;
      break;
    case 2:
      spec.edge_probability = 0.7;
      break;
    default:
      spec.edge_probability = 0.1;
      spec.hubs = 1 + static_cast<std::int64_t>(rng() % 2);
      break;
  }
  spec.self_loop_probability = (rng() % 3 == 0) ? 0.2 : 0.0;
  return spec;
}

/// The subset-DP oracle of a graph with a bound (-1: none), as a vector of max_length + 1 entries
/// (no bound: the size of an unbounded result, max(n, 2) + 1).
template <typename graph_t>
std::vector<std::uint64_t> cc_oracle(const resources& res, const graph_t& g, int max_length) {
  const auto csr = g.to_csr(res);
  return testing::oracle_simple_cycles(csr.view(), max_length);
}

/// The brute-force count of a graph with a bound (-1: none).
template <typename graph_t>
std::vector<std::uint64_t> cc_brute(const resources& res, const graph_t& g, int max_length) {
  const auto csr = g.to_csr(res);
  return testing::brute_force_simple_cycles(csr.view(), max_length);
}

}  // namespace dyng::test
