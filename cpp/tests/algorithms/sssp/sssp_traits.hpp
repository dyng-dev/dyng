// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_traits.hpp
 * @brief The conformance-kit traits of sssp (PLAN Section 8.2; cpp/tests/conformance).
 */
#pragma once

#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/dijkstra.hpp>

#include <cstdint>
#include <functional>
#include <ostream>
#include <string_view>
#include <vector>

namespace dyng::conformance {

/// A host copy of an sssp result.
struct sssp_snapshot {
  std::vector<std::int64_t> distances;  ///< per vertex
  std::vector<std::int64_t> parents;    ///< per vertex (-1: none)

  /// Bitwise equality (determinism::bitwise).
  friend bool operator==(const sssp_snapshot& a, const sssp_snapshot& b) {
    return a.distances == b.distances && a.parents == b.parents;
  }
  /// For GoogleTest's messages.
  friend void PrintTo(const sssp_snapshot& s, std::ostream* os) {  // NOLINT: gtest API
    *os << "{distances " << ::testing::PrintToString(s.distances) << ", parents "
        << ::testing::PrintToString(s.parents) << "}";
  }
};

/// sssp: a canonical shortest-path tree from vertex 0, bit-exact on every backend.
template <>
struct test_traits<tags::sssp> {
  static constexpr std::string_view name = "sssp";             ///< as in the manifest
  static constexpr oracle_kind oracle = oracle_kind::compute;  ///< update chain == compute
  static constexpr determinism level = determinism::bitwise;   ///< identical trees
  static constexpr bool history_independent = true;            ///< the tree of a graph is unique
  /// The instantiated graph types (DYNG_FOR_EACH_GRAPH_TYPE).
  using graph_types = type_list<graph<std::int32_t, std::int32_t, std::int32_t>,
                                graph<std::int32_t, std::int64_t, std::int32_t>,
                                graph<std::int64_t, std::int64_t, std::int32_t>>;
  template <typename graph_t>
  using result = sssp::result<typename graph_t::vertex_type>;  ///< the result
  using stats = sssp::stats;                                   ///< update()'s stats
  using snapshot = sssp_snapshot;                              ///< a host copy

  /// The requirements: the in-edges are stored (the pull step reads them).
  static void require(graph_properties& props) {
    props.store_transposed = true;
  }
  /// MOSP's graphs as well (rows in insertion order, parallel edges kept).
  static std::vector<graph_properties> extra_properties() {
    return {graph_properties::mosp_compatible()};
  }
  /// The generated graphs.
  static graph_shape shape(size_class size) {
    switch (size) {
      case size_class::tiny:
        return {6, 10, 5};
      case size_class::small:
        return {40, 160, 9};
      case size_class::medium:
        return {200, 900, 20};
    }
    return {};
  }
  /// compute() from vertex 0 with the CUDA engine `e`.
  template <typename graph_t>
  static result<graph_t> compute(const resources& res, const graph_t& g, engine e) {
    sssp::options opt;
    opt.cuda_engine = e;
    return sssp::compute(res, g, typename graph_t::vertex_type{0}, opt);
  }
  /// update().
  template <typename graph_t>
  static stats update(
      const resources& res, graph_t& g,
      const edge_batch_view<typename graph_t::vertex_type, typename graph_t::weight_type>& batch,
      result<graph_t>& r) {
    return sssp::update(res, g, batch, r);
  }
  /// The tree on the host.
  template <typename graph_t>
  static snapshot take(const resources& res, const result<graph_t>& r) {
    snapshot s;
    s.distances = to_vector(res, r.distances());
    for (const auto p : to_vector(res, r.parents())) {
      s.parents.push_back(static_cast<std::int64_t>(p));
    }
    return s;
  }
  /// Dijkstra with lowest-id ties (dyng::testing), the independent oracle of C2.
  template <typename graph_t>
  static snapshot oracle_of(const resources& res, const graph_t& g) {
    const auto csr = g.to_csr(res);
    const auto tree = testing::dijkstra(csr.view(), typename graph_t::vertex_type{0});
    snapshot s;
    s.distances = tree.distances;
    for (const auto p : tree.parents) {
      s.parents.push_back(static_cast<std::int64_t>(p));
    }
    return s;
  }
  /// The counters that are equal across runs and backends.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.invalidated};
  }
  /// Options compute() and set_options() reject.
  template <typename graph_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const graph_t& g) {
    using vertex_t = typename graph_t::vertex_type;
    return {
        [&] { (void)sssp::compute(res, g, vertex_t{-1}); },
        [&] { (void)sssp::compute(res, g, static_cast<vertex_t>(g.num_vertices())); },
        [&] {
          sssp::options opt;
          opt.objective = 1;  // the graph has one weight column
          (void)sssp::compute(res, g, vertex_t{0}, opt);
        },
        [&] {
          sssp::options opt;
          opt.delta = -1;
          (void)sssp::compute(res, g, vertex_t{0}, opt);
        },
        [&] {
          auto r = sssp::compute(res, g, vertex_t{0});
          sssp::options opt = r.get_options();
          opt.objective = 1;  // fixed at compute()
          r.set_options(opt);
        },
    };
  }
  /// C8: host synchronizations of the algorithm phase (the fused kernel's control block).
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? 1 : 0;
  }
};

}  // namespace dyng::conformance
