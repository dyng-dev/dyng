// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_traits.hpp
 * @brief The conformance-kit traits of mosp (PLAN Section 8.2; cpp/tests/conformance).
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
#include <dyng/mosp.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/testing/mosp_oracle.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string_view>
#include <vector>

namespace dyng::conformance {

/// A host copy of a mosp result: the K trees, the MOSP tree and the path costs.
struct mosp_snapshot {
  std::vector<std::vector<std::int64_t>> distances;  ///< per objective, per vertex
  std::vector<std::vector<std::int64_t>> parents;    ///< per objective, per vertex (-1: none)
  std::vector<std::int64_t> combined_distances;      ///< per vertex (units of 1/L)
  std::vector<std::int64_t> combined_parents;        ///< the MOSP tree
  std::vector<std::int64_t> path_costs;              ///< n * K, vertex-major

  /// Bitwise equality (determinism::bitwise).
  friend bool operator==(const mosp_snapshot& a, const mosp_snapshot& b) {
    return a.distances == b.distances && a.parents == b.parents &&
           a.combined_distances == b.combined_distances &&
           a.combined_parents == b.combined_parents && a.path_costs == b.path_costs;
  }
  /// For GoogleTest's messages.
  friend void PrintTo(const mosp_snapshot& s, std::ostream* os) {  // NOLINT: gtest API
    *os << "{combined distances " << ::testing::PrintToString(s.combined_distances)
        << ", combined parents " << ::testing::PrintToString(s.combined_parents) << ", costs "
        << ::testing::PrintToString(s.path_costs) << ", parents "
        << ::testing::PrintToString(s.parents) << "}";
  }
};

/// mosp: K = 3 objectives with the preferences {2, 1, 3} (L = 6), from vertex 0; every tree
/// canonical, bit-exact on every backend.
template <>
struct test_traits<tags::mosp> {
  static constexpr std::string_view name = "mosp";             ///< as in the manifest
  static constexpr oracle_kind oracle = oracle_kind::compute;  ///< update chain == compute
  static constexpr determinism level = determinism::bitwise;   ///< identical trees
  static constexpr bool history_independent = true;            ///< the trees of a graph are unique
  static constexpr int num_weights = 3;                        ///< the kit's graphs have K = 3
  /// The instantiated graph types (DYNG_FOR_EACH_GRAPH_TYPE).
  using graph_types = type_list<graph<std::int32_t, std::int32_t, std::int32_t>,
                                graph<std::int32_t, std::int64_t, std::int32_t>,
                                graph<std::int64_t, std::int64_t, std::int32_t>>;
  template <typename graph_t>
  using result = mosp::result<typename graph_t::vertex_type>;  ///< the result
  using stats = mosp::stats;                                   ///< update()'s stats
  using snapshot = mosp_snapshot;                              ///< a host copy

  /// The preferences of the kit's results.
  static std::vector<std::int32_t> preferences() {
    return {2, 1, 3};
  }
  /// The requirements: the in-edges are stored (the sssp updates pull).
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
    mosp::options opt;
    opt.preferences = preferences();
    opt.cuda_engine = e;
    return mosp::compute(res, g, typename graph_t::vertex_type{0}, opt);
  }
  /// update().
  template <typename graph_t>
  static stats update(
      const resources& res, graph_t& g,
      const edge_batch_view<typename graph_t::vertex_type, typename graph_t::weight_type>& batch,
      result<graph_t>& r) {
    return mosp::update(res, g, batch, r);
  }
  /// Everything on the host.
  template <typename graph_t>
  static snapshot take(const resources& res, const result<graph_t>& r) {
    snapshot s;
    for (int k = 0; k < r.num_objectives(); ++k) {
      s.distances.push_back(to_vector(res, r.distances(k)));
      std::vector<std::int64_t> p;
      for (const auto x : to_vector(res, r.parents(k))) {
        p.push_back(static_cast<std::int64_t>(x));
      }
      s.parents.push_back(std::move(p));
    }
    s.combined_distances = to_vector(res, r.combined_distances());
    for (const auto x : to_vector(res, r.combined_parents())) {
      s.combined_parents.push_back(static_cast<std::int64_t>(x));
    }
    s.path_costs = to_vector(res, r.path_costs());
    return s;
  }
  /// The independent oracle of C2 (dyng::testing): Dijkstra per objective, the reference combined
  /// graph of the K Dijkstra trees and Dijkstra on it, the reference path costs.
  template <typename graph_t>
  static snapshot oracle_of(const resources& res, const graph_t& g) {
    using vertex_t = typename graph_t::vertex_type;
    const auto csr = g.to_csr(res);
    snapshot s;
    std::vector<std::vector<vertex_t>> trees;
    for (int k = 0; k < num_weights; ++k) {
      const auto tree = testing::dijkstra(csr.view(), vertex_t{0}, k);
      s.distances.push_back(tree.distances);
      std::vector<std::int64_t> p(tree.parents.begin(), tree.parents.end());
      s.parents.push_back(std::move(p));
      trees.push_back(tree.parents);
    }
    const auto combined = testing::combined_graph_reference(trees, vertex_t{0}, preferences());
    const auto mosp_tree = testing::dijkstra(combined.view(), vertex_t{0});
    s.combined_distances = mosp_tree.distances;
    s.combined_parents.assign(mosp_tree.parents.begin(), mosp_tree.parents.end());
    s.path_costs =
        testing::mosp_path_costs_reference(csr.view(), mosp_tree.parents, vertex_t{0}, num_weights);
    return s;
  }
  /// The counters that are equal across runs and backends.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    std::vector<std::int64_t> out{s.combined_edges, s.preference_scale,
                                  static_cast<std::int64_t>(s.objectives.size())};
    for (const sssp::stats& o : s.objectives) {
      out.push_back(o.invalidated);
      out.push_back(o.affected);
    }
    return out;
  }
  /// Options compute() and set_options() reject.
  template <typename graph_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const graph_t& g) {
    using vertex_t = typename graph_t::vertex_type;
    const auto with = [&res, &g](const std::function<void(mosp::options&)>& change) {
      return [&res, &g, change] {
        mosp::options opt;
        change(opt);
        (void)mosp::compute(res, g, vertex_t{0}, opt);
      };
    };
    return {
        [&] { (void)mosp::compute(res, g, vertex_t{-1}); },
        [&] { (void)mosp::compute(res, g, static_cast<vertex_t>(g.num_vertices())); },
        with([](mosp::options& o) { o.preferences = {1, 0, 1}; }),        // a preference below 1
        with([](mosp::options& o) { o.preferences = {1, 2}; }),           // not one per objective
        with([](mosp::options& o) { o.preferences = {1 << 20, 3, 1}; }),  // lcm above 2^20
        with([](mosp::options& o) { o.num_objectives = num_weights + 1; }),
        with([](mosp::options& o) { o.num_objectives = -1; }),
        with([](mosp::options& o) { o.delta = -1; }),
        [&] {
          auto r = compute(res, g, engine::automatic);
          mosp::options opt = r.get_options();
          opt.preferences = {1, 1, 1};  // fixed at compute()
          r.set_options(opt);
        },
        [&] {
          auto r = compute(res, g, engine::automatic);
          mosp::options opt = r.get_options();
          opt.num_objectives = 2;  // fixed at compute()
          r.set_options(opt);
        },
        [&] {
          auto r = compute(res, g, engine::automatic);
          (void)r.distances(num_weights);  // no such objective
        },
    };
  }
  /// C8: host synchronizations of the algorithm phase on CUDA: one per objective (the fused sssp
  /// engine's control block), then the combined graph's size, the combined solve (which counts
  /// `affected`) and the download of the MOSP tree for the path costs (compute_path_costs is on).
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? num_weights + 3 : 0;
  }
};

}  // namespace dyng::conformance
