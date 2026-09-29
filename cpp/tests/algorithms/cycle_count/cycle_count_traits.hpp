// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_traits.hpp
 * @brief The conformance-kit traits of cycle_count (PLAN Section 8.2; cpp/tests/conformance).
 */
#pragma once

#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace dyng::conformance {

/// cycle_count: the histogram of directed simple cycles of length <= 5, exact on every backend.
template <>
struct test_traits<tags::cycle_count> {
  static constexpr std::string_view name = "cycle_count";         ///< as in the manifest
  static constexpr oracle_kind oracle = oracle_kind::compute;     ///< update chain == compute
  static constexpr determinism level = determinism::exact_value;  ///< identical counts
  static constexpr bool history_independent = true;               ///< counts of a graph
  static constexpr int bound = 5;                                 ///< options::max_length
  /// The instantiated graph types (int32 vertex ids; weighted and unweighted).
  using graph_types = type_list<graph<std::int32_t, std::int32_t, std::int32_t>,
                                graph<std::int32_t, std::int64_t, std::int32_t>,
                                graph<std::int32_t, std::int32_t, unweighted>,
                                graph<std::int32_t, std::int64_t, unweighted>>;
  template <typename graph_t>
  using result = cycle_count::result;           ///< the result
  using stats = cycle_count::stats;             ///< update()'s stats
  using snapshot = std::vector<std::uint64_t>;  ///< counts[len], len <= bound

  /// The requirements: sorted rows without parallel edges.
  static void require(graph_properties& props) {
    props.order = row_order::sorted;
    props.parallel_edges = multi_edges::forbid;
  }
  /// The generated graphs.
  static graph_shape shape(size_class size) {
    switch (size) {
      case size_class::tiny:
        return {6, 12, 5};
      case size_class::small:
        return {24, 70, 9};
      case size_class::medium:
        return {60, 200, 9};
    }
    return {};
  }
  /// compute() with the bound and the CUDA engine `e`.
  template <typename graph_t>
  static result<graph_t> compute(const resources& res, const graph_t& g, engine e) {
    cycle_count::options opt;
    opt.max_length = bound;
    opt.cuda_engine = e;
    return cycle_count::compute(res, g, opt);
  }
  /// update().
  template <typename graph_t>
  static stats update(
      const resources& res, graph_t& g,
      const edge_batch_view<typename graph_t::vertex_type, typename graph_t::weight_type>& batch,
      result<graph_t>& r) {
    return cycle_count::update(res, g, batch, r);
  }
  /// The histogram on the host, lengths 0..bound.
  template <typename graph_t>
  static snapshot take(const resources& res, const result<graph_t>& r) {
    snapshot h = to_vector(res, r.counts());
    h.resize(static_cast<std::size_t>(bound) + 1, 0);
    return h;
  }
  /// The brute-force count (dyng::testing), the independent oracle of C2.
  template <typename graph_t>
  static snapshot oracle_of(const resources& res, const graph_t& g) {
    const auto csr = g.to_csr(res);
    return testing::brute_force_simple_cycles(csr.view(), bound);
  }
  /// The counters that are equal across runs and backends.
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.deletions, s.insertions, static_cast<std::int64_t>(s.cycles_removed),
            static_cast<std::int64_t>(s.cycles_added)};
  }
  /// Options and graphs compute() and set_options() reject.
  template <typename graph_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const graph_t& g) {
    return {
        [&] {
          cycle_count::options opt;
          opt.max_length = 1;
          (void)cycle_count::compute(res, g, opt);
        },
        [&] {
          cycle_count::options opt;
          opt.max_length = -2;
          (void)cycle_count::compute(res, g, opt);
        },
        [&] {
          auto r = compute(res, g, engine::automatic);
          cycle_count::options opt = r.get_options();
          opt.max_length = bound + 1;  // fixed at compute()
          r.set_options(opt);
        },
        [&] {
          // A graph requirement: sorted rows (PLAN Section 5.1, "Graph requirements").
          edge_list<typename graph_t::vertex_type, typename graph_t::weight_type> list;
          list.num_vertices = 2;
          graph_properties props = g.properties();
          props.order = row_order::append;
          props.parallel_edges = multi_edges::allow;
          props.semantics = batch_semantics::upsert_last_wins();
          const graph_t append = graph_t::from_edges(res, list.view(), props);
          (void)cycle_count::compute(res, append);
        },
    };
  }
  /// C8: host synchronizations of the algorithm phase on CUDA (the insert phase's item counts, the
  /// copy of both histograms).
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? 2 : 0;
  }
};

}  // namespace dyng::conformance
