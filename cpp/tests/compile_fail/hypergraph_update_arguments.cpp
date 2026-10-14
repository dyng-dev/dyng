// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_update_arguments.cpp
 * @brief Compile-failure tests of dyng::update() over hypergraphs (ADR 0043): compiled with
 *        -fsyntax-only and one of the defines below (cpp/tests/CMakeLists.txt checks the message);
 *        without a define it compiles.
 *
 *   DYNG_CF_HYPER_OWNING_BATCH      the owning hyperedge_batch instead of its view
 *   DYNG_CF_HYPER_EDGE_BATCH        a graph's edge_batch_view with a hypergraph
 *   DYNG_CF_GRAPH_HYPER_BATCH       a hypergraph's batch view with a graph
 *   DYNG_CF_HYPER_RESULT_ON_GRAPH   a hypergraph algorithm's result with a graph (its
 *                                   update_traits static_asserts detail::is_hypergraph_v, the
 *                                   pattern of the hypergraph scaffold)
 *   DYNG_CF_GRAPH_RESULT_ON_HYPER   a graph algorithm's result (sssp) with a hypergraph
 */
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>

namespace fake {
/// A hypergraph algorithm's result.
struct result {};
/// Its stats.
struct stats : dyng::update_stats {
  dyng::hyper_apply_summary batch;  ///< the commit
};
}  // namespace fake

namespace dyng::detail {
template <>
struct update_traits<fake::result> {
  using stats_type = fake::stats;
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      fake::result& /*r*/, fake::stats& /*s*/) {
    static_assert(is_hypergraph_v<container_t>,
                  "fake: a hypergraph algorithm; dyng::update() takes a dyng::hypergraph<V, W> "
                  "for its result");
    return nullptr;
  }
};
}  // namespace dyng::detail

using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using graph = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;

void call(const dyng::resources& res, hgraph& h, graph& g,
          const dyng::hyperedge_batch<std::int32_t, dyng::unweighted>& hb,
          const dyng::edge_batch<std::int32_t, std::int32_t>& eb, fake::result& r,
          dyng::sssp::result<std::int32_t>& s) {
#if defined(DYNG_CF_HYPER_OWNING_BATCH)
  (void)dyng::update(res, h, hb, r);
#elif defined(DYNG_CF_HYPER_EDGE_BATCH)
  (void)dyng::update(res, h, eb.view(), r);
#elif defined(DYNG_CF_GRAPH_HYPER_BATCH)
  (void)dyng::update(res, g, hb.view(), s);
#elif defined(DYNG_CF_HYPER_RESULT_ON_GRAPH)
  (void)dyng::update(res, g, eb.view(), r);
#elif defined(DYNG_CF_GRAPH_RESULT_ON_HYPER)
  (void)dyng::update(res, h, hb.view(), s);
#else
  (void)dyng::update(res, h, hb.view(), r);
  (void)dyng::update(res, g, eb.view(), s);
#endif
}
