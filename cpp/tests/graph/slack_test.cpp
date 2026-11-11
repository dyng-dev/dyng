// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slack_test.cpp
 * @brief row_layout::slack on the host (ADR 0103): with_capacity()'s headroom over empty, partial
 *        and full capacity hints, rows that overflow moving to the tail with their entries in
 *        order, hubs and repeated overflows, every row overflowing in one batch, deletions of
 *        moved rows, compactions, vertex growth, the strong guarantee, the compact-only guard,
 *        and the rows after every batch of vertex-arrival streams with deliberately small
 *        capacities against the kit's model (support/slack_model.hpp).
 */
#include "graph/graph_impl.hpp"
#include "graph/slotted_rows.hpp"
#include "support/slack_model.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/sssp.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::array_view;
using dyng::batch_semantics;
using dyng::graph_properties;
using dyng::host_view;
using dyng::row_layout;
using dyng::vertex_status;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using lp_graph64 = dyng::graph<std::int32_t, std::int64_t, float>;
using lp_batch = dyng::edge_batch<std::int32_t, float>;

graph_properties slack_props(bool growth = false, double headroom = 0.125) {
  graph_properties p;
  p.directed = false;
  p.layout = row_layout::slack;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = batch_semantics::net_effect();
  p.semantics.allow_vertex_growth = growth;
  p.headroom = headroom;
  return p;
}

template <typename graph_t>
graph_t make(const dyng::resources& res, std::int64_t n,
             const std::vector<typename graph_t::edge_type>& hints, bool growth = false) {
  return graph_t::with_capacity(res, static_cast<std::int32_t>(n), host_view(hints),
                                slack_props(growth));
}

/// The row of v as (start, end, limit).
template <typename graph_t>
std::vector<std::int64_t> row_of(const graph_t& g, std::size_t v) {
  const auto& gi = dyng::detail::graph_access::impl(g);
  return {static_cast<std::int64_t>(gi.host_storage().row_ptr[v]),
          static_cast<std::int64_t>(gi.slotted.row_end[v]),
          static_cast<std::int64_t>(gi.slotted.row_limit[v])};
}

TEST(Slack, CapacityHintsGetTheHeadroom) {
  const dyng::resources res = dyng::resources::sequential();
  const lp_graph g = make<lp_graph>(res, 4, {4, 0, 8});  // a partial hint: vertex 3 has none
  const auto view = g.view();
  EXPECT_EQ(view.layout, row_layout::slack);
  // ceil(4 * 1.125) = 5, 0, ceil(8 * 1.125) = 9, 0.
  EXPECT_EQ(std::vector<std::int32_t>(view.out.row_ptr.begin(), view.out.row_ptr.end()),
            (std::vector<std::int32_t>{0, 5, 5, 14, 14}));
  EXPECT_EQ(row_of(g, 2), (std::vector<std::int64_t>{5, 5, 14}));
  EXPECT_EQ(g.num_vertices(), 4);
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_EQ(g.num_active_vertices(), 0);
  g.check_integrity(res);
  EXPECT_EQ(dyng::detail::slack_capacity(1, 0.125), 2);
  EXPECT_EQ(dyng::detail::slack_capacity(8, 0.125), 9);
  EXPECT_EQ(dyng::detail::slack_capacity(0, 0.125), 0);
  EXPECT_EQ(dyng::detail::slack_capacity(5, 0.0), 5);
  // Empty hints; a hint per vertex at most; the headroom's range; the edge_t limit.
  EXPECT_NO_THROW((void)make<lp_graph>(res, 3, {}));
  EXPECT_THROW((void)make<lp_graph>(res, 1, {1, 1}), dyng::invalid_argument_error);
  EXPECT_THROW((void)make<lp_graph>(res, 2, {1, -1}), dyng::invalid_argument_error);
  graph_properties p = slack_props();
  p.headroom = -0.5;
  EXPECT_THROW((void)lp_graph::with_capacity(res, 2, array_view<const std::int32_t>(), p),
               dyng::invalid_argument_error);
  const std::vector<std::int32_t> huge = {1 << 30, 1 << 30};
  EXPECT_THROW((void)make<lp_graph>(res, 2, huge), dyng::capacity_error);
  // from_edges and from_csr refuse slack rows, and the other algorithms refuse slack graphs.
  dyng::edge_list<std::int32_t, float> none;
  EXPECT_THROW((void)lp_graph::from_edges(res, none.view(), slack_props()),
               dyng::not_supported_error);
}

TEST(Slack, AnOverflowingRowMovesToTheTailWithItsEntriesInOrder) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, 5, {1, 1, 1, 1, 1});  // 2 slots each
  dyng::test::slack_model model(5);
  lp_batch b0(1);
  for (std::int32_t v = 0; v < 5; ++v) b0.insert_vertex(v, v == 0 ? 1 : -1);
  b0.insert_edge(0, 1, {0.5F});
  b0.insert_edge(0, 2, {0.25F});
  dyng::apply_summary s = g.apply(res, b0.view());
  model.apply(b0.view(), s.compactions > 0);
  EXPECT_EQ(s.relocated_rows, 0);  // row 0 holds 2 of its 2 slots
  model.expect_equal(g, "b0");
  lp_batch b1(1);
  b1.insert_edge(0, 3, {0.75F});  // row 0 overflows: 3 entries -> ceil(3 * 1.125) = 4 slots
  b1.insert_edge(4, 3, {0.125F});
  s = g.apply(res, b1.view());
  model.apply(b1.view(), s.compactions > 0);
  EXPECT_EQ(s.relocated_rows, 1);
  EXPECT_EQ(s.compactions, 0);
  EXPECT_EQ(s.inserted_edges, 4);
  EXPECT_EQ(row_of(g, 0), (std::vector<std::int64_t>{10, 13, 14}));
  model.expect_equal(g, "b1");
  g.check_integrity(res);
  const auto view = g.view();
  EXPECT_EQ(view.out.col_ind[0], -1);  // the old slots are dead
  EXPECT_EQ(view.out.col_ind[1], -1);
  EXPECT_EQ(view.out.col_ind[10], 1);
  EXPECT_EQ(view.out.col_ind[11], 2);
  EXPECT_EQ(view.out.col_ind[12], 3);
  // to_csr is the compacted current graph.
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.num_edges(), g.num_edges());
  EXPECT_EQ(std::vector<std::int32_t>(c.col_ind.begin(), c.col_ind.begin() + 3),
            (std::vector<std::int32_t>{1, 2, 3}));
}

/// Apply random vertex batches (hubs, overflows, deletions of moved rows) and compare with the
/// model after every batch; returns the moves and compactions seen.
template <typename graph_t>
std::pair<std::int64_t, std::int64_t> random_stream(graph_t& g, dyng::test::slack_model& model,
                                                    std::int64_t n, int batches, int edges,
                                                    std::uint64_t seed, int hub_share) {
  const dyng::resources res = dyng::resources::sequential();
  std::mt19937_64 rng(seed);
  std::vector<std::int32_t> present;
  std::vector<char> used(static_cast<std::size_t>(n), 0);
  std::int32_t next = 0;
  std::int64_t moves = 0;
  std::int64_t compactions = 0;
  for (int b = 0; b < batches; ++b) {
    lp_batch batch(1);
    const int arrivals = std::min<std::int32_t>(static_cast<std::int32_t>(n) - next, 20);
    for (int i = 0; i < arrivals; ++i) {
      batch.insert_vertex(next, static_cast<std::int8_t>((next % 7) == 0 ? next % 2 : -1));
      present.push_back(next);
      used[static_cast<std::size_t>(next)] = 1;
      ++next;
    }
    for (int e = 0; e < edges && present.size() > 1; ++e) {
      const auto u = (static_cast<int>(rng() % 100) < hub_share) ? present[0]
                                                                 : present[rng() % present.size()];
      const auto v = present[rng() % present.size()];
      batch.insert_edge(u, v, {static_cast<float>(rng() % 1000 + 1) / 1000.0F});
    }
    if (b % 3 == 2 && present.size() > 4) {  // departures (unlabelled; the hub stays)
      for (int k = 0; k < 2; ++k) {
        const std::size_t i = 1 + rng() % (present.size() - 1);
        const std::int32_t v = present[i];
        if ((v % 7) == 0) continue;
        batch.delete_vertex(v);
        present.erase(present.begin() + static_cast<std::ptrdiff_t>(i));
      }
    }
    const dyng::apply_summary s = g.apply(res, batch.view());
    model.apply(batch.view(), s.compactions > 0);
    moves += s.relocated_rows;
    compactions += s.compactions;
    model.expect_equal(g, "batch " + std::to_string(b));
    g.check_integrity(res);
    if (::testing::Test::HasFatalFailure()) break;
  }
  return {moves, compactions};
}

TEST(Slack, HubsRepeatedOverflowsDeletionsAndCompactions) {
  const dyng::resources res = dyng::resources::sequential();
  for (const int hub_share : {0, 30, 90}) {
    SCOPED_TRACE(hub_share);
    lp_graph g = make<lp_graph>(res, 400, {});
    dyng::test::slack_model model(400);
    const auto [moves, compactions] =
        random_stream(g, model, 400, 30, 60, 7 + hub_share, hub_share);
    EXPECT_GT(moves, 0);
    EXPECT_GT(compactions, 0);  // the dead slots outgrew the owned ones at least once
    const auto& rows = dyng::detail::graph_access::impl(g).slotted;
    EXPECT_EQ(rows.compactions, compactions);
    EXPECT_LE(2 * static_cast<std::int64_t>(rows.dead_slots),
              static_cast<std::int64_t>(g.view().out.row_ptr[400]));
  }
  // The 64-bit offsets.
  lp_graph64 g64 =
      lp_graph64::with_capacity(res, 300, array_view<const std::int64_t>(), slack_props());
  dyng::test::slack_model m64(300);
  (void)random_stream(g64, m64, 300, 20, 50, 99, 50);
}

TEST(Slack, EveryRowOverflowsInOneBatch) {
  const dyng::resources res = dyng::resources::sequential();
  const std::int64_t n = 64;
  lp_graph g = make<lp_graph>(res, n, std::vector<std::int32_t>(n, 1));
  dyng::test::slack_model model(n);
  lp_batch b0(1);
  for (std::int32_t v = 0; v < n; ++v) b0.insert_vertex(v);
  dyng::apply_summary s = g.apply(res, b0.view());
  model.apply(b0.view(), s.compactions > 0);
  lp_batch b1(1);  // three edges per vertex: every row needs 3 entries > 2 slots
  for (std::int32_t v = 0; v < n; ++v) {
    b1.insert_edge(v, static_cast<std::int32_t>((v + 1) % n), {0.5F});
    b1.insert_edge(v, static_cast<std::int32_t>((v + 7) % n), {0.25F});
  }
  b1.insert_edge(0, 0, {0.75F});  // a self-loop: two entries in row 0
  s = g.apply(res, b1.view());
  model.apply(b1.view(), s.compactions > 0);
  EXPECT_EQ(s.relocated_rows + (s.compactions > 0 ? n : 0), n);
  model.expect_equal(g, "b1");
  g.check_integrity(res);
}

TEST(Slack, ForcedCompactionKeepsTheLiveEntriesInOrder) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, 200, {});
  dyng::test::slack_model model(200);
  (void)random_stream(g, model, 200, 9, 40, 3, 40);
  auto& gi = const_cast<dyng::detail::graph_impl<std::int32_t, std::int32_t, float>&>(
      dyng::detail::graph_access::impl(g));
  dyng::detail::compact_slack_host(gi.host_edges_for_write(), gi.slotted);
  model.compact();
  model.expect_equal(g, "compacted");
  g.check_integrity(res);
  EXPECT_EQ(gi.slotted.dead_slots, 0);
}

TEST(Slack, VertexGrowthUnderAllowVertexGrowth) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph fixed = make<lp_graph>(res, 2, {});
  lp_batch grow(1);
  grow.insert_vertex(0);
  grow.insert_vertex(5, 1);
  grow.insert_edge(0, 5, {0.5F});
  grow.delete_vertex(7);  // a growth retires an id too
  EXPECT_THROW((void)fixed.apply(res, grow.view()), dyng::invalid_argument_error);
  EXPECT_EQ(fixed.num_vertices(), 2);
  lp_graph g = make<lp_graph>(res, 2, {}, true);
  dyng::test::slack_model model(2);
  const dyng::apply_summary s = g.apply(res, grow.view());
  model.apply(grow.view(), s.compactions > 0);
  EXPECT_EQ(g.num_vertices(), 8);
  EXPECT_EQ(s.num_vertices_after, 8);
  EXPECT_EQ(g.num_edges(), 2);
  EXPECT_EQ(g.view().status[7], vertex_status::deleted);
  EXPECT_EQ(g.view().status[3], vertex_status::inactive);
  model.expect_equal(g, "grown");
  g.check_integrity(res);
  // A failing batch does not grow the graph (the strong guarantee).
  lp_batch bad(1);
  bad.insert_vertex(20);
  bad.insert_vertex(20);
  EXPECT_THROW((void)g.apply(res, bad.view()), dyng::invalid_argument_error);
  EXPECT_EQ(g.num_vertices(), 8);
  g.check_integrity(res);
}

TEST(Slack, FailedBatchesChangeNothing) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, 50, {});
  dyng::test::slack_model model(50);
  (void)random_stream(g, model, 50, 4, 30, 5, 50);
  const auto before = g.to_csr(res);
  const auto view = g.view();
  const std::vector<std::int32_t> starts(view.out.row_ptr.begin(), view.out.row_ptr.end());
  const std::uint64_t version = g.version();
  for (int k = 0; k < 4; ++k) {
    lp_batch bad(1);
    bad.insert_edge(0, 1, {0.5F});
    if (k == 0) bad.insert_vertex(0);      // re-insertion
    if (k == 1) bad.delete_vertex(60);     // out of range
    if (k == 2) bad.insert_vertex(49, 3);  // a bad label
    if (k == 3) bad.delete_edge(0, 1);     // an edge deletion
    EXPECT_ANY_THROW((void)g.apply(res, bad.view()));
    const auto after = g.view();
    EXPECT_EQ(std::vector<std::int32_t>(after.out.row_ptr.begin(), after.out.row_ptr.end()),
              starts);
    EXPECT_EQ(g.version(), version);
  }
  const auto same = g.to_csr(res);
  EXPECT_EQ(same.col_ind, before.col_ind);
  EXPECT_EQ(same.row_ptr, before.row_ptr);
  model.expect_equal(g, "unchanged");
}

TEST(Slack, CompactOnlyAlgorithmsRefuseSlackRows) {
  const dyng::resources res = dyng::resources::sequential();
  graph_properties p = slack_props();
  const auto g = dyng::graph<std::int32_t, std::int32_t, std::int32_t>::with_capacity(
      res, 3, array_view<const std::int32_t>(), p);
  try {
    (void)dyng::sssp::compute(res, g, 0);
    ADD_FAILURE() << "sssp accepted slack rows";
  } catch (const dyng::not_supported_error& e) {
    EXPECT_NE(std::string(e.what()).find("slack"), std::string::npos) << e.what();
  }
}

/// The rows after every batch of a vertex-arrival stream whose capacity hints are a fraction of
/// the full degrees (0: no hints) equal the model; with the full degrees nothing moves.
template <typename graph_t, typename stream_t>
void replay_stream(const stream_t& stream, int percent, std::int64_t& moves,
                   std::int64_t& compactions) {
  const dyng::resources res = dyng::resources::sequential();
  const std::int64_t n = static_cast<std::int64_t>(stream.true_labels.size());
  std::vector<typename graph_t::edge_type> hints;
  if (percent > 0) {
    for (const auto c : stream.row_capacity) {
      hints.push_back(static_cast<typename graph_t::edge_type>(c * percent / 100));
    }
  }
  graph_t g =
      graph_t::with_capacity(res, static_cast<std::int32_t>(n), host_view(hints), slack_props());
  dyng::test::slack_model model(n);
  moves = 0;
  compactions = 0;
  for (std::size_t b = 0; b < stream.batches.size(); ++b) {
    const dyng::apply_summary s = g.apply(res, stream.batches[b].view());
    model.apply(stream.batches[b].view(), s.compactions > 0);
    moves += s.relocated_rows;
    compactions += s.compactions;
    model.expect_equal(g, "batch " + std::to_string(b));
    g.check_integrity(res);
    if (::testing::Test::HasFatalFailure()) return;
  }
}

TEST(Slack, VertexArrivalStreamsWithSmallCapacities) {
  const dyng::resources res = dyng::resources::sequential();
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = 5000;
  o.num_batches = 5;
  const auto legacy =
      dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(res, o);
  o.num_batches = 10;
  const auto library =
      dyng::generators::generate_vertex_arrival_stream<std::int32_t, std::int64_t>(res, o);
  for (const int percent : {0, 25, 50, 100}) {
    SCOPED_TRACE(percent);
    std::int64_t moves = 0;
    std::int64_t compactions = 0;
    replay_stream<lp_graph>(legacy, percent, moves, compactions);
    if (percent == 100) {
      EXPECT_EQ(moves, 0);  // the published capacities: nothing moves
      EXPECT_EQ(compactions, 0);
    } else {
      EXPECT_GT(moves, 0);
    }
    replay_stream<lp_graph64>(library, percent, moves, compactions);
  }
}

}  // namespace
