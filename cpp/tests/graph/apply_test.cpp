// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file apply_test.cpp
 * @brief graph::apply: batch semantics, summaries, the weight-increase classification, and a
 *        randomized comparison with an independent row-list model.
 */
#include "graph/apply_host.hpp"
#include "graph/graph_impl.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using dyng::batch_semantics;
using dyng::graph_properties;
using graph32 = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using batch32 = dyng::edge_batch<std::int32_t, std::int32_t>;

graph32 small_graph(const graph_properties& props = graph_properties::mosp_compatible()) {
  // 0 -> 1 (5), 0 -> 2 (7), 1 -> 2 (3), 2 -> 0 (4)
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 3;
  e.num_weights = 1;
  e.add_edge(0, 1, {5});
  e.add_edge(0, 2, {7});
  e.add_edge(1, 2, {3});
  e.add_edge(2, 0, {4});
  return graph32::from_edges(dyng::resources::sequential(), e.view(), props);
}

// ADR 0009: the default edge offset type is int32 (the originals' type), with checked
// construction: an edge count past the type's range throws capacity_error naming int64.
TEST(GraphEdgeType, DefaultIsInt32WithCheckedConstruction) {
  static_assert(std::is_same_v<dyng::graph<>::edge_type, std::int32_t>);
  static_assert(std::is_same_v<dyng::graph<>::vertex_type, std::int32_t>);
  constexpr std::int64_t max32 = INT32_MAX;
  EXPECT_EQ(dyng::detail::checked_edge_count<std::int32_t>(max32), INT32_MAX);
  EXPECT_EQ(dyng::detail::checked_edge_count<std::int64_t>(max32 + 1), max32 + 1);
  try {
    (void)dyng::detail::checked_edge_count<std::int32_t>(max32 + 1);
    ADD_FAILURE() << "expected capacity_error";
  } catch (const dyng::capacity_error& e) {
    EXPECT_NE(std::string(e.what()).find("int64"), std::string::npos) << e.what();
  }
}

TEST(GraphApply, UpsertAppendDeleteAndVersion) {
  const auto res = dyng::resources::sequential();
  auto g = small_graph();
  batch32 b;
  b.insert_edge(0, 1, {9});  // upsert
  b.insert_edge(1, 0, {2});  // new, appended to row 1
  b.insert_edge(1, 0, {1});  // last insertion wins
  b.delete_edge(0, 2);
  b.delete_edge(2, 1);  // missing
  const auto summary = g.apply(res, b.view());
  EXPECT_EQ(g.version(), 1u);
  EXPECT_EQ(summary.inserted_edges, 1);
  EXPECT_EQ(summary.updated_edges, 2);
  EXPECT_EQ(summary.deleted_edges, 1);
  EXPECT_EQ(summary.ignored_deletions, 1);
  EXPECT_EQ(summary.num_vertices_after, 3);
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 1, 3, 4}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{1, 2, 0, 0}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{9, 3, 1, 4}));
  g.check_integrity(res);
  EXPECT_EQ(g.view().version, 1u);
  (void)g.apply(res, batch32().view());
  EXPECT_EQ(g.version(), 2u);
}

TEST(GraphApply, WeightIncreaseClassification) {
  const auto res = dyng::resources::sequential();
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 3;
  e.num_weights = 2;
  e.add_edge(0, 1, {5, 5});
  e.add_edge(0, 1, {6, 6});  // parallel edge
  e.add_edge(1, 2, {3, 3});
  auto g = graph32::from_edges(res, e.view(), graph_properties::mosp_compatible());
  batch32 b(2);
  b.insert_edge(1, 2, {4, 2});  // increase in objective 0 only
  b.insert_edge(0, 1, {1, 9});  // after the deletion below: overwrites the second (6, 6) edge
  b.insert_edge(2, 0, {1, 1});  // new edge: never an increase
  b.delete_edge(0, 1);          // removes the first (5, 5)
  dyng::detail::apply_delta<std::int32_t> delta;
  (void)dyng::detail::graph_access::apply(res, g, b.view(), &delta);
  EXPECT_EQ(delta.num_weights, 2);
  EXPECT_EQ(delta.insert_src, (std::vector<std::int32_t>{1, 0, 2}));
  EXPECT_EQ(delta.insert_dst, (std::vector<std::int32_t>{2, 1, 0}));
  // MOSP compares the final first (0,1) = (1, 9) with the ORIGINAL first (0,1) = (5, 5).
  EXPECT_EQ(delta.weight_increased, (std::vector<std::uint8_t>{1, 0, 0, 1, 0, 0}));
  EXPECT_EQ(delta.delete_src, (std::vector<std::int32_t>{0}));
  EXPECT_EQ(delta.delete_dst, (std::vector<std::int32_t>{1}));
}

TEST(GraphApply, ExistingInsertPolicies) {
  const auto res = dyng::resources::sequential();
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.on_existing_insert = batch_semantics::existing_insert::ignore;
  auto g = small_graph(props);
  batch32 b;
  b.insert_edge(0, 1, {9});
  const auto summary = g.apply(res, b.view());
  EXPECT_EQ(summary.ignored_insertions, 1);
  EXPECT_EQ(g.to_csr(res).weights[0], 5);

  props.semantics.on_existing_insert = batch_semantics::existing_insert::error;
  auto h = small_graph(props);
  EXPECT_THROW((void)h.apply(res, b.view()), dyng::invalid_argument_error);
}

TEST(GraphApply, MissingDeletePolicies) {
  const auto res = dyng::resources::sequential();
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.on_missing_delete = batch_semantics::missing_delete::error;
  auto g = small_graph(props);
  batch32 b;
  b.delete_edge(1, 0);
  EXPECT_THROW((void)g.apply(res, b.view()), dyng::invalid_argument_error);
  batch32 out_of_range;
  out_of_range.delete_edge(0, 7);
  EXPECT_THROW((void)g.apply(res, out_of_range.view()), dyng::invalid_argument_error);
  auto h = small_graph();
  EXPECT_EQ(h.apply(res, out_of_range.view()).ignored_deletions, 1);
}

TEST(GraphApply, SelfLoopPolicies) {
  const auto res = dyng::resources::sequential();
  batch32 b;
  b.insert_edge(1, 1, {2});
  b.delete_edge(2, 2);
  auto keep = small_graph();
  const auto kept = keep.apply(res, b.view());
  EXPECT_EQ(kept.inserted_edges, 1);
  EXPECT_EQ(kept.ignored_deletions, 1);
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.on_self_loop = batch_semantics::self_loop::drop;
  auto drop = small_graph(props);
  const auto dropped = drop.apply(res, b.view());
  EXPECT_EQ(dropped.dropped_self_loops, 2);
  EXPECT_EQ(dropped.inserted_edges, 0);
  EXPECT_EQ(drop.num_edges(), 4);
  props.semantics.on_self_loop = batch_semantics::self_loop::error;
  auto error = small_graph(props);
  EXPECT_THROW((void)error.apply(res, b.view()), dyng::invalid_argument_error);
}

TEST(GraphApply, InsertionsFirst) {
  const auto res = dyng::resources::sequential();
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.deletions_first = false;
  auto g = small_graph(props);
  batch32 b;
  b.insert_edge(1, 0, {8});
  b.delete_edge(1, 0);  // removes the edge inserted by the same batch
  const auto summary = g.apply(res, b.view());
  EXPECT_EQ(summary.inserted_edges, 1);
  EXPECT_EQ(summary.deleted_edges, 1);
  EXPECT_EQ(g.num_edges(), 4);
  auto h = small_graph();
  EXPECT_EQ(h.apply(res, b.view()).ignored_deletions, 1);
  EXPECT_EQ(h.num_edges(), 5);
}

TEST(GraphApply, VertexGrowth) {
  const auto res = dyng::resources::sequential();
  auto g = small_graph();
  batch32 b;
  b.insert_edge(5, 1, {1});
  const auto summary = g.apply(res, b.view());
  EXPECT_EQ(summary.inserted_vertices, 3);
  EXPECT_EQ(summary.num_vertices_after, 6);
  EXPECT_EQ(g.num_vertices(), 6);
  EXPECT_EQ(g.view().in.num_vertices(), 6);
  g.check_integrity(res);
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.allow_vertex_growth = false;
  auto h = small_graph(props);
  EXPECT_THROW((void)h.apply(res, b.view()), dyng::invalid_argument_error);
}

TEST(GraphApply, RejectsMalformedBatches) {
  const auto res = dyng::resources::sequential();
  auto g = small_graph();
  batch32 negative;
  negative.insert_edge(-1, 0, {1});
  EXPECT_THROW((void)g.apply(res, negative.view()), dyng::invalid_argument_error);
  batch32 negative_delete;
  negative_delete.delete_edge(0, -2);
  EXPECT_THROW((void)g.apply(res, negative_delete.view()), dyng::invalid_argument_error);
  batch32 two_weights(2);
  two_weights.insert_edge(0, 1, {1, 2});
  EXPECT_THROW((void)g.apply(res, two_weights.view()), dyng::invalid_argument_error);
  batch32 deletions_only(2);  // the weight count does not matter without insertions
  deletions_only.delete_edge(0, 1);
  EXPECT_NO_THROW((void)g.apply(res, deletions_only.view()));
  std::vector<std::int32_t> vertices{1};
  dyng::edge_batch_view<std::int32_t, std::int32_t> vertex_ops;
  vertex_ops.insert_vertices = dyng::host_view(vertices);
  EXPECT_THROW((void)g.apply(res, vertex_ops), dyng::not_supported_error);
  dyng::edge_batch_view<std::int32_t, std::int32_t> ragged = batch32().view();
  ragged.insert_src = dyng::host_view(vertices);
  EXPECT_THROW((void)g.apply(res, ragged), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 1u);  // only the valid batch was applied
}

TEST(GraphApply, UndirectedChangesBothDirections) {
  const auto res = dyng::resources::sequential();
  graph_properties props;
  props.directed = false;
  dyng::edge_list<std::int32_t, std::int32_t> e;
  e.num_vertices = 3;
  e.num_weights = 1;
  e.add_edge(0, 1, {4});
  auto g = graph32::from_edges(res, e.view(), props);
  batch32 b;
  b.insert_edge(2, 1, {6});
  b.insert_edge(1, 0, {3});  // upsert of both directions of {0, 1}
  const auto summary = g.apply(res, b.view());
  EXPECT_EQ(summary.inserted_edges, 2);
  EXPECT_EQ(summary.updated_edges, 2);
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 1, 3, 4}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{1, 0, 2, 1}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{3, 3, 6, 6}));
  g.check_integrity(res);
  batch32 d;
  d.delete_edge(1, 2);
  EXPECT_EQ(g.apply(res, d.view()).deleted_edges, 2);
  g.check_integrity(res);
  EXPECT_EQ(g.num_edges(), 2);
}

TEST(GraphApply, SortedRowsStaySorted) {
  const auto res = dyng::resources::sequential();
  auto g = small_graph(graph_properties{});
  batch32 b;
  b.insert_edge(2, 1, {1});
  b.insert_edge(1, 0, {1});
  b.insert_edge(0, 0, {1});
  (void)g.apply(res, b.view());
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{0, 1, 2, 0, 2, 0, 1}));
  g.check_integrity(res);
}

// ------------------------------------------------------------------------------------------------
// Randomized comparison with an independent model: rows are lists of (neighbour, weights),
// operations are applied one at a time exactly as batch_semantics documents them.
// ------------------------------------------------------------------------------------------------

struct model_edge {
  std::int64_t v;
  std::vector<std::int32_t> w;
};
using model_rows = std::vector<std::vector<model_edge>>;

struct model_result {
  dyng::apply_summary summary;
  std::vector<std::uint8_t> increased;
};

template <typename vertex_t, typename edge_t>
model_rows rows_of(const dyng::csr<vertex_t, edge_t, std::int32_t>& c) {
  model_rows rows(static_cast<std::size_t>(c.num_vertices()));
  for (std::size_t u = 0; u < rows.size(); ++u) {
    for (auto e = static_cast<std::size_t>(c.row_ptr[u]);
         e < static_cast<std::size_t>(c.row_ptr[u + 1]); ++e) {
      model_edge edge{static_cast<std::int64_t>(c.col_ind[e]), {}};
      for (int k = 0; k < c.num_weights; ++k) {
        edge.w.push_back(c.weight(static_cast<edge_t>(e), k));
      }
      rows[u].push_back(edge);
    }
  }
  return rows;
}

model_result model_apply(model_rows& rows, int K, const graph_properties& props,
                         const std::vector<std::pair<std::int64_t, std::int64_t>>& ins,
                         const std::vector<std::vector<std::int32_t>>& ins_w,
                         const std::vector<std::pair<std::int64_t, std::int64_t>>& del) {
  const auto& s = props.semantics;
  model_result result;
  const model_rows before = rows;
  std::vector<std::pair<std::int64_t, std::int64_t>> eff_ins;
  std::vector<std::vector<std::int32_t>> eff_w;
  std::vector<std::pair<std::int64_t, std::int64_t>> eff_del;
  std::int64_t n_after = static_cast<std::int64_t>(rows.size());
  for (std::size_t i = 0; i < ins.size(); ++i) {
    auto [u, v] = ins[i];
    if (u == v && s.on_self_loop == batch_semantics::self_loop::drop) {
      ++result.summary.dropped_self_loops;
      continue;
    }
    n_after = std::max(n_after, std::max(u, v) + 1);
    eff_ins.push_back({u, v});
    eff_w.push_back(ins_w[i]);
    if (!props.directed && u != v) {
      eff_ins.push_back({v, u});
      eff_w.push_back(ins_w[i]);
    }
  }
  for (auto [u, v] : del) {
    if (u == v && s.on_self_loop == batch_semantics::self_loop::drop) {
      ++result.summary.dropped_self_loops;
      continue;
    }
    if (u >= n_after || v >= n_after) {
      result.summary.ignored_deletions += (!props.directed && u != v) ? 2 : 1;
      continue;
    }
    eff_del.push_back({u, v});
    if (!props.directed && u != v) {
      eff_del.push_back({v, u});
    }
  }
  rows.resize(static_cast<std::size_t>(n_after));
  const auto do_deletes = [&] {
    for (auto [u, v] : eff_del) {
      auto& row = rows[static_cast<std::size_t>(u)];
      // A structured binding cannot be captured in C++17: copy it first.
      const auto head = v;
      auto it =
          std::find_if(row.begin(), row.end(), [&](const model_edge& e) { return e.v == head; });
      if (it == row.end()) {
        ++result.summary.ignored_deletions;
      } else {
        row.erase(it);
        ++result.summary.deleted_edges;
      }
    }
  };
  const auto do_inserts = [&] {
    for (std::size_t i = 0; i < eff_ins.size(); ++i) {
      const auto u = eff_ins[i].first;
      const auto v = eff_ins[i].second;
      auto& row = rows[static_cast<std::size_t>(u)];
      auto it = std::find_if(row.begin(), row.end(), [&](const model_edge& e) { return e.v == v; });
      if (it == row.end()) {
        row.push_back({v, eff_w[i]});
        ++result.summary.inserted_edges;
      } else if (s.on_existing_insert == batch_semantics::existing_insert::upsert) {
        it->w = eff_w[i];
        ++result.summary.updated_edges;
      } else {
        ++result.summary.ignored_insertions;
      }
    }
  };
  if (s.deletions_first) {
    do_deletes();
    do_inserts();
  } else {
    do_inserts();
    do_deletes();
  }
  if (props.order == dyng::row_order::sorted) {
    for (auto& row : rows) {
      std::stable_sort(row.begin(), row.end(),
                       [](const model_edge& a, const model_edge& b) { return a.v < b.v; });
    }
  }
  result.summary.inserted_vertices = n_after - static_cast<std::int64_t>(before.size());
  result.summary.num_vertices_after = n_after;
  // First occurrence before vs after.
  for (const auto& [u, head] : eff_ins) {
    const auto v = head;  // captured below: a structured binding cannot be captured in C++17
    std::vector<std::uint8_t> flags(static_cast<std::size_t>(K), 0);
    if (u < static_cast<std::int64_t>(before.size())) {
      const auto& old_row = before[static_cast<std::size_t>(u)];
      const auto& new_row = rows[static_cast<std::size_t>(u)];
      auto a = std::find_if(old_row.begin(), old_row.end(),
                            [&](const model_edge& e) { return e.v == v; });
      auto b = std::find_if(new_row.begin(), new_row.end(),
                            [&](const model_edge& e) { return e.v == v; });
      if (a != old_row.end() && b != new_row.end()) {
        for (int k = 0; k < K; ++k) {
          flags[static_cast<std::size_t>(k)] =
              b->w[static_cast<std::size_t>(k)] > a->w[static_cast<std::size_t>(k)];
        }
      }
    }
    result.increased.insert(result.increased.end(), flags.begin(), flags.end());
  }
  return result;
}

template <typename graph_t>
class GraphApplyRandom : public ::testing::Test {};

using graph_types = ::testing::Types<dyng::graph<std::int32_t, std::int32_t, std::int32_t>,
                                     dyng::graph<std::int32_t, std::int64_t, std::int32_t>,
                                     dyng::graph<std::int64_t, std::int64_t, std::int32_t>>;
TYPED_TEST_SUITE(GraphApplyRandom, graph_types, dyng::test::type_index_name);

TYPED_TEST(GraphApplyRandom, MatchesRowListModel) {
  using graph_t = TypeParam;
  using vertex_t = typename graph_t::vertex_type;
  const auto res = dyng::resources::sequential();
  std::mt19937 rng(20260927);
  for (int trial = 0; trial < 300; ++trial) {
    graph_properties props;
    props.order = rng() % 2 ? dyng::row_order::sorted : dyng::row_order::append;
    props.parallel_edges = rng() % 2 ? dyng::multi_edges::allow : dyng::multi_edges::forbid;
    props.directed = rng() % 4 != 0;
    props.semantics.on_self_loop =
        rng() % 3 == 0 ? batch_semantics::self_loop::drop : batch_semantics::self_loop::keep;
    props.semantics.on_existing_insert = rng() % 4 == 0 ? batch_semantics::existing_insert::ignore
                                                        : batch_semantics::existing_insert::upsert;
    props.semantics.deletions_first = rng() % 4 != 0;
    const int n = 1 + static_cast<int>(rng() % 10);
    const int K = static_cast<int>(rng() % 4);
    dyng::edge_list<vertex_t, std::int32_t> e;
    e.num_vertices = n;
    e.num_weights = K;
    const int m = static_cast<int>(rng() % 30);
    for (int i = 0; i < m; ++i) {
      e.src.push_back(static_cast<vertex_t>(rng() % n));
      e.dst.push_back(static_cast<vertex_t>(rng() % n));
      for (int k = 0; k < K; ++k) {
        e.weights.push_back(1 + static_cast<std::int32_t>(rng() % 9));
      }
    }
    auto g = graph_t::from_edges(res, e.view(), props);
    g.check_integrity(res);
    model_rows rows = rows_of(g.to_csr(res));
    for (int step = 0; step < 4; ++step) {
      dyng::edge_batch<vertex_t, std::int32_t> b(K);
      std::vector<std::pair<std::int64_t, std::int64_t>> ins;
      std::vector<std::vector<std::int32_t>> ins_w;
      std::vector<std::pair<std::int64_t, std::int64_t>> del;
      const auto bound = static_cast<unsigned>(g.num_vertices() + 2);
      const int num_ins = static_cast<int>(rng() % 8);
      const int num_del = static_cast<int>(rng() % 8);
      for (int i = 0; i < num_ins; ++i) {
        const auto u = static_cast<vertex_t>(rng() % bound);
        const auto v = static_cast<vertex_t>(rng() % bound);
        std::vector<std::int32_t> w;
        for (int k = 0; k < K; ++k) {
          w.push_back(1 + static_cast<std::int32_t>(rng() % 9));
        }
        b.insert_edge(u, v, dyng::host_view(w.data(), w.size()));
        ins.push_back({u, v});
        ins_w.push_back(w);
      }
      for (int j = 0; j < num_del; ++j) {
        const auto u = static_cast<vertex_t>(rng() % bound);
        const auto v = static_cast<vertex_t>(rng() % bound);
        b.delete_edge(u, v);
        del.push_back({u, v});
      }
      dyng::detail::apply_delta<vertex_t> delta;
      const auto summary = dyng::detail::graph_access::apply(res, g, b.view(), &delta);
      const auto expected = model_apply(rows, K, props, ins, ins_w, del);
      SCOPED_TRACE("trial " + std::to_string(trial) + " step " + std::to_string(step));
      ASSERT_NO_THROW(g.check_integrity(res));
      const auto got = rows_of(g.to_csr(res));
      ASSERT_EQ(got.size(), rows.size());
      for (std::size_t u = 0; u < rows.size(); ++u) {
        ASSERT_EQ(got[u].size(), rows[u].size()) << "row " << u;
        for (std::size_t i = 0; i < rows[u].size(); ++i) {
          ASSERT_EQ(got[u][i].v, rows[u][i].v) << "row " << u;
          ASSERT_EQ(got[u][i].w, rows[u][i].w) << "row " << u;
        }
      }
      EXPECT_EQ(summary.inserted_edges, expected.summary.inserted_edges);
      EXPECT_EQ(summary.updated_edges, expected.summary.updated_edges);
      EXPECT_EQ(summary.deleted_edges, expected.summary.deleted_edges);
      EXPECT_EQ(summary.ignored_insertions, expected.summary.ignored_insertions);
      EXPECT_EQ(summary.ignored_deletions, expected.summary.ignored_deletions);
      EXPECT_EQ(summary.dropped_self_loops, expected.summary.dropped_self_loops);
      EXPECT_EQ(summary.inserted_vertices, expected.summary.inserted_vertices);
      EXPECT_EQ(summary.num_vertices_after, expected.summary.num_vertices_after);
      EXPECT_EQ(delta.weight_increased, expected.increased);
      EXPECT_EQ(g.version(), static_cast<std::uint64_t>(step + 1));
    }
  }
}

}  // namespace
