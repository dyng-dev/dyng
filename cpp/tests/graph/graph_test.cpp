// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_test.cpp
 * @brief graph construction, properties, views, transposition and integrity.
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::graph_properties;
using dyng::multi_edges;
using dyng::row_order;
using graph32 = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using edges32 = dyng::edge_list<std::int32_t, std::int32_t>;

template <typename view_t>
std::vector<typename view_t::value_type> values(const view_t& view) {
  return {view.begin(), view.end()};
}

edges32 sample_edges() {
  // Row 0: 2, 1, 2 (parallel), 0 (self-loop); row 1: 0; row 2: -.
  edges32 e;
  e.num_vertices = 3;
  e.num_weights = 2;
  e.add_edge(0, 2, {1, 10});
  e.add_edge(1, 0, {2, 20});
  e.add_edge(0, 1, {3, 30});
  e.add_edge(0, 2, {4, 40});
  e.add_edge(0, 0, {5, 50});
  return e;
}

TEST(GraphProperties, Defaults) {
  const graph_properties props;
  EXPECT_TRUE(props.directed);
  EXPECT_TRUE(props.store_transposed);
  EXPECT_EQ(props.num_weights, 1);
  EXPECT_EQ(props.layout, dyng::row_layout::compact);
  EXPECT_EQ(props.order, row_order::sorted);
  EXPECT_EQ(props.parallel_edges, multi_edges::forbid);
  const dyng::batch_semantics s = dyng::batch_semantics::upsert_last_wins();
  const dyng::batch_semantics d{};
  EXPECT_EQ(s.on_existing_insert, d.on_existing_insert);
  EXPECT_EQ(s.on_missing_delete, d.on_missing_delete);
  EXPECT_EQ(s.on_self_loop, d.on_self_loop);
  EXPECT_EQ(s.deletions_first, d.deletions_first);
  EXPECT_EQ(s.allow_vertex_growth, d.allow_vertex_growth);
  EXPECT_EQ(s.on_existing_insert, dyng::batch_semantics::existing_insert::upsert);
  EXPECT_EQ(s.on_missing_delete, dyng::batch_semantics::missing_delete::ignore);
  EXPECT_EQ(s.on_self_loop, dyng::batch_semantics::self_loop::keep);
  EXPECT_TRUE(s.deletions_first);
}

TEST(GraphProperties, MospCompatible) {
  constexpr graph_properties props = graph_properties::mosp_compatible();
  EXPECT_TRUE(props.directed);
  EXPECT_TRUE(props.store_transposed);
  EXPECT_EQ(props.order, row_order::append);
  EXPECT_EQ(props.parallel_edges, multi_edges::allow);
  EXPECT_EQ(props.semantics.on_existing_insert, dyng::batch_semantics::existing_insert::upsert);
  EXPECT_EQ(props.semantics.on_self_loop, dyng::batch_semantics::self_loop::keep);
}

TEST(Graph, EmptyGraph) {
  const auto res = dyng::resources::sequential();
  graph32 g;
  EXPECT_EQ(g.num_vertices(), 0);
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_EQ(g.version(), 0u);
  EXPECT_EQ(g.space(), dyng::memory_space::host);
  EXPECT_NO_THROW(g.check_integrity(res));
  const auto view = g.view();
  EXPECT_EQ(view.num_vertices(), 0);
  EXPECT_EQ(view.out.row_ptr.size(), 1u);
}

TEST(Graph, FromEdgesAppendAllowKeepsInputOrderAndParallelEdges) {
  const auto res = dyng::resources::sequential();
  const auto g =
      graph32::from_edges(res, sample_edges().view(), graph_properties::mosp_compatible());
  EXPECT_EQ(g.num_vertices(), 3);
  EXPECT_EQ(g.num_edges(), 5);
  EXPECT_EQ(g.num_weights(), 2);
  EXPECT_EQ(g.properties().num_weights, 2);
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 4, 5, 5}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{2, 1, 2, 0, 0}));
  // Objective-major columns.
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{1, 3, 4, 5, 2, 10, 30, 40, 50, 20}));
  EXPECT_EQ(c.weight(2, 1), 40);
  g.check_integrity(res);
}

TEST(Graph, FromEdgesSortedForbidMergesDuplicatesLastWins) {
  const auto res = dyng::resources::sequential();
  const auto g = graph32::from_edges(res, sample_edges().view(), graph_properties{});
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 3, 4, 4}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{0, 1, 2, 0}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{5, 3, 4, 2, 50, 30, 40, 20}));
  g.check_integrity(res);
}

TEST(Graph, FromEdgesAppendForbidKeepsFirstPosition) {
  const auto res = dyng::resources::sequential();
  graph_properties props = graph_properties::mosp_compatible();
  props.parallel_edges = multi_edges::forbid;
  const auto c = graph32::from_edges(res, sample_edges().view(), props).to_csr(res);
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{2, 1, 0, 0}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{4, 3, 5, 2, 40, 30, 50, 20}));
}

TEST(Graph, FromEdgesSortedAllowIsStable) {
  const auto res = dyng::resources::sequential();
  graph_properties props;
  props.parallel_edges = multi_edges::allow;
  const auto c = graph32::from_edges(res, sample_edges().view(), props).to_csr(res);
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{0, 1, 2, 2, 0}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{5, 3, 1, 4, 2, 50, 30, 10, 40, 20}));
}

TEST(Graph, FromEdgesSelfLoopPolicies) {
  const auto res = dyng::resources::sequential();
  graph_properties props = graph_properties::mosp_compatible();
  props.semantics.on_self_loop = dyng::batch_semantics::self_loop::drop;
  const auto g = graph32::from_edges(res, sample_edges().view(), props);
  EXPECT_EQ(g.num_edges(), 4);
  g.check_integrity(res);
  props.semantics.on_self_loop = dyng::batch_semantics::self_loop::error;
  EXPECT_THROW((void)graph32::from_edges(res, sample_edges().view(), props),
               dyng::invalid_argument_error);
}

TEST(Graph, FromEdgesUndirectedStoresBothDirections) {
  const auto res = dyng::resources::sequential();
  edges32 e;
  e.num_vertices = 3;
  e.num_weights = 1;
  e.add_edge(0, 1, {7});
  e.add_edge(2, 1, {8});
  e.add_edge(1, 1, {9});
  graph_properties props;
  props.directed = false;
  const auto g = graph32::from_edges(res, e.view(), props);
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 1, 4, 5}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{1, 0, 1, 2, 1}));
  EXPECT_EQ(c.weights, (std::vector<std::int32_t>{7, 7, 9, 8, 8}));
  EXPECT_FALSE(g.is_directed());
  g.check_integrity(res);
}

TEST(Graph, FromEdgesRejectsBadInput) {
  const auto res = dyng::resources::sequential();
  edges32 e;
  e.num_vertices = 2;
  e.num_weights = 1;
  e.add_edge(0, 2, {1});
  EXPECT_THROW((void)graph32::from_edges(res, e.view()), dyng::invalid_argument_error);
  e.dst[0] = -1;
  EXPECT_THROW((void)graph32::from_edges(res, e.view()), dyng::invalid_argument_error);
  e.dst[0] = 1;
  e.weights.push_back(3);
  EXPECT_THROW((void)graph32::from_edges(res, e.view()), dyng::invalid_argument_error);
  EXPECT_THROW(e.add_edge(0, 1, {1, 2}), dyng::invalid_argument_error);
  graph_properties slotted;
  slotted.layout = dyng::row_layout::slotted;
  e.weights.pop_back();
  EXPECT_THROW((void)graph32::from_edges(res, e.view(), slotted), dyng::not_supported_error);
  EXPECT_THROW(graph32{slotted}, dyng::not_supported_error);
}

TEST(Graph, FromCsrValidatesAndRoundTrips) {
  const auto res = dyng::resources::sequential();
  dyng::csr<std::int32_t, std::int32_t, std::int32_t> c;
  c.row_ptr = {0, 2, 2, 3};
  c.col_ind = {2, 1, 0};
  c.num_weights = 1;
  c.weights = {5, 6, 7};
  const auto g = graph32::from_csr(res, c.view(), graph_properties::mosp_compatible());
  const auto back = g.to_csr(res);
  EXPECT_EQ(back.row_ptr, c.row_ptr);
  EXPECT_EQ(back.col_ind, c.col_ind);
  EXPECT_EQ(back.weights, c.weights);

  auto bad = c;
  bad.row_ptr[0] = 1;
  EXPECT_THROW((void)graph32::from_csr(res, bad.view()), dyng::invalid_argument_error);
  bad = c;
  bad.row_ptr = {0, 2, 1, 3};
  EXPECT_THROW((void)graph32::from_csr(res, bad.view()), dyng::invalid_argument_error);
  bad = c;
  bad.row_ptr = {0, 2, 2, 2};
  EXPECT_THROW((void)graph32::from_csr(res, bad.view()), dyng::invalid_argument_error);
  bad = c;
  bad.col_ind[0] = 3;
  EXPECT_THROW((void)graph32::from_csr(res, bad.view()), dyng::invalid_argument_error);
  bad = c;
  bad.weights.pop_back();
  EXPECT_THROW((void)graph32::from_csr(res, bad.view()), dyng::invalid_argument_error);
  graph_properties undirected;
  undirected.directed = false;
  EXPECT_THROW((void)graph32::from_csr(res, c.view(), undirected), dyng::invalid_argument_error);
}

TEST(Graph, ViewAndTransposedStorage) {
  const auto res = dyng::resources::sequential();
  const auto g =
      graph32::from_edges(res, sample_edges().view(), graph_properties::mosp_compatible());
  const auto view = g.view();
  EXPECT_TRUE(view.has_transposed);
  EXPECT_EQ(view.version, 0u);
  EXPECT_EQ(view.num_edges(), 5);
  EXPECT_EQ(view.num_weights(), 2);
  // In-edges: row v lists the sources of the edges into v, in out-edge order.
  EXPECT_EQ(values(view.in.row_ptr), (std::vector<std::int32_t>{0, 2, 3, 5}));
  EXPECT_EQ(values(view.in.col_ind), (std::vector<std::int32_t>{0, 1, 0, 0, 0}));
  EXPECT_EQ(values(view.in.weight_column(0)), (std::vector<std::int32_t>{5, 2, 3, 1, 4}));
  EXPECT_EQ(values(view.in.weight_column(1)), (std::vector<std::int32_t>{50, 20, 30, 10, 40}));
  EXPECT_EQ(values(view.out.weight_column(1)), (std::vector<std::int32_t>{10, 30, 40, 50, 20}));
  EXPECT_THROW((void)view.out.weight_column(2), dyng::invalid_argument_error);

  graph_properties props = graph_properties::mosp_compatible();
  props.store_transposed = false;
  const auto h = graph32::from_edges(res, sample_edges().view(), props);
  EXPECT_FALSE(h.has_transposed());
  EXPECT_FALSE(h.view().has_transposed);
  EXPECT_TRUE(h.view().in.row_ptr.empty());
}

TEST(Graph, MoveCloneAndReserve) {
  const auto res = dyng::resources::sequential();
  auto g = graph32::from_edges(res, sample_edges().view(), graph_properties::mosp_compatible());
  auto copy = g.clone(res);
  graph32 moved = std::move(g);
  EXPECT_EQ(moved.num_edges(), 5);
  EXPECT_EQ(copy.num_edges(), 5);
  EXPECT_EQ(g.num_edges(), 0);  // NOLINT(bugprone-use-after-move): moved-from state is defined
  EXPECT_THROW((void)g.view(), dyng::invalid_argument_error);  // NOLINT(bugprone-use-after-move)
  moved.reserve(res, 100);
  EXPECT_EQ(moved.num_edges(), 5);
  EXPECT_THROW(moved.reserve(res, -1), dyng::invalid_argument_error);
}

TEST(Graph, SixtyFourBitTypes) {
  const auto res = dyng::resources::sequential();
  dyng::edge_list<std::int64_t, std::int32_t> e;
  e.num_vertices = 4;
  e.num_weights = 1;
  e.add_edge(3, 0, {2});
  e.add_edge(0, 3, {1});
  const auto g = dyng::graph<std::int64_t, std::int64_t, std::int32_t>::from_edges(res, e.view());
  EXPECT_EQ(g.num_vertices(), 4);
  EXPECT_EQ(g.view().out.row_ptr[4], 2);
  const auto h = dyng::graph<>::from_edges(res, sample_edges().view());
  EXPECT_EQ(h.num_edges(), 4);
}

}  // namespace

namespace {

// The OpenMP transposition (graph built and updated with resources::openmp) is identical to the
// sequential one, including the order of parallel edges inside a row.
TEST(GraphTranspose, OpenmpEqualsSequential) {
  if (!dyng::backend_available(dyng::backend::openmp)) {
    GTEST_SKIP() << "OpenMP backend not available in this build";
  }
  using graph64 = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
  std::mt19937 rng(4242);
  for (int trial = 0; trial < 20; ++trial) {
    const std::int32_t n = 1 + static_cast<std::int32_t>(rng() % 3000);
    const std::size_t m = rng() % 20000;
    dyng::edge_list<std::int32_t, std::int32_t> list;
    list.num_vertices = n;
    list.num_weights = 2;
    for (std::size_t i = 0; i < m; ++i) {
      // Few distinct heads: long in-rows with many parallel edges.
      const auto u = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n));
      const auto v = static_cast<std::int32_t>(
          rng() % std::min<std::uint32_t>(static_cast<std::uint32_t>(n), 50));
      list.add_edge(u, v, {static_cast<std::int32_t>(i), static_cast<std::int32_t>(rng() % 9)});
    }
    const auto seq = dyng::resources::sequential();
    const auto omp = dyng::resources::openmp(7);
    const auto a = graph64::from_edges(seq, list.view(), graph_properties::mosp_compatible());
    const auto b = graph64::from_edges(omp, list.view(), graph_properties::mosp_compatible());
    const auto va = a.view().in;
    const auto vb = b.view().in;
    ASSERT_TRUE(
        std::equal(va.row_ptr.begin(), va.row_ptr.end(), vb.row_ptr.begin(), vb.row_ptr.end()));
    ASSERT_TRUE(
        std::equal(va.col_ind.begin(), va.col_ind.end(), vb.col_ind.begin(), vb.col_ind.end()));
    ASSERT_TRUE(
        std::equal(va.weights.begin(), va.weights.end(), vb.weights.begin(), vb.weights.end()));
    EXPECT_NO_THROW(b.check_integrity(omp));
  }
}

}  // namespace

TEST(Graph, ImpossibleAllocationsAreOutOfMemoryErrors) {
  // Every exception that leaves dynG derives from dyng::error (PLAN Section 4.7.3): host
  // allocation failures (std::bad_alloc, std::length_error) become out_of_memory_error.
  using graph64 = dyng::graph<std::int64_t, std::int64_t, std::int32_t>;
  const auto res = dyng::resources::sequential();
  dyng::edge_list<std::int64_t, std::int32_t> huge;
  huge.num_vertices = std::int64_t{1} << 62;  // 2^62 + 1 row offsets cannot be allocated
  huge.num_weights = 1;
  try {
    (void)graph64::from_edges(res, huge.view());
    FAIL() << "from_edges with 2^62 vertices did not throw";
  } catch (const dyng::out_of_memory_error& e) {
    EXPECT_NE(std::string(e.what()).find("graph::from_edges"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("4611686018427387904 vertices"), std::string::npos)
        << e.what();
  }

  graph64 small = graph64::from_edges(res, dyng::edge_list<std::int64_t, std::int32_t>{}.view());
  EXPECT_THROW(small.reserve(res, std::int64_t{1} << 62), dyng::out_of_memory_error);

  dyng::edge_batch<std::int64_t, std::int32_t> batch(1);
  EXPECT_THROW(batch.reserve(std::size_t{1} << 62, 0), dyng::out_of_memory_error);
  if (DYNG_TEST_SANITIZED) {
    // The sizes above exceed max_size() (std::length_error); a merely impossible size (8 TiB)
    // makes the sanitizer allocators abort instead of throwing std::bad_alloc.
    return;
  }
  // Vertex growth to 2^40 vertices: the new row offsets (8 TiB) cannot be allocated.
  dyng::graph_properties props;
  props.semantics.allow_vertex_growth = true;
  graph64 grow =
      graph64::from_edges(res, dyng::edge_list<std::int64_t, std::int32_t>{}.view(), props);
  dyng::edge_batch<std::int64_t, std::int32_t> far(0);
  far.insert_edge(0, std::int64_t{1} << 40, {});
  EXPECT_THROW((void)grow.apply(res, far.view()), dyng::out_of_memory_error);
}
