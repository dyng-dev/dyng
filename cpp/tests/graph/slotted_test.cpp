// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slotted_test.cpp
 * @brief row_layout::slotted on the host (ADR 0069): graph::with_capacity(), vertex batches under
 *        batch_semantics::net_effect(), the queries, the errors with the strong guarantee,
 *        LabelPropagation-CUDA@a276a3a's nine input_contract cases, the edge_batch vertex
 *        operations, the 64-bit offsets, the compact-only guard of the device copy (ADR 0070),
 *        and the rows, row ends and states after every batch of a DynLP stream against
 *        dyng::testing::lp_host_model.
 */
#include "graph/graph_impl.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/testing/lp_host_model.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
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

graph_properties slotted_props() {
  graph_properties p;
  p.directed = false;
  p.layout = row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = batch_semantics::net_effect();
  return p;
}

template <typename graph_t>
graph_t make(const dyng::resources& res, const std::vector<typename graph_t::edge_type>& cap) {
  return graph_t::with_capacity(res, static_cast<std::int32_t>(cap.size()), host_view(cap),
                                slotted_props());
}

/// Everything observable of a slotted graph (for the strong guarantee: byte for byte).
struct snapshot {
  std::vector<std::int32_t> row_ptr;
  std::vector<std::int32_t> col_ind;
  std::vector<float> weights;
  std::vector<std::int32_t> row_end;
  std::vector<vertex_status> status;
  std::uint64_t version = 0;
  std::uint64_t state = 0;
  std::int32_t edges = 0;
  std::int32_t active = 0;

  bool operator==(const snapshot& o) const {
    return row_ptr == o.row_ptr && col_ind == o.col_ind &&
           std::memcmp(weights.data(), o.weights.data(), weights.size() * sizeof(float)) == 0 &&
           weights.size() == o.weights.size() && row_end == o.row_end && status == o.status &&
           version == o.version && state == o.state && edges == o.edges && active == o.active;
  }
};

snapshot snap(const lp_graph& g) {
  const auto v = g.view();
  snapshot s;
  s.row_ptr.assign(v.out.row_ptr.begin(), v.out.row_ptr.end());
  s.col_ind.assign(v.out.col_ind.begin(), v.out.col_ind.end());
  s.weights.assign(v.out.weights.begin(), v.out.weights.end());
  s.row_end.assign(v.row_end.begin(), v.row_end.end());
  s.status.assign(v.status.begin(), v.status.end());
  s.version = g.version();
  s.state = dyng::detail::graph_access::impl(g).state_id;
  s.edges = g.num_edges();
  s.active = g.num_active_vertices();
  return s;
}

/// LabelPropagation-CUDA's input_contract_test graph: vertices 0, 1, 2; edges 0-2 (0.9) and 1-2
/// (0.8); 0 and 1 are ground truth (labels 0 and 1).
lp_batch first_batch() {
  lp_batch b;
  b.insert_vertex(0, 0);
  b.insert_vertex(1, 1);
  b.insert_vertex(2);
  b.insert_edge(0, 2, {0.9F});
  b.insert_edge(1, 2, {0.8F});
  return b;
}

const std::vector<std::int32_t> three_vertex_capacity = {1, 1, 2};

TEST(Slotted, PropertiesPreset) {
  constexpr batch_semantics s = batch_semantics::net_effect();
  EXPECT_TRUE(s.as_net_effect);
  EXPECT_FALSE(s.as_sets);
  EXPECT_FALSE(s.allow_vertex_growth);
  EXPECT_TRUE(s.deletions_first);
  EXPECT_EQ(s.on_self_loop, batch_semantics::self_loop::keep);
  EXPECT_FALSE(batch_semantics{}.as_net_effect);
  EXPECT_FALSE(batch_semantics::set().as_net_effect);
}

TEST(Slotted, WithCapacityIsEmpty) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int32_t> cap = {2, 0, 3, 1};
  const lp_graph g = make<lp_graph>(res, cap);
  EXPECT_EQ(g.num_vertices(), 4);
  EXPECT_EQ(g.num_active_vertices(), 0);
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_EQ(g.num_weights(), 1);
  EXPECT_EQ(g.version(), 0U);
  EXPECT_EQ(g.properties().layout, row_layout::slotted);
  EXPECT_FALSE(g.has_transposed());  // the rows are symmetric
  const auto v = g.view();
  EXPECT_EQ(v.layout, row_layout::slotted);
  EXPECT_FALSE(v.has_transposed);
  EXPECT_FALSE(v.directed);
  ASSERT_EQ(v.out.row_ptr.size(), 5U);
  EXPECT_EQ(v.out.row_ptr[4], 6);
  ASSERT_EQ(v.out.col_ind.size(), 6U);
  for (const auto c : v.out.col_ind) {
    EXPECT_EQ(c, -1);
  }
  ASSERT_EQ(v.status.size(), 4U);
  ASSERT_EQ(v.row_end.size(), 4U);
  for (std::size_t u = 0; u < 4; ++u) {
    EXPECT_EQ(v.status[u], vertex_status::inactive);
    EXPECT_EQ(v.row_end[u], v.out.row_ptr[u]);
  }
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.num_vertices(), 4);
  EXPECT_EQ(c.num_edges(), 0);
  EXPECT_NO_THROW(g.check_integrity(res));
}

TEST(Slotted, CompactViewsHaveNoStatus) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int32_t> src = {0};
  const std::vector<std::int32_t> dst = {1};
  const std::vector<float> w = {0.5F};
  const auto g = lp_graph::from_edges(res, {2, host_view(src), host_view(dst), host_view(w), 1});
  EXPECT_TRUE(g.view().status.empty());
  EXPECT_TRUE(g.view().row_end.empty());
  EXPECT_EQ(g.num_active_vertices(), 2);
}

TEST(Slotted, WithCapacityRefusesOtherProperties) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int32_t> cap = {1, 1};
  const auto build = [&](const graph_properties& p) {
    return lp_graph::with_capacity(res, 2, host_view(cap), p);
  };
  const auto refused = [&](graph_properties p, const char* property) {
    try {
      (void)build(p);
      ADD_FAILURE() << "accepted " << property;
    } catch (const dyng::not_supported_error& e) {
      EXPECT_NE(std::string(e.what()).find(property), std::string::npos) << e.what();
    }
  };
  graph_properties p = slotted_props();
  p.directed = true;
  refused(p, "directed");
  p = slotted_props();
  p.order = dyng::row_order::sorted;
  refused(p, "order");
  p = slotted_props();
  p.parallel_edges = dyng::multi_edges::forbid;
  refused(p, "parallel_edges");
  p = slotted_props();
  p.semantics = batch_semantics::upsert_last_wins();
  refused(p, "semantics");
  p = slotted_props();
  p.semantics.on_self_loop = batch_semantics::self_loop::drop;
  refused(p, "semantics");
  p = slotted_props();
  p.layout = row_layout::slack;
  refused(p, "slack");
  p = slotted_props();
  p.layout = row_layout::compact;
  EXPECT_THROW((void)build(p), dyng::invalid_argument_error);
  p = slotted_props();
  p.num_weights = 2;
  EXPECT_THROW((void)build(p), dyng::invalid_argument_error);
  const std::vector<std::int32_t> negative = {1, -1};
  EXPECT_THROW((void)lp_graph::with_capacity(res, 2, host_view(negative), slotted_props()),
               dyng::invalid_argument_error);
  EXPECT_THROW((void)lp_graph::with_capacity(res, 3, host_view(cap), slotted_props()),
               dyng::invalid_argument_error);
  EXPECT_THROW(
      (void)lp_graph::with_capacity(res, -1, array_view<const std::int32_t>(), slotted_props()),
      dyng::invalid_argument_error);
}

TEST(Slotted, CompactConstructorsRefuseSlottedRows) {
  const dyng::resources res = dyng::resources::sequential();
  graph_properties p = slotted_props();
  EXPECT_THROW((void)lp_graph(p), dyng::not_supported_error);
  const std::vector<std::int32_t> src = {0};
  const std::vector<std::int32_t> dst = {1};
  const std::vector<float> w = {0.5F};
  const dyng::edge_list_view<std::int32_t, float> edges{2, host_view(src), host_view(dst),
                                                        host_view(w), 1};
  EXPECT_THROW((void)lp_graph::from_edges(res, edges, p), dyng::not_supported_error);
  graph_properties compact_net;  // compact rows with the vertex-batch semantics
  compact_net.semantics = batch_semantics::net_effect();
  EXPECT_THROW((void)lp_graph::from_edges(res, edges, compact_net), dyng::not_supported_error);
  // Vertex operations on compact rows name the way to slotted rows.
  auto g = lp_graph::from_edges(res, edges);
  lp_batch b;
  b.insert_vertex(1);
  try {
    (void)g.apply(res, b.view());
    ADD_FAILURE() << "accepted a vertex batch on compact rows";
  } catch (const dyng::not_supported_error& e) {
    EXPECT_NE(std::string(e.what()).find("with_capacity"), std::string::npos) << e.what();
  }
}

TEST(Slotted, CapacityBeyondTheOffsetType) {
  const dyng::resources res = dyng::resources::sequential();
  // Three rows of 2^30 slots: 3 * 2^30 > 2^31 - 1, refused before anything is allocated.
  const std::vector<std::int32_t> cap = {1 << 30, 1 << 30, 1 << 30};
  try {
    (void)lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
    ADD_FAILURE() << "accepted 3 * 2^30 slots with 32-bit offsets";
  } catch (const dyng::capacity_error& e) {
    EXPECT_NE(std::string(e.what()).find("64-bit"), std::string::npos) << e.what();
  }
  // Exactly 2^31 - 1 is accepted by the check (the arithmetic does not overflow on the way).
  const std::vector<std::int32_t> edge = {std::numeric_limits<std::int32_t>::max() - 5, 5};
  std::int64_t sum = 0;
  for (const auto c : edge) {
    sum += c;
  }
  EXPECT_EQ(sum, std::numeric_limits<std::int32_t>::max());
  const std::vector<std::int32_t> over = {std::numeric_limits<std::int32_t>::max() - 5, 6};
  EXPECT_THROW((void)lp_graph::with_capacity(res, 2, host_view(over), slotted_props()),
               dyng::capacity_error);
}

TEST(Slotted, VertexBatchAppendsInBatchOrder) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, {2, 2, 3, 1});
  lp_batch b = first_batch();
  b.insert_vertex(3);
  b.insert_edge(2, 3, {0.25F});
  b.insert_edge(1, 0, {0.5F});
  const dyng::apply_summary s = g.apply(res, b.view());
  EXPECT_EQ(s.inserted_vertices, 4);
  EXPECT_EQ(s.deleted_vertices, 0);
  EXPECT_EQ(s.inserted_edges, 8);  // both directions
  EXPECT_EQ(s.dropped_insertions, 0);
  EXPECT_EQ(s.num_vertices_after, 4);
  EXPECT_EQ(g.version(), 1U);
  EXPECT_EQ(g.num_active_vertices(), 4);
  EXPECT_EQ(g.num_edges(), 8);
  const auto v = g.view();
  // Row 0: 2 (edge 0), 1 (edge 3); row 1: 2 (edge 1), 0 (edge 3); row 2: 0, 1, 3; row 3: 2.
  const std::vector<std::int32_t> col = {2, 1, 2, 0, 0, 1, 3, 2};
  const std::vector<float> w = {0.9F, 0.5F, 0.8F, 0.5F, 0.9F, 0.8F, 0.25F, 0.25F};
  ASSERT_EQ(v.out.col_ind.size(), col.size());
  for (std::size_t e = 0; e < col.size(); ++e) {
    EXPECT_EQ(v.out.col_ind[e], col[e]) << e;
    EXPECT_EQ(v.out.weights[e], w[e]) << e;
  }
  for (std::size_t u = 0; u < 4; ++u) {
    EXPECT_EQ(v.row_end[u], v.out.row_ptr[u + 1]);
    EXPECT_EQ(v.status[u], vertex_status::active);
  }
  EXPECT_NO_THROW(g.check_integrity(res));
  // The applied record of the commit: Ins_eff with labels, each live edge once.
  dyng::detail::apply_delta<std::int32_t> delta;
  lp_graph h = make<lp_graph>(res, {2, 2, 3, 1});
  (void)dyng::detail::graph_access::apply(res, h, b.view(), &delta);
  EXPECT_EQ(delta.inserted_vertices, (std::vector<std::int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(delta.inserted_labels, (std::vector<std::int8_t>{0, 1, -1, -1}));
  EXPECT_TRUE(delta.deleted_vertices.empty());
  EXPECT_EQ(delta.insert_src, (std::vector<std::int32_t>{0, 1, 2, 1}));
  EXPECT_EQ(delta.insert_dst, (std::vector<std::int32_t>{2, 2, 3, 0}));
  EXPECT_EQ(delta.insert_index, (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_TRUE(delta.delete_src.empty());
}

TEST(Slotted, DeletionsAndTheNetEffect) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, {2, 2, 2, 2, 0});
  lp_batch b0;
  b0.insert_vertex(0);
  b0.insert_vertex(1);
  b0.insert_vertex(2);
  b0.insert_edge(0, 1, {0.5F});
  b0.insert_edge(1, 2, {0.6F});
  (void)g.apply(res, b0.view());
  EXPECT_EQ(g.num_edges(), 4);
  // Batch 1: 3 arrives and leaves (never enters), 1 leaves, 4 (never inserted) is retired; the
  // edges to 3 and to 1 are dropped; an edge-only insertion between present vertices (0, 2).
  lp_batch b1;
  b1.insert_vertex(3);
  b1.delete_vertex(1);
  b1.delete_vertex(3);
  b1.delete_vertex(4);
  b1.insert_edge(0, 3, {0.7F});
  b1.insert_edge(2, 1, {0.1F});
  b1.insert_edge(0, 2, {0.2F});
  dyng::detail::apply_delta<std::int32_t> delta;
  const auto s = dyng::detail::graph_access::apply(res, g, b1.view(), &delta);
  EXPECT_EQ(s.inserted_vertices, 0);
  EXPECT_EQ(s.deleted_vertices, 1);  // Del ∩ V_t
  EXPECT_EQ(s.inserted_edges, 2);
  EXPECT_EQ(s.dropped_insertions, 4);
  EXPECT_TRUE(delta.inserted_vertices.empty());
  EXPECT_EQ(delta.deleted_vertices, (std::vector<std::int32_t>{1}));
  EXPECT_EQ(delta.insert_index, (std::vector<std::size_t>{2}));
  const auto v = g.view();
  EXPECT_EQ(v.status[0], vertex_status::active);
  EXPECT_EQ(v.status[1], vertex_status::deleted);
  EXPECT_EQ(v.status[2], vertex_status::active);
  EXPECT_EQ(v.status[3], vertex_status::deleted);  // Ins ∩ Del ends deleted
  EXPECT_EQ(v.status[4], vertex_status::deleted);  // retired
  EXPECT_EQ(g.num_active_vertices(), 2);
  // The deleted vertex keeps its entries (rows are never compacted); they are not edges.
  EXPECT_EQ(v.row_end[1] - v.out.row_ptr[1], 2);
  EXPECT_EQ(g.num_edges(), 2);
  const auto c = g.to_csr(res);
  EXPECT_EQ(c.num_edges(), 2);
  EXPECT_EQ(c.row_ptr, (std::vector<std::int32_t>{0, 1, 1, 2, 2, 2}));
  EXPECT_EQ(c.col_ind, (std::vector<std::int32_t>{2, 0}));
  EXPECT_EQ(c.weights, (std::vector<float>{0.2F, 0.2F}));
  EXPECT_NO_THROW(g.check_integrity(res));
  // A clone is equal and independent.
  lp_graph k = g.clone(res);
  EXPECT_TRUE(snap(k).row_ptr == snap(g).row_ptr && snap(k).status == snap(g).status &&
              snap(k).col_ind == snap(g).col_ind && snap(k).row_end == snap(g).row_end);
  EXPECT_EQ(k.num_edges(), 2);
  EXPECT_EQ(k.num_active_vertices(), 2);
  lp_batch b2;
  b2.delete_vertex(0);
  (void)k.apply(res, b2.view());
  EXPECT_EQ(k.num_edges(), 0);
  EXPECT_EQ(g.num_edges(), 2);
  EXPECT_NO_THROW(k.check_integrity(res));
}

TEST(Slotted, SelfLoopsAndParallelEdgesAreKept) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, {4, 2});
  lp_batch b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  b.insert_edge(0, 0, {0.5F});
  b.insert_edge(0, 1, {0.25F});
  b.insert_edge(0, 1, {0.75F});
  const auto s = g.apply(res, b.view());
  EXPECT_EQ(s.inserted_edges, 6);
  EXPECT_EQ(g.num_edges(), 6);
  const auto v = g.view();
  EXPECT_EQ(v.out.col_ind[0], 0);
  EXPECT_EQ(v.out.col_ind[1], 0);
  EXPECT_EQ(v.out.col_ind[2], 1);
  EXPECT_EQ(v.out.col_ind[3], 1);
  EXPECT_NO_THROW(g.check_integrity(res));
  lp_batch d;
  d.delete_vertex(0);
  (void)g.apply(res, d.view());
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_NO_THROW(g.check_integrity(res));
}

TEST(Slotted, ErrorsKeepTheGraphUnchanged) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, {2, 2, 2, 1});
  lp_batch b0;
  b0.insert_vertex(0, 1);
  b0.insert_vertex(1);
  b0.insert_edge(0, 1, {0.5F});
  (void)g.apply(res, b0.view());
  const snapshot before = snap(g);
  const auto unchanged = [&](const lp_batch& b, const char* what, auto error) {
    using error_t = decltype(error);
    EXPECT_THROW((void)g.apply(res, b.view()), error_t) << what;
    EXPECT_TRUE(snap(g) == before) << what;
    EXPECT_NO_THROW(g.check_integrity(res)) << what;  // the scratch is clear again
  };
  lp_batch b;
  b.insert_vertex(4);
  unchanged(b, "an inserted id out of range", dyng::invalid_argument_error{""});
  b.clear();
  b.delete_vertex(-1);
  unchanged(b, "a deleted id out of range", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2);
  b.insert_edge(2, 9, {0.5F});
  unchanged(b, "an edge endpoint out of range", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(1);
  unchanged(b, "a re-insertion", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2);
  b.insert_vertex(2);
  unchanged(b, "a duplicate insertion", dyng::invalid_argument_error{""});
  b.clear();
  b.delete_vertex(1);
  b.delete_vertex(1);
  unchanged(b, "a duplicate deletion", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2, 1);
  b.delete_vertex(2);
  unchanged(b, "a seed label on a vertex the batch deletes", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2, 5);
  unchanged(b, "a label other than -1, 0, 1", dyng::invalid_argument_error{""});
  {
    // Labels that do not match the insertions one to one (a label without a vertex).
    const std::vector<std::int32_t> ins = {2};
    const std::vector<std::int8_t> labels = {0, 1};
    dyng::edge_batch_view<std::int32_t, float> view;
    view.insert_vertices = host_view(ins);
    view.insert_vertex_labels = host_view(labels);
    EXPECT_THROW((void)g.apply(res, view), dyng::invalid_argument_error);
    EXPECT_TRUE(snap(g) == before);
  }
  b.clear();
  b.insert_vertex(2);
  b.insert_edge(0, 2, {0.1F});
  b.insert_edge(1, 2, {0.1F});  // row 0 is full (2 slots: 1 and 2 used)
  b.insert_edge(0, 2, {0.1F});
  unchanged(b, "a row overflow", dyng::capacity_error{""});
  b.clear();
  b.delete_edge(0, 1);
  unchanged(b, "an edge deletion", dyng::not_supported_error{""});
  b.clear();
  b.insert_vertex(3);
  b.insert_edge(3, 0, {0.1F});
  b.insert_edge(3, 1, {0.1F});  // row 3 has one slot
  unchanged(b, "an overflow of a new vertex's row", dyng::capacity_error{""});
  // A valid batch still works afterwards.
  b.clear();
  b.insert_vertex(2);
  b.insert_edge(1, 2, {0.3F});
  EXPECT_NO_THROW((void)g.apply(res, b.view()));
  EXPECT_NO_THROW(g.check_integrity(res));
  EXPECT_EQ(g.num_edges(), 4);
}

/// LabelPropagation-CUDA@a276a3a:tests/input_contract_test.cu's nine scenarios on the same
/// three-vertex graph (ADR 0069 maps each): valid and insdel are accepted; reinsert, dupid,
/// dupsame, range, insdelre and insdel2 raise invalid_argument_error with the graph unchanged;
/// dupedge raises capacity_error (the second 0-2 overflows row 0).
TEST(Slotted, OriginalInputContract) {
  const dyng::resources res = dyng::resources::sequential();
  const auto scenario = [](const std::string& name) {
    std::vector<lp_batch> batches;
    lp_batch b0 = first_batch();
    lp_batch b1;
    lp_batch b2;
    if (name == "dupedge") {
      b0.insert_edge(0, 2, {0.9F});
      return std::vector<lp_batch>{b0};
    }
    if (name == "dupid") {
      b1.insert_vertex(2);
      return std::vector<lp_batch>{b0, b1};
    }
    if (name == "dupsame") {
      b0.insert_vertex(2);
      return std::vector<lp_batch>{b0};
    }
    if (name == "insdel" || name == "insdel2") {
      lp_batch c;  // vertex 2 never enters the graph, so it gets no edges
      c.insert_vertex(0, 0);
      c.insert_vertex(1, 1);
      c.insert_vertex(2);
      if (name == "insdel2") {
        c.insert_vertex(2);
      }
      c.delete_vertex(2);
      return std::vector<lp_batch>{c};
    }
    if (name == "insdelre") {
      b1.insert_vertex(2);
      b1.delete_vertex(2);
      return std::vector<lp_batch>{b0, b1};
    }
    if (name == "range") {
      b1.delete_vertex(3);
      return std::vector<lp_batch>{b0, b1};
    }
    b1.delete_vertex(2);
    if (name == "valid") {
      return std::vector<lp_batch>{b0, b1};
    }
    b2.insert_vertex(2);  // reinsert
    b2.insert_edge(0, 2, {0.9F});
    b2.insert_edge(1, 2, {0.8F});
    return std::vector<lp_batch>{b0, b1, b2};
  };
  for (const std::string name : {"valid", "insdel", "reinsert", "dupid", "dupsame", "range",
                                 "dupedge", "insdelre", "insdel2"}) {
    lp_graph g = make<lp_graph>(res, three_vertex_capacity);
    const auto batches = scenario(name);
    std::string error;
    snapshot before;
    for (const auto& b : batches) {
      before = snap(g);
      try {
        (void)g.apply(res, b.view());
      } catch (const dyng::capacity_error&) {
        error = "capacity";
        break;
      } catch (const dyng::invalid_argument_error&) {
        error = "invalid";
        break;
      }
    }
    if (!error.empty()) {
      EXPECT_TRUE(snap(g) == before) << name << ": the refused batch changed the graph";
    }
    EXPECT_NO_THROW(g.check_integrity(res)) << name;
    if (name == "valid" || name == "insdel") {
      EXPECT_EQ(error, "") << name;
    } else if (name == "dupedge") {
      EXPECT_EQ(error, "capacity") << name;
    } else {
      EXPECT_EQ(error, "invalid") << name;
    }
  }
  // insdel: 2 is deleted, 0 and 1 active, no edges.
  lp_graph g = make<lp_graph>(res, three_vertex_capacity);
  for (const auto& b : scenario("insdel")) {
    (void)g.apply(res, b.view());
  }
  EXPECT_EQ(g.view().status[2], vertex_status::deleted);
  EXPECT_EQ(g.num_active_vertices(), 2);
  EXPECT_EQ(g.num_edges(), 0);
}

TEST(Slotted, EdgeBatchVertexOperations) {
  lp_batch b;
  EXPECT_TRUE(b.empty());
  b.insert_vertex(3);
  EXPECT_FALSE(b.empty());
  b.insert_vertex(4, 1);
  b.delete_vertex(7);
  EXPECT_EQ(b.insert_vertices(), (std::vector<std::int32_t>{3, 4}));
  EXPECT_EQ(b.insert_vertex_labels(), (std::vector<std::int8_t>{-1, 1}));
  EXPECT_EQ(b.delete_vertices(), (std::vector<std::int32_t>{7}));
  const auto v = b.view();
  EXPECT_EQ(v.insert_vertices.size(), 2U);
  EXPECT_EQ(v.insert_vertex_labels.size(), 2U);
  EXPECT_EQ(v.delete_vertices.size(), 1U);
  EXPECT_FALSE(v.empty());
  b.clear();
  EXPECT_TRUE(b.empty());
  EXPECT_TRUE(b.view().empty());
  lp_batch only_deletion;
  only_deletion.delete_vertex(1);
  EXPECT_FALSE(only_deletion.empty());
}

TEST(Slotted, SixtyFourBitOffsets) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int64_t> cap = {1, 1, 2};
  lp_graph64 g = lp_graph64::with_capacity(res, 3, host_view(cap), slotted_props());
  dyng::edge_batch<std::int32_t, float> b;
  b.insert_vertex(0, 0);
  b.insert_vertex(1, 1);
  b.insert_vertex(2);
  b.insert_edge(0, 2, {0.9F});
  b.insert_edge(1, 2, {0.8F});
  (void)g.apply(res, b.view());
  EXPECT_EQ(g.num_edges(), 4);
  EXPECT_EQ(g.view().row_end[2], 4);
  EXPECT_NO_THROW(g.check_integrity(res));
  // 64-bit offsets take more than 2^31 - 1 slots in the check (nothing is allocated here).
  const std::vector<std::int64_t> big = {std::int64_t{1} << 40, -1};
  EXPECT_THROW((void)lp_graph64::with_capacity(res, 2, host_view(big), slotted_props()),
               dyng::invalid_argument_error);
}

TEST(Slotted, IntegerWeightsAndUnweightedGraphs) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int32_t> cap = {1, 1};
  using int_graph = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
  int_graph g = int_graph::with_capacity(res, 2, host_view(cap), slotted_props());
  dyng::edge_batch<std::int32_t, std::int32_t> b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  b.insert_edge(0, 1, {7});
  (void)g.apply(res, b.view());
  EXPECT_EQ(g.to_csr(res).weights, (std::vector<std::int32_t>{7, 7}));
  graph_properties p = slotted_props();
  p.num_weights = 0;
  using unweighted_graph = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>;
  unweighted_graph u = unweighted_graph::with_capacity(res, 2, host_view(cap), p);
  dyng::edge_batch<std::int32_t, dyng::unweighted> ub;
  ub.insert_vertex(0);
  ub.insert_vertex(1);
  ub.insert_edge(0, 1);
  (void)u.apply(res, ub.view());
  EXPECT_EQ(u.num_edges(), 2);
  EXPECT_NO_THROW(u.check_integrity(res));
}

TEST(Slotted, ReserveChangesNothing) {
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(res, {1, 1});
  const snapshot before = snap(g);
  g.reserve(res, 1000);
  EXPECT_TRUE(snap(g) == before);
  EXPECT_THROW(g.reserve(res, -1), dyng::invalid_argument_error);
}

/// The rows, row ends, slots and states after every batch of a DynLP stream (5000 vertices, 5
/// batches, the legacy generator) equal lp_host_model's, and check_integrity() passes.
TEST(Slotted, DynlpStreamAgainstTheHostModel) {
  const dyng::resources res = dyng::resources::sequential();
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = 5000;
  opt.num_batches = 5;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(res, opt);
  lp_graph g = lp_graph::with_capacity(res, static_cast<std::int32_t>(opt.total_vertices),
                                       host_view(stream.row_capacity), slotted_props());
  std::vector<std::int64_t> cap(stream.row_capacity.begin(), stream.row_capacity.end());
  dyng::testing::lp_host_model model(cap);
  for (std::size_t b = 0; b < stream.batches.size(); ++b) {
    const auto& batch = stream.batches[b];
    (void)g.apply(res, batch.view());
    dyng::testing::lp_model_batch mb;
    mb.insert_vertices = batch.insert_vertices;
    mb.insert_labels = batch.insert_vertex_labels;
    mb.delete_vertices = batch.delete_vertices;
    mb.edge_sources = batch.edge_sources;
    mb.edge_destinations = batch.edge_destinations;
    mb.edge_weights = batch.edge_weights;
    const auto report = model.apply(mb, stream.similarity_threshold);
    EXPECT_EQ(report.dropped_overflow, 0);
    const auto v = g.view();
    for (std::size_t u = 0; u < static_cast<std::size_t>(g.num_vertices()); ++u) {
      ASSERT_EQ(static_cast<std::int64_t>(v.row_end[u]), model.row_end()[u]) << "batch " << b;
      const auto st = model.states()[u];
      const vertex_status expected =
          st == dyng::testing::lp_host_model::vertex_state::active    ? vertex_status::active
          : st == dyng::testing::lp_host_model::vertex_state::deleted ? vertex_status::deleted
                                                                      : vertex_status::inactive;
      ASSERT_EQ(v.status[u], expected) << "batch " << b << " vertex " << u;
    }
    ASSERT_EQ(v.out.col_ind.size(), model.slot_columns().size());
    for (std::size_t e = 0; e < v.out.col_ind.size(); ++e) {
      ASSERT_EQ(v.out.col_ind[e], model.slot_columns()[e]) << "batch " << b << " slot " << e;
      std::uint32_t a = 0;
      std::uint32_t m = 0;
      std::memcpy(&a, &v.out.weights[e], 4);
      std::memcpy(&m, &model.slot_weights()[e], 4);
      ASSERT_EQ(a, m) << "batch " << b << " slot " << e;
    }
    EXPECT_NO_THROW(g.check_integrity(res)) << "batch " << b;
    std::int64_t present = 0;
    for (const auto p : model.present()) {
      present += p;
    }
    EXPECT_EQ(g.num_active_vertices(), present);
  }
}

}  // namespace
