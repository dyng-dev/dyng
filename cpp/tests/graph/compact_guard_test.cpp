// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file compact_guard_test.cpp
 * @brief The compact-only guard (AD11, ADR 0070): every algorithm of 0.1-0.3 that reads graphs
 *        (sssp, mosp, cycle_count, dynamic_bfs, triangle_delta), each of its entries (compute,
 *        update, from_arrays) and dyng::update()'s composition refuse a graph with
 *        row_layout::slotted with a not_supported_error that names the layout, before anything
 *        changes; compact graphs are unaffected.
 */
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/dynamic_bfs.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/triangle_delta.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using dyng::host_view;

dyng::graph_properties slotted_props(int num_weights) {
  dyng::graph_properties p;
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  p.num_weights = num_weights;
  return p;
}

/// Calls `f` and checks that it throws not_supported_error naming row_layout::slotted.
template <typename function_t>
void expect_refused(function_t&& f, const char* what) {
  try {
    f();
    ADD_FAILURE() << what << ": accepted a slotted graph";
  } catch (const dyng::not_supported_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("row_layout::slotted"), std::string::npos) << what << ": " << message;
    EXPECT_NE(message.find("row_layout::compact"), std::string::npos) << what << ": " << message;
  }
}

using weighted = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using plain = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>;

const std::vector<std::int32_t> capacity = {2, 2, 2};

weighted slotted_weighted(const dyng::resources& res) {
  weighted g = weighted::with_capacity(res, 3, host_view(capacity), slotted_props(1));
  dyng::edge_batch<std::int32_t, std::int32_t> b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  b.insert_vertex(2);
  b.insert_edge(0, 1, {3});
  (void)g.apply(res, b.view());
  return g;
}

weighted compact_weighted(const dyng::resources& res) {
  const std::vector<std::int32_t> src = {0, 1};
  const std::vector<std::int32_t> dst = {1, 0};
  const std::vector<std::int32_t> w = {3, 3};
  return weighted::from_edges(res, {3, host_view(src), host_view(dst), host_view(w), 1});
}

TEST(CompactGuard, ComputeRefusesSlottedRows) {
  const dyng::resources res = dyng::resources::sequential();
  const weighted g = slotted_weighted(res);
  expect_refused([&] { (void)dyng::sssp::compute(res, g, 0); }, "sssp::compute");
  expect_refused([&] { (void)dyng::mosp::compute(res, g, 0); }, "mosp::compute");
  expect_refused([&] { (void)dyng::cycle_count::compute(res, g); }, "cycle_count::compute");
  expect_refused([&] { (void)dyng::dynamic_bfs::compute(res, g); }, "dynamic_bfs::compute");
  expect_refused([&] { (void)dyng::triangle_delta::compute(res, g); }, "triangle_delta::compute");
  const plain u = plain::with_capacity(res, 3, host_view(capacity), slotted_props(0));
  expect_refused([&] { (void)dyng::cycle_count::compute(res, u); }, "cycle_count::compute");
  expect_refused([&] { (void)dyng::dynamic_bfs::compute(res, u); }, "dynamic_bfs::compute");
}

TEST(CompactGuard, UpdateAndCompositionRefuseSlottedRows) {
  const dyng::resources res = dyng::resources::sequential();
  const weighted c = compact_weighted(res);
  weighted g = slotted_weighted(res);
  auto tree = dyng::sssp::compute(res, c, 0);
  auto bfs = dyng::dynamic_bfs::compute(res, c);
  dyng::graph_properties undirected;
  undirected.directed = false;
  const std::vector<std::int32_t> src = {0};
  const std::vector<std::int32_t> dst = {1};
  const std::vector<std::int32_t> w = {3};
  const weighted cu =
      weighted::from_edges(res, {3, host_view(src), host_view(dst), host_view(w), 1}, undirected);
  auto tri = dyng::triangle_delta::compute(res, cu);
  auto cycles = dyng::cycle_count::compute(res, c);
  auto multi = dyng::mosp::compute(res, c, 0);
  dyng::edge_batch<std::int32_t, std::int32_t> b;
  b.insert_edge(1, 2, {4});
  const std::uint64_t version = g.version();
  expect_refused([&] { (void)dyng::sssp::update(res, g, b.view(), tree); }, "sssp::update");
  expect_refused([&] { (void)dyng::mosp::update(res, g, b.view(), multi); }, "mosp::update");
  expect_refused([&] { (void)dyng::cycle_count::update(res, g, b.view(), cycles); },
                 "cycle_count::update");
  expect_refused([&] { (void)dyng::dynamic_bfs::update(res, g, b.view(), bfs); },
                 "dynamic_bfs::update");
  expect_refused([&] { (void)dyng::triangle_delta::update(res, g, b.view(), tri); },
                 "triangle_delta::update");
  expect_refused([&] { (void)dyng::update(res, g, b.view(), tree, bfs); }, "dyng::update");
  EXPECT_EQ(g.version(), version);  // nothing was applied
  EXPECT_NO_THROW(g.check_integrity(res));
  // The results are still usable on their compact graph.
  weighted c2 = compact_weighted(res);
  auto tree2 = dyng::sssp::compute(res, c2, 0);
  EXPECT_NO_THROW((void)dyng::sssp::update(res, c2, b.view(), tree2));
}

TEST(CompactGuard, ImportedTreesRefuseSlottedRows) {
  const dyng::resources res = dyng::resources::sequential();
  const weighted g = slotted_weighted(res);
  const std::vector<std::int64_t> distances = {0, 3, dyng::infinite_distance<std::int64_t>()};
  const std::vector<std::int32_t> parents = {-1, 0, -1};
  expect_refused(
      [&] {
        (void)dyng::sssp::result<std::int32_t>::from_arrays(res, g, 0, host_view(distances),
                                                            host_view(parents));
      },
      "sssp::result::from_arrays");
}

}  // namespace
