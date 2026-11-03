// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file compact_guard_test.cpp
 * @brief The compact-only guard (AD11, ADR 0070): every algorithm of 0.1-0.3 that reads graphs
 *        (sssp, mosp, cycle_count, dynamic_bfs, triangle_delta), each of its entries (compute,
 *        update, from_arrays) and dyng::update()'s composition refuse a graph with
 *        row_layout::slotted with a not_supported_error that names the layout, before anything
 *        changes; compact graphs are unaffected. Each algorithm's checks are compiled when the
 *        algorithm is built (DYNG_ALGORITHM_<NAME>, a subset build of DYNG_ALGORITHMS).
 */
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/update.hpp>
#if defined(DYNG_ALGORITHM_CYCLE_COUNT)
#include <dyng/cycle_count.hpp>
#endif
#if defined(DYNG_ALGORITHM_DYNAMIC_BFS)
#include <dyng/dynamic_bfs.hpp>
#endif
#if defined(DYNG_ALGORITHM_MOSP)
#include <dyng/mosp.hpp>
#endif
#if defined(DYNG_ALGORITHM_SSSP)
#include <dyng/sssp.hpp>
#endif
#if defined(DYNG_ALGORITHM_TRIANGLE_DELTA)
#include <dyng/triangle_delta.hpp>
#endif

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
  const plain u = plain::with_capacity(res, 3, host_view(capacity), slotted_props(0));
  (void)g;
  (void)u;
#if defined(DYNG_ALGORITHM_SSSP)
  expect_refused([&] { (void)dyng::sssp::compute(res, g, 0); }, "sssp::compute");
#endif
#if defined(DYNG_ALGORITHM_MOSP)
  expect_refused([&] { (void)dyng::mosp::compute(res, g, 0); }, "mosp::compute");
#endif
#if defined(DYNG_ALGORITHM_CYCLE_COUNT)
  expect_refused([&] { (void)dyng::cycle_count::compute(res, g); }, "cycle_count::compute");
  expect_refused([&] { (void)dyng::cycle_count::compute(res, u); }, "cycle_count::compute");
#endif
#if defined(DYNG_ALGORITHM_DYNAMIC_BFS)
  expect_refused([&] { (void)dyng::dynamic_bfs::compute(res, g); }, "dynamic_bfs::compute");
  expect_refused([&] { (void)dyng::dynamic_bfs::compute(res, u); }, "dynamic_bfs::compute");
#endif
#if defined(DYNG_ALGORITHM_TRIANGLE_DELTA)
  expect_refused([&] { (void)dyng::triangle_delta::compute(res, g); }, "triangle_delta::compute");
#endif
}

TEST(CompactGuard, UpdateAndCompositionRefuseSlottedRows) {
  const dyng::resources res = dyng::resources::sequential();
  const weighted c = compact_weighted(res);
  weighted g = slotted_weighted(res);
  (void)c;
  dyng::edge_batch<std::int32_t, std::int32_t> b;
  b.insert_edge(1, 2, {4});
  const std::uint64_t version = g.version();
#if defined(DYNG_ALGORITHM_SSSP)
  auto tree = dyng::sssp::compute(res, c, 0);
  expect_refused([&] { (void)dyng::sssp::update(res, g, b.view(), tree); }, "sssp::update");
#endif
#if defined(DYNG_ALGORITHM_MOSP)
  auto multi = dyng::mosp::compute(res, c, 0);
  expect_refused([&] { (void)dyng::mosp::update(res, g, b.view(), multi); }, "mosp::update");
#endif
#if defined(DYNG_ALGORITHM_CYCLE_COUNT)
  auto cycles = dyng::cycle_count::compute(res, c);
  expect_refused([&] { (void)dyng::cycle_count::update(res, g, b.view(), cycles); },
                 "cycle_count::update");
#endif
#if defined(DYNG_ALGORITHM_DYNAMIC_BFS)
  auto bfs = dyng::dynamic_bfs::compute(res, c);
  expect_refused([&] { (void)dyng::dynamic_bfs::update(res, g, b.view(), bfs); },
                 "dynamic_bfs::update");
#endif
#if defined(DYNG_ALGORITHM_TRIANGLE_DELTA)
  dyng::graph_properties undirected;
  undirected.directed = false;
  const std::vector<std::int32_t> src = {0};
  const std::vector<std::int32_t> dst = {1};
  const std::vector<std::int32_t> w = {3};
  const weighted cu =
      weighted::from_edges(res, {3, host_view(src), host_view(dst), host_view(w), 1}, undirected);
  auto tri = dyng::triangle_delta::compute(res, cu);
  expect_refused([&] { (void)dyng::triangle_delta::update(res, g, b.view(), tri); },
                 "triangle_delta::update");
#endif
#if defined(DYNG_ALGORITHM_SSSP) && defined(DYNG_ALGORITHM_DYNAMIC_BFS)
  expect_refused([&] { (void)dyng::update(res, g, b.view(), tree, bfs); }, "dyng::update");
#endif
  EXPECT_EQ(g.version(), version);  // nothing was applied
  EXPECT_NO_THROW(g.check_integrity(res));
#if defined(DYNG_ALGORITHM_SSSP)
  // The results are still usable on their compact graph.
  weighted c2 = compact_weighted(res);
  auto tree2 = dyng::sssp::compute(res, c2, 0);
  EXPECT_NO_THROW((void)dyng::sssp::update(res, c2, b.view(), tree2));
#endif
}

/// The guard comes before Step 0 (ADR 0070, M10a review): an invalid vertex batch (a
/// re-insertion, an edge deletion under net_effect) on a slotted graph is refused with the
/// guard's named error, not with Step 0's invalid_argument_error or its net_effect message, on
/// every backend; nothing is applied.
TEST(CompactGuard, InvalidBatchesAreRefusedByTheGuardFirst) {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  if (!dyng::backend_available(dyng::backend::cuda)) {
    GTEST_SKIP() << "no CUDA backend";
  }
  const std::vector<dyng::resources> backends = {dyng::resources::cuda(0)};
#else
  const std::vector<dyng::resources> backends = {dyng::resources::sequential()};
#endif
  for (const dyng::resources& res : backends) {
    SCOPED_TRACE(dyng::to_string(res.get_backend()));
    const weighted c = compact_weighted(res);
    weighted g = slotted_weighted(res);
    (void)c;
    std::vector<dyng::edge_batch<std::int32_t, std::int32_t>> invalid(3);
    invalid[0].insert_vertex(1);  // vertex 1 was inserted before
    invalid[1].delete_edge(0, 1);  // net_effect() has no edge deletions
    invalid[2].insert_vertex(7);   // out of range
    const std::uint64_t version = g.version();
    for (const auto& b : invalid) {
#if defined(DYNG_ALGORITHM_SSSP)
      auto tree = dyng::sssp::compute(res, c, 0);
      expect_refused([&] { (void)dyng::sssp::update(res, g, b.view(), tree); }, "sssp::update");
#endif
#if defined(DYNG_ALGORITHM_MOSP)
      auto multi = dyng::mosp::compute(res, c, 0);
      expect_refused([&] { (void)dyng::mosp::update(res, g, b.view(), multi); }, "mosp::update");
#endif
#if defined(DYNG_ALGORITHM_CYCLE_COUNT)
      auto cycles = dyng::cycle_count::compute(res, c);
      expect_refused([&] { (void)dyng::cycle_count::update(res, g, b.view(), cycles); },
                     "cycle_count::update");
#endif
#if defined(DYNG_ALGORITHM_DYNAMIC_BFS)
      auto bfs = dyng::dynamic_bfs::compute(res, c);
      expect_refused([&] { (void)dyng::dynamic_bfs::update(res, g, b.view(), bfs); },
                     "dynamic_bfs::update");
#endif
#if defined(DYNG_ALGORITHM_SSSP) && defined(DYNG_ALGORITHM_DYNAMIC_BFS)
      expect_refused([&] { (void)dyng::update(res, g, b.view(), tree, bfs); }, "dyng::update");
#endif
    }
    EXPECT_EQ(g.version(), version);
    EXPECT_NO_THROW(g.check_integrity(res));
  }
}

/// The guard's suggested fix works as written (M10a review): from_csr() of the slotted graph's
/// to_csr() with g.properties(), the layout compact and the semantics upsert_last_wins() keeps
/// every entry in slot order, and the compact copy is accepted.
TEST(CompactGuard, TheSuggestedFixWorksAsWritten) {
  const dyng::resources res = dyng::resources::sequential();
  weighted g = slotted_weighted(res);
  dyng::edge_batch<std::int32_t, std::int32_t> b;
  b.insert_edge(0, 1, {5});  // a parallel edge
  b.insert_edge(2, 2, {1});  // a self-loop
  (void)g.apply(res, b.view());
#if defined(DYNG_ALGORITHM_SSSP)
  try {
    (void)dyng::sssp::compute(res, g, 0);
    ADD_FAILURE() << "the guard accepted a slotted graph";
  } catch (const dyng::not_supported_error& e) {
    EXPECT_NE(std::string(e.what()).find("upsert_last_wins"), std::string::npos) << e.what();
  }
#endif
  dyng::graph_properties props = g.properties();
  props.layout = dyng::row_layout::compact;
  props.semantics = dyng::batch_semantics::upsert_last_wins();
  const auto rows = g.to_csr(res);
  const weighted c = weighted::from_csr(res, rows.view(), props);
  EXPECT_EQ(c.num_edges(), g.num_edges());
  const auto back = c.to_csr(res);
  EXPECT_EQ(back.row_ptr, rows.row_ptr);
  EXPECT_EQ(back.col_ind, rows.col_ind);
  EXPECT_EQ(back.weights, rows.weights);
#if defined(DYNG_ALGORITHM_SSSP)
  // With the consumer's own requirements on top (sssp reads the in-edges).
  props.store_transposed = true;
  const weighted t = weighted::from_csr(res, rows.view(), props);
  EXPECT_NO_THROW((void)dyng::sssp::compute(res, t, 0));
#endif
}

#if defined(DYNG_ALGORITHM_SSSP)
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
#endif

}  // namespace
