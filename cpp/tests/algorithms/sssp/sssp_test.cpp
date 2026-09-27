// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_test.cpp
 * @brief Hand cases of sssp: small graphs with known trees, ties, delete-all, empty batches,
 *        vertex growth, the distance-only fallback, stale-result detection, input validation
 *        (validate_inputs), options, the result's value semantics, profiler stages and
 *        dyng::update() over several results.
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/check_sssp.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using result_t = dyng::sssp::result<std::int32_t>;
using batch_t = dyng::edge_batch<std::int32_t, std::int32_t>;
constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();

struct edge {
  std::int32_t u;
  std::int32_t v;
  std::int32_t w;
};

graph_t make_graph(const dyng::resources& res, std::int32_t n, const std::vector<edge>& edges,
                   dyng::graph_properties props = dyng::graph_properties::mosp_compatible()) {
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  for (const edge& e : edges) {
    list.add_edge(e.u, e.v, {e.w});
  }
  return graph_t::from_edges(res, list.view(), props);
}

std::vector<std::int64_t> distances_of(const result_t& r) {
  const auto d = r.distances();
  return {d.begin(), d.end()};
}

std::vector<std::int32_t> parents_of(const result_t& r) {
  const auto p = r.parents();
  return {p.begin(), p.end()};
}

class SsspBackend : public ::testing::TestWithParam<dyng::backend> {
 protected:
  dyng::resources res_ = dyng::test::make_resources(GetParam(), 4);
};

INSTANTIATE_TEST_SUITE_P(HostBackends, SsspBackend,
                         ::testing::ValuesIn(dyng::test::host_backends()),
                         dyng::test::backend_name{});

// The MOSP_ESCHER test_mosp_update "ties" case: 0->1 (5), 0->2 (1), 2->3 (1), 1->4 (2), 3->4 (2).
TEST_P(SsspBackend, ComputeAndUpdateGiveTheCanonicalTree) {
  auto g = make_graph(res_, 5, {{0, 1, 5}, {0, 2, 1}, {2, 3, 1}, {1, 4, 2}, {3, 4, 2}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 5, 1, 2, 4}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, 0, 0, 2, 3}));
  EXPECT_EQ(r.source(), 0);
  EXPECT_EQ(r.graph_version(), g.version());
  EXPECT_EQ(r.space(), dyng::memory_space::host);
  // Inserting 2->1 (1) gives d(1) = 2 and a second tight path to 4 via 1: its parent must become
  // the lower id, 1.
  batch_t b;
  b.insert_edge(2, 1, {1});
  const dyng::sssp::stats st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 2, 1, 2, 4}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, 2, 0, 2, 1}));
  EXPECT_EQ(st.invalidated, 0);
  EXPECT_EQ(st.affected, 2);  // vertex 1 (distance) and vertex 4 (parent)
  EXPECT_EQ(st.batch.inserted_edges, 1);
  EXPECT_TRUE(st.converged);
  EXPECT_FALSE(st.fallback_used);
  EXPECT_TRUE(st.packed_parents);
  EXPECT_EQ(st.engine_used,
            GetParam() == dyng::backend::openmp ? dyng::engine::fused : dyng::engine::operators);
  EXPECT_EQ(r.graph_version(), g.version());
  EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
}

// The MOSP_ESCHER "disconnect" case: the original counted to infinity around 1 <-> 2.
TEST_P(SsspBackend, DeletingATreeEdgeDoesNotCountToInfinity) {
  auto g = make_graph(res_, 4, {{0, 1, 1}, {0, 3, 50}, {1, 2, 1}, {2, 1, 1}, {3, 1, 50}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  b.delete_edge(0, 1);
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 100, 101, 50}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, 3, 1, 0}));
  EXPECT_EQ(st.invalidated, 2);  // 1 and its child 2
  EXPECT_EQ(st.affected, 2);
}

TEST_P(SsspBackend, DeleteAllLeavesOnlyTheSourceReachable) {
  auto g = make_graph(res_, 5, {{0, 1, 3}, {1, 2, 4}, {0, 3, 2}, {3, 4, 7}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  for (const auto& [u, v] : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {0, 3}, {3, 4}}) {
    b.delete_edge(u, v);
  }
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, inf, inf, inf, inf}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, -1, -1, -1, -1}));
  EXPECT_EQ(st.invalidated, 4);
  EXPECT_EQ(st.affected, 4);
  // Re-inserting edges makes them reachable again.
  batch_t again;
  again.insert_edge(0, 4, {9});
  again.insert_edge(4, 2, {1});
  (void)dyng::sssp::update(res_, g, again.view(), r);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, inf, 10, inf, 9}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, -1, 4, -1, 0}));
}

TEST_P(SsspBackend, EmptyBatchChangesNothingButTheVersion) {
  auto g = make_graph(res_, 4, {{0, 1, 2}, {1, 2, 2}, {0, 2, 4}, {2, 3, 1}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  const auto before_d = distances_of(r);
  const auto before_p = parents_of(r);
  const batch_t empty;
  const auto st = dyng::sssp::update(res_, g, empty.view(), r);
  EXPECT_EQ(st.affected, 0);
  EXPECT_EQ(st.invalidated, 0);
  EXPECT_EQ(st.iterations, 0);
  EXPECT_EQ(distances_of(r), before_d);
  EXPECT_EQ(parents_of(r), before_p);
  EXPECT_EQ(g.version(), 1u);
  EXPECT_EQ(r.graph_version(), 1u);
}

TEST_P(SsspBackend, WeightIncreaseOnATreeEdgeInvalidatesTheSubtree) {
  // 0->1 (1), 1->2 (1), 0->2 (5): the tree edge 0->1 becomes 10, so 1 and 2 are invalidated.
  auto g = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}, {0, 2, 5}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  b.insert_edge(0, 1, {10});  // an upsert that raises the weight
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(st.invalidated, 2);
  EXPECT_EQ(st.batch.updated_edges, 1);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 10, 5}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, 0, 0}));
}

TEST_P(SsspBackend, ParallelEdgesAndSelfLoops) {
  // Two parallel edges 0->1 (7 and 3), a self-loop on 1, 1->2 (1).
  auto g = make_graph(res_, 3, {{0, 1, 7}, {0, 1, 3}, {1, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 3, 4}));
  // Deleting (0,1) removes the FIRST parallel edge (weight 7); vertex 1 is a root (conservative)
  // and pulls the remaining edge.
  batch_t b;
  b.delete_edge(0, 1);
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(st.invalidated, 2);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 3, 4}));
  b.clear();
  b.delete_edge(0, 1);
  (void)dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, inf, inf}));
}

TEST_P(SsspBackend, VertexGrowth) {
  auto g = make_graph(res_, 3, {{0, 1, 2}, {1, 2, 2}}, dyng::graph_properties{});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  b.insert_edge(2, 5, {1});  // vertices 3, 4 and 5 appear; 3 and 4 stay unreachable
  b.insert_edge(5, 3, {4});
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(g.num_vertices(), 6);
  EXPECT_EQ(st.batch.num_vertices_after, 6);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 2, 4, 9, inf, 5}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{-1, 0, 1, 5, -1, 2}));
  EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
}

TEST_P(SsspBackend, UndirectedGraph) {
  dyng::graph_properties props;
  props.directed = false;
  auto g = make_graph(res_, 4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}}, props);
  result_t r = dyng::sssp::compute(res_, g, 3);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{3, 2, 1, 0}));
  batch_t b;
  b.delete_edge(1, 2);  // both directions
  b.insert_edge(0, 3, {5});
  (void)dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{5, 6, 1, 0}));
  EXPECT_EQ(parents_of(r), (std::vector<std::int32_t>{3, 0, 3, -1}));
}

TEST_P(SsspBackend, MultiObjectiveGraphUsesTheChosenColumn) {
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 3;
  list.num_weights = 2;
  list.add_edge(0, 1, {1, 10});
  list.add_edge(1, 2, {1, 10});
  list.add_edge(0, 2, {5, 1});
  auto g = graph_t::from_edges(res_, list.view(), dyng::graph_properties::mosp_compatible());
  dyng::sssp::options opt;
  opt.objective = 1;
  result_t r1 = dyng::sssp::compute(res_, g, 0, opt);
  result_t r0 = dyng::sssp::compute(res_, g, 0);
  EXPECT_EQ(distances_of(r0), (std::vector<std::int64_t>{0, 1, 2}));
  EXPECT_EQ(distances_of(r1), (std::vector<std::int64_t>{0, 10, 1}));
  dyng::edge_batch<std::int32_t, std::int32_t> b(2);
  b.insert_edge(0, 2, {5, 30});  // raises objective 1 only
  const auto [s0, s1] = dyng::update(res_, g, b.view(), r0, r1);
  EXPECT_EQ(s0.invalidated, 0);
  EXPECT_EQ(s1.invalidated, 1);
  EXPECT_EQ(distances_of(r1), (std::vector<std::int64_t>{0, 10, 20}));
  EXPECT_EQ(parents_of(r1), (std::vector<std::int32_t>{-1, 0, 1}));
  EXPECT_EQ(r0.graph_version(), g.version());
  EXPECT_EQ(r1.graph_version(), g.version());
}

// Weights of 2 * 10^9 on a 320 x 320 grid (mospTest runLargeWeightTies): (n - 1) * max weight does
// not fit next to the parent ids, so the OpenMP engine keeps distances only and recovers the
// lowest-id parents after the search; most vertices have two tight in-neighbours.
TEST_P(SsspBackend, DistanceOnlyFallbackKeepsCanonicalParents) {
  const std::int32_t side = 320;
  const std::int32_t w = 2000000000;
  std::vector<edge> edges;
  for (std::int32_t y = 0; y < side; ++y) {
    for (std::int32_t x = 0; x < side; ++x) {
      const std::int32_t v = y * side + x;
      if (x + 1 < side) {
        edges.push_back({v, v + 1, w});
        edges.push_back({v + 1, v, w});
      }
      if (y + 1 < side) {
        edges.push_back({v, v + side, w});
        edges.push_back({v + side, v, w});
      }
    }
  }
  auto g = make_graph(res_, side * side, edges);
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  for (std::int32_t v = 1; v < side * side; v += 97) {
    b.delete_edge(v - 1, v);
  }
  b.insert_edge(0, side * side - 1, {w});
  const auto st = dyng::sssp::update(res_, g, b.view(), r);
  EXPECT_EQ(st.packed_parents, GetParam() == dyng::backend::sequential);
  EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
  const result_t fresh = dyng::sssp::compute(res_, g, 0);
  EXPECT_EQ(distances_of(r), distances_of(fresh));
  EXPECT_EQ(parents_of(r), parents_of(fresh));
}

TEST_P(SsspBackend, StaleResultIsDetected) {
  auto g = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t b;
  b.insert_edge(0, 2, {1});
  (void)g.apply(res_, b.view());  // the structure changes, the result does not
  EXPECT_THROW((void)dyng::sssp::update(res_, g, b.view(), r), dyng::stale_result_error);
  EXPECT_THROW((void)dyng::update(res_, g, b.view(), r), dyng::stale_result_error);
  // A fresh result works again.
  r = dyng::sssp::compute(res_, g, 0);
  EXPECT_NO_THROW((void)dyng::sssp::update(res_, g, b.view(), r));
}

TEST_P(SsspBackend, ResultOfAnotherGraphIsStale) {
  // Two different graphs at the same version (0) with the same vertex count.
  auto a = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}});
  auto b = make_graph(res_, 3, {{0, 2, 5}, {2, 1, 5}});
  ASSERT_EQ(a.version(), b.version());
  result_t ra = dyng::sssp::compute(res_, a, 0);
  const batch_t empty;
  EXPECT_THROW((void)dyng::sssp::update(res_, b, empty.view(), ra), dyng::stale_result_error);
  EXPECT_THROW((void)dyng::update(res_, b, empty.view(), ra), dyng::stale_result_error);
  EXPECT_EQ(b.version(), 0u);  // rejected before the batch was applied

  // A graph variable reassigned while a result is kept (the new graph is at version 0 again).
  result_t old = dyng::sssp::compute(res_, a, 0);
  a = make_graph(res_, 3, {{0, 2, 5}, {2, 1, 5}});
  EXPECT_THROW((void)dyng::sssp::update(res_, a, empty.view(), old), dyng::stale_result_error);

  // A clone has the same content, so a result of the original may be updated on the clone; after
  // that, the original and the clone are different states.
  auto g = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  auto copy = g.clone(res_);
  batch_t insert;
  insert.insert_edge(0, 2, {1});
  EXPECT_NO_THROW((void)dyng::sssp::update(res_, copy, insert.view(), r));
  EXPECT_EQ(r.distances()[2], 1);
  batch_t other;
  other.insert_edge(1, 0, {1});
  (void)g.apply(res_, other.view());
  ASSERT_EQ(g.version(), copy.version());  // both at version 1, different content
  EXPECT_THROW((void)dyng::sssp::update(res_, g, empty.view(), r), dyng::stale_result_error);
  // result::clone() keeps the graph it matches.
  result_t rc = r.clone(res_);
  EXPECT_NO_THROW((void)dyng::sssp::update(res_, copy, empty.view(), rc));
}

TEST_P(SsspBackend, ImpossibleGrowthIsAnOutOfMemoryError) {
  // Vertex growth to 2^40 vertices cannot be allocated: sssp::update reports it as
  // out_of_memory_error (a dyng::error), not as std::bad_alloc.
  using graph64 = dyng::graph<std::int64_t, std::int64_t, std::int32_t>;
  dyng::edge_list<std::int64_t, std::int32_t> list;
  list.num_vertices = 2;
  list.num_weights = 1;
  list.add_edge(0, 1, {1});
  const auto g0 = graph64::from_edges(res_, list.view(), dyng::graph_properties::mosp_compatible());
  auto g = g0.clone(res_);
  auto r = dyng::sssp::compute(res_, g, std::int64_t{0});
  dyng::edge_batch<std::int64_t, std::int32_t> far;
  far.insert_edge(1, std::int64_t{1} << 40, {1});
  try {
    (void)dyng::sssp::update(res_, g, far.view(), r);
    FAIL() << "growth to 2^40 vertices did not throw";
  } catch (const dyng::out_of_memory_error& e) {
    EXPECT_NE(std::string(e.what()).find("graph::apply"), std::string::npos) << e.what();
  }
  // Nothing was applied: the failure happened while the new graph was built.
  EXPECT_EQ(g.version(), 0u);
  EXPECT_NO_THROW(
      (void)dyng::sssp::update(res_, g, dyng::edge_batch<std::int64_t, std::int32_t>{}.view(), r));
}

TEST_P(SsspBackend, InvalidBatchChangesNothing) {
  auto g = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res_, g, 0);
  batch_t zero;
  zero.insert_edge(0, 2, {0});  // sssp needs weights >= 1
  EXPECT_THROW((void)dyng::sssp::update(res_, g, zero.view(), r), dyng::invalid_argument_error);
  batch_t negative;
  negative.insert_edge(-1, 2, {3});
  EXPECT_THROW((void)dyng::sssp::update(res_, g, negative.view(), r), dyng::invalid_argument_error);
  dyng::edge_batch<std::int32_t, std::int32_t> two(2);
  two.insert_edge(0, 2, {1, 1});  // K mismatch
  EXPECT_THROW((void)dyng::sssp::update(res_, g, two.view(), r), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0u);
  EXPECT_EQ(r.graph_version(), 0u);
  batch_t ok;
  ok.insert_edge(0, 2, {1});
  EXPECT_NO_THROW((void)dyng::sssp::update(res_, g, ok.view(), r));
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, 1, 1}));
}

TEST_P(SsspBackend, ComputeRejectsInvalidArguments) {
  auto g = make_graph(res_, 3, {{0, 1, 1}, {1, 2, 1}});
  EXPECT_THROW((void)dyng::sssp::compute(res_, g, 3), dyng::invalid_argument_error);
  EXPECT_THROW((void)dyng::sssp::compute(res_, g, -1), dyng::invalid_argument_error);
  dyng::sssp::options opt;
  opt.objective = 1;
  EXPECT_THROW((void)dyng::sssp::compute(res_, g, 0, opt), dyng::invalid_argument_error);
  opt = {};
  opt.delta = -5;
  EXPECT_THROW((void)dyng::sssp::compute(res_, g, 0, opt), dyng::invalid_argument_error);
  auto zero = make_graph(res_, 2, {{0, 1, 0}});
  EXPECT_THROW((void)dyng::sssp::compute(res_, zero, 0), dyng::invalid_argument_error);
  dyng::graph_properties no_in = dyng::graph_properties::mosp_compatible();
  no_in.store_transposed = false;
  auto out_only = make_graph(res_, 2, {{0, 1, 1}}, no_in);
  EXPECT_THROW((void)dyng::sssp::compute(res_, out_only, 0), dyng::invalid_argument_error);
}

TEST_P(SsspBackend, FromArraysValidatesAndCanonicalizes) {
  // 0->1 (1), 0->2 (1), 1->3 (1), 2->3 (1): d(3) = 2 with parents 1 or 2 (canonical: 1).
  auto g = make_graph(res_, 4, {{0, 1, 1}, {0, 2, 1}, {1, 3, 1}, {2, 3, 1}});
  const std::vector<std::int64_t> d{0, 1, 1, 2};
  const std::vector<std::int32_t> p{-1, 0, 0, 2};
  const auto dv = dyng::host_view(d);
  const auto pv = dyng::host_view(p);
  const result_t canonical = result_t::from_arrays(res_, g, 0, dv, pv);
  EXPECT_EQ(parents_of(canonical), (std::vector<std::int32_t>{-1, 0, 0, 1}));
  const result_t kept = result_t::from_arrays(res_, g, 0, dv, pv, /*canonicalize=*/false);
  EXPECT_EQ(parents_of(kept), p);
  EXPECT_EQ(kept.graph_version(), g.version());

  auto reject = [&](std::vector<std::int64_t> dd, std::vector<std::int32_t> pp,
                    std::int32_t source = 0, bool validate = true) {
    dyng::sssp::options opt;
    opt.validate_inputs = validate;
    EXPECT_THROW((void)result_t::from_arrays(res_, g, source, dyng::host_view(dd),
                                             dyng::host_view(pp), false, opt),
                 dyng::invalid_argument_error);
  };
  reject({0, 1, 1}, {-1, 0, 0});                  // wrong size
  reject(d, p, 4);                                // source out of range
  reject({5, 1, 1, 2}, p);                        // source distance != 0
  reject(d, {3, 0, 0, 2});                        // source with a parent
  reject({0, 1, inf, 2}, {-1, 0, 0, 2});          // unreachable vertex with a parent
  reject({0, 1, 1, 2}, {-1, 0, -1, 2});           // reachable vertex without a parent
  reject({0, 1, 1, -2}, {-1, 0, 0, 2});           // negative distance
  reject({0, 1, 1, 1000}, {-1, 0, 0, 2});         // above (n - 1) * max weight
  reject({0, 1, 1, 2}, {-1, 3, 0, 1});            // parent cycle 1 <-> 3
  reject({0, 1, 1, 2}, {-1, 0, 0, 7}, 0, false);  // parent out of range: always checked
  // INF-like values are normalized to infinite_distance().
  const std::vector<std::int64_t> big{0, 1, 1, inf / 2 + 3};
  const std::vector<std::int32_t> none{-1, 0, 0, -1};
  const result_t normalized =
      result_t::from_arrays(res_, g, 0, dyng::host_view(big), dyng::host_view(none), false);
  EXPECT_EQ(distances_of(normalized)[3], inf);
}

TEST(SsspOpenmp, CyclicTreeWithoutValidationIsReportedAndPoisonsTheResult) {
  DYNG_SKIP_IF_NO_OPENMP();
  // mospTest runRegressions "cyclic-tree": 1 <-> 2 below the deleted edge 0 -> 3.
  const auto res = dyng::resources::openmp(2);
  auto g = make_graph(res, 4, {{0, 3, 5}, {1, 2, 1}, {2, 1, 1}, {3, 1, 1}, {3, 2, 1}});
  const std::vector<std::int64_t> d{0, 6, 6, 5};
  const std::vector<std::int32_t> p{-1, 2, 1, 0};
  dyng::sssp::options opt;
  opt.validate_inputs = true;
  EXPECT_THROW(
      (void)result_t::from_arrays(res, g, 0, dyng::host_view(d), dyng::host_view(p), false, opt),
      dyng::invalid_argument_error);
  opt.validate_inputs = false;
  result_t r = result_t::from_arrays(res, g, 0, dyng::host_view(d), dyng::host_view(p), false, opt);
  batch_t b;
  b.delete_edge(0, 3);
  EXPECT_THROW((void)dyng::sssp::update(res, g, b.view(), r), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 1u);  // the batch was applied
  const batch_t empty;
  EXPECT_THROW((void)dyng::sssp::update(res, g, empty.view(), r), dyng::stale_result_error);
}

TEST(SsspResult, OptionsCloneAndMove) {
  const auto res = dyng::resources::sequential();
  auto g = make_graph(res, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res, g, 0);
  dyng::sssp::options opt = r.get_options();
  opt.delta = 17;
  opt.validate_inputs = false;
  r.set_options(opt);
  EXPECT_EQ(r.get_options().delta, 17);
  opt.objective = 1;
  EXPECT_THROW(r.set_options(opt), dyng::invalid_argument_error);
  opt.objective = 0;
  opt.delta = -1;
  EXPECT_THROW(r.set_options(opt), dyng::invalid_argument_error);

  result_t copy = r.clone(res);
  batch_t b;
  b.delete_edge(0, 1);
  (void)dyng::sssp::update(res, g, b.view(), r);
  EXPECT_EQ(distances_of(copy), (std::vector<std::int64_t>{0, 1, 2}));
  EXPECT_EQ(distances_of(r), (std::vector<std::int64_t>{0, inf, inf}));
  EXPECT_EQ(copy.graph_version(), 0u);

  result_t moved = std::move(r);
  EXPECT_EQ(moved.graph_version(), 1u);
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is part of the contract
  EXPECT_EQ(r.source(), -1);
  EXPECT_EQ(r.graph_version(), 0u);
  EXPECT_THROW((void)r.distances(), dyng::invalid_argument_error);
  EXPECT_THROW((void)r.parents(), dyng::invalid_argument_error);
  EXPECT_THROW((void)r.get_options(), dyng::invalid_argument_error);
  EXPECT_THROW((void)dyng::sssp::update(res, g, b.view(), r), dyng::invalid_argument_error);
}

TEST(SsspResult, DeviceBackendIsNotSupportedYet) {
  const auto res = dyng::resources::sequential();
  auto g = make_graph(res, 2, {{0, 1, 1}});
  EXPECT_THROW((void)dyng::resources::cuda(), dyng::error);
}

TEST(SsspProfiler, StagesFollowTheHookNames) {
  auto res = dyng::resources::sequential();
  dyng::profiler prof;
  res.attach_profiler(&prof);
  auto g = make_graph(res, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t r = dyng::sssp::compute(res, g, 0);
  batch_t b;
  b.delete_edge(1, 2);
  (void)dyng::sssp::update(res, g, b.view(), r);
  res.attach_profiler(nullptr);
  std::vector<std::string> names;
  for (const auto& s : prof.stages()) {
    names.push_back(s.name);
  }
  for (const char* expected :
       {"sssp.compute", "sssp.reset", "sssp.seed", "sssp.loop", "sssp.finalize", "sssp.update",
        "sssp.prepare", "sssp.commit", "graph.apply", "sssp.identify_affected"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end()) << expected;
  }
}

TEST(SsspComposition, RejectsDuplicatesAndStaleResultsBeforeApplying) {
  const auto res = dyng::resources::sequential();
  auto g = make_graph(res, 3, {{0, 1, 1}, {1, 2, 1}});
  result_t a = dyng::sssp::compute(res, g, 0);
  result_t c = dyng::sssp::compute(res, g, 2);
  batch_t b;
  b.insert_edge(2, 0, {1});
  EXPECT_THROW((void)dyng::update(res, g, b.view(), a, a), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0u);
  result_t old = a.clone(res);
  (void)dyng::update(res, g, b.view(), a, c);
  EXPECT_EQ(g.version(), 1u);
  EXPECT_EQ(distances_of(c), (std::vector<std::int64_t>{1, 2, 0}));
  EXPECT_THROW((void)dyng::update(res, g, b.view(), a, old), dyng::stale_result_error);
  EXPECT_EQ(g.version(), 1u);  // nothing applied
  EXPECT_EQ(a.graph_version(), 1u);
}

// The quickstart of README.md.
TEST_P(SsspBackend, ReadmeQuickstart) {
  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 1;
  edges.add_edge(0, 1, {4});
  edges.add_edge(0, 2, {1});
  edges.add_edge(2, 1, {2});
  edges.add_edge(1, 3, {1});
  auto g = dyng::graph<std::int32_t, std::int64_t, std::int32_t>::from_edges(res_, edges.view());
  auto tree = dyng::sssp::compute(res_, g, 0);
  dyng::edge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_edge(2, 1);
  batch.insert_edge(2, 3, {1});
  const dyng::sssp::stats st = dyng::sssp::update(res_, g, batch.view(), tree);
  EXPECT_EQ(distances_of(tree), (std::vector<std::int64_t>{0, 4, 1, 2}));
  EXPECT_EQ(parents_of(tree), (std::vector<std::int32_t>{-1, 0, 0, 2}));
  EXPECT_EQ(st.invalidated, 2);
}

}  // namespace
