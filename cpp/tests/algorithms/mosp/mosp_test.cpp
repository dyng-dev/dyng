// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_test.cpp
 * @brief Hand cases of mosp: the worked example of thesis Chapter 4 (costs (15, 3, 20) for
 *        Pref {4, 1, 4}, (15, 24, 7) for Pref {4, 4, 1}; MOSP-CUDA@e220ee2 mospTest group
 *        thesis-example), K from 1 to 4, the default preferences, options and their checks,
 *        num_objectives, from_arrays, clone, compute_path_costs, the stats, the profiler stages,
 *        stale results, vertex growth, dyng::update() with an sssp result, the distance-only
 *        mode of the combined solve, and the path costs of a graph large enough for the parallel
 *        traversal (openmp, cuda).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/testing/mosp_oracle.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using vertex_t = std::int32_t;
using graph_t = dyng::graph<vertex_t, std::int32_t, std::int32_t>;
using result_t = dyng::mosp::result<vertex_t>;
using batch_t = dyng::edge_batch<vertex_t, std::int32_t>;
constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();

struct edge {
  vertex_t u;
  vertex_t v;
  std::vector<std::int32_t> w;
};

graph_t make_graph(const dyng::resources& res, vertex_t n, int K, const std::vector<edge>& edges,
                   dyng::graph_properties props = dyng::graph_properties::mosp_compatible()) {
  dyng::edge_list<vertex_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = K;
  for (const edge& e : edges) {
    list.src.push_back(e.u);
    list.dst.push_back(e.v);
    list.weights.insert(list.weights.end(), e.w.begin(), e.w.end());
  }
  props.num_weights = K;
  return graph_t::from_edges(res, list.view(), props);
}

std::vector<std::int64_t> costs_of(const result_t& r, vertex_t v) {
  const auto all = dyng::test::host_copy(r.path_costs());
  const auto K = static_cast<std::size_t>(r.num_objectives());
  return {all.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(v) * K),
          all.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(v) + 1) * K)};
}

std::vector<vertex_t> path_to(const result_t& r, vertex_t v) {
  const auto parent = dyng::test::host_copy(r.combined_parents());
  std::vector<vertex_t> path{v};
  while (parent[static_cast<std::size_t>(path.back())] >= 0) {
    path.push_back(parent[static_cast<std::size_t>(path.back())]);
  }
  return path;
}

/// Everything a result holds, on the host.
struct snapshot {
  std::vector<std::vector<std::int64_t>> distances;
  std::vector<std::vector<vertex_t>> parents;
  std::vector<std::int64_t> combined_distances;
  std::vector<vertex_t> combined_parents;
  std::vector<std::int64_t> costs;
  bool operator==(const snapshot& o) const {
    return distances == o.distances && parents == o.parents &&
           combined_distances == o.combined_distances && combined_parents == o.combined_parents &&
           costs == o.costs;
  }
};

snapshot take(const result_t& r) {
  snapshot s;
  for (int k = 0; k < r.num_objectives(); ++k) {
    s.distances.push_back(dyng::test::host_copy(r.distances(k)));
    s.parents.push_back(dyng::test::host_copy(r.parents(k)));
  }
  s.combined_distances = dyng::test::host_copy(r.combined_distances());
  s.combined_parents = dyng::test::host_copy(r.combined_parents());
  s.costs = dyng::test::host_copy(r.path_costs());
  return s;
}

/// The vertices whose combined distance or MOSP parent differ between two snapshots (stats::affected
/// of the update between them; vertices new in `after` count against an unreachable `before`).
std::int64_t changed_vertices(const snapshot& before, const snapshot& after) {
  std::int64_t changed = 0;
  for (std::size_t v = 0; v < after.combined_parents.size(); ++v) {
    const bool old = v < before.combined_parents.size();
    const std::int64_t d = old ? before.combined_distances[v] : inf;
    const vertex_t p = old ? before.combined_parents[v] : vertex_t{-1};
    changed += (d != after.combined_distances[v] || p != after.combined_parents[v]) ? 1 : 0;
  }
  return changed;
}

/// The references of the graph `g` (host copy): Dijkstra per objective, the combined graph and
/// Dijkstra on it, the path costs.
snapshot reference(const dyng::resources& res, const graph_t& g, vertex_t source, int K,
                   const std::vector<std::int32_t>& pref) {
  const auto csr = g.to_csr(res);
  snapshot s;
  for (int k = 0; k < K; ++k) {
    const auto t = dyng::testing::dijkstra(csr.view(), source, k);
    s.distances.push_back(t.distances);
    s.parents.push_back(t.parents);
  }
  const auto combined = dyng::testing::combined_graph_reference(s.parents, source, pref);
  const auto mosp_tree = dyng::testing::dijkstra(combined.view(), source);
  s.combined_distances = mosp_tree.distances;
  s.combined_parents = mosp_tree.parents;
  s.costs = dyng::testing::mosp_path_costs_reference(csr.view(), mosp_tree.parents, source, K);
  return s;
}

class MospBackend : public ::testing::TestWithParam<dyng::backend> {
 protected:
  dyng::resources res_ = dyng::test::make_resources(GetParam(), 4);
};

INSTANTIATE_TEST_SUITE_P(Backends, MospBackend, ::testing::ValuesIn(dyng::test::suite_backends()),
                         dyng::test::backend_name{});
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MospBackend);

/// The graph of thesis Chapter 4, Fig. "Finding a single MOSP" (MOSP's mospTest thesis-example):
/// u1..u7 are 0..6, source u1; the edge u3->u6 of the preliminaries figure is omitted, as there.
graph_t thesis_graph(const dyng::resources& res) {
  std::vector<edge> edges = {{0, 1, {2, 1, 5}}, {0, 2, {4, 1, 1}},  {2, 1, {10, 15, 2}},
                             {1, 3, {2, 4, 2}}, {2, 3, {5, 16, 3}}, {3, 4, {1, 1, 1}},
                             {4, 1, {4, 3, 2}}, {4, 5, {1, 2, 2}},  {4, 6, {5, 6, 2}},
                             {5, 6, {1, 1, 1}}};
  std::stable_sort(edges.begin(), edges.end(),
                   [](const edge& a, const edge& b) { return a.u < b.u; });
  return make_graph(res, 7, 3, edges);
}

/// The thesis' batch: delete (u2,u4), (u5,u2); insert (u4,u6):(10,2,12), (u2,u6):(12,1,14).
batch_t thesis_batch() {
  batch_t b(3);
  b.delete_edge(1, 3);
  b.delete_edge(4, 1);
  b.insert_edge(3, 5, {10, 2, 12});
  b.insert_edge(1, 5, {12, 1, 14});
  return b;
}

TEST_P(MospBackend, TheThesisWorkedExample) {
  struct expectation {
    std::vector<std::int32_t> pref;
    std::vector<vertex_t> path_to_u7;  // u7, its parent, ..., u1
    std::vector<std::int64_t> cost_of_u7;
  };
  const std::vector<expectation> expectations = {
      {{4, 1, 4}, {6, 5, 1, 0}, {15, 3, 20}},     // sub-figures (d), (e)
      {{4, 4, 1}, {6, 4, 3, 2, 0}, {15, 24, 7}},  // sub-figures (f), (g)
  };
  // The updated SOSP trees of sub-figures (a), (b), (c).
  const std::vector<std::vector<vertex_t>> trees = {
      {-1, 0, 0, 2, 3, 4, 5}, {-1, 0, 0, 2, 3, 1, 5}, {-1, 2, 0, 2, 3, 4, 4}};
  for (const expectation& e : expectations) {
    SCOPED_TRACE("Pref {" + std::to_string(e.pref[0]) + "," + std::to_string(e.pref[1]) + "," +
                 std::to_string(e.pref[2]) + "}");
    graph_t g = thesis_graph(res_);
    dyng::mosp::options opt;
    opt.preferences = e.pref;
    result_t r = dyng::mosp::compute(res_, g, 0, opt);
    EXPECT_EQ(r.preference_scale(), 4);
    const dyng::mosp::stats st = dyng::mosp::update(res_, g, thesis_batch().view(), r);
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(dyng::test::host_copy(r.parents(k)), trees[static_cast<std::size_t>(k)])
          << "objective " << k + 1 << " tree differs from the thesis";
    }
    EXPECT_EQ(path_to(r, 6), e.path_to_u7);
    EXPECT_EQ(costs_of(r, 6), e.cost_of_u7);
    EXPECT_EQ(st.preference_scale, 4);
    EXPECT_EQ(st.objectives.size(), 3U);
    EXPECT_TRUE(take(r) == reference(res_, g, 0, 3, e.pref));
    EXPECT_TRUE(take(r) == take(dyng::mosp::compute(res_, g, 0, opt)));
  }
}

TEST_P(MospBackend, KFromOneToFourEqualsTheReferences) {
  for (int K = 1; K <= 4; ++K) {
    SCOPED_TRACE("K " + std::to_string(K));
    std::vector<edge> edges;
    for (vertex_t u = 0; u < 12; ++u) {
      for (vertex_t d : {1, 3, 5}) {
        std::vector<std::int32_t> w;
        for (int k = 0; k < K; ++k) {
          w.push_back(1 + (u * 7 + d * 3 + k * 5) % 4);
        }
        edges.push_back({u, static_cast<vertex_t>((u + d) % 12), w});
      }
    }
    graph_t g = make_graph(res_, 12, K, edges);
    std::vector<std::int32_t> pref;
    for (int k = 0; k < K; ++k) {
      pref.push_back(k + 1);
    }
    for (const auto& p : {std::vector<std::int32_t>{}, pref}) {
      dyng::mosp::options opt;
      opt.preferences = p;
      result_t r = dyng::mosp::compute(res_, g, 3, opt);
      EXPECT_EQ(r.num_objectives(), K);
      EXPECT_TRUE(take(r) == reference(res_, g, 3, K, p));
      batch_t b(K);
      b.delete_edge(3, 4);
      b.delete_edge(9, 0);
      std::vector<std::int32_t> w(static_cast<std::size_t>(K), 2);
      b.insert_edge(5, 11, dyng::host_view(w));
      graph_t h = g.clone(res_);
      (void)dyng::mosp::update(res_, h, b.view(), r);
      EXPECT_TRUE(take(r) == reference(res_, h, 3, K, p));
    }
  }
}

TEST_P(MospBackend, TheDefaultPreferencesWeighAnEdgeByItsTrees) {
  // Two objectives that disagree: 0->1->3 is best for objective 0, 0->2->3 for objective 1.
  graph_t g =
      make_graph(res_, 4, 2, {{0, 1, {1, 5}}, {0, 2, {5, 1}}, {1, 3, {1, 5}}, {2, 3, {5, 1}}});
  result_t r = dyng::mosp::compute(res_, g, 0);
  EXPECT_EQ(r.preference_scale(), 1);
  // (0,1) and (0,2) are in both trees: weight K + 1 - 2 = 1; (1,3) and (2,3) are in one tree
  // each: weight 2. The tie at vertex 3 goes to the lowest parent id.
  EXPECT_EQ(dyng::test::host_copy(r.combined_distances()), (std::vector<std::int64_t>{0, 1, 1, 3}));
  EXPECT_EQ(dyng::test::host_copy(r.combined_parents()), (std::vector<vertex_t>{-1, 0, 0, 1}));
  EXPECT_EQ(costs_of(r, 3), (std::vector<std::int64_t>{2, 10}));
  // Preferring objective 1 (a lower value is a higher priority): Pref {2, 1}, L = 2, base
  // L * (K + 1) = 6; an edge of both trees weighs 6 - 1 - 2 = 3, of T_0 only 5, of T_1 only 4.
  dyng::mosp::options opt;
  opt.preferences = {2, 1};
  result_t q = dyng::mosp::compute(res_, g, 0, opt);
  EXPECT_EQ(q.preference_scale(), 2);
  EXPECT_EQ(dyng::test::host_copy(q.combined_distances()), (std::vector<std::int64_t>{0, 3, 3, 7}));
  EXPECT_EQ(dyng::test::host_copy(q.combined_parents()), (std::vector<vertex_t>{-1, 0, 0, 2}));
  EXPECT_EQ(costs_of(q, 3), (std::vector<std::int64_t>{10, 2}));
}

TEST_P(MospBackend, UnreachableVerticesHaveNoParentAndInfiniteCosts) {
  graph_t g = make_graph(res_, 4, 2, {{0, 1, {1, 1}}, {2, 3, {1, 1}}});
  result_t r = dyng::mosp::compute(res_, g, 0);
  EXPECT_EQ(dyng::test::host_copy(r.combined_distances()),
            (std::vector<std::int64_t>{0, 1, inf, inf}));
  EXPECT_EQ(dyng::test::host_copy(r.combined_parents()), (std::vector<vertex_t>{-1, 0, -1, -1}));
  EXPECT_EQ(costs_of(r, 2), (std::vector<std::int64_t>{inf, inf}));
  EXPECT_EQ(costs_of(r, 0), (std::vector<std::int64_t>{0, 0}));
}

TEST_P(MospBackend, OptionsAreChecked) {
  graph_t g = thesis_graph(res_);
  const auto bad = [&](const dyng::mosp::options& opt) {
    EXPECT_THROW((void)dyng::mosp::compute(res_, g, 0, opt), dyng::invalid_argument_error);
  };
  dyng::mosp::options opt;
  opt.preferences = {1, 2};
  bad(opt);
  opt.preferences = {1, 0, 1};
  bad(opt);
  opt.preferences = {1 << 20, 3, 1};  // lcm 3 * 2^20
  bad(opt);
  opt.preferences = {1 << 20, 1 << 19, 1};  // lcm 2^20: accepted
  EXPECT_NO_THROW((void)dyng::mosp::compute(res_, g, 0, opt));
  opt = {};
  opt.num_objectives = 4;
  bad(opt);
  opt.num_objectives = -1;
  bad(opt);
  opt = {};
  opt.delta = -2;
  bad(opt);
  EXPECT_THROW((void)dyng::mosp::compute(res_, g, 7), dyng::invalid_argument_error);
  // No in-edges stored.
  dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
  props.store_transposed = false;
  graph_t plain = make_graph(res_, 2, 1, {{0, 1, {1}}}, props);
  EXPECT_THROW((void)dyng::mosp::compute(res_, plain, 0), dyng::invalid_argument_error);
  // No weight column: K = 0.
  dyng::edge_list<vertex_t, std::int32_t> empty_list;
  empty_list.num_vertices = 2;
  empty_list.num_weights = 0;
  dyng::graph_properties no_weights = dyng::graph_properties::mosp_compatible();
  no_weights.num_weights = 0;
  graph_t unweighted = graph_t::from_edges(res_, empty_list.view(), no_weights);
  EXPECT_THROW((void)dyng::mosp::compute(res_, unweighted, 0), dyng::invalid_argument_error);
  // set_options: tunables only.
  result_t r = dyng::mosp::compute(res_, g, 0);
  dyng::mosp::options changed = r.get_options();
  changed.preferences = {1, 1, 1};
  EXPECT_THROW(r.set_options(changed), dyng::invalid_argument_error);
  changed = r.get_options();
  changed.num_objectives = 3;
  EXPECT_THROW(r.set_options(changed), dyng::invalid_argument_error);
  changed = r.get_options();
  changed.delta = 7;
  changed.cuda_engine = dyng::engine::operators;
  EXPECT_NO_THROW(r.set_options(changed));
  EXPECT_EQ(r.get_options().delta, 7);
  EXPECT_THROW((void)r.distances(3), dyng::invalid_argument_error);
  EXPECT_THROW((void)r.parents(-1), dyng::invalid_argument_error);
  // The update still gives the same result with the new tunables.
  (void)dyng::mosp::update(res_, g, thesis_batch().view(), r);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, 3, {}));
}

TEST_P(MospBackend, NumObjectivesSelectsTheFirstColumns) {
  graph_t g = thesis_graph(res_);
  dyng::mosp::options opt;
  opt.num_objectives = 2;
  opt.preferences = {3, 1};
  result_t r = dyng::mosp::compute(res_, g, 0, opt);
  EXPECT_EQ(r.num_objectives(), 2);
  EXPECT_EQ(dyng::test::host_copy(r.path_costs()).size(), 14U);
  (void)dyng::mosp::update(res_, g, thesis_batch().view(), r);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, 2, {3, 1}));
}

TEST_P(MospBackend, ABatchWithAnotherWeightCountIsRejected) {
  graph_t g = thesis_graph(res_);
  result_t r = dyng::mosp::compute(res_, g, 0);
  const snapshot before = take(r);
  const std::uint64_t version = g.version();
  batch_t b(2);
  b.insert_edge(0, 6, {1, 1});
  EXPECT_THROW((void)dyng::mosp::update(res_, g, b.view(), r), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), version);
  EXPECT_TRUE(take(r) == before);
  // A batch without insertions carries no weights: accepted with any count.
  batch_t deletions;
  deletions.delete_edge(0, 1);
  const dyng::mosp::stats st = dyng::mosp::update(res_, g, deletions.view(), r);
  EXPECT_EQ(st.batch.deleted_edges, 1);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, 3, {}));
}

TEST_P(MospBackend, FromArraysAdoptsTheTreesAndBuildsTheMospTree) {
  graph_t g = thesis_graph(res_);
  dyng::mosp::options opt;
  opt.preferences = {4, 1, 4};
  const result_t computed = dyng::mosp::compute(res_, g, 0, opt);
  std::vector<std::vector<std::int64_t>> d;
  std::vector<std::vector<vertex_t>> p;
  for (int k = 0; k < 3; ++k) {
    d.push_back(dyng::test::host_copy(computed.distances(k)));
    p.push_back(dyng::test::host_copy(computed.parents(k)));
  }
  std::vector<dyng::array_view<const std::int64_t>> dv;
  std::vector<dyng::array_view<const vertex_t>> pv;
  for (int k = 0; k < 3; ++k) {
    dv.push_back(dyng::host_view(std::as_const(d[static_cast<std::size_t>(k)])));
    pv.push_back(dyng::host_view(std::as_const(p[static_cast<std::size_t>(k)])));
  }
  result_t r = result_t::from_arrays(res_, g, 0, dyng::host_view(std::as_const(dv)),
                                     dyng::host_view(std::as_const(pv)), true, opt);
  EXPECT_TRUE(take(r) == take(computed));
  EXPECT_EQ(r.graph_version(), g.version());
  (void)dyng::mosp::update(res_, g, thesis_batch().view(), r);
  EXPECT_EQ(costs_of(r, 6), (std::vector<std::int64_t>{15, 3, 20}));
  // K must match the options.
  dv.pop_back();
  pv.pop_back();
  EXPECT_THROW((void)result_t::from_arrays(res_, g, 0, dyng::host_view(std::as_const(dv)),
                                           dyng::host_view(std::as_const(pv)), true, opt),
               dyng::invalid_argument_error);
}

TEST_P(MospBackend, ComputePathCostsCanBeSwitchedOff) {
  graph_t g = thesis_graph(res_);
  dyng::mosp::options opt;
  opt.compute_path_costs = false;
  result_t r = dyng::mosp::compute(res_, g, 0, opt);
  EXPECT_THROW((void)r.path_costs(), dyng::invalid_argument_error);
  (void)dyng::mosp::update(res_, g, thesis_batch().view(), r);
  EXPECT_THROW((void)r.path_costs(), dyng::invalid_argument_error);
  opt.compute_path_costs = true;
  r.set_options(opt);
  (void)dyng::mosp::update(res_, g, batch_t(3).view(), r);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, 3, {}));
}

TEST_P(MospBackend, StatsAndProfilerStages) {
  graph_t g = thesis_graph(res_);
  result_t r = dyng::mosp::compute(res_, g, 0);
  const snapshot initial = take(r);
  dyng::profiler prof;
  res_.attach_profiler(&prof);
  const dyng::mosp::stats st = dyng::mosp::update(res_, g, thesis_batch().view(), r);
  res_.attach_profiler(nullptr);
  EXPECT_EQ(st.affected, changed_vertices(initial, take(r)));
  EXPECT_EQ(st.batch.deleted_edges, 2);
  EXPECT_EQ(st.batch.inserted_edges, 2);
  EXPECT_EQ(st.preference_scale, 1);
  EXPECT_GT(st.combined_edges, 0);
  EXPECT_GT(st.affected, 0);
  EXPECT_NE(st.engine_used, dyng::engine::automatic);
  EXPECT_TRUE(st.converged);
  EXPECT_FALSE(st.fallback_used);
  ASSERT_EQ(st.objectives.size(), 3U);
  std::map<std::string, int> calls;
  for (const dyng::stage_sample& s : prof.samples()) {
    ++calls[s.name];
  }
  EXPECT_EQ(calls["mosp.update"], 1);
  EXPECT_EQ(calls["mosp.commit"], 1);
  EXPECT_EQ(calls["mosp.objective"], 3);
  EXPECT_EQ(calls["mosp.combine"], 1);
  EXPECT_EQ(calls["mosp.combined_sssp"], 1);
  EXPECT_EQ(calls["mosp.finalize"], 1);
  EXPECT_EQ(calls["mosp.path_costs"], 1);
  // An empty batch changes nothing.
  const snapshot before = take(r);
  const dyng::mosp::stats none = dyng::mosp::update(res_, g, batch_t(3).view(), r);
  EXPECT_EQ(none.affected, 0);
  EXPECT_TRUE(take(r) == before);
}

TEST_P(MospBackend, StaleResultsAreDetected) {
  graph_t g = thesis_graph(res_);
  result_t r = dyng::mosp::compute(res_, g, 0);
  (void)g.apply(res_, thesis_batch().view());
  EXPECT_THROW((void)dyng::mosp::update(res_, g, batch_t(3).view(), r), dyng::stale_result_error);
  result_t fresh = dyng::mosp::compute(res_, g, 0);
  EXPECT_NO_THROW((void)dyng::mosp::update(res_, g, batch_t(3).view(), fresh));
}

TEST_P(MospBackend, NewVerticesJoinTheTrees) {
  graph_t g = thesis_graph(res_);
  result_t r = dyng::mosp::compute(res_, g, 0);
  batch_t b(3);
  b.insert_edge(6, 7, {1, 2, 3});
  b.insert_edge(8, 2, {1, 1, 1});  // vertex 8 is not reachable
  const snapshot before = take(r);
  const dyng::mosp::stats st = dyng::mosp::update(res_, g, b.view(), r);
  ASSERT_EQ(g.num_vertices(), 9);
  EXPECT_EQ(st.affected, changed_vertices(before, take(r)));
  EXPECT_TRUE(take(r) == reference(res_, g, 0, 3, {}));
  EXPECT_EQ(dyng::test::host_copy(r.combined_parents())[8], -1);
}

TEST_P(MospBackend, CloneCopiesEverything) {
  graph_t g = thesis_graph(res_);
  dyng::mosp::options opt;
  opt.preferences = {4, 4, 1};
  result_t r = dyng::mosp::compute(res_, g, 0, opt);
  result_t c = r.clone(res_);
  EXPECT_TRUE(take(c) == take(r));
  (void)dyng::mosp::update(res_, g, thesis_batch().view(), c);
  EXPECT_EQ(costs_of(c, 6), (std::vector<std::int64_t>{15, 24, 7}));
  EXPECT_THROW((void)dyng::mosp::update(res_, g, batch_t(3).view(), r), dyng::stale_result_error);
  // To the sequential backend and back.
  const dyng::resources seq = dyng::resources::sequential();
  result_t host = c.clone(seq);
  EXPECT_EQ(host.space(), dyng::memory_space::host);
  EXPECT_TRUE(take(host) == take(c));
  result_t back = host.clone(res_);
  EXPECT_TRUE(take(back) == take(c));
  // A moved-from result.
  result_t moved = std::move(back);
  EXPECT_EQ(back.num_objectives(), 0);  // NOLINT(bugprone-use-after-move): the documented state
  EXPECT_THROW((void)back.combined_parents(), dyng::invalid_argument_error);
}

TEST_P(MospBackend, OneUpdateOfMospAndSsspEqualsSeparateUpdates) {
  graph_t g = thesis_graph(res_);
  graph_t alone = g.clone(res_);
  result_t r = dyng::mosp::compute(res_, g, 0);
  dyng::sssp::options so;
  so.objective = 2;
  auto tree = dyng::sssp::compute(res_, g, 0, so);
  result_t r_alone = dyng::mosp::compute(res_, alone, 0);
  auto [ms, ss] = dyng::update(res_, g, thesis_batch().view(), r, tree);
  (void)dyng::mosp::update(res_, alone, thesis_batch().view(), r_alone);
  EXPECT_TRUE(take(r) == take(r_alone));
  EXPECT_EQ(ms.objectives.size(), 3U);
  EXPECT_EQ(ss.batch.deleted_edges, 2);
  EXPECT_EQ(dyng::test::host_copy(tree.parents()), dyng::test::host_copy(r.parents(2)));
  // The same result twice is rejected.
  EXPECT_THROW((void)dyng::update(res_, g, batch_t(3).view(), r, r), dyng::invalid_argument_error);
}

// The combined solve in the distance-only mode of the host engines: L = 2^20 makes the combined
// weights 2^21 - 1, and (n - 1) of them do not fit next to the 22 parent bits of n = 2^21 vertices
// (the sequential engine recovers the parents over the combined graph's in-edges).
TEST_P(MospBackend, PathCostsOfALargeGraphEqualTheReference) {
  // A 170 x 170 grid (28,900 vertices: above the 2^14 vertices from which the openmp and cuda
  // backends compute the path costs with threads) with both edge directions and seeded weights;
  // its breadth-first levels are narrower and wider than 8 nodes per thread (4 threads).
  constexpr vertex_t side = 170;
  constexpr int K = 3;
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto next_weight = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::int32_t>(1 + (state >> 33) % 100);
  };
  std::vector<edge> edges;
  for (vertex_t r = 0; r < side; ++r) {
    for (vertex_t c = 0; c < side; ++c) {
      const vertex_t v = r * side + c;
      if (c + 1 < side) {
        edges.push_back({v, v + 1, {next_weight(), next_weight(), next_weight()}});
        edges.push_back({v + 1, v, {next_weight(), next_weight(), next_weight()}});
      }
      if (r + 1 < side) {
        edges.push_back({v, v + side, {next_weight(), next_weight(), next_weight()}});
        edges.push_back({v + side, v, {next_weight(), next_weight(), next_weight()}});
      }
    }
  }
  std::stable_sort(edges.begin(), edges.end(),
                   [](const edge& a, const edge& b) { return a.u < b.u; });
  graph_t g = make_graph(res_, side * side, K, edges);
  dyng::mosp::options opt;
  opt.preferences = {2, 1, 3};
  result_t r = dyng::mosp::compute(res_, g, 0, opt);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, K, opt.preferences));
  batch_t b(K);
  for (vertex_t i = 0; i < 400; ++i) {
    const vertex_t v = (i * 7919) % (side * side - 1);
    if (v % side + 1 < side) {
      b.delete_edge(v, v + 1);
    }
    b.insert_edge(v, (v * 31 + 17) % (side * side), {next_weight(), next_weight(), next_weight()});
  }
  (void)dyng::mosp::update(res_, g, b.view(), r);
  EXPECT_TRUE(take(r) == reference(res_, g, 0, K, opt.preferences));
}

TEST_P(MospBackend, TheCombinedSolveInTheDistanceOnlyMode) {
  const vertex_t n = vertex_t{1} << 21;
  dyng::edge_list<vertex_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  for (vertex_t v = 1; v < n; ++v) {  // a binary tree, plus a few cross edges with ties
    list.add_edge((v - 1) / 2, v, {1 + v % 3});
    if (v % 1000 == 0) {
      list.add_edge(v - 1, v, {1});
    }
  }
  dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
  props.num_weights = 1;
  graph_t g = graph_t::from_edges(res_, list.view(), props);
  dyng::mosp::options opt;
  opt.preferences = {1 << 20};
  result_t r = dyng::mosp::compute(res_, g, 0, opt);
  EXPECT_EQ(r.preference_scale(), 1 << 20);
  const auto combined = dyng::test::host_copy(r.combined_distances());
  const auto parents = dyng::test::host_copy(r.combined_parents());
  const auto tree = dyng::test::host_copy(r.parents(0));
  // K = 1: the MOSP tree is the objective's tree, every edge weighs 2^21 - 1.
  EXPECT_EQ(parents, tree);
  const std::int64_t w = (std::int64_t{1} << 21) - 1;
  for (vertex_t v : {vertex_t{1}, vertex_t{2}, n - 1}) {
    std::int64_t hops = 0;
    for (vertex_t x = v; x != 0; x = parents[static_cast<std::size_t>(x)]) {
      ++hops;
    }
    EXPECT_EQ(combined[static_cast<std::size_t>(v)], hops * w) << "vertex " << v;
  }
  // An update in the same mode: `affected` (on cuda counted by the solve's distance-only unpack,
  // with either engine) is the number of changed vertices.
  std::vector<dyng::engine> engines{dyng::engine::automatic};
  if (GetParam() == dyng::backend::cuda) {
    engines.push_back(dyng::engine::operators);
  }
  for (const dyng::engine e : engines) {
    graph_t h = graph_t::from_edges(res_, list.view(), props);
    dyng::mosp::options with = opt;
    with.cuda_engine = e;
    result_t s = dyng::mosp::compute(res_, h, 0, with);
    batch_t b(1);
    b.insert_edge(0, n / 2 + 1, {1});  // a shortcut to a deep subtree
    b.delete_edge(0, 1);               // the left half hangs from the shortcut or is cut off
    const snapshot before = take(s);
    const dyng::mosp::stats st = dyng::mosp::update(res_, h, b.view(), s);
    const snapshot after = take(s);
    EXPECT_EQ(st.affected, changed_vertices(before, after));
    EXPECT_GT(st.affected, 0);
    EXPECT_EQ(after.combined_parents, after.parents[0]);
  }
}

}  // namespace
