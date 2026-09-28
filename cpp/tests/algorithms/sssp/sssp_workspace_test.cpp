// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_workspace_test.cpp
 * @brief Results share the scratch memory of their resources handle (ADR 0015): one workspace for
 *        the K objectives, no scratch allocation in a steady-state update (the OpenMP engine's
 *        per-thread lists excepted: they grow to the largest share of a round a thread has
 *        taken, which the dynamic schedule decides), results that stay
 *        independent, a failed update that discards its workspace, concurrent computes on copies
 *        of one handle; and the parallel validation of imported trees.
 */
#include "algorithms/sssp/problem.hpp"
#include "framework/workspace.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using result_t = dyng::sssp::result<std::int32_t>;
using batch_t = dyng::edge_batch<std::int32_t, std::int32_t>;
using dyng::detail::resources_access;

constexpr int num_objectives = 3;

/// A random multi-weight graph (parallel edges and self-loops allowed, as MOSP's).
graph_t random_graph(const dyng::resources& res, std::int32_t n, int m, std::uint32_t seed) {
  std::mt19937 rng(seed);
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = num_objectives;
  for (std::int32_t v = 1; v < n; ++v) {  // a spanning path keeps most vertices reachable
    list.add_edge(v - 1, v, {1 + static_cast<std::int32_t>(rng() % 20), 5, 1});
  }
  for (int i = 0; i < m; ++i) {
    list.add_edge(
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        {1 + static_cast<std::int32_t>(rng() % 20), 1 + static_cast<std::int32_t>(rng() % 3),
         1 + static_cast<std::int32_t>(rng() % 1000)});
  }
  return graph_t::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
}

/// Deletes the out-edges of `count` random vertices' first neighbours and inserts as many edges.
batch_t random_batch(const graph_t& g, int count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const auto out = g.view().out;
  const std::int32_t n = g.num_vertices();
  batch_t b(num_objectives);
  for (int i = 0; i < count; ++i) {
    const auto u = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n));
    if (out.row_ptr[static_cast<std::size_t>(u) + 1] > out.row_ptr[static_cast<std::size_t>(u)]) {
      b.delete_edge(u, out.col_ind[static_cast<std::size_t>(out.row_ptr[u])]);
    }
    b.insert_edge(
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        {1 + static_cast<std::int32_t>(rng() % 20), 1 + static_cast<std::int32_t>(rng() % 3),
         1 + static_cast<std::int32_t>(rng() % 1000)});
  }
  return b;
}

std::vector<std::int64_t> distances_of(const result_t& r) {
  const auto d = r.distances();
  return {d.begin(), d.end()};
}

std::vector<std::int32_t> parents_of(const result_t& r) {
  const auto p = r.parents();
  return {p.begin(), p.end()};
}

/// The update postcondition: equal to compute() on the current graph (on a fresh handle).
void expect_equals_compute(const dyng::resources& res, const graph_t& g, const result_t& r,
                           int objective) {
  dyng::sssp::options opt;
  opt.objective = objective;
  const result_t fresh = dyng::sssp::compute(res, g, r.source(), opt);
  EXPECT_EQ(distances_of(r), distances_of(fresh)) << "objective " << objective;
  EXPECT_EQ(parents_of(r), parents_of(fresh)) << "objective " << objective;
}

class SsspWorkspace : public ::testing::TestWithParam<dyng::backend> {
 protected:
  dyng::resources res_ = dyng::test::make_resources(GetParam(), 4);
};

INSTANTIATE_TEST_SUITE_P(HostBackends, SsspWorkspace,
                         ::testing::ValuesIn(dyng::test::host_backends()),
                         dyng::test::backend_name{});

/// The pooled workspace's bytes, split into the part that depends only on the graph and the
/// batches and the part held by the OpenMP engine's per-thread lists, whose capacity follows the
/// largest share of a round each thread has taken (the dynamic schedule decides the shares).
struct workspace_split {
  std::size_t fixed = 0;
  std::size_t thread_lists = 0;
};

workspace_split split_of(const dyng::resources& res) {
  auto lease =
      resources_access::workspaces(res).acquire<dyng::detail::sssp_workspace<std::int32_t>>();
  return {lease->bytes() - lease->thread_list_bytes(), lease->thread_list_bytes()};
}

TEST_P(SsspWorkspace, TheObjectivesShareOneWorkspaceAndASteadyStateAllocatesNone) {
  auto g = random_graph(res_, 3000, 9000, 11);
  const auto& pool = resources_access::workspaces(res_);
  std::vector<result_t> results;
  for (int k = 0; k < num_objectives; ++k) {
    dyng::sssp::options opt;
    opt.objective = k;
    results.push_back(dyng::sssp::compute(res_, g, 0, opt));
  }
  EXPECT_EQ(pool.statistics().created, 1u);  // MOSP's one SospWorkspace for the K objectives
  std::vector<result_t*> pointers;
  for (result_t& r : results) {
    pointers.push_back(&r);
  }
  const auto list = dyng::array_view<result_t* const>(pointers.data(), pointers.size());
  // A stable workload: the same batch deleted and re-inserted, twice.
  const batch_t forth = random_batch(g, 200, 5);
  batch_t back(num_objectives);
  for (std::size_t i = 0; i < forth.num_deletions(); ++i) {
    back.insert_edge(forth.view().delete_src[i], forth.view().delete_dst[i], {3, 2, 7});
  }
  workspace_split warm;
  std::size_t lists_before = 0;
  for (int round = 0; round < 5; ++round) {
    (void)dyng::update_each(res_, g, forth.view(), list);
    (void)dyng::update_each(res_, g, back.view(), list);
    const workspace_split now = split_of(res_);
    EXPECT_EQ(pool.statistics().idle_bytes, now.fixed + now.thread_lists) << "round " << round;
    // The per-thread lists are kept, never shrunk (a growth is amortized, as a vector's).
    EXPECT_GE(now.thread_lists, lists_before) << "round " << round;
    lists_before = now.thread_lists;
    if (round < 2) {
      // The warm-up: the first rounds size the change lists (the first `forth` inserts new
      // edges, later ones update them, so the set of weight increases settles after round 1).
      warm = now;
      continue;
    }
    EXPECT_EQ(now.fixed, warm.fixed) << "round " << round;
    EXPECT_EQ(pool.statistics().created, 1u) << "round " << round;
  }
  EXPECT_GT(warm.fixed, 0u);
  if (GetParam() == dyng::backend::sequential) {
    // No per-thread lists: the whole workspace is steady.
    EXPECT_EQ(lists_before, 0u);
    EXPECT_EQ(res_.workspace_bytes(), warm.fixed);
  }
  EXPECT_EQ(res_.workspace_bytes(), pool.statistics().idle_bytes);
  for (int k = 0; k < num_objectives; ++k) {
    expect_equals_compute(dyng::test::make_resources(GetParam(), 4), g,
                          results[static_cast<std::size_t>(k)], k);
  }
}

TEST_P(SsspWorkspace, FromArraysAndCloneSizeTheSharedWorkspaceOnce) {
  const auto g = random_graph(res_, 500, 2000, 12);
  const result_t base = dyng::sssp::compute(res_, g, 0);
  const auto fresh = dyng::test::make_resources(GetParam(), 4);
  std::vector<result_t> imported;
  for (int copy = 0; copy < 4; ++copy) {
    imported.push_back(result_t::from_arrays(fresh, g, 0, base.distances(), base.parents()));
  }
  imported.push_back(base.clone(fresh));
  const auto s = resources_access::workspaces(fresh).statistics();
  EXPECT_EQ(s.created, 1u);
  EXPECT_EQ(s.idle, 1u);
  EXPECT_GT(s.idle_bytes, 0u);
}

TEST_P(SsspWorkspace, ResultsOfDifferentGraphsOnOneHandleStayIndependent) {
  // Two graphs of different sizes, their results updated in turns through one handle (the shared
  // workspace is resized for the larger one and reused by both), and one result moved to another
  // handle halfway: every result equals compute() on its own graph after every batch.
  auto small = random_graph(res_, 400, 1500, 21);
  auto large = random_graph(res_, 2500, 8000, 22);
  result_t a = dyng::sssp::compute(res_, small, 0);
  dyng::sssp::options opt;
  opt.objective = 2;
  result_t b = dyng::sssp::compute(res_, large, 7, opt);
  const auto other = dyng::test::make_resources(GetParam(), 2);
  for (int round = 0; round < 4; ++round) {
    const batch_t bs = random_batch(small, 30, 100 + static_cast<std::uint32_t>(round));
    const batch_t bl = random_batch(large, 60, 200 + static_cast<std::uint32_t>(round));
    const dyng::resources& for_a = round < 2 ? res_ : other;
    (void)dyng::sssp::update(for_a, small, bs.view(), a);
    (void)dyng::sssp::update(res_, large, bl.view(), b);
    expect_equals_compute(other, small, a, 0);
    expect_equals_compute(other, large, b, 2);
  }
  EXPECT_EQ(resources_access::workspaces(res_).statistics().created, 1u);
}

TEST_P(SsspWorkspace, AFailedUpdateDiscardsItsWorkspace) {
  // A corrupt imported tree (parent cycle 1 <-> 2) makes the engine throw in the middle of a run;
  // its workspace is dropped, and the next update through the same handle is exact.
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 4;
  list.num_weights = 1;
  for (const auto& [u, v] : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 1}, {2, 3}}) {
    list.add_edge(u, v, {1});
  }
  auto g = graph_t::from_edges(res_, list.view(), dyng::graph_properties::mosp_compatible());
  const std::vector<std::int64_t> d{0, 1, 2, 3};
  const std::vector<std::int32_t> p{-1, 2, 1, 2};
  dyng::sssp::options opt;
  opt.validate_inputs = false;
  result_t bad =
      result_t::from_arrays(res_, g, 0, dyng::host_view(d), dyng::host_view(p), false, opt);
  batch_t b;
  b.delete_edge(0, 1);
  EXPECT_THROW((void)dyng::sssp::update(res_, g, b.view(), bad), dyng::invalid_argument_error);
  const auto s = resources_access::workspaces(res_).statistics();
  EXPECT_EQ(s.discarded, 1u);
  EXPECT_EQ(s.leased, 0u);

  auto h = random_graph(res_, 800, 3000, 31);
  result_t r = dyng::sssp::compute(res_, h, 0);
  const batch_t hb = random_batch(h, 50, 32);
  (void)dyng::sssp::update(res_, h, hb.view(), r);
  expect_equals_compute(dyng::test::make_resources(GetParam(), 2), h, r, 0);
}

TEST_P(SsspWorkspace, ConcurrentComputesOnCopiesOfOneHandleLeaseDistinctWorkspaces) {
  const auto g = random_graph(res_, 1500, 5000, 41);
  const result_t expected = dyng::sssp::compute(dyng::test::make_resources(GetParam(), 2), g, 3);
  constexpr int threads = 4;
  std::vector<std::vector<std::int64_t>> distances(threads);
  std::vector<std::vector<std::int32_t>> parents(threads);
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t, copy = res_] {  // a copy shares the handle and its pool
      for (int i = 0; i < 5; ++i) {
        const result_t r = dyng::sssp::compute(copy, g, 3);
        distances[static_cast<std::size_t>(t)] = distances_of(r);
        parents[static_cast<std::size_t>(t)] = parents_of(r);
      }
    });
  }
  for (std::thread& w : workers) {
    w.join();
  }
  for (int t = 0; t < threads; ++t) {
    EXPECT_EQ(distances[static_cast<std::size_t>(t)], distances_of(expected));
    EXPECT_EQ(parents[static_cast<std::size_t>(t)], parents_of(expected));
  }
  const auto s = resources_access::workspaces(res_).statistics();
  EXPECT_LE(s.created, static_cast<std::uint64_t>(threads + 1));
  EXPECT_EQ(s.leased, 0u);
}

TEST(SsspValidate, TheParallelChecksReportLikeTheSequentialOnes) {
  DYNG_SKIP_IF_NO_OPENMP();
  // A path 0 -> 1 -> ... -> n-1 (n above the parallel threshold) and trees with one defect each:
  // the OpenMP backend must accept the valid tree and reject each defect with the message of the
  // sequential backend.
  const std::int32_t n = 20000;
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  for (std::int32_t v = 1; v < n; ++v) {
    list.add_edge(v - 1, v, {1});
  }
  const auto seq = dyng::resources::sequential();
  const auto omp = dyng::resources::openmp(8);
  const auto g = graph_t::from_edges(seq, list.view(), dyng::graph_properties::mosp_compatible());
  std::vector<std::int64_t> d(static_cast<std::size_t>(n));
  std::vector<std::int32_t> p(static_cast<std::size_t>(n));
  for (std::int32_t v = 0; v < n; ++v) {
    d[static_cast<std::size_t>(v)] = v;
    p[static_cast<std::size_t>(v)] = v - 1;
  }
  const auto message = [&](const dyng::resources& res, const std::vector<std::int64_t>& dist,
                           const std::vector<std::int32_t>& par) -> std::string {
    try {
      (void)result_t::from_arrays(res, g, 0, dyng::host_view(dist), dyng::host_view(par), false);
    } catch (const dyng::invalid_argument_error& e) {
      return e.what();
    }
    return "accepted";
  };
  EXPECT_EQ(message(omp, d, p), "accepted");
  const std::int64_t inf = dyng::infinite_distance<std::int64_t>();
  struct defect {
    const char* name;
    std::int32_t vertex;
    std::int64_t distance;
    std::int32_t parent;
  };
  const defect defects[] = {
      {"parent cycle", 15000, 15000, 15500},  // 15000 -> 15500 -> ... -> 15000
      {"unreachable with a parent", 19000, inf, 18999},
      {"reachable without a parent", 17000, 17000, -1},
      {"unreachable parent", 12001, 12001, 19999},
      {"distance out of range", 9000, -5, 8999},
      {"source with a parent", 0, 0, 3},
  };
  for (const defect& x : defects) {
    SCOPED_TRACE(x.name);
    auto dist = d;
    auto par = p;
    dist[static_cast<std::size_t>(x.vertex)] = x.distance;
    par[static_cast<std::size_t>(x.vertex)] = x.parent;
    if (std::string(x.name) == "unreachable parent") {
      dist[19999] = inf;
      par[19999] = -1;
    }
    const std::string sequential = message(seq, dist, par);
    EXPECT_NE(sequential, "accepted");
    EXPECT_EQ(message(omp, dist, par), sequential);
  }
}

}  // namespace
