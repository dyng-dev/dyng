// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_cuda_test.cpp
 * @brief The CUDA backend of cycle_count (label gpu; skipped without a device): the length bound of
 *        the device searches (2..64, rejections beyond), the engine option, every scheduler and
 *        kind of work item, cross-backend equality (cuda = openmp = sequential) on randomized
 *        graphs and chains of batches, the resident graph (the device set apply equals the host
 *        apply byte for byte, the host copy is downloaded only when read, the insertion ids), the
 *        host-commit fallbacks (weight columns, other batch semantics) and the corner cases of
 *        batches (empty, deletions of non-edges, insertions of existing edges, new vertex ids).
 *
 * The shared suites (cycle_count_test.cpp, cycle_count_random_test.cpp, cycle_count_fixture_test.cpp)
 * run on cuda in the same executable (DYNG_TEST_CUDA=1).
 */
#include "algorithms/cycle_count/problem.hpp"
#include "graph/graph_impl.hpp"
#include "graph/normalized_batch.hpp"
#include "support/cycle_count_support.hpp"
#include "support/test_seeds.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/update.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::backend;
using dyng::resources;
using dyng::unweighted;
using dyng::test::cc_batch;
using dyng::test::cc_counts;
using dyng::test::cc_edge;
using dyng::test::cc_graph;
using dyng::test::seed_trace;
using dyng::test::test_seeds;
namespace cycle_count = dyng::cycle_count;
namespace legacy = dyng::generators::legacy;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_u32 = dyng::graph<std::int32_t, std::int32_t, unweighted>;
using graph_w = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using hist = std::vector<std::uint64_t>;
using batch_u = dyng::edge_batch<std::int32_t, unweighted>;

cycle_count::options bound(int k) {
  cycle_count::options opt;
  opt.max_length = k;
  return opt;
}

/// A ring 0 -> 1 -> ... -> n-1 -> 0 plus a dense block on the first `block` vertices (cycles of
/// every length from 2 up to `block`, and the ring of length n).
std::vector<cc_edge> ring_with_block(std::int32_t n, std::int32_t block) {
  std::set<cc_edge> edges;
  for (std::int32_t v = 0; v < n; ++v) {
    edges.emplace(v, (v + 1) % n);
  }
  for (std::int32_t u = 0; u < block; ++u) {
    for (std::int32_t v = 0; v < block; ++v) {
      if (u != v && (u * 7 + v * 3) % 4 != 0) {
        edges.emplace(u, v);
      }
    }
  }
  return {edges.begin(), edges.end()};
}

class CycleCountCuda : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!dyng::backend_available(backend::cuda)) {
      GTEST_SKIP() << "no CUDA device or CUDA not built";
    }
    cuda_ = resources::cuda();
  }

  resources cuda_ = resources::sequential();
  resources seq_ = resources::sequential();
};

// ---- the length bound: 2..64 (kMaxDeviceCycleLength), every capacity of the dispatch -----------

TEST_F(CycleCountCuda, EveryBoundUpToSixtyFourMatchesTheSequentialCount) {
  // 64 vertices: a ring of length 64 and cycles of lengths 2..9 in the block.
  const std::vector<cc_edge> edges = ring_with_block(64, 9);
  const graph_u g = cc_graph<graph_u>(cuda_, 64, edges);
  const graph_u32 g32 = cc_graph<graph_u32>(cuda_, 64, edges);
  const graph_u gs = cc_graph<graph_u>(seq_, 64, edges);
  for (int k = 2; k <= 64; ++k) {
    SCOPED_TRACE(::testing::Message() << "k=" << k);
    const hist expected = cc_counts(cycle_count::compute(seq_, gs, bound(k)));
    EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g, bound(k))), expected);
    EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g32, bound(k))), expected);
  }
  // Without a bound the effective bound is n = 64: accepted.
  const cycle_count::result r = cycle_count::compute(cuda_, g);
  EXPECT_GT(r.count(64), 0U);  // the ring, and the rings through chords of the block
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(seq_, gs)));
  // A bound above 64 is the vertex count here, as the original's effective_length.
  EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g, bound(1000))),
            dyng::test::cc_resize(cc_counts(cycle_count::compute(seq_, gs, bound(1000))), 1001));
}

TEST_F(CycleCountCuda, BoundsBeyondSixtyFourAreRejectedWithAClearError) {
  const std::vector<cc_edge> edges = ring_with_block(65, 6);
  const graph_u g = cc_graph<graph_u>(cuda_, 65, edges);
  for (const int k : {65, 100, -1}) {
    SCOPED_TRACE(::testing::Message() << "k=" << k);
    try {
      (void)cycle_count::compute(cuda_, g, bound(k));
      FAIL() << "a bound of 65 was accepted";
    } catch (const dyng::invalid_argument_error& e) {
      EXPECT_NE(std::string(e.what()).find("64"), std::string::npos) << e.what();
    }
  }
  // k = 64 on 65 vertices counts every cycle up to length 64 (the ring of 65 is not one of them).
  const graph_u gs = cc_graph<graph_u>(seq_, 65, edges);
  const cycle_count::result r = cycle_count::compute(cuda_, g, bound(64));
  EXPECT_EQ(r.count(65), 0U);
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(seq_, gs, bound(64))));
}

TEST_F(CycleCountCuda, UpdatesAtEveryCapacityBoundaryMatchTheSequentialUpdate) {
  const std::vector<cc_edge> edges = ring_with_block(40, 10);
  for (const int k : {2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 40, 64}) {
    SCOPED_TRACE(::testing::Message() << "k=" << k);
    graph_u g = cc_graph<graph_u>(cuda_, 40, edges);
    graph_u gs = cc_graph<graph_u>(seq_, 40, edges);
    cycle_count::result r = cycle_count::compute(cuda_, g, bound(k));
    cycle_count::result rs = cycle_count::compute(seq_, gs, bound(k));
    // Cut the ring, close a shorter one, add chords and a vertex.
    const batch_u b = cc_batch<unweighted>({{20, 21}, {3, 1}, {0, 5}},
                                           {{20, 30}, {35, 3}, {1, 3}, {39, 40}, {40, 0}});
    const cycle_count::stats st = cycle_count::update(cuda_, g, b.view(), r);
    const cycle_count::stats ss = cycle_count::update(seq_, gs, b.view(), rs);
    EXPECT_EQ(cc_counts(r), cc_counts(rs));
    EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(seq_, gs, bound(k))));
    EXPECT_EQ(st.cycles_added, ss.cycles_added);
    EXPECT_EQ(st.cycles_removed, ss.cycles_removed);
    EXPECT_EQ(st.affected, ss.affected);
    EXPECT_EQ(st.deletions, ss.deletions);
    EXPECT_EQ(st.insertions, ss.insertions);
    EXPECT_EQ(st.engine_used, dyng::engine::fused);
    EXPECT_EQ(ss.engine_used, dyng::engine::operators);
  }
}

// ---- the engine option -------------------------------------------------------------------------

TEST_F(CycleCountCuda, OperatorsEngineIsNotSupported) {
  graph_u g = cc_graph<graph_u>(cuda_, 5, ring_with_block(5, 4));
  cycle_count::options opt = bound(4);
  opt.cuda_engine = dyng::engine::operators;
  EXPECT_THROW((void)cycle_count::compute(cuda_, g, opt), dyng::not_supported_error);
  // The option is ignored on the host backends.
  const graph_u gs = cc_graph<graph_u>(seq_, 5, ring_with_block(5, 4));
  const hist expected = cc_counts(cycle_count::compute(seq_, gs, opt));
  for (const dyng::engine e : {dyng::engine::automatic, dyng::engine::fused}) {
    opt.cuda_engine = e;
    EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g, opt)), expected);
  }
  // A result computed with the fused engine is updated with the fused kernels.
  opt.cuda_engine = dyng::engine::fused;
  cycle_count::result r = cycle_count::compute(cuda_, g, opt);
  const batch_u b = cc_batch<unweighted>({{0, 1}}, {{2, 0}});
  EXPECT_EQ(cycle_count::update(cuda_, g, b.view(), r).engine_used, dyng::engine::fused);
  // An update of a result whose options ask for operators (a tunable: set_options()) is rejected
  // before anything changes; switching back makes the result usable again.
  cycle_count::result bad = r.clone(cuda_);
  cycle_count::options operators = bad.get_options();
  operators.cuda_engine = dyng::engine::operators;
  bad.set_options(operators);
  const std::uint64_t version = g.version();
  EXPECT_THROW((void)cycle_count::update(cuda_, g, b.view(), bad), dyng::not_supported_error);
  EXPECT_EQ(g.version(), version);
  operators.cuda_engine = dyng::engine::automatic;
  bad.set_options(operators);
  (void)cycle_count::update(cuda_, g, b.view(), bad);
  EXPECT_EQ(cc_counts(bad), cc_counts(cycle_count::compute(cuda_, g, opt)));
}

// ---- the length bound of an edgeless graph, and the copy policy ------------------------------

TEST_F(CycleCountCuda, EdgelessGraphsCountBeforeTheLengthCheck) {
  // 100 isolated vertices without a bound: the original returns the empty histogram before its
  // length check, so compute() succeeds; the first batch that adds edges needs the bound.
  graph_u g = cc_graph<graph_u>(cuda_, 100, {});
  cycle_count::result r = cycle_count::compute(cuda_, g, cycle_count::options{});
  const graph_u gs = cc_graph<graph_u>(seq_, 100, {});
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(seq_, gs, cycle_count::options{})));
  const batch_u b = cc_batch<unweighted>({}, {{0, 1}, {1, 0}});
  EXPECT_THROW((void)cycle_count::update(cuda_, g, b.view(), r), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0U);  // nothing was applied
  EXPECT_EQ(r.total(), 0U);    // and the result is still usable
  // With a bound the same graph updates.
  cycle_count::result bounded = cycle_count::compute(cuda_, g, bound(4));
  (void)cycle_count::update(cuda_, g, b.view(), bounded);
  EXPECT_EQ(bounded.count(2), 1U);
}

// copy_policy::error turns the implicit copy of a device batch into invalid_argument_error before
// anything changes, and the message names the function the user called.
TEST_F(CycleCountCuda, TheCopyPolicyErrorNamesCycleCountUpdate) {
  graph_u g = cc_graph<graph_u>(cuda_, 5, ring_with_block(5, 4));
  cycle_count::result r = cycle_count::compute(cuda_, g, bound(4));
  const batch_u b = cc_batch<unweighted>({{0, 1}}, {{2, 0}});
  const auto host = b.view();
  const auto device_src = dyng::to_space(cuda_, host.insert_src, dyng::memory_space::device);
  cuda_.synchronize();
  auto view = host;
  view.insert_src = device_src.view();
  resources res = resources::cuda();  // an independent handle for the policy
  res.set_copy_policy(dyng::copy_policy::error);
  try {
    (void)cycle_count::update(res, g, view, r);
    FAIL() << "copy_policy::error allowed an implicit copy";
  } catch (const dyng::invalid_argument_error& error) {
    const std::string what = error.what();
    EXPECT_NE(what.find("cycle_count::update"), std::string::npos) << what;
    EXPECT_EQ(what.find("dyng::update"), std::string::npos) << what;
    EXPECT_NE(what.find("insert_src"), std::string::npos) << what;
  }
  EXPECT_EQ(g.version(), 0U);
  (void)cycle_count::update(cuda_, g, view, r);  // allowed under the default policy
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(cuda_, g, bound(4))));
}

// ---- every scheduler and kind of work item ----------------------------------------------------

TEST_F(CycleCountCuda, EverySchedulerAndWorkItemAgree) {
  using items = cycle_count::cuda_work_items;
  for (const std::uint64_t seed : test_seeds(8128, 25)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    // Sparse to medium graphs with and without a hub (hubs make the root trees skewed, which the
    // finer work items split), small enough for the sequential count.
    dyng::test::cc_spec spec;
    spec.vertex_count = 10 + static_cast<std::int64_t>(rng() % 31);
    spec.edge_probability = 0.04 + 0.03 * static_cast<double>(rng() % 5);
    spec.hubs = static_cast<std::int64_t>(rng() % 2);
    spec.self_loop_probability = 0.05;
    const std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
    const graph_u g = cc_graph<graph_u>(cuda_, spec.vertex_count, edges);
    const graph_u32 g32 = cc_graph<graph_u32>(cuda_, spec.vertex_count, edges);
    const graph_u gs = cc_graph<graph_u>(seq_, spec.vertex_count, edges);
    for (const int k : {2, 3, 4, 5, 6}) {
      const hist expected = cc_counts(cycle_count::compute(seq_, gs, bound(k)));
      for (const auto& [scheduler, work] :
           {std::pair{cycle_count::cuda_scheduler::naive, items::automatic},
            std::pair{cycle_count::cuda_scheduler::work_queue, items::automatic},
            std::pair{cycle_count::cuda_scheduler::work_queue, items::roots},
            std::pair{cycle_count::cuda_scheduler::work_queue, items::edges},
            std::pair{cycle_count::cuda_scheduler::work_queue, items::two_hop}}) {
        cycle_count::options opt = bound(k);
        opt.scheduler = scheduler;
        opt.work_items = work;
        SCOPED_TRACE(::testing::Message()
                     << "k=" << k << " scheduler " << static_cast<int>(scheduler) << " items "
                     << static_cast<int>(work));
        EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g, opt)), expected);
        EXPECT_EQ(cc_counts(cycle_count::compute(cuda_, g32, opt)), expected);
      }
    }
  }
}

// ---- cross-backend equality: cuda = openmp = sequential ----------------------------------------

TEST_F(CycleCountCuda, RandomChainsOfBatchesAgreeOnEveryBackend) {
  std::vector<resources> hosts{seq_};
  if (dyng::backend_available(backend::openmp)) {
    hosts.push_back(resources::openmp(4));
  }
  for (const std::uint64_t seed : test_seeds(20260928, 40)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    // Small graphs, sparse to dense (the host backends recount them quickly).
    dyng::test::cc_spec spec;
    spec.vertex_count = 4 + static_cast<std::int64_t>(rng() % 27);
    spec.edge_probability = std::min(
        0.6, (1.5 + static_cast<double>(rng() % 5)) / static_cast<double>(spec.vertex_count));
    spec.hubs = spec.vertex_count <= 12 ? static_cast<std::int64_t>(rng() % 2) : 0;
    const std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
    const int k = 2 + static_cast<int>(rng() % 6);
    graph_u g = cc_graph<graph_u>(cuda_, spec.vertex_count, edges);
    cycle_count::result r = cycle_count::compute(cuda_, g, bound(k));
    std::vector<graph_u> host_graphs;
    std::vector<cycle_count::result> host_results;
    for (const resources& h : hosts) {
      host_graphs.push_back(cc_graph<graph_u>(h, spec.vertex_count, edges));
      host_results.push_back(cycle_count::compute(h, host_graphs.back(), bound(k)));
    }
    for (int step = 0; step < 5; ++step) {
      // Arbitrary changes: absent deletions, present insertions, self-loops, duplicates, new
      // vertex ids (up to two past the graph), delete-then-reinsert pairs; sometimes empty.
      std::uniform_int_distribution<std::int32_t> vertex(0, g.num_vertices() + 1);
      std::vector<cc_edge> deletions;
      std::vector<cc_edge> insertions;
      const std::size_t num_del = step == 2 ? 0 : rng() % 12;
      const std::size_t num_ins = step == 2 ? 0 : rng() % 12;
      const auto csr = g.to_csr(cuda_);  // the host copy of a resident graph (downloaded)
      for (std::size_t i = 0; i < num_del; ++i) {
        if (!csr.col_ind.empty() && rng() % 2 == 0) {
          const auto e = static_cast<std::size_t>(rng() % csr.col_ind.size());
          const auto u =
              static_cast<std::int32_t>(std::upper_bound(csr.row_ptr.begin(), csr.row_ptr.end(),
                                                         static_cast<std::int64_t>(e)) -
                                        csr.row_ptr.begin() - 1);
          deletions.emplace_back(u, csr.col_ind[e]);
        } else {
          deletions.emplace_back(vertex(rng), vertex(rng));
        }
      }
      for (std::size_t i = 0; i < num_ins; ++i) {
        insertions.emplace_back(vertex(rng), vertex(rng));
      }
      if (step == 3 && !deletions.empty()) {
        insertions.push_back(deletions.front());
      }
      const batch_u b = cc_batch<unweighted>(deletions, insertions);
      SCOPED_TRACE(::testing::Message() << "step " << step << " k=" << k);
      const cycle_count::stats st = cycle_count::update(cuda_, g, b.view(), r);
      for (std::size_t h = 0; h < hosts.size(); ++h) {
        const cycle_count::stats sh =
            cycle_count::update(hosts[h], host_graphs[h], b.view(), host_results[h]);
        EXPECT_EQ(cc_counts(r), cc_counts(host_results[h]));
        EXPECT_EQ(st.cycles_added, sh.cycles_added);
        EXPECT_EQ(st.cycles_removed, sh.cycles_removed);
        EXPECT_EQ(st.deletions, sh.deletions);
        EXPECT_EQ(st.insertions, sh.insertions);
        EXPECT_EQ(st.affected, sh.affected);
        EXPECT_EQ(st.batch.inserted_edges, sh.batch.inserted_edges);
        EXPECT_EQ(st.batch.deleted_edges, sh.batch.deleted_edges);
        EXPECT_EQ(st.batch.ignored_deletions, sh.batch.ignored_deletions);
        EXPECT_EQ(st.batch.ignored_insertions, sh.batch.ignored_insertions);
        EXPECT_EQ(st.batch.dropped_self_loops, sh.batch.dropped_self_loops);
        EXPECT_EQ(st.batch.cancelled_pairs, sh.batch.cancelled_pairs);
        EXPECT_EQ(st.batch.num_vertices_after, sh.batch.num_vertices_after);
        EXPECT_EQ(g.to_csr(cuda_).col_ind, host_graphs[h].to_csr(hosts[h]).col_ind);
        EXPECT_EQ(g.to_csr(cuda_).row_ptr, host_graphs[h].to_csr(hosts[h]).row_ptr);
      }
      EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(cuda_, g, bound(k))));
    }
  }
}

TEST_F(CycleCountCuda, GeneratedBatchesOnLargerGraphsAgreeWithTheSequentialUpdate) {
  for (const std::uint64_t seed : test_seeds(31415, 12)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec;
    spec.vertex_count = 300 + static_cast<std::int64_t>(rng() % 500);
    spec.edge_probability =
        (2.0 + static_cast<double>(rng() % 4)) / static_cast<double>(spec.vertex_count);
    const std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
    const int k = 3 + static_cast<int>(rng() % 3);
    graph_u32 g = cc_graph<graph_u32>(cuda_, spec.vertex_count, edges);
    graph_u32 gs = cc_graph<graph_u32>(seq_, spec.vertex_count, edges);
    cycle_count::result r = cycle_count::compute(cuda_, g, bound(k));
    cycle_count::result rs = cycle_count::compute(seq_, gs, bound(k));
    ASSERT_EQ(cc_counts(r), cc_counts(rs));
    for (int step = 0; step < 3; ++step) {
      legacy::cycle_enum_batch_options params;
      params.num_deletions = 100;
      params.num_insertions = 100;
      params.seed = rng();
      const auto b = legacy::cycle_enum_batch(gs.to_csr(seq_).view(), params);
      (void)cycle_count::update(cuda_, g, b.view(), r);
      (void)cycle_count::update(seq_, gs, b.view(), rs);
      EXPECT_EQ(cc_counts(r), cc_counts(rs)) << "step " << step;
    }
  }
}

// ---- the resident graph -------------------------------------------------------------------------

TEST_F(CycleCountCuda, TheGraphStaysOnTheDeviceAcrossUpdates) {
  dyng::profiler prof;
  resources res = cuda_;
  res.attach_profiler(&prof);
  const std::vector<cc_edge> edges = ring_with_block(30, 8);
  graph_u g = cc_graph<graph_u>(res, 30, edges);
  graph_u gs = cc_graph<graph_u>(seq_, 30, edges);
  cycle_count::result r = cycle_count::compute(res, g, bound(6));
  cycle_count::result rs = cycle_count::compute(seq_, gs, bound(6));
  const auto& impl = dyng::detail::graph_access::impl(g);
  EXPECT_TRUE(impl.has_device_edges());
  EXPECT_TRUE(impl.host_current());
  for (int i = 0; i < 4; ++i) {
    const batch_u b =
        cc_batch<unweighted>({{i, i + 1}, {5, 3}}, {{i + 1, i}, {3, 5}, {29, 30 + i}});
    (void)cycle_count::update(res, g, b.view(), r);
    (void)cycle_count::update(seq_, gs, b.view(), rs);
    EXPECT_TRUE(impl.has_device_edges());
    EXPECT_FALSE(impl.host_current()) << "the batch was merged on the device";
    // The counts never download the host copy.
    EXPECT_EQ(g.num_vertices(), gs.num_vertices());
    EXPECT_EQ(g.num_edges(), gs.num_edges());
    EXPECT_FALSE(impl.host_current());
    EXPECT_EQ(cc_counts(r), cc_counts(rs));
  }
  res.attach_profiler(nullptr);
  std::int64_t uploads = 0;
  for (const dyng::stage_record& s : prof.stages()) {
    if (s.name == "graph.upload") {
      uploads = s.calls;
    }
  }
  EXPECT_EQ(uploads, 1);  // once, by compute()
  // Reading the host copy downloads it once; it equals the host backend's graph byte for byte.
  const auto csr = g.to_csr(cuda_);
  EXPECT_TRUE(impl.host_current());
  EXPECT_EQ(csr.row_ptr, gs.to_csr(seq_).row_ptr);
  EXPECT_EQ(csr.col_ind, gs.to_csr(seq_).col_ind);
  EXPECT_NO_THROW(g.check_integrity(cuda_));
  // A clone of a stale graph copies the downloaded state; the result still matches it.
  const batch_u b = cc_batch<unweighted>({{0, 1}}, {});
  (void)cycle_count::update(cuda_, g, b.view(), r);
  EXPECT_FALSE(impl.host_current());
  graph_u copy = g.clone(cuda_);
  EXPECT_EQ(copy.to_csr(cuda_).col_ind, g.to_csr(cuda_).col_ind);
  (void)cycle_count::update(cuda_, copy, b.view(), r);  // r follows the clone (same state)
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(cuda_, copy, bound(6))));
}

TEST_F(CycleCountCuda, DeviceSetApplyEqualsTheHostApply) {
  for (const std::uint64_t seed : test_seeds(1618, 60)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 1, 70);
    std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
    // Every third seed an undirected graph under set semantics (both directions of each change).
    dyng::graph_properties props = dyng::graph_properties::cycle_enum_compatible();
    if (seed % 3 == 0) {
      props.directed = false;
      const std::size_t m = edges.size();
      for (std::size_t i = 0; i < m; ++i) {
        edges.emplace_back(edges[i].second, edges[i].first);
      }
    }
    SCOPED_TRACE(props.directed ? "directed" : "undirected");
    graph_u g = cc_graph<graph_u>(cuda_, spec.vertex_count, edges, props);
    graph_u32 g32 = cc_graph<graph_u32>(cuda_, spec.vertex_count, edges, props);
    graph_u gs = cc_graph<graph_u>(seq_, spec.vertex_count, edges, props);
    (void)dyng::detail::graph_access::device_out(cuda_, g);  // resident: the device path
    (void)dyng::detail::graph_access::device_out(cuda_, g32);
    std::uniform_int_distribution<std::int32_t> vertex(
        0, static_cast<std::int32_t>(spec.vertex_count) + 3);
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
    for (std::uint64_t i = rng() % 30; i > 0; --i) {
      deletions.emplace_back(vertex(rng), vertex(rng));
    }
    for (std::uint64_t i = rng() % 30; i > 0; --i) {
      insertions.emplace_back(vertex(rng), vertex(rng));
    }
    for (std::size_t i = 0; i < edges.size() && i < 5; ++i) {
      deletions.push_back(edges[rng() % edges.size()]);  // existing edges
    }
    if (!deletions.empty()) {
      insertions.push_back(deletions.back());  // delete, then re-insert
    }
    const batch_u b = cc_batch<unweighted>(deletions, insertions);
    dyng::detail::apply_delta<std::int32_t> delta;
    dyng::detail::apply_delta<std::int32_t> delta_s;
    const dyng::apply_summary s = dyng::detail::graph_access::apply(cuda_, g, b.view(), &delta);
    const dyng::apply_summary s32 = g32.apply(cuda_, b.view());
    const dyng::apply_summary ss = dyng::detail::graph_access::apply(seq_, gs, b.view(), &delta_s);
    const auto& impl = dyng::detail::graph_access::impl(g);
    EXPECT_FALSE(impl.host_current());
    EXPECT_EQ(s.inserted_edges, ss.inserted_edges);
    EXPECT_EQ(s.deleted_edges, ss.deleted_edges);
    EXPECT_EQ(s.ignored_deletions, ss.ignored_deletions);
    EXPECT_EQ(s.ignored_insertions, ss.ignored_insertions);
    EXPECT_EQ(s.cancelled_pairs, ss.cancelled_pairs);
    EXPECT_EQ(s.dropped_self_loops, ss.dropped_self_loops);
    EXPECT_EQ(s.num_vertices_after, ss.num_vertices_after);
    EXPECT_EQ(s32.inserted_edges, ss.inserted_edges);
    EXPECT_EQ(delta.insert_src, delta_s.insert_src);
    EXPECT_EQ(delta.insert_dst, delta_s.insert_dst);
    EXPECT_EQ(delta.delete_src, delta_s.delete_src);
    EXPECT_EQ(delta.delete_dst, delta_s.delete_dst);
    EXPECT_EQ(g.num_vertices(), gs.num_vertices());
    EXPECT_EQ(g.num_edges(), gs.num_edges());
    EXPECT_EQ(g.version(), gs.version());
    // The insertion id of every edge: its position in the normalized insertions, else none.
    const auto& d = dyng::detail::graph_access::device_out(cuda_, g);
    std::vector<std::int32_t> ids(d.insertion_ids.size());
    if (!ids.empty()) {
      ASSERT_EQ(cudaSuccess, cudaMemcpy(ids.data(), d.insertion_ids.data(),
                                        ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    }
    const auto host = gs.to_csr(seq_);
    for (std::int32_t u = 0; u < gs.num_vertices(); ++u) {
      for (std::int64_t e = host.row_ptr[u]; e < host.row_ptr[u + 1]; ++e) {
        std::int32_t want = dyng::detail::no_change_id;
        for (std::size_t i = 0; i < delta_s.insert_src.size(); ++i) {
          if (delta_s.insert_src[i] == u && delta_s.insert_dst[i] == host.col_ind[e]) {
            want = static_cast<std::int32_t>(i);
          }
        }
        EXPECT_EQ(ids[static_cast<std::size_t>(e)], want) << u << " -> " << host.col_ind[e];
      }
    }
    // The host copies, downloaded now, equal the host apply byte for byte.
    EXPECT_EQ(g.to_csr(cuda_).row_ptr, host.row_ptr);
    EXPECT_EQ(g.to_csr(cuda_).col_ind, host.col_ind);
    const auto c32 = g32.to_csr(cuda_);
    EXPECT_EQ(c32.col_ind, host.col_ind);
    ASSERT_EQ(c32.row_ptr.size(), host.row_ptr.size());
    for (std::size_t i = 0; i < c32.row_ptr.size(); ++i) {
      EXPECT_EQ(static_cast<std::int64_t>(c32.row_ptr[i]), host.row_ptr[i]);
    }
    EXPECT_TRUE(impl.host_current());
  }
}

TEST_F(CycleCountCuda, SeveralResultsShareOneDeviceCommit) {
  // Two cycle_count results with different bounds on one resident graph: one Step 0, one upload
  // of the change lists, one device merge; each result equals its own recount.
  const std::vector<cc_edge> edges = ring_with_block(24, 7);
  graph_u g = cc_graph<graph_u>(cuda_, 24, edges);
  cycle_count::result r4 = cycle_count::compute(cuda_, g, bound(4));
  cycle_count::result r7 = cycle_count::compute(cuda_, g, bound(7));
  const batch_u b = cc_batch<unweighted>({{0, 1}, {2, 3}, {4, 5}}, {{1, 0}, {3, 2}, {23, 24}});
  const auto [s4, s7] = dyng::update(cuda_, g, b.view(), r4, r7);
  EXPECT_FALSE(dyng::detail::graph_access::impl(g).host_current());  // merged on the device
  EXPECT_EQ(cc_counts(r4), cc_counts(cycle_count::compute(seq_, g.clone(seq_), bound(4))));
  EXPECT_EQ(cc_counts(r7), cc_counts(cycle_count::compute(seq_, g.clone(seq_), bound(7))));
  EXPECT_EQ(s4.deletions, 3);   // the ring edges 0->1, 2->3, 4->5
  EXPECT_EQ(s4.insertions, 1);  // 1->0 and 3->2 exist (the block): only 23->24 is new
  EXPECT_EQ(s7.deletions, s4.deletions);
  EXPECT_EQ(s7.insertions, s4.insertions);
  EXPECT_THROW((void)dyng::update(cuda_, g, b.view(), r4, r4), dyng::invalid_argument_error);
}

// ---- the host-commit fallbacks: weight columns, other batch semantics --------------------------

TEST_F(CycleCountCuda, HostCommitsUploadTheNextGraph) {
  std::vector<std::pair<const char*, dyng::graph_properties>> variants;
  variants.emplace_back("default (upsert)", dyng::graph_properties{});
  {
    dyng::graph_properties p = dyng::graph_properties::cycle_enum_compatible();
    p.directed = false;
    variants.emplace_back("undirected set", p);
  }
  for (const auto& [name, props] : variants) {
    SCOPED_TRACE(name);
    for (const std::uint64_t seed : test_seeds(2718, 10)) {
      std::mt19937_64 rng(seed);
      SCOPED_TRACE(seed_trace(seed));
      dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 3, 20);
      std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
      if (!props.directed) {
        const std::size_t m = edges.size();
        for (std::size_t i = 0; i < m; ++i) {
          edges.emplace_back(edges[i].second, edges[i].first);
        }
      }
      // Weighted (one weight column): the batch is applied on the host.
      graph_w g = cc_graph<graph_w>(cuda_, spec.vertex_count, edges, props);
      graph_w gs = cc_graph<graph_w>(seq_, spec.vertex_count, edges, props);
      const int k = 2 + static_cast<int>(rng() % 6);
      cycle_count::result r = cycle_count::compute(cuda_, g, bound(k));
      cycle_count::result rs = cycle_count::compute(seq_, gs, bound(k));
      std::uniform_int_distribution<std::int32_t> vertex(
          0, static_cast<std::int32_t>(spec.vertex_count));
      std::vector<cc_edge> deletions;
      std::vector<cc_edge> insertions;
      for (std::uint64_t i = rng() % 8; i > 0; --i) {
        deletions.emplace_back(vertex(rng), vertex(rng));
      }
      for (std::uint64_t i = rng() % 8; i > 0; --i) {
        insertions.emplace_back(vertex(rng), vertex(rng));
      }
      const auto b = cc_batch<std::int32_t>(deletions, insertions);
      (void)cycle_count::update(cuda_, g, b.view(), r);
      (void)cycle_count::update(seq_, gs, b.view(), rs);
      EXPECT_EQ(cc_counts(r), cc_counts(rs));
      EXPECT_TRUE(dyng::detail::graph_access::impl(g).host_current());
    }
  }
}

// ---- batches that change nothing, empty graphs --------------------------------------------------

TEST_F(CycleCountCuda, CornerCasesOfBatchesAndGraphs) {
  // An empty graph: no cycle, no work.
  graph_u empty = cc_graph<graph_u>(cuda_, 0, {});
  cycle_count::result r0 = cycle_count::compute(cuda_, empty, bound(4));
  EXPECT_EQ(r0.total(), 0U);
  const batch_u grow = cc_batch<unweighted>({}, {{0, 1}, {1, 0}});
  (void)cycle_count::update(cuda_, empty, grow.view(), r0);
  EXPECT_EQ(r0.count(2), 1U);
  // Edges but no cycle.
  graph_u dag = cc_graph<graph_u>(cuda_, 4, {{0, 1}, {1, 2}, {2, 3}});
  EXPECT_EQ(cycle_count::compute(cuda_, dag, bound(4)).total(), 0U);

  graph_u g = cc_graph<graph_u>(cuda_, 4, {{0, 1}, {1, 2}, {2, 0}, {1, 0}, {2, 3}, {3, 1}});
  graph_u gs = cc_graph<graph_u>(seq_, 4, {{0, 1}, {1, 2}, {2, 0}, {1, 0}, {2, 3}, {3, 1}});
  cycle_count::result r = cycle_count::compute(cuda_, g, bound(5));
  cycle_count::result rs = cycle_count::compute(seq_, gs, bound(5));
  const hist before = cc_counts(r);
  struct named_batch {
    const char* name;
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
  };
  for (const named_batch& c :
       std::vector<named_batch>{{"empty", {}, {}},
                                {"deletions of non-edges", {{0, 3}, {3, 0}, {7, 8}}, {}},
                                {"insertions of existing edges", {}, {{0, 1}, {2, 3}}},
                                {"self-loops", {{1, 1}}, {{2, 2}}},
                                {"delete and re-insert", {{0, 1}}, {{0, 1}}}}) {
    SCOPED_TRACE(c.name);
    const batch_u b = cc_batch<unweighted>(c.deletions, c.insertions);
    const cycle_count::stats st = cycle_count::update(cuda_, g, b.view(), r);
    (void)cycle_count::update(seq_, gs, b.view(), rs);
    EXPECT_EQ(cc_counts(r), before);
    EXPECT_EQ(cc_counts(r), cc_counts(rs));
    EXPECT_EQ(st.affected, 0);
    EXPECT_EQ(g.num_edges(), 6);
  }
  // New vertex ids: a 3-cycle through two new vertices.
  const batch_u fresh = cc_batch<unweighted>({}, {{3, 4}, {4, 5}, {5, 3}});
  (void)cycle_count::update(cuda_, g, fresh.view(), r);
  (void)cycle_count::update(seq_, gs, fresh.view(), rs);
  EXPECT_EQ(g.num_vertices(), 6);
  EXPECT_EQ(cc_counts(r), cc_counts(rs));
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(seq_, gs, bound(5))));
  // A stale result is rejected on cuda too.
  const batch_u del = cc_batch<unweighted>({{0, 1}}, {});
  (void)g.apply(cuda_, del.view());
  EXPECT_THROW((void)cycle_count::update(cuda_, g, del.view(), r), dyng::stale_result_error);
  // A result moves to any backend (the histogram is a host array).
  const cycle_count::result r2 = rs.clone(cuda_);
  EXPECT_EQ(cc_counts(r2), cc_counts(rs));
  EXPECT_EQ(r2.space(), dyng::memory_space::host);
}

}  // namespace
