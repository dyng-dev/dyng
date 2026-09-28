// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_random_test.cpp
 * @brief Randomized differential tests of cycle_count against independent oracles, the ports of
 *        CycleEnumeration-GPU@0a976ad's randomized_static_parity_test, randomized_update_parity_test,
 *        dynamic_update_parity_test and dynamic_update_openmp_test (CPU parts), plus dynG's own
 *        cases: every batch_semantics, undirected graphs, chains of batches, inverse batches.
 *
 * The oracles share no code with the enumerators: testing::oracle_simple_cycles (a subset dynamic
 * program), testing::brute_force_simple_cycles and testing::edge_set_after_batch (the graph after
 * a batch computed without the library's apply). This executable is also built against the two
 * recorded mutations (tests/CMakeLists.txt, DYNG_MUTATION_TESTS), which must make it fail.
 *
 * Every trial draws from its own seed and prints it on failure; replay one with
 * `DYNG_TEST_SEED=<seed> ctest -R <test>`, widen the campaign with `DYNG_TEST_SEEDS=<n>` (PLAN
 * Section 8.1).
 */
#include "support/cycle_count_support.hpp"
#include "support/test_seeds.hpp"

#include <dyng/core/error.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <tuple>
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
using dyng::test::cc_resources;
using dyng::test::seed_trace;
using dyng::test::test_seeds;
namespace cycle_count = dyng::cycle_count;
namespace legacy = dyng::generators::legacy;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_u32 = dyng::graph<std::int32_t, std::int32_t, unweighted>;
using graph_w = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using hist = std::vector<std::uint64_t>;
using batch_u = dyng::edge_batch<std::int32_t, unweighted>;

/// The trial index of a seed of test_seeds(base, n) (any value for a replayed foreign seed).
int trial_of(std::uint64_t seed, std::uint64_t base) {
  return static_cast<int>((seed - base) & 0x7fffffffU);
}

/// At least two thirds of the seeds must have given a checkable case (a replay of one seed: 0).
std::size_t required_checks(std::size_t seeds) {
  return seeds > 1 ? seeds * 2 / 3 : 0;
}

cycle_count::options bound(int k) {
  cycle_count::options opt;
  opt.max_length = k;
  return opt;
}

/// The oracle histogram of the graph after `batch`, computed without the library's apply.
template <typename graph_t, typename batch_t>
hist recount_after(const resources& res, const graph_t& g0, const batch_t& batch, int k) {
  const auto csr0 = g0.to_csr(res);
  const auto after = dyng::testing::edge_set_after_batch(csr0.view(), batch.view());
  return dyng::testing::oracle_simple_cycles(after.view(), k);
}

class CycleCountRandom : public ::testing::TestWithParam<backend> {
 protected:
  resources res_ = cc_resources(GetParam());
};

// RandomizedStaticParityTest.CpuCountersMatchOracle: 400 random graphs (sparse, dense, hub-heavy,
// with and without self-loops), k in 2..8 or no bound.
TEST_P(CycleCountRandom, StaticMatchesOracle) {
  for (const std::uint64_t seed : test_seeds(20260925, 400)) {
    std::mt19937_64 rng(seed);
    const int trial = trial_of(seed, 20260925);
    SCOPED_TRACE(seed_trace(seed));
    const dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 2, 10);
    const graph_u g =
        cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = trial % 5 == 0 ? -1 : 2 + trial % 7;
    SCOPED_TRACE(::testing::Message()
                 << "trial " << trial << " n=" << spec.vertex_count << " k=" << k);
    const hist expected = dyng::test::cc_oracle(res_, g, k);
    EXPECT_EQ(cc_counts(cycle_count::compute(res_, g, bound(k))), expected);
    EXPECT_EQ(dyng::test::cc_brute(res_, g, k), expected);
  }
}

// Larger sparse and hub-heavy graphs against the brute force (the host part of
// CudaCountersMatchBruteForceOnLargerGraphs), 32-bit and 64-bit offsets.
TEST_P(CycleCountRandom, StaticMatchesBruteForceOnLargerGraphs) {
  struct shape {
    std::int64_t vertices;
    double probability;
    std::int64_t hubs;
    int k;
  };
  for (const std::uint64_t seed : test_seeds(99, 1)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    for (const shape& c : {shape{40, 0.12, 0, 6}, shape{60, 0.05, 2, 5}, shape{30, 0.3, 0, 5},
                           shape{80, 0.03, 3, 6}, shape{50, 0.06, 1, 7}, shape{25, 0.5, 0, 4}}) {
      dyng::test::cc_spec spec;
      spec.vertex_count = c.vertices;
      spec.edge_probability = c.probability;
      spec.hubs = c.hubs;
      spec.self_loop_probability = 0.05;
      const std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
      const graph_u g = cc_graph<graph_u>(res_, c.vertices, edges);
      const graph_u32 g32 = cc_graph<graph_u32>(res_, c.vertices, edges);
      SCOPED_TRACE(::testing::Message() << "n=" << c.vertices << " k=" << c.k);
      const hist expected = dyng::test::cc_brute(res_, g, c.k);
      EXPECT_EQ(cc_counts(cycle_count::compute(res_, g, bound(c.k))), expected);
      EXPECT_EQ(cc_counts(cycle_count::compute(res_, g32, bound(c.k))), expected);
    }
  }
}

// RandomizedUpdateParityTest.ValidBatchesMatchRecount: generated (valid) batches.
TEST_P(CycleCountRandom, GeneratedBatchesMatchRecount) {
  const std::vector<std::uint64_t> seeds = test_seeds(8675309, 300);
  std::size_t checked = 0;
  for (const std::uint64_t seed : seeds) {
    std::mt19937_64 rng(seed);
    const int trial = trial_of(seed, 8675309);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 3, 11);
    spec.self_loop_probability = 0.0;  // the parser never produces self-loops
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = 2 + trial % 7;
    legacy::cycle_enum_batch_options params;
    params.num_deletions = static_cast<std::int64_t>(rng() % 6);
    params.num_insertions = static_cast<std::int64_t>(rng() % 6);
    params.seed = rng();
    batch_u batch;
    try {
      batch = legacy::cycle_enum_batch(g.to_csr(res_).view(), params);
    } catch (const dyng::invalid_argument_error&) {
      continue;  // not enough edges or non-edges for this request
    }
    SCOPED_TRACE(::testing::Message()
                 << "trial " << trial << " n=" << spec.vertex_count << " k=" << k);
    const hist expected = recount_after(res_, g, batch, k);
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    EXPECT_EQ(cc_counts(r), dyng::test::cc_oracle(res_, g, k));
    const cycle_count::stats st = cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), expected);
    EXPECT_EQ(st.deletions, params.num_deletions);
    EXPECT_EQ(st.insertions, params.num_insertions);
    ++checked;
  }
  EXPECT_GE(checked, required_checks(seeds.size()));
}

// RandomizedUpdateParityTest.LargerGraphsMatchBruteForceRecount.
TEST_P(CycleCountRandom, LargerGraphsMatchBruteForceRecount) {
  for (const std::uint64_t seed : test_seeds(1234, 12)) {
    std::mt19937_64 rng(seed);
    const int trial = trial_of(seed, 1234);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec;
    spec.vertex_count = 40 + static_cast<std::int64_t>(rng() % 40);
    spec.edge_probability = 0.04 + 0.02 * static_cast<double>(trial % 4);
    spec.hubs = trial % 3;
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = 3 + trial % 4;
    legacy::cycle_enum_batch_options params;
    params.num_deletions = 20;
    params.num_insertions = 20;
    params.seed = rng();
    const batch_u batch = legacy::cycle_enum_batch(g.to_csr(res_).view(), params);
    SCOPED_TRACE(::testing::Message() << "trial " << trial << " k=" << k);
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, k));
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, k));
  }
}

// UpdateBatchValidationTest.ChangesThatDoNotAlterTheGraphAreNoOps: each invalid kind of change.
TEST_P(CycleCountRandom, ChangesThatDoNotAlterTheGraphAreNoOps) {
  // 0->1, 1->2, 2->0, 1->0: one 2-cycle and one 3-cycle.
  const std::vector<cc_edge> base = {{0, 1}, {1, 2}, {2, 0}, {1, 0}};
  struct named_batch {
    const char* name;
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
  };
  const std::vector<named_batch> cases = {
      {"delete missing edge", {{2, 1}}, {}},
      {"insert existing edge", {}, {{0, 1}}},
      {"insert self-loop", {}, {{1, 1}}},
      {"delete self-loop", {{1, 1}}, {}},
      {"delete and reinsert", {{0, 1}}, {{0, 1}}},
      {"duplicate changes", {{1, 0}, {1, 0}}, {{0, 2}, {0, 2}}},
      {"insert through new vertices", {}, {{2, 4}, {4, 0}}},
      {"mixed", {{2, 1}, {1, 0}, {3, 3}}, {{0, 1}, {2, 5}, {5, 1}, {5, 5}}},
  };
  for (const named_batch& c : cases) {
    SCOPED_TRACE(c.name);
    graph_u g = cc_graph<graph_u>(res_, 3, base);
    const batch_u batch = cc_batch<unweighted>(c.deletions, c.insertions);
    const hist expected = recount_after(res_, g, batch, 4);
    cycle_count::result r = cycle_count::compute(res_, g, bound(4));
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), expected);
    EXPECT_EQ(dyng::test::cc_oracle(res_, g, 4), expected);  // the library's apply agrees
  }
  graph_u g = cc_graph<graph_u>(res_, 3, base);
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  const batch_u grow = cc_batch<unweighted>({}, {{2, 4}, {4, 0}});
  (void)cycle_count::update(res_, g, grow.view(), r);
  EXPECT_EQ(r.count(2), 1U);
  EXPECT_EQ(r.count(3), 1U);
  EXPECT_EQ(r.count(4), 1U);
}

// UpdateBatchValidationTest.ArbitraryBatchesMatchRecount: ids past the graph, coinciding endpoints,
// existing insertions, absent deletions, delete-then-reinsert.
TEST_P(CycleCountRandom, ArbitraryBatchesMatchRecount) {
  for (const std::uint64_t seed : test_seeds(4711, 200)) {
    std::mt19937_64 rng(seed);
    const int trial = trial_of(seed, 4711);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 3, 9);
    spec.self_loop_probability = 0.0;
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = 2 + trial % 6;
    std::uniform_int_distribution<std::int32_t> vertex(
        0, static_cast<std::int32_t>(spec.vertex_count + 1));
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
    const std::size_t num_del = rng() % 6;
    const std::size_t num_ins = rng() % 6;
    for (std::size_t i = 0; i < num_del; ++i) {
      deletions.emplace_back(vertex(rng), vertex(rng));
    }
    for (std::size_t i = 0; i < num_ins; ++i) {
      insertions.emplace_back(vertex(rng), vertex(rng));
    }
    if (trial % 3 == 0 && !deletions.empty()) {
      insertions.push_back(deletions.front());
    }
    const batch_u batch = cc_batch<unweighted>(deletions, insertions);
    SCOPED_TRACE(::testing::Message() << "trial " << trial << " k=" << k);
    const hist expected = recount_after(res_, g, batch, k);
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), expected);
  }
}

// DynamicUpdateParityTest.UpdateMatchesRecompute: 200 random graphs and generated batches, update
// against a full recount of the post-batch graph (k = 6).
TEST_P(CycleCountRandom, UpdateMatchesRecompute) {
  constexpr std::int32_t vertices = 8;
  constexpr int k = 6;
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  const std::vector<std::uint64_t> seeds = test_seeds(12345, 200);
  std::size_t checked = 0;
  for (const std::uint64_t seed : seeds) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    std::vector<cc_edge> edges;
    for (std::int32_t u = 0; u < vertices; ++u) {
      for (std::int32_t v = 0; v < vertices; ++v) {
        if (u != v && coin(rng) < 0.35) {
          edges.emplace_back(u, v);
        }
      }
    }
    graph_u g = cc_graph<graph_u>(res_, vertices, edges);
    std::uniform_int_distribution<std::int64_t> small(0, 3);
    legacy::cycle_enum_batch_options params;
    params.num_deletions = std::min<std::int64_t>(small(rng), g.num_edges());
    params.num_insertions = small(rng);
    params.seed = rng();
    batch_u batch;
    try {
      batch = legacy::cycle_enum_batch(g.to_csr(res_).view(), params);
    } catch (const dyng::invalid_argument_error&) {
      continue;
    }
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    (void)cycle_count::update(res_, g, batch.view(), r);
    ASSERT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(k))));
    ++checked;
  }
  EXPECT_GE(checked, required_checks(seeds.size()) / 2);
}

// DynamicUpdateParityTest.AllDeleteAndAllInsertMatchRecompute.
TEST_P(CycleCountRandom, DeleteOnlyAndInsertOnlyMatchRecompute) {
  const std::uint64_t seed = test_seeds(99, 1).front();
  SCOPED_TRACE(seed_trace(seed));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  std::vector<cc_edge> edges;
  for (std::int32_t u = 0; u < 8; ++u) {
    for (std::int32_t v = 0; v < 8; ++v) {
      if (u != v && coin(rng) < 0.4) {
        edges.emplace_back(u, v);
      }
    }
  }
  for (const auto& [del, ins, seed] : {std::tuple<int, int, int>{3, 0, 1}, {0, 4, 2}}) {
    graph_u g = cc_graph<graph_u>(res_, 8, edges);
    cycle_count::result r = cycle_count::compute(res_, g, bound(6));
    legacy::cycle_enum_batch_options params;
    params.num_deletions = del;
    params.num_insertions = ins;
    params.seed = static_cast<std::uint64_t>(seed);
    const batch_u batch = legacy::cycle_enum_batch(g.to_csr(res_).view(), params);
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(6))));
  }
  // Deleting every edge empties the histogram.
  graph_u g = cc_graph<graph_u>(res_, 8, edges);
  cycle_count::result r = cycle_count::compute(res_, g, bound(8));
  const batch_u all = cc_batch<unweighted>(edges, {});
  const cycle_count::stats st = cycle_count::update(res_, g, all.view(), r);
  EXPECT_EQ(r.total(), 0U);
  EXPECT_EQ(g.num_edges(), 0);
  EXPECT_EQ(st.deletions, static_cast<std::int64_t>(edges.size()));
}

// Unbounded results: the update counts every cycle through the change edges.
TEST_P(CycleCountRandom, UnboundedUpdatesMatchOracle) {
  for (const std::uint64_t seed : test_seeds(31337, 60)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 2, 8);
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    cycle_count::result r = cycle_count::compute(res_, g);
    std::uniform_int_distribution<std::int32_t> vertex(
        0, static_cast<std::int32_t>(spec.vertex_count + 1));
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
    for (std::uint64_t i = rng() % 5; i > 0; --i) {
      deletions.emplace_back(vertex(rng), vertex(rng));
    }
    for (std::uint64_t i = rng() % 5; i > 0; --i) {
      insertions.emplace_back(vertex(rng), vertex(rng));
    }
    const batch_u batch = cc_batch<unweighted>(deletions, insertions);
    const hist expected = recount_after(res_, g, batch, -1);
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(cc_counts(r), expected);
    EXPECT_EQ(r.bound(), std::max<std::int64_t>(g.num_vertices(), 2));
  }
}

// PLAN 5.1: every batch_semantics preset is accepted (Step 0 reduces a batch to the change of the
// edge set): upsert (the default), ignore, insertions first, kept self-loops, undirected graphs,
// weighted graphs; chains of batches equal compute() (conformance C2) and a batch followed by its
// inverse restores the histogram (C5).
TEST_P(CycleCountRandom, EverySemanticsMatchesCompute) {
  std::vector<std::pair<const char*, dyng::graph_properties>> variants;
  variants.emplace_back("cycle_enum_compatible", dyng::graph_properties::cycle_enum_compatible());
  variants.emplace_back("default", dyng::graph_properties{});
  {
    dyng::graph_properties p;
    p.semantics.on_existing_insert = dyng::batch_semantics::existing_insert::ignore;
    p.semantics.on_self_loop = dyng::batch_semantics::self_loop::keep;
    variants.emplace_back("ignore, self-loops kept", p);
  }
  {
    dyng::graph_properties p;
    p.semantics.deletions_first = false;
    variants.emplace_back("insertions first", p);
  }
  {
    dyng::graph_properties p = dyng::graph_properties::cycle_enum_compatible();
    p.directed = false;
    variants.emplace_back("undirected set", p);
  }
  {
    dyng::graph_properties p;
    p.directed = false;
    variants.emplace_back("undirected upsert", p);
  }
  for (const auto& [name, props] : variants) {
    for (const std::uint64_t seed : test_seeds(777, 25)) {
      std::mt19937_64 rng(seed);
      const int trial = trial_of(seed, 777);
      SCOPED_TRACE(seed_trace(seed));
      dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 3, 9);
      std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
      if (!props.directed) {  // a symmetric edge list
        const std::size_t m = edges.size();
        for (std::size_t i = 0; i < m; ++i) {
          edges.emplace_back(edges[i].second, edges[i].first);
        }
      }
      graph_w g = cc_graph<graph_w>(res_, spec.vertex_count, edges, props);
      const int k = trial % 4 == 0 ? -1 : 2 + trial % 6;
      cycle_count::result r = cycle_count::compute(res_, g, bound(k));
      std::uniform_int_distribution<std::int32_t> vertex(
          0, static_cast<std::int32_t>(spec.vertex_count));
      for (int step = 0; step < 3; ++step) {
        std::vector<cc_edge> deletions;
        std::vector<cc_edge> insertions;
        for (std::uint64_t i = rng() % 6; i > 0; --i) {
          deletions.emplace_back(vertex(rng), vertex(rng));
        }
        for (std::uint64_t i = rng() % 6; i > 0; --i) {
          insertions.emplace_back(vertex(rng), vertex(rng));
        }
        if (step == 1 && !deletions.empty()) {
          insertions.push_back(deletions.back());
        }
        const auto batch = cc_batch<std::int32_t>(deletions, insertions);
        SCOPED_TRACE(::testing::Message()
                     << name << " trial " << trial << " step " << step << " k=" << k);
        (void)cycle_count::update(res_, g, batch.view(), r);
        EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(k))));
        EXPECT_EQ(cc_counts(r),
                  dyng::test::cc_resize(dyng::test::cc_oracle(res_, g, k), cc_counts(r).size()));
      }
    }
  }
}

TEST_P(CycleCountRandom, InverseBatchRestoresTheHistogram) {
  for (const std::uint64_t seed : test_seeds(4242, 40)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 4, 10);
    spec.self_loop_probability = 0.0;
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    cycle_count::result r = cycle_count::compute(res_, g, bound(5));
    const hist before = cc_counts(r);
    legacy::cycle_enum_batch_options params;
    params.num_deletions = std::min<std::int64_t>(3, g.num_edges());
    params.num_insertions = 3;
    params.seed = rng();
    batch_u batch;
    try {
      batch = legacy::cycle_enum_batch(g.to_csr(res_).view(), params);
    } catch (const dyng::invalid_argument_error&) {
      continue;
    }
    // A generated batch is valid for inversion: its deletions exist and its insertions do not.
    batch_u inverse;
    for (std::size_t i = 0; i < batch.num_insertions(); ++i) {
      inverse.delete_edge(batch.insert_src()[i], batch.insert_dst()[i]);
    }
    for (std::size_t i = 0; i < batch.num_deletions(); ++i) {
      inverse.insert_edge(batch.delete_src()[i], batch.delete_dst()[i]);
    }
    const cycle_count::stats forward = cycle_count::update(res_, g, batch.view(), r);
    const cycle_count::stats back = cycle_count::update(res_, g, inverse.view(), r);
    EXPECT_EQ(cc_counts(r), before);
    EXPECT_EQ(forward.cycles_added, back.cycles_removed);
    EXPECT_EQ(forward.cycles_removed, back.cycles_added);
  }
}

// Deep searches (default options, no bound): the reversed path n-1 -> ... -> 1 -> 0 plus the edge
// 0 -> n-1 is one n-cycle. The static search from root 0 and the update search through the edge
// both walk a path of n vertices; the searches keep it on explicit stacks, so this neither
// overflows the thread's stack (the original's recursion does at n of about 70,000 to 200,000)
// nor costs more than O(n).
TEST_P(CycleCountRandom, LongRingsNeedNoThreadStack) {
  constexpr std::int32_t n = 300000;
  std::vector<cc_edge> path;
  path.reserve(n);
  for (std::int32_t v = 1; v < n; ++v) {
    path.emplace_back(v, v - 1);
  }
  graph_u g = cc_graph<graph_u>(res_, n, path);
  cycle_count::result r = cycle_count::compute(res_, g);
  EXPECT_EQ(r.total(), 0U);
  EXPECT_EQ(r.bound(), n);
  const batch_u close = cc_batch<unweighted>({}, {{0, n - 1}});
  const cycle_count::stats st = cycle_count::update(res_, g, close.view(), r);
  EXPECT_EQ(st.cycles_added, 1U);
  EXPECT_EQ(r.count(n), 1U);
  EXPECT_EQ(r.total(), 1U);
  EXPECT_EQ(cycle_count::compute(res_, g).count(n), 1U);  // the static search from root 0
  const batch_u open = cc_batch<unweighted>({{n / 2, n / 2 - 1}}, {});
  EXPECT_EQ(cycle_count::update(res_, g, open.view(), r).cycles_removed, 1U);
  EXPECT_EQ(r.total(), 0U);
}

// Bounds at the CUDA limit (64) and far above the vertex count: the histogram covers
// min(k, max(n, 2)) lengths, so a huge k costs nothing, and the counts equal the oracle's.
TEST_P(CycleCountRandom, LargeBoundsMatchOracle) {
  for (const std::uint64_t seed : test_seeds(6464, 30)) {
    std::mt19937_64 rng(seed);
    const int trial = trial_of(seed, 6464);
    SCOPED_TRACE(seed_trace(seed));
    dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 3, 9);
    graph_u g = cc_graph<graph_u>(res_, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = trial % 2 == 0 ? 64 : 2000000000;
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    EXPECT_EQ(r.bound(), std::max<std::int64_t>(g.num_vertices(), 2));
    EXPECT_EQ(r.counts().size(), static_cast<std::size_t>(r.bound()) + 1);
    EXPECT_EQ(r.get_options().max_length, k);
    EXPECT_EQ(cc_counts(r),
              dyng::test::cc_resize(dyng::test::cc_oracle(res_, g, -1), cc_counts(r).size()));
    std::uniform_int_distribution<std::int32_t> vertex(
        0, static_cast<std::int32_t>(spec.vertex_count + 2));
    std::vector<cc_edge> deletions;
    std::vector<cc_edge> insertions;
    for (std::uint64_t i = rng() % 5; i > 0; --i) {
      deletions.emplace_back(vertex(rng), vertex(rng));
    }
    for (std::uint64_t i = 1 + rng() % 5; i > 0; --i) {
      insertions.emplace_back(vertex(rng), vertex(rng));
    }
    const batch_u batch = cc_batch<unweighted>(deletions, insertions);
    (void)cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(r.bound(), std::max<std::int64_t>(g.num_vertices(), 2));
    EXPECT_EQ(cc_counts(r),
              dyng::test::cc_resize(dyng::test::cc_oracle(res_, g, -1), cc_counts(r).size()));
  }
}

// A batch that grows the graph far beyond its size (the per-thread marks of the insert phase must
// be resized for the new vertices): a ring through 601 new vertices closed at an old one.
TEST_P(CycleCountRandom, UpdatesThatGrowTheGraphFarBeyondItsSize) {
  for (const int k : {-1, 4, 700}) {
    SCOPED_TRACE(::testing::Message() << "k=" << k);
    graph_u g = cc_graph<graph_u>(res_, 5, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 3}});
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    std::vector<cc_edge> ring{{0, 5}};
    for (std::int32_t v = 5; v < 605; ++v) {
      ring.emplace_back(v, v + 1);
    }
    ring.emplace_back(605, 0);
    ring.emplace_back(10, 11);  // listed twice: a duplicate insertion
    const batch_u batch = cc_batch<unweighted>({{2, 0}}, ring);
    const cycle_count::stats st = cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ(g.num_vertices(), 606);
    EXPECT_EQ(st.cycles_removed, 1U);  // 0 -> 1 -> 2 -> 0
    EXPECT_EQ(r.count(2), 1U);         // 3 <-> 4
    EXPECT_EQ(r.count(3), 0U);
    EXPECT_EQ(r.count(602), k == 4 ? 0U : 1U);  // 0 -> 5 -> ... -> 605 -> 0
    EXPECT_EQ(r.total(), k == 4 ? 1U : 2U);
    EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(k))));
    EXPECT_EQ(r.bound(), k == 4 ? 4 : 606);
  }
}

// The cost of an unbounded update depends on its searches, not on the vertex count: a graph of
// 100,000 disjoint triangles, 2,000 deletions and 2,000 insertions. Before the counters grew on
// demand, the unbounded update touched max(n, 2) counters per change edge (about 350 times the
// bounded update in Release); a loose factor keeps the test robust on a loaded machine.
TEST_P(CycleCountRandom, UnboundedUpdateCostsAboutTheBoundedOne) {
  constexpr std::int32_t triangles = 100000;
  std::vector<cc_edge> edges;
  for (std::int32_t t = 0; t < triangles; ++t) {
    edges.emplace_back(3 * t, 3 * t + 1);
    edges.emplace_back(3 * t + 1, 3 * t + 2);
    edges.emplace_back(3 * t + 2, 3 * t);
  }
  std::vector<cc_edge> deletions;
  std::vector<cc_edge> insertions;
  for (std::int32_t t = 0; t < 2000; ++t) {
    deletions.emplace_back(3 * t, 3 * t + 1);
    insertions.emplace_back(3 * t + 3 * 5000, 3 * t + 2 + 3 * 5000);  // closes a 2-cycle
  }
  const batch_u batch = cc_batch<unweighted>(deletions, insertions);
  const auto best_of_three = [&](int k) {
    double best = 1e30;
    for (int round = 0; round < 3; ++round) {
      graph_u g = cc_graph<graph_u>(res_, 3 * triangles, edges);
      cycle_count::result r = cycle_count::compute(res_, g, bound(k));
      const auto start = std::chrono::steady_clock::now();
      const cycle_count::stats st = cycle_count::update(res_, g, batch.view(), r);
      const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
      best = std::min(best, took.count());
      EXPECT_EQ(st.cycles_removed, 2000U);
      EXPECT_EQ(st.cycles_added, 2000U);
      EXPECT_EQ(r.count(2), 2000U);
      EXPECT_EQ(r.count(3), static_cast<std::uint64_t>(triangles) - 2000U);
    }
    return best;
  };
  const double bounded = best_of_three(3);
  const double unbounded = best_of_three(-1);
  EXPECT_LT(unbounded, 10.0 * bounded + 0.05) << "bounded " << bounded << " s";
}

INSTANTIATE_TEST_SUITE_P(Backends, CycleCountRandom,
                         ::testing::ValuesIn(dyng::test::host_backends()),
                         [](const ::testing::TestParamInfo<backend>& info) {
                           return dyng::test::cc_name(info.param);
                         });

// DynamicUpdateOpenMPTest: one thread equals the sequential update; 2 and 4 threads equal the
// recount; results do not depend on the thread count (determinism, conformance C3 / C6).
TEST(CycleCountRandomOpenmp, ThreadCountsAgree) {
  if (!dyng::backend_available(backend::openmp)) {
    GTEST_SKIP() << "OpenMP is not built";
  }
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  const resources seq = resources::sequential();
  for (const std::uint64_t seed : test_seeds(654, 40)) {
    std::mt19937_64 rng(seed);
    SCOPED_TRACE(seed_trace(seed));
    std::vector<cc_edge> edges;
    for (std::int32_t u = 0; u < 8; ++u) {
      for (std::int32_t v = 0; v < 8; ++v) {
        if (u != v && coin(rng) < 0.4) {
          edges.emplace_back(u, v);
        }
      }
    }
    legacy::cycle_enum_batch_options params;
    params.num_deletions = 3;
    params.num_insertions = 3;
    params.seed = rng();
    graph_u gs = cc_graph<graph_u>(seq, 8, edges);
    batch_u batch;
    try {
      batch = legacy::cycle_enum_batch(gs.to_csr(seq).view(), params);
    } catch (const dyng::invalid_argument_error&) {
      continue;
    }
    cycle_count::result rs = cycle_count::compute(seq, gs, bound(6));
    const cycle_count::stats ss = cycle_count::update(seq, gs, batch.view(), rs);
    for (const int threads : {1, 2, 4, 7}) {
      const resources omp = resources::openmp(threads);
      graph_u g = cc_graph<graph_u>(omp, 8, edges);
      cycle_count::result r = cycle_count::compute(omp, g, bound(6));
      const cycle_count::stats so = cycle_count::update(omp, g, batch.view(), r);
      EXPECT_EQ(cc_counts(r), cc_counts(rs)) << "threads " << threads;
      EXPECT_EQ(so.cycles_added, ss.cycles_added);
      EXPECT_EQ(so.cycles_removed, ss.cycles_removed);
      EXPECT_EQ(so.affected, ss.affected);
    }
  }
}

}  // namespace
