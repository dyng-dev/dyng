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
 */
#include "support/cycle_count_support.hpp"

#include <dyng/core/error.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
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
namespace cycle_count = dyng::cycle_count;
namespace legacy = dyng::generators::legacy;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_u32 = dyng::graph<std::int32_t, std::int32_t, unweighted>;
using graph_w = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using hist = std::vector<std::uint64_t>;
using batch_u = dyng::edge_batch<std::int32_t, unweighted>;

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
  std::mt19937_64 rng(20260925);
  for (int trial = 0; trial < 400; ++trial) {
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
  std::mt19937_64 rng(99);
  struct shape {
    std::int64_t vertices;
    double probability;
    std::int64_t hubs;
    int k;
  };
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

// RandomizedUpdateParityTest.ValidBatchesMatchRecount: generated (valid) batches.
TEST_P(CycleCountRandom, GeneratedBatchesMatchRecount) {
  std::mt19937_64 rng(8675309);
  int checked = 0;
  for (int trial = 0; trial < 300; ++trial) {
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
  EXPECT_GT(checked, 200);
}

// RandomizedUpdateParityTest.LargerGraphsMatchBruteForceRecount.
TEST_P(CycleCountRandom, LargerGraphsMatchBruteForceRecount) {
  std::mt19937_64 rng(1234);
  for (int trial = 0; trial < 12; ++trial) {
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
  std::mt19937_64 rng(4711);
  for (int trial = 0; trial < 200; ++trial) {
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
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  int checked = 0;
  for (int trial = 0; trial < 200; ++trial) {
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
    ASSERT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(k))))
        << "trial " << trial;
    ++checked;
  }
  EXPECT_GT(checked, 100);
}

// DynamicUpdateParityTest.AllDeleteAndAllInsertMatchRecompute.
TEST_P(CycleCountRandom, DeleteOnlyAndInsertOnlyMatchRecompute) {
  std::mt19937_64 rng(99);
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
  std::mt19937_64 rng(31337);
  for (int trial = 0; trial < 60; ++trial) {
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
    SCOPED_TRACE(::testing::Message() << "trial " << trial);
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
  std::mt19937_64 rng(777);
  for (const auto& [name, props] : variants) {
    for (int trial = 0; trial < 25; ++trial) {
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
                  dyng::test::cc_resize(dyng::test::cc_oracle(res_, g, k), r.counts().size()));
      }
    }
  }
}

TEST_P(CycleCountRandom, InverseBatchRestoresTheHistogram) {
  std::mt19937_64 rng(4242);
  for (int trial = 0; trial < 40; ++trial) {
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
    EXPECT_EQ(cc_counts(r), before) << "trial " << trial;
    EXPECT_EQ(forward.cycles_added, back.cycles_removed);
    EXPECT_EQ(forward.cycles_removed, back.cycles_added);
  }
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
  std::mt19937_64 rng(654);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  const resources seq = resources::sequential();
  for (int trial = 0; trial < 40; ++trial) {
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
      EXPECT_EQ(cc_counts(r), cc_counts(rs)) << "trial " << trial << " threads " << threads;
      EXPECT_EQ(so.cycles_added, ss.cycles_added);
      EXPECT_EQ(so.cycles_removed, ss.cycles_removed);
      EXPECT_EQ(so.affected, ss.affected);
    }
  }
}

}  // namespace
