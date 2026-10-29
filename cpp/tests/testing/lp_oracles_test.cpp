// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lp_oracles_test.cpp
 * @brief The label_propagation oracles of dyng::testing (ADR 0068): union_find, harmonic_solution
 *        (closed forms, excluded components, the thread-count independence, the comparison
 *        helpers) and lp_host_model (slot rows, the net effect, the flags, V_aff, G', the
 *        component numbering, the supernode weights and the initial labels, hand-computed). The
 *        replay against LabelPropagation-CUDA's traces is dyng_lp_replay_tests and
 *        parity/compare.py label_propagation model.
 */
#include <dyng/testing/harmonic_solution.hpp>
#include <dyng/testing/lp_host_model.hpp>
#include <dyng/testing/union_find.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using dyng::testing::harmonic_options;
using dyng::testing::harmonic_solution;
using dyng::testing::lp_host_model;
using dyng::testing::lp_model_batch;
using dyng::testing::union_find;

TEST(UnionFind, RootIsTheSmallestMember) {
  union_find uf(10);
  EXPECT_EQ(uf.size(), 10);
  EXPECT_TRUE(uf.unite(7, 9));
  EXPECT_TRUE(uf.unite(9, 3));
  EXPECT_FALSE(uf.unite(3, 7));
  EXPECT_TRUE(uf.unite(5, 4));
  EXPECT_EQ(uf.find(9), 3);
  EXPECT_EQ(uf.find(7), 3);
  EXPECT_EQ(uf.find(5), 4);
  EXPECT_TRUE(uf.unite(5, 9));
  for (const int v : {3, 4, 5, 7, 9}) {
    EXPECT_EQ(uf.find(v), 3);
  }
  EXPECT_TRUE(uf.same(4, 7));
  EXPECT_FALSE(uf.same(0, 1));
  EXPECT_THROW((void)uf.find(10), std::out_of_range);
  EXPECT_THROW(union_find(-1), std::invalid_argument);
}

TEST(UnionFind, IndependentOfTheUnionOrder) {
  std::mt19937 rng(5);
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i < 300; ++i) {
    edges.emplace_back(static_cast<int>(rng() % 200), static_cast<int>(rng() % 200));
  }
  union_find a(200);
  for (const auto& [u, v] : edges) {
    a.unite(u, v);
  }
  std::shuffle(edges.begin(), edges.end(), rng);
  union_find b(200);
  for (const auto& [u, v] : edges) {
    b.unite(v, u);
  }
  for (int v = 0; v < 200; ++v) {
    EXPECT_EQ(a.find(v), b.find(v));
    EXPECT_LE(a.find(v), v);
  }
}

/// A CSR from undirected weighted edges.
struct csr {
  std::vector<std::int64_t> row_ptr;
  std::vector<std::int32_t> col;
  std::vector<float> w;
};
csr make_csr(int n, const std::vector<std::tuple<int, int, float>>& edges) {
  std::vector<std::vector<std::pair<int, float>>> rows(static_cast<std::size_t>(n));
  for (const auto& [u, v, w] : edges) {
    rows[static_cast<std::size_t>(u)].emplace_back(v, w);
    rows[static_cast<std::size_t>(v)].emplace_back(u, w);
  }
  csr g;
  g.row_ptr.push_back(0);
  for (const auto& r : rows) {
    for (const auto& [v, w] : r) {
      g.col.push_back(v);
      g.w.push_back(w);
    }
    g.row_ptr.push_back(static_cast<std::int64_t>(g.col.size()));
  }
  return g;
}

TEST(HarmonicSolution, PathInterpolatesLinearly) {
  // 0 - 1 - 2 - 3 - 4 with unit weights, seeds 0 -> 0 and 4 -> 1: F* = i / 4.
  const csr g = make_csr(5, {{0, 1, 1.0F}, {1, 2, 1.0F}, {2, 3, 1.0F}, {3, 4, 1.0F}});
  const std::vector<std::uint8_t> present(5, 1);
  const std::vector<std::int8_t> seeds = {0, -1, -1, -1, 1};
  const auto f = harmonic_solution(g.row_ptr, g.col, g.w, present, seeds);
  ASSERT_TRUE(f.converged);
  EXPECT_LE(f.relative_residual, 1e-12);
  EXPECT_EQ(f.num_seeds, 2);
  EXPECT_EQ(f.num_unlabelled, 3);
  EXPECT_EQ(f.num_determined, 3);
  EXPECT_EQ(f.num_excluded, 0);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(f.solution[static_cast<std::size_t>(i)], i / 4.0, 1e-12);
  }
}

TEST(HarmonicSolution, WeightedStarAndExcludedComponents) {
  // Centre 1 between seed 0 (class 0, weight 3) and seed 2 (class 1, weight 1): F* = 1 / 4.
  // Vertices 3 - 4 form a seed-free component: excluded, 0.5. Vertex 5 is not present.
  const csr g = make_csr(6, {{0, 1, 3.0F}, {1, 2, 1.0F}, {3, 4, 1.0F}, {1, 5, 100.0F}});
  const std::vector<std::uint8_t> present = {1, 1, 1, 1, 1, 0};
  const std::vector<std::int8_t> seeds = {0, -1, 1, -1, -1, -1};
  const auto f = harmonic_solution(g.row_ptr, g.col, g.w, present, seeds);
  ASSERT_TRUE(f.converged);
  EXPECT_NEAR(f.solution[1], 0.25, 1e-12);
  EXPECT_EQ(f.solution[3], 0.5);
  EXPECT_EQ(f.solution[4], 0.5);
  EXPECT_EQ(f.num_excluded, 2);
  EXPECT_EQ(f.determined, (std::vector<std::uint8_t>{0, 1, 0, 0, 0, 0}));
  const std::vector<float> labels = {0.0F, 0.3F, 1.0F, 0.9F, 0.1F, 0.7F};
  const auto e = dyng::testing::label_error_against(labels, f, present, seeds);
  EXPECT_EQ(e.compared, 1);
  EXPECT_EQ(e.excluded, 2);
  EXPECT_NEAR(e.mean, 0.05, 1e-7);
  EXPECT_EQ(e.flips, 0);
  const std::vector<float> flipped = {0.0F, 0.6F, 1.0F, 0.0F, 0.0F, 0.0F};
  const auto c = dyng::testing::class_agreement_with(flipped, f, present, seeds, 0.2);
  EXPECT_EQ(c.flips, 1);
  EXPECT_EQ(c.flips_beyond_margin, 1);  // |0.25 - 0.5| = 0.25 > 0.2
  EXPECT_NEAR(c.largest_flip_distance, 0.25, 1e-12);
  EXPECT_EQ(
      dyng::testing::class_agreement_with(flipped, f, present, seeds, 0.3).flips_beyond_margin, 0);
}

TEST(HarmonicSolution, NoSeedNeighbourGivesZero) {
  // A component whose seeds are all of class 0: b = 0, x = 0 exactly, residual 0.
  const csr g = make_csr(3, {{0, 1, 0.5F}, {1, 2, 0.5F}});
  const auto f = harmonic_solution(g.row_ptr, g.col, g.w, std::vector<std::uint8_t>(3, 1),
                                   std::vector<std::int8_t>{0, -1, -1});
  EXPECT_TRUE(f.converged);
  EXPECT_EQ(f.iterations, 0);
  EXPECT_EQ(f.solution[1], 0.0);
  EXPECT_EQ(f.solution[2], 0.0);
}

TEST(HarmonicSolution, IndependentOfTheThreadCount) {
  // A random sparse graph large enough for two threads (50000 vertices per thread and more).
  const int n = 100001;
  std::mt19937 rng(11);
  std::vector<std::tuple<int, int, float>> edges;
  for (int i = 1; i < n; ++i) {
    edges.emplace_back(static_cast<int>(rng() % static_cast<unsigned>(i)), i,
                       0.01F + static_cast<float>(rng() % 1000) / 1000.0F);
  }
  for (int i = 0; i < n; ++i) {
    edges.emplace_back(static_cast<int>(rng() % n), static_cast<int>(rng() % n), 0.5F);
  }
  std::vector<std::tuple<int, int, float>> simple;
  for (const auto& e : edges) {
    if (std::get<0>(e) != std::get<1>(e)) {
      simple.push_back(e);
    }
  }
  const csr g = make_csr(n, simple);
  std::vector<std::uint8_t> present(static_cast<std::size_t>(n), 1);
  std::vector<std::int8_t> seeds(static_cast<std::size_t>(n), -1);
  for (int i = 0; i < n; i += 5) {
    seeds[static_cast<std::size_t>(i)] = static_cast<std::int8_t>((i / 5) % 2);
  }
  harmonic_options one;
  const auto a = harmonic_solution(g.row_ptr, g.col, g.w, present, seeds, one);
  harmonic_options four;
  four.num_threads = 4;
  const auto b = harmonic_solution(g.row_ptr, g.col, g.w, present, seeds, four);
  ASSERT_TRUE(a.converged);
  EXPECT_GT(a.iterations, 10);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.relative_residual, b.relative_residual);
  EXPECT_EQ(a.solution, b.solution);  // bitwise
}

TEST(HarmonicSolution, RejectsBadInput) {
  const csr g = make_csr(2, {{0, 1, 1.0F}});
  EXPECT_THROW((void)harmonic_solution(g.row_ptr, g.col, g.w, std::vector<std::uint8_t>(1, 1),
                                       std::vector<std::int8_t>(2, -1)),
               std::invalid_argument);
  EXPECT_THROW((void)harmonic_solution(g.row_ptr, g.col, g.w, std::vector<std::uint8_t>(2, 1),
                                       std::vector<std::int8_t>{2, -1}),
               std::invalid_argument);
  harmonic_options bad;
  bad.max_iterations = 0;
  EXPECT_THROW((void)harmonic_solution(g.row_ptr, g.col, g.w, std::vector<std::uint8_t>(2, 1),
                                       std::vector<std::int8_t>(2, -1), bad),
               std::invalid_argument);
}

lp_model_batch arrivals(std::vector<std::int32_t> ids, std::vector<std::int8_t> labels) {
  lp_model_batch b;
  b.insert_vertices = std::move(ids);
  b.insert_labels = std::move(labels);
  return b;
}

void add_edge(lp_model_batch& b, std::int32_t u, std::int32_t v, float w) {
  b.edge_sources.push_back(u);
  b.edge_destinations.push_back(v);
  b.edge_weights.push_back(w);
}

TEST(LpHostModel, SlotsNetEffectAndStages) {
  // Full-graph degrees: 0:2 1:3 2:2 3:1 4:2 5:2 (capacities); tau = 0.5.
  lp_host_model m({2, 3, 2, 1, 2, 2});
  EXPECT_EQ(m.row_ptr(), (std::vector<std::int64_t>{0, 2, 5, 7, 8, 10, 12}));
  // b0: seeds 0 (class 0) and 1 (class 1), edge 0-1.
  lp_model_batch b0 = arrivals({0, 1}, {0, 1});
  add_edge(b0, 0, 1, 0.75F);
  auto r0 = m.apply(b0, 0.5F);
  EXPECT_EQ(r0.inserted, (std::vector<std::int32_t>{0, 1}));
  EXPECT_TRUE(r0.affected.empty());                             // seeds are not affected
  EXPECT_EQ(r0.sparsified, (std::vector<std::int32_t>{0, 1}));  // 0.75 > tau, both new
  EXPECT_EQ(r0.num_components, 1);
  EXPECT_EQ(r0.component_of_inserted, (std::vector<std::int32_t>{0, 0}));
  // W: vertex 0's entry 1 (class 1) and vertex 1's entry 0 (class 0), 0.75 * 2^32 each.
  EXPECT_EQ(r0.w0, (std::vector<std::uint64_t>{3221225472ULL}));
  EXPECT_EQ(r0.w1, (std::vector<std::uint64_t>{3221225472ULL}));
  EXPECT_EQ(r0.initial_labels, (std::vector<float>{0.0F, 1.0F}));
  EXPECT_EQ(m.row_end()[0], 1);
  EXPECT_EQ(m.slot_columns()[0], 1);
  EXPECT_EQ(m.slot_columns()[1], -1);

  // b1: 2, 3, 4 arrive (unlabelled), 5 arrives and departs (Ins ∩ Del: ends deleted); edges
  // 1-2 (0.25), 2-4 (0.9: G'), 3-1 (0.6), 4-5 (dropped: 5 absent), 2-3 (0.5: not > tau).
  lp_model_batch b1 = arrivals({2, 3, 4, 5}, {-1, -1, -1, -1});
  b1.delete_vertices = {5};
  add_edge(b1, 1, 2, 0.25F);
  add_edge(b1, 2, 4, 0.9F);
  add_edge(b1, 1, 3, 0.6F);
  add_edge(b1, 4, 5, 0.3F);
  add_edge(b1, 2, 3, 0.5F);
  auto r1 = m.apply(b1, 0.5F);
  EXPECT_EQ(r1.inserted, (std::vector<std::int32_t>{2, 3, 4}));
  EXPECT_TRUE(r1.deleted.empty());  // 5 was not in V_t
  EXPECT_EQ(m.states()[5], lp_host_model::vertex_state::deleted);
  EXPECT_EQ(r1.dropped_absent, 1);
  EXPECT_EQ(r1.dropped_overflow, 2);  // 2-3 finds rows 2 and 3 full (2 and 1 slots)
  EXPECT_EQ(m.row_overflow_flag(), 1);
  EXPECT_EQ(r1.affected, (std::vector<std::int32_t>{2, 3, 4}));
  EXPECT_EQ(r1.sparsified, (std::vector<std::int32_t>{2, 4}));
  // Components on [2, 3, 4]: {2, 4} root 2 -> id 0, {3} -> id 1.
  EXPECT_EQ(r1.num_components, 2);
  EXPECT_EQ(r1.component_of_inserted, (std::vector<std::int32_t>{0, 1, 0}));
  // Component 0: vertex 2's row [1 (0.25, class 1), 4]; vertex 4's row [2]: W1 = 0.25 * 2^32.
  // Component 1: vertex 3's row [1 (0.6f = 10066330 * 2^-24)]: W1 = 10066330 * 2^8.
  EXPECT_EQ(r1.w0, (std::vector<std::uint64_t>{0, 0}));
  EXPECT_EQ(r1.w1, (std::vector<std::uint64_t>{1073741824ULL, 2576980480ULL}));
  EXPECT_EQ(r1.initial_labels, (std::vector<float>{1.0F, 1.0F, 1.0F}));
  // Row 2 holds 1, 4 in the batch's edge order; row 3 holds 1 only (the rest overflowed).
  EXPECT_EQ(m.slot_columns()[5], 1);
  EXPECT_EQ(m.slot_columns()[6], 4);
  EXPECT_EQ(m.row_end()[2], 7);
  EXPECT_EQ(m.row_end()[3], 8);

  // b2: 4 departs (Del_eff); its pre-batch row [2] makes 2 affected. Re-inserting 5 (retired)
  // raises the reinserted flag.
  lp_model_batch b2 = arrivals({5}, {-1});
  b2.delete_vertices = {4};
  auto r2 = m.apply(b2, 0.5F);
  EXPECT_EQ(r2.deleted, (std::vector<std::int32_t>{4}));
  EXPECT_EQ(m.reinserted_flag(), 6);
  EXPECT_EQ(m.previously_present()[4], 1);
  EXPECT_EQ(m.present()[4], 0);
  EXPECT_EQ(r2.affected, (std::vector<std::int32_t>{2, 5}));
}

TEST(LpHostModel, WeightRoundingAndNeutralLabels) {
  // An inserted vertex whose only neighbours are unlabelled starts at 0.5; a weight whose scaled
  // value is not an integer rounds to nearest even.
  lp_host_model m({1, 2, 1});
  lp_model_batch b0 = arrivals({0}, {1});
  auto r0 = m.apply(b0, 0.5F);
  EXPECT_EQ(r0.initial_labels, (std::vector<float>{1.0F}));
  lp_model_batch b1 = arrivals({1, 2}, {-1, -1});
  add_edge(b1, 0, 1, 1e-10F);  // 1e-10 * 2^32 = 0.4295: rounds to 0
  add_edge(b1, 1, 2, 0.3F);
  auto r1 = m.apply(b1, 0.5F);
  EXPECT_EQ(r1.num_components, 2);
  EXPECT_EQ(r1.w1[0], 0U);
  EXPECT_EQ(r1.initial_labels, (std::vector<float>{0.5F, 0.5F}));
  EXPECT_EQ(lp_host_model::average_weight({1, 0}, {0.25F, 0.75F}), 0.5F);
  EXPECT_EQ(lp_host_model::average_weight({}, {}), 0.5F);
}

TEST(LpHostModel, RefusesBadBatchesBeforeAnyChange) {
  lp_host_model m({1, 1});
  EXPECT_THROW((void)m.apply(arrivals({2}, {-1}), 0.5F), std::invalid_argument);
  EXPECT_THROW((void)m.apply(arrivals({0}, {}), 0.5F), std::invalid_argument);
  EXPECT_THROW((void)m.apply(arrivals({0}, {3}), 0.5F), std::invalid_argument);
  lp_model_batch seeded_and_deleted = arrivals({0}, {1});
  seeded_and_deleted.delete_vertices = {0};
  EXPECT_THROW((void)m.apply(seeded_and_deleted, 0.5F), std::invalid_argument);
  lp_model_batch bad_edge = arrivals({0}, {-1});
  add_edge(bad_edge, 0, 7, 0.5F);
  EXPECT_THROW((void)m.apply(bad_edge, 0.5F), std::invalid_argument);
  EXPECT_EQ(m.states()[0], lp_host_model::vertex_state::inactive);
  EXPECT_THROW(lp_host_model({-1}), std::invalid_argument);
}

}  // namespace
