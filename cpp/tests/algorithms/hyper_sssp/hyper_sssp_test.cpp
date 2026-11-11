// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_test.cpp
 * @brief The hand cases of hyper_sssp (PLAN Section 4.8, file 12; M11a step 4): compute() and
 *        update() against dyng::testing's oracle, the public source-vertex mode against the
 *        virtual-endpoint mode on H + {s} (HD2), distance_to() and vertex_distances() (HD11),
 *        the requirements and the refusals (HD9, HD14), the weight limit (HD11), the work budget
 *        and its three outcomes (HD8; ADR 0092), stale results and clones. Compiled into the host
 *        test executable (sequential; openmp is refused) and, with DYNG_TEST_CUDA=1, into the CUDA
 *        one, where every case runs on both CUDA engines: the as-is engine (engine::fused,
 *        MOSP_ESCHER's kernels) and the shared sssp engine (engine::operators, M11b; ADR 0095).
 */
#include "algorithms/hyper_sssp/hyper_sssp_support.hpp"
#include "algorithms/hyper_sssp/virtual_endpoints.hpp"
#include "framework/budgets.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::resources;
using namespace dyng::test::hsssp;  // NOLINT(google-build-using-namespace): the test helpers
namespace hs = dyng::hyper_sssp;
namespace hv = dyng::detail::hyper_sssp_virtual;
using rows = dyng::test::hyper::rows;
constexpr std::int64_t inf = std::numeric_limits<std::int64_t>::max() / 4;

/// A backend and, on cuda, the engine of every compute() of a case.
struct config {
  backend b = backend::sequential;
  dyng::engine e = dyng::engine::automatic;
};

/// The configurations of the executable: the host backends, or cuda with each engine.
std::vector<config> configs() {
  std::vector<config> out;
  for (const backend b : dyng::test::suite_backends()) {
    if (b == backend::cuda) {
      out.push_back({b, dyng::engine::fused});
      out.push_back({b, dyng::engine::operators});
    } else {
      out.push_back({b, dyng::engine::automatic});
    }
  }
  return out;
}

class HyperSssp : public ::testing::TestWithParam<config> {
 protected:
  void SetUp() override {
    if (be() == backend::openmp) GTEST_SKIP() << "openmp is refused (HD14): RefusesOpenmp";
  }
  backend be() const {
    return GetParam().b;
  }
  /// The shared engine (engine::operators on cuda).
  bool shared() const {
    return GetParam().b == backend::cuda && GetParam().e == dyng::engine::operators;
  }
  /// `o` with the case's engine.
  hs::options opts(hs::options o = {}) const {
    o.cuda_engine = GetParam().e;
    return o;
  }
  resources res() const {
    return dyng::test::hyper::resources_of(be());
  }
  /// A path 0-1-2-3 of hyperedges and a heavy shortcut 4 = {0, 9} to hyperedge 3's vertex 9 (12
  /// vertices; vertex 11 in no hyperedge).
  whgraph path() const {
    return make(res(), 12, rows{{0, 1}, {1, 2}, {2, 3}, {3, 9}, {0, 9}, {10}}, {1, 2, 3, 4, 20, 1});
  }
};

TEST(HyperSsspBackends, RefusesOpenmp) {
  if (!dyng::backend_available(backend::openmp)) GTEST_SKIP() << "built without OpenMP";
  const resources omp = dyng::test::hyper::resources_of(backend::openmp);
  const whgraph h = make(omp, 4, rows{{0, 1}, {1, 2}}, {1, 1});
  try {
    (void)hs::compute(omp, h, 0);
    FAIL() << "openmp was accepted";
  } catch (const dyng::not_supported_error& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("sequential"), std::string::npos) << m;
    EXPECT_NE(m.find("cuda"), std::string::npos) << m;
    EXPECT_NE(m.find("dyng.Resources(\"sequential\")"), std::string::npos) << m;
  }
}

TEST_P(HyperSssp, ComputeEqualsTheOracleAndTheDistancesIncludeTheWeights) {
  const whgraph h = path();
  const result_t r = hs::compute(res(), h, 0, opts());
  EXPECT_TRUE(matches_oracle(res(), h, r, 0, true));
  const auto [d, p] = host_of(res(), r);
  ASSERT_EQ(d.size(), 6U);
  EXPECT_EQ(d[0], 1);  // holds the source: its own weight, parent -1
  EXPECT_EQ(p[0], -1);
  EXPECT_EQ(d[4], 20);  // holds the source too
  EXPECT_EQ(p[4], -1);
  EXPECT_EQ(d[3], 10);  // 0 -> 1 -> 2 -> 3: 1 + 2 + 3 + 4 (not through 4: 20 + 4)
  EXPECT_EQ(p[3], 2);
  EXPECT_EQ(d[5], inf);  // unreachable
  EXPECT_EQ(p[5], -1);
  // distance_to(v): min over the live hyperedges of v of D(h), 0 for the source (HD11).
  EXPECT_EQ(r.distance_to(0), 0);
  EXPECT_EQ(r.distance_to(1), 1);
  EXPECT_EQ(r.distance_to(9), 10);
  EXPECT_EQ(r.distance_to(10), inf);
  EXPECT_EQ(r.distance_to(11), inf);
  const auto vd = to_vector(res(), r.vertex_distances());
  ASSERT_EQ(vd.size(), 12U);
  EXPECT_EQ(vd[0], 0);
  EXPECT_EQ(vd[3], 6);
  EXPECT_EQ(vd[9], 10);
  EXPECT_THROW((void)r.distance_to(12), dyng::invalid_argument_error);
  EXPECT_EQ(r.source(), 0);
  EXPECT_EQ(r.graph_version(), h.version());
}

TEST_P(HyperSssp, UpdatesEqualComputeAndTheOracle) {
  std::mt19937_64 rng(be() == backend::cuda ? 41 : 42);
  for (int trial = 0; trial < 6; ++trial) {
    const std::int32_t n = 30 + trial * 7;
    rows init;
    std::vector<std::int32_t> w;
    std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
    std::uniform_int_distribution<std::int32_t> weight(1, trial % 2 == 0 ? 9 : 1 << 20);
    for (int i = 0; i < 40; ++i) {
      init.push_back({vertex(rng), vertex(rng), vertex(rng)});
      w.push_back(weight(rng));
    }
    init.push_back({0, 1});  // id 40 keeps the source
    w.push_back(1);
    whgraph h =
        make(res(), n, init, w,
             slack(trial % 3 == 0 ? dyng::id_reuse::insert_first : dyng::id_reuse::erase_first));
    result_t r = hs::compute(res(), h, 0, opts());
    ASSERT_TRUE(matches_oracle(res(), h, r, 0, true));
    for (int t = 0; t < 8; ++t) {
      const wbatch b = random_batch(rng, h, 40, trial % 2 == 0 ? 9 : 1 << 20);
      const hs::stats s = hs::update(res(), h, b.view(), r);
      EXPECT_TRUE(s.converged);
      EXPECT_EQ(s.batch.inserted_hyperedges, static_cast<std::int64_t>(b.view().num_insertions()));
      ASSERT_TRUE(matches_oracle(res(), h, r, 0, true)) << "trial " << trial << " batch " << t;
      const result_t c = hs::compute(res(), h, 0, opts());
      ASSERT_EQ(host_of(res(), r), host_of(res(), c)) << "update != compute";
    }
  }
}

TEST_P(HyperSssp, ThePublicModeEqualsTheVirtualModeOnHPlusTheSourceRow) {
  // HD2: hyper_sssp from vertex s on H equals the virtual-endpoint mode on H + {s} (the source
  // row appended, weight 0 inside hyper_sssp), with the source row's id as parent read as -1.
  std::mt19937_64 rng(7);
  const std::int32_t n = 25;
  rows init;
  std::vector<std::int32_t> w;
  std::uniform_int_distribution<std::int32_t> vertex(1, n - 1);
  for (int i = 0; i < 30; ++i) {
    init.push_back({vertex(rng), vertex(rng)});
    w.push_back(1 + i % 5);
  }
  init.push_back({0, 3});
  w.push_back(2);
  init.push_back({0, 7, 8});
  w.push_back(3);
  rows plus = init;
  std::vector<std::int32_t> wp = w;
  plus.push_back({0});  // the source row, id 32
  wp.push_back(1);
  whgraph a = make(res(), n, init, w);
  whgraph b = make(res(), n, plus, wp);
  dyng::detail::hyper_sssp_endpoints ends;
  ends.source_hyperedge = 32;
  ends.zero_weight_a = 32;
  result_t ra = hs::compute(res(), a, 0, opts());
  result_t rb = hv::compute(res(), b, ends, opts());
  const auto same = [&](const char* where) {
    const auto [da, pa] = host_of(res(), ra);
    const auto [db, pb] = host_of(res(), rb);
    for (std::size_t i = 0; i < da.size(); ++i) {
      ASSERT_EQ(da[i], db[i]) << where << ": distance of " << i;
      ASSERT_EQ(pa[i], pb[i] == 32 ? -1 : pb[i]) << where << ": parent of " << i;
    }
  };
  same("compute");
  for (int t = 0; t < 10; ++t) {
    // The same batch on both (never touching the source row; s gained and lost by edits).
    wbatch x = random_batch(rng, a, 30, 5, 3);
    if (t % 2 == 0) x.insert_incidence(static_cast<std::int64_t>(t), 0);
    if (t % 3 == 0) x.delete_incidence(31, 0);
    if (t % 3 == 1) x.insert_incidence(31, 0);
    const hs::stats sa = hs::update(res(), a, x.view(), ra);
    const hs::stats sb = hs::update(res(), b, x.view(), rb);
    EXPECT_EQ(sa.invalidated, sb.invalidated);
    same(("batch " + std::to_string(t)).c_str());
    ASSERT_TRUE(matches_oracle(res(), a, ra, 0, true));
    ASSERT_TRUE(matches_oracle(res(), b, rb, 32, false, {32}));
  }
}

TEST_P(HyperSssp, TheRequirementsAreChecked) {
  const resources r = res();
  // escher storage: not_supported_error naming slack_csr.
  dyng::hypergraph_properties escher;
  escher.line_graph = dyng::line_graph_policy::automatic;
  const whgraph e = make(r, 4, rows{{0, 1}, {1, 2}}, {1, 1}, escher);
  try {
    (void)hs::compute(r, e, 0, opts());
    FAIL() << "escher storage was accepted";
  } catch (const dyng::not_supported_error& x) {
    EXPECT_NE(std::string(x.what()).find("slack_csr"), std::string::npos) << x.what();
  }
  // No line graph.
  dyng::hypergraph_properties none = slack();
  none.line_graph = dyng::line_graph_policy::none;
  const whgraph h0 = make(r, 4, rows{{0, 1}, {1, 2}}, {1, 1}, none);
  EXPECT_THROW((void)hs::compute(r, h0, 0, opts()), dyng::not_supported_error);
  // line_graph_policy::automatic builds it (and keeps it under the updates).
  dyng::hypergraph_properties automatic = slack();
  automatic.line_graph = dyng::line_graph_policy::automatic;
  whgraph ha = make(r, 4, rows{{0, 1}, {1, 2}}, {1, 1}, automatic);
  result_t ra = hs::compute(r, ha, 0, opts());
  wbatch b;
  b.insert_hyperedge({2, 3}, 5);
  (void)hs::update(r, ha, b.view(), ra);
  EXPECT_EQ(ra.distance_to(3), 7);
  // The source out of range; invalid options.
  const whgraph h = path();
  EXPECT_THROW((void)hs::compute(r, h, -1), dyng::invalid_argument_error);
  EXPECT_THROW((void)hs::compute(r, h, 12, opts()), dyng::invalid_argument_error);
  hs::options o;
  o.work_budget = -1;
  EXPECT_THROW((void)hs::compute(r, h, 0, opts(o)), dyng::invalid_argument_error);
  o = {};
  o.work_budget = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)hs::compute(r, h, 0, opts(o)), dyng::invalid_argument_error);
  o = {};
  o.max_iterations = 0;
  EXPECT_THROW((void)hs::compute(r, h, 0, opts(o)), dyng::invalid_argument_error);
  o = {};
  o.on_budget = static_cast<hs::budget_action>(7);
  EXPECT_THROW((void)hs::compute(r, h, 0, opts(o)), dyng::invalid_argument_error);
  // Every engine runs everywhere: on cuda fused and operators are the two engines (M11b), on the
  // host the choice is ignored.
  for (const dyng::engine e :
       {dyng::engine::automatic, dyng::engine::fused, dyng::engine::operators}) {
    o = {};
    o.cuda_engine = e;
    EXPECT_NO_THROW((void)hs::compute(r, h, 0, o));
  }
}

TEST_P(HyperSssp, WeightsAboveTwoToThe28AreRefusedBeforeTheCommit) {
  const resources r = res();
  constexpr std::int32_t heavy = (1 << 28) + 1;
  const whgraph h1 = make(r, 4, rows{{0, 1}, {1, 2}}, {1, heavy});
  EXPECT_THROW((void)hs::compute(r, h1, 0, opts()), dyng::invalid_argument_error);
  hs::options off;
  off.validate_inputs = false;
  const result_t loose = hs::compute(r, h1, 0, opts(off));  // still exact (the 64-bit fallback)
  EXPECT_EQ(loose.distance_to(2), 1 + static_cast<std::int64_t>(heavy));
  whgraph h = make(r, 4, rows{{0, 1}, {1, 2}}, {1, 1 << 28});  // 2^28 is allowed
  result_t res28 = hs::compute(r, h, 0, opts());
  wbatch b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({2, 3}, heavy);
  const std::uint64_t v = h.version();
  EXPECT_THROW((void)hs::update(r, h, b.view(), res28), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), v);  // strong guarantee
  EXPECT_EQ(h.num_hyperedges(), 2);
  EXPECT_EQ(res28.graph_version(), v);
}

TEST_P(HyperSssp, WeightsOfTwoToThe28OnLongPathsStayExact) {
  // 40 hyperedges of weight 2^28 in a chain: the distances pass 2^32 - 1 (the CUDA engine's packed
  // width) from the 16th on, so its update falls back (distance_width); the host is exact as is.
  const resources r = res();
  rows chain;
  std::vector<std::int32_t> w;
  for (int i = 0; i < 40; ++i) {
    chain.push_back({i, i + 1});
    w.push_back(1 << 28);
  }
  whgraph h = make(r, 42, chain, w);
  result_t sp = hs::compute(r, h, 0, opts());
  EXPECT_EQ(sp.distance_to(40), 40 * (std::int64_t{1} << 28));
  wbatch b;
  b.delete_hyperedge(20);
  b.insert_hyperedge({20, 21}, 1 << 28);
  const hs::stats s = hs::update(r, h, b.view(), sp);
  ASSERT_TRUE(matches_oracle(r, h, sp, 0, true));
  if (be() == backend::cuda && !shared()) {
    EXPECT_TRUE(s.fallback_used);
    EXPECT_EQ(s.fallback_reason, hs::fallback_reason::distance_width);
  } else {
    // The host engine is exact as is; the shared engine's words hold n x 2^28 and the parents
    // (sssp's packing, ADR 0095).
    EXPECT_FALSE(s.fallback_used);
  }
}

TEST_P(HyperSssp, TheSharedEngineFallsBackWhenItsWordsCannotHoldTheParents) {
  // Without validate_inputs the packing must hold 2^31 - 1 per hyperedge: with 2^17 or more ids
  // the bound and the parent bits pass 64 bits, so the shared engine's compute() takes the ported
  // recompute and its update falls back (distance_width), exactly (ADR 0095). A binary tree of
  // 140,000 two-vertex hyperedges keeps the line graph sparse and shallow.
  if (!shared()) GTEST_SKIP() << "the shared engine's packing (engine::operators on cuda)";
  const resources r = res();
  constexpr std::int32_t m = 140000;
  rows tree;
  std::vector<std::int32_t> w;
  for (std::int32_t v = 1; v <= m; ++v) {
    tree.push_back({(v - 1) / 2, v});
    w.push_back(1 + v % 7);
  }
  whgraph h = make(r, m + 1, tree, w);
  hs::options o;
  o.validate_inputs = false;
  result_t sp = hs::compute(r, h, 0, opts(o));
  ASSERT_TRUE(matches_oracle(r, h, sp, 0, true));
  wbatch b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({0, 4}, 1);
  const hs::stats s = hs::update(r, h, b.view(), sp);
  EXPECT_TRUE(s.fallback_used);
  EXPECT_EQ(s.fallback_reason, hs::fallback_reason::distance_width);
  ASSERT_TRUE(matches_oracle(r, h, sp, 0, true));
  // With validate_inputs the bound is the largest weight (7): the words hold the parents.
  whgraph h2 = make(r, m + 1, tree, w);
  result_t s2 = hs::compute(r, h2, 0, opts());
  const hs::stats t = hs::update(r, h2, b.view(), s2);
  EXPECT_FALSE(t.fallback_used);
  ASSERT_TRUE(matches_oracle(r, h2, s2, 0, true));
  EXPECT_EQ(host_of(r, sp), host_of(r, s2));
}

TEST_P(HyperSssp, TheWorkBudgetRecomputesOrThrows) {
  const resources r = res();
  // work_budget 0: an update with work falls back (MOSP_ESCHER's test_hsosp_matches_dijkstra
  // pattern); +infinity never does.
  for (const double budget : {0.0, std::numeric_limits<double>::infinity()}) {
    whgraph h = path();
    hs::options o;
    o.work_budget = budget;
    result_t sp = hs::compute(r, h, 0, opts(o));
    wbatch b;
    b.delete_hyperedge(1);
    b.insert_hyperedge({1, 9}, 1);
    const hs::stats s = hs::update(r, h, b.view(), sp);
    EXPECT_EQ(s.fallback_used, budget == 0.0);
    EXPECT_EQ(s.fallback_reason,
              budget == 0.0 ? hs::fallback_reason::work_budget : hs::fallback_reason::none);
    EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  }
  // on_budget = error: convergence_error after the commit, the result poisoned (basic).
  whgraph h = path();
  hs::options o;
  o.work_budget = 0;
  o.on_budget = hs::budget_action::error;
  result_t sp = hs::compute(r, h, 0, opts(o));
  wbatch b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({1, 9}, 1);
  const std::uint64_t v = h.version();
  EXPECT_THROW((void)hs::update(r, h, b.view(), sp), dyng::convergence_error);
  EXPECT_EQ(h.version(), v + 1);  // the hypergraph holds G_{t+1}
  EXPECT_THROW((void)sp.hyperedge_distances(), dyng::stale_result_error);
  // max_iterations 1 with on_budget = error (cuda: push rounds; the host engine has none).
  whgraph h2 = path();
  hs::options it;
  it.max_iterations = 1;
  it.work_budget = std::numeric_limits<double>::infinity();  // only the iteration trigger
  it.on_budget = hs::budget_action::error;
  result_t s2 = hs::compute(r, h2, 0, opts(it));
  wbatch c;
  c.delete_hyperedge(0);  // everything below the source's hyperedge 0 is invalidated
  c.insert_hyperedge({0, 1}, 1);
  if (be() == backend::cuda) {
    EXPECT_THROW((void)hs::update(r, h2, c.view(), s2), dyng::convergence_error);
  } else {
    const hs::stats s = hs::update(r, h2, c.view(), s2);
    EXPECT_FALSE(s.fallback_used);
    EXPECT_TRUE(matches_oracle(r, h2, s2, 0, true));
  }
}

/// The triggers of HD8 at deterministic points well away from their edges: a path of m hyperedges
/// {i, i + 1} of weight 2 (the source vertex 0 in hyperedge 0 only) and a batch inserting {0, 1} at
/// weight 1, which lowers every distance but hyperedge 0's by one, one hyperedge per push round.
struct budget_case {
  static constexpr std::int32_t m = 64;
  static whgraph make_path(const resources& res) {
    rows r;
    std::vector<std::int32_t> w;
    for (std::int32_t i = 0; i < m; ++i) {
      r.push_back({i, i + 1});
      w.push_back(2);
    }
    return make(res, m + 1, r, w);
  }
  static wbatch batch() {
    wbatch b;
    b.insert_hyperedge({0, 1}, 1);
    return b;
  }
  /// The live line-graph entries after the batch (each pair in both rows): the path's m - 1
  /// pairs and the new hyperedge's two.
  static constexpr double entries = 2.0 * (m - 1 + 2);
};

TEST_P(HyperSssp, TheWorkTriggerFiresAboveTheLimitAndNotBelow) {
  const resources r = res();
  // The work of the batch without a budget.
  whgraph h0 = budget_case::make_path(r);
  hs::options free;
  free.work_budget = std::numeric_limits<double>::infinity();
  result_t s0 = hs::compute(r, h0, 0, opts(free));
  const hs::stats base = hs::update(r, h0, budget_case::batch().view(), s0);
  ASSERT_FALSE(base.fallback_used);
  const double work = static_cast<double>(base.relaxations);
  ASSERT_GE(work, 2.0 * (budget_case::m - 2)) << "the batch relaxes the whole path";
  // work > work_budget x max(1, live entries): about 1.25x the limit must fall back, about 0.8x
  // must not (a scaled limit or a different count of the entries moves one of the two).
  for (const double ratio : {0.8, 1.25}) {
    whgraph h = budget_case::make_path(r);
    hs::options o;
    o.work_budget = work / (ratio * budget_case::entries);
    result_t sp = hs::compute(r, h, 0, opts(o));
    const hs::stats s = hs::update(r, h, budget_case::batch().view(), sp);
    EXPECT_EQ(s.fallback_used, ratio > 1.0) << "work " << work << " at " << ratio << "x the limit";
    EXPECT_EQ(s.fallback_reason,
              ratio > 1.0 ? hs::fallback_reason::work_budget : hs::fallback_reason::none);
    EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  }
}

TEST_P(HyperSssp, TheIterationTriggerAllowsExactlyMaxIterationsPushRounds) {
  if (be() != backend::cuda) GTEST_SKIP() << "push rounds are the cuda engines' (HD8)";
  const resources r = res();
  whgraph h0 = budget_case::make_path(r);
  hs::options free;
  free.work_budget = std::numeric_limits<double>::infinity();
  result_t s0 = hs::compute(r, h0, 0, opts(free));
  const hs::stats base = hs::update(r, h0, budget_case::batch().view(), s0);
  ASSERT_FALSE(base.fallback_used);
  const std::int64_t k = base.iterations;
  ASSERT_GE(k, budget_case::m - 2) << "one push round per hyperedge of the path";
  // The cap is tested before each push round: k rounds are allowed with max_iterations = k, and
  // k - 1 falls back (or throws under budget_action::error).
  for (const std::int64_t cap : {k, k - 1}) {
    whgraph h = budget_case::make_path(r);
    hs::options o = free;
    o.max_iterations = cap;
    result_t sp = hs::compute(r, h, 0, opts(o));
    const hs::stats s = hs::update(r, h, budget_case::batch().view(), sp);
    EXPECT_EQ(s.fallback_used, cap < k) << "k " << k << ", max_iterations " << cap;
    EXPECT_EQ(s.fallback_reason,
              cap < k ? hs::fallback_reason::iterations : hs::fallback_reason::none);
    EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  }
}

TEST_P(HyperSssp, AnEmptyHypergraphComputesAndUpdates) {
  // No hyperedge: the result's state is never sized 0 (a launch of zero blocks would leave a
  // pending CUDA error for the next checked call, the store's commit), and insertions then work.
  const resources r = res();
  whgraph h = make(r, 5, rows{}, {});
  result_t sp = hs::compute(r, h, 0, opts());
  EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  EXPECT_EQ(sp.distance_to(0), 0);
  EXPECT_EQ(sp.distance_to(1), inf);
  wbatch b;
  b.insert_hyperedge({0, 1}, 3);
  b.insert_hyperedge({1, 2}, 4);
  const hs::stats s = hs::update(r, h, b.view(), sp);
  EXPECT_EQ(s.batch.inserted_hyperedges, 2);
  EXPECT_NO_THROW(h.check_integrity(r));
  EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  EXPECT_EQ(sp.distance_to(2), 7);
  const result_t c = hs::compute(r, h, 0, opts());
  EXPECT_EQ(host_of(r, sp), host_of(r, c));
}

TEST_P(HyperSssp, StaleResultsAndAccessorsOfAGoneHypergraph) {
  const resources r = res();
  std::optional<whgraph> h(path());
  result_t sp = hs::compute(r, *h, 0, opts());
  wbatch b;
  b.insert_hyperedge({5, 6}, 1);
  (void)h->apply(r, b.view());
  EXPECT_THROW((void)hs::update(r, *h, wbatch{}.view(), sp), dyng::stale_result_error);
  EXPECT_THROW((void)sp.distance_to(1), dyng::stale_result_error);  // the hypergraph changed
  result_t sq = hs::compute(r, *h, 0, opts());
  EXPECT_EQ(sq.distance_to(1), 1);
  h.reset();
  EXPECT_THROW((void)sq.distance_to(1), dyng::stale_result_error);  // it no longer exists
  EXPECT_NO_THROW((void)sq.hyperedge_distances());                  // the arrays are the result's
}

TEST_P(HyperSssp, ClonesKeepTheResultOnEveryBackend) {
  const resources r = res();
  whgraph h = path();
  result_t sp = hs::compute(r, h, 0, opts());
  result_t host = sp.clone(resources::sequential());
  EXPECT_EQ(host.space(), dyng::memory_space::host);
  EXPECT_EQ(host_of(resources::sequential(), host), host_of(r, sp));
  // The clone updates on the hypergraph moved to its backend.
  whgraph hh = h.to_backend(resources::sequential());
  wbatch b;
  b.delete_hyperedge(2);
  (void)hs::update(resources::sequential(), hh, b.view(), host);
  (void)hs::update(r, h, b.view(), sp);
  EXPECT_EQ(host_of(resources::sequential(), host), host_of(r, sp));
  EXPECT_THROW((void)sp.clone(dyng::test::hyper::resources_of(backend::openmp)),
               dyng::not_supported_error);
}

TEST_P(HyperSssp, AGrowthOfTheResultsStateStaysWithinTheBudget) {
  // Invariant I9 under strict budgets (an excess throws internal_error in a DYNG_DEBUG_BUDGETS
  // build): a batch that grows the store's node capacity grows the result's device state (its
  // fills, the copy of the old state and the release of the old pinned counters), and so does the
  // first update of a clone on the hypergraph moved to another backend and back.
  const dyng::detail::framework::strict_budgets_scope strict;
  const resources r = res();
  whgraph h = make(r, 4000, rows{{0, 1}}, {1});
  result_t sp = hs::compute(r, h, 0, opts());
  wbatch grow;
  for (std::int32_t i = 1; i < 3000; ++i) grow.insert_hyperedge({i, i + 1}, 1 + i % 3);
  EXPECT_NO_THROW((void)hs::update(r, h, grow.view(), sp));
  EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
  // A clone through the host and back, on the hypergraph moved the same way.
  whgraph back = h.to_backend(resources::sequential()).to_backend(r);
  result_t c = sp.clone(resources::sequential()).clone(r);
  wbatch more;
  for (std::int32_t i = 0; i < 1500; ++i) more.insert_hyperedge({i, i + 2}, 2);
  more.delete_hyperedge(5);
  EXPECT_NO_THROW((void)hs::update(r, back, more.view(), c));
  EXPECT_TRUE(matches_oracle(r, back, c, 0, true));
}

TEST_P(HyperSssp, TwoResultsOfDifferentSourcesInOneUpdate) {
  const resources r = res();
  whgraph h = path();
  result_t a = hs::compute(r, h, 0, opts());
  result_t c = hs::compute(r, h, 9, opts());
  wbatch b;
  b.delete_hyperedge(1);
  b.insert_incidence(5, 2);  // {10} + 2 joins hyperedges 2 and 5
  (void)dyng::update(r, h, b.view(), a, c);
  EXPECT_TRUE(matches_oracle(r, h, a, 0, true));
  EXPECT_TRUE(matches_oracle(r, h, c, 9, true));
  EXPECT_EQ(c.distance_to(10), 3 + 4 + 1);  // 3 -> 2 -> 5
}

TEST_P(HyperSssp, TheTwoCudaEnginesGiveTheSameBytes) {
  // M11b criterion 3 (C4 by hand, both modes): engine::operators == engine::fused byte for byte
  // after compute() and every update, the public mode and the virtual-endpoint mode, with
  // hyperedge and incidence changes and both id-reuse rules.
  if (!shared()) GTEST_SKIP() << "both engines run on cuda (the engine::operators case)";
  const resources r = res();
  hs::options fused;
  fused.cuda_engine = dyng::engine::fused;
  std::mt19937_64 rng(2026);
  for (int trial = 0; trial < 4; ++trial) {
    const std::int32_t n = 40 + 10 * trial;
    rows init;
    std::vector<std::int32_t> w;
    std::uniform_int_distribution<std::int32_t> vertex(1, n - 1);
    std::uniform_int_distribution<std::int32_t> weight(1, trial % 2 == 0 ? 9 : 1 << 20);
    for (int i = 0; i < 60; ++i) {
      init.push_back({vertex(rng), vertex(rng), vertex(rng)});
      w.push_back(weight(rng));
    }
    init.push_back({0, 1});
    w.push_back(1);
    rows plus = init;
    std::vector<std::int32_t> wp = w;
    plus.push_back({0});  // the virtual mode's source row, id 61
    wp.push_back(1);
    const dyng::id_reuse reuse =
        trial % 2 == 0 ? dyng::id_reuse::insert_first : dyng::id_reuse::erase_first;
    whgraph a = make(r, n, init, w, slack(reuse));
    whgraph af = make(r, n, init, w, slack(reuse));
    whgraph v = make(r, n, plus, wp, slack(reuse));
    whgraph vf = make(r, n, plus, wp, slack(reuse));
    dyng::detail::hyper_sssp_endpoints ends;
    ends.source_hyperedge = 61;
    ends.zero_weight_a = 61;
    result_t ra = hs::compute(r, a, 0, opts());
    result_t rf = hs::compute(r, af, 0, fused);
    result_t va = hv::compute(r, v, ends, opts());
    result_t vf2 = hv::compute(r, vf, ends, fused);
    ASSERT_EQ(host_of(r, ra), host_of(r, rf)) << "trial " << trial << ": compute()";
    ASSERT_EQ(host_of(r, va), host_of(r, vf2)) << "trial " << trial << ": compute() (virtual)";
    for (int t = 0; t < 8; ++t) {
      // The same batch on the four (never touching the source row).
      wbatch x = random_batch(rng, a, 30, trial % 2 == 0 ? 9 : 1 << 20, 3);
      if (t % 2 == 0) x.insert_incidence(static_cast<std::int64_t>(t), 0);
      if (t % 3 == 0) x.delete_incidence(60, 0);
      const hs::stats sa = hs::update(r, a, x.view(), ra);
      const hs::stats sf = hs::update(r, af, x.view(), rf);
      const hs::stats sv = hs::update(r, v, x.view(), va);
      const hs::stats svf = hs::update(r, vf, x.view(), vf2);
      EXPECT_EQ(sa.engine_used, dyng::engine::operators);
      EXPECT_EQ(sf.engine_used, dyng::engine::fused);
      ASSERT_EQ(host_of(r, ra), host_of(r, rf)) << "trial " << trial << " batch " << t;
      ASSERT_EQ(host_of(r, va), host_of(r, vf2)) << "trial " << trial << " batch " << t;
      EXPECT_EQ(sa.invalidated, sf.invalidated);
      EXPECT_EQ(sv.invalidated, svf.invalidated);
      ASSERT_TRUE(matches_oracle(r, a, ra, 0, true));
    }
  }
}

TEST_P(HyperSssp, TurningValidateInputsOnLaterKeepsTheResultExact) {
  // M11b review: the shared engine's packing bound is the largest weight the checks have seen,
  // which bounds the live weights only while every compute() and update() of the result checked
  // them. A result computed (or updated) without validate_inputs and then turned to it by
  // set_options() has no such bound: the update must stay exact on every engine.
  const resources r = res();
  rows chain;
  std::vector<std::int32_t> w;
  for (std::int32_t i = 0; i < 20; ++i) {
    chain.push_back({i, i + 1});
    w.push_back(1000);
  }
  for (const bool update_unchecked : {false, true}) {
    SCOPED_TRACE(update_unchecked ? "an unchecked update first" : "an unchecked compute()");
    whgraph h = make(r, 24, chain, w);
    hs::options o = opts();
    o.validate_inputs = update_unchecked;
    result_t sp = hs::compute(r, h, 0, o);
    ASSERT_TRUE(matches_oracle(r, h, sp, 0, true));
    if (update_unchecked) {
      // validate_inputs off for one update that inserts heavier weights than compute() saw
      o.validate_inputs = false;
      sp.set_options(o);
      wbatch heavy;
      heavy.insert_hyperedge({21, 22}, 5000);
      (void)hs::update(r, h, heavy.view(), sp);
      ASSERT_TRUE(matches_oracle(r, h, sp, 0, true));
    }
    o.validate_inputs = true;
    sp.set_options(o);
    wbatch b;
    b.insert_hyperedge({20, 21}, 1);
    b.insert_hyperedge({22, 23}, 1);
    const hs::stats s = hs::update(r, h, b.view(), sp);
    EXPECT_FALSE(s.fallback_used) << hs::to_string(s.fallback_reason);
    EXPECT_TRUE(matches_oracle(r, h, sp, 0, true));
    EXPECT_EQ(sp.distance_to(21), 20 * 1000 + 1);
    EXPECT_EQ(sp.distance_to(23), update_unchecked ? 20 * 1000 + 1 + 5000 + 1
                                                  : dyng::infinite_distance<std::int64_t>());
  }
}

TEST_P(HyperSssp, AMovedCloneReadsItsHypergraphAfterOneUpdate) {
  // M11b review: clone() keeps the link to the hypergraph the source result matches (its readers
  // read that hypergraph, whose contents the moved copy shares); once that hypergraph is gone or
  // changed, distance_to() and vertex_distances() throw stale_result_error until an update() of
  // the clone on the moved hypergraph (an empty batch is enough) links it there.
  const resources r = res();
  const resources seq = resources::sequential();
  std::optional<whgraph> h(path());
  const result_t sp = hs::compute(r, *h, 0, opts());
  whgraph moved = h->to_backend(seq);
  result_t c = sp.clone(seq);
  EXPECT_EQ(c.distance_to(9), sp.distance_to(9));  // the source hypergraph still exists
  h.reset();
  try {
    (void)c.distance_to(9);
    ADD_FAILURE() << "read a hypergraph that no longer exists";
  } catch (const dyng::stale_result_error& e) {
    EXPECT_NE(std::string(e.what()).find("update()"), std::string::npos) << e.what();
  }
  EXPECT_THROW((void)c.vertex_distances(), dyng::stale_result_error);
  (void)hs::update(seq, moved, wbatch{}.view(), c);
  EXPECT_EQ(c.distance_to(9), 1 + 2 + 3 + 4);
  EXPECT_EQ(c.vertex_distances().size(), 12U);
  EXPECT_TRUE(matches_oracle(seq, moved, c, 0, true));
}

std::string config_name(const ::testing::TestParamInfo<config>& info) {
  std::string name(dyng::to_string(info.param.b));
  if (info.param.b == backend::cuda && info.param.e == dyng::engine::operators) {
    name += "_operators";
  }
  return name;
}

INSTANTIATE_TEST_SUITE_P(Backends, HyperSssp, ::testing::ValuesIn(configs()), config_name);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HyperSssp);

}  // namespace
