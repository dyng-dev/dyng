// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_fixed_point_test.cpp
 * @brief A fixed-point problem on the hypergraph through update_enactor (PLAN risk R3; M11a
 *        step 4, ADR 0092): the fake test_hyper_hops (framework/fake_hyper_fixed_point.hpp) on
 *        slack_csr storage, kept equal to compute() under hyperedge batches and incidence edits;
 *        the Tier A hook order and stages of a fixed point on G_{t+1}; two results of different
 *        sources in one dyng::update(); the convergence cap's three on_limit actions; and the
 *        problem's own work budget (over_budget: the recompute hook, or convergence_error with the
 *        result poisoned and the hypergraph at G_{t+1}). Compiled into the host test executable
 *        and, with DYNG_TEST_CUDA=1, into the CUDA one.
 */
#include "framework/fake_hyper_fixed_point.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::hypergraph_properties;
using dyng::resources;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using wbatch = dyng::hyperedge_batch<std::int32_t, std::int32_t>;
namespace fakes = ::dyng::test::framework_fakes;
namespace hfp = ::dyng::detail::framework;

class HypergraphFixedPoint : public ::testing::TestWithParam<backend> {
 protected:
  resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
  static hypergraph_properties slack() {
    hypergraph_properties p;
    p.storage = dyng::hypergraph_storage::slack_csr;
    p.line_graph = dyng::line_graph_policy::slack_csr;
    return p;
  }
  /// A path of hyperedges 0-1-2-3-4 plus a shortcut 5 = {0, 9} between 0 and 4, on 12 vertices.
  whgraph make() const {
    return whgraph::from_hyperedges(
        res(),
        list_of<std::int32_t>(12, rows{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 9}, {0, 9}, {10, 11}},
                              {1, 1, 1, 1, 1, 1, 1})
            .view(),
        slack());
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

/// How often a stage ran.
std::int64_t calls(const dyng::profiler& prof, const std::string& name) {
  for (const auto& st : prof.stages()) {
    if (st.name == name) return st.calls;
  }
  return 0;
}

void expect_equal(const fakes::hyper_hops_result& got, const fakes::hyper_hops_result& want,
                  const std::string& where) {
  const std::size_t n = std::max(got.hops.size(), want.hops.size());
  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t gh = i < got.hops.size() ? got.hops[i] : fakes::hops_infinity;
    const std::int64_t wh = i < want.hops.size() ? want.hops[i] : fakes::hops_infinity;
    const std::int64_t gp = i < got.parent.size() ? got.parent[i] : -1;
    const std::int64_t wp = i < want.parent.size() ? want.parent[i] : -1;
    ASSERT_EQ(gh, wh) << where << ": hops of id " << i;
    ASSERT_EQ(gp, wp) << where << ": parent of id " << i;
  }
}

/// A random batch that never deletes the source (id 0) nor removes its vertices.
wbatch random_batch(std::mt19937_64& rng, const whgraph& h, std::int32_t n) {
  dyng::test::hyper::rows current;
  const std::vector<std::int64_t> live = dyng::test::hyper::live_ids(h, current);
  std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
  std::uniform_int_distribution<int> coin(0, 3);
  wbatch b;
  for (const std::int64_t id : live) {
    if (id == 0) continue;
    if (coin(rng) == 0) b.delete_hyperedge(id);
    if (coin(rng) == 0) b.insert_incidence(id, vertex(rng));
    if (coin(rng) == 0 && !current[static_cast<std::size_t>(id)].empty()) {
      b.delete_incidence(id, current[static_cast<std::size_t>(id)].front());
    }
  }
  for (int i = 0; i < 4; ++i) {
    b.insert_hyperedge({vertex(rng), vertex(rng)}, 1);
  }
  return b;
}

TEST_P(HypergraphFixedPoint, UpdatesEqualComputeUnderBatchesAndIncidenceEdits) {
  std::mt19937_64 rng(20261029);
  constexpr std::int32_t n = 40;
  dyng::test::hyper::rows init;
  for (int i = 0; i < 30; ++i) {
    init.push_back({static_cast<std::int32_t>(i % n), static_cast<std::int32_t>((i * 7 + 3) % n)});
  }
  whgraph h = whgraph::from_hyperedges(
      res(), list_of<std::int32_t>(n, init, std::vector<std::int32_t>(init.size(), 1)).view(),
      slack());
  fakes::hyper_hops_result r = fakes::compute_hops(res(), h, 0);
  for (int t = 0; t < 25; ++t) {
    const wbatch b = random_batch(rng, h, n);
    const fakes::hyper_hops_stats s = fakes::update_hops(res(), h, b.view(), r);
    EXPECT_FALSE(s.fallback_used);
    EXPECT_TRUE(s.converged);
    EXPECT_EQ(r.graph_version(), h.version());
    expect_equal(r, fakes::compute_hops(res(), h, 0), "batch " + std::to_string(t));
  }
}

TEST_P(HypergraphFixedPoint, TheTierAHooksOfAFixedPointRunOnTheNewGraph) {
  whgraph h = make();
  fakes::hyper_hops_result r = fakes::compute_hops(res(), h, 0);
  EXPECT_EQ(r.hops[4], 2);  // 0 -> 5 -> 4
  EXPECT_EQ(r.parent[4], 5);
  dyng::profiler prof;
  resources rr = res();
  rr.attach_profiler(&prof);
  wbatch b;
  b.delete_hyperedge(5);  // the shortcut goes: 4 is invalidated and found again through 3
  const fakes::hyper_hops_stats s = fakes::update_hops(rr, h, b.view(), r);
  rr.attach_profiler(nullptr);
  EXPECT_EQ(r.hops[4], 4);
  EXPECT_EQ(r.parent[4], 3);
  EXPECT_GE(s.invalidated, 1);
  EXPECT_GE(s.iterations, 1);
  for (const char* stage :
       {"test_hyper_hops.update", "test_hyper_hops.commit", "test_hyper_hops.identify_affected",
        "test_hyper_hops.seed", "test_hyper_hops.loop", "test_hyper_hops.finalize"}) {
    EXPECT_GE(calls(prof, stage), 1) << stage;
  }
  EXPECT_EQ(calls(prof, "test_hyper_hops.recompute"), 0);
  expect_equal(r, fakes::compute_hops(res(), h, 0), "after the deletion");
}

TEST_P(HypergraphFixedPoint, TwoResultsOfDifferentSourcesShareOneCommit) {
  whgraph h = make();
  fakes::hyper_hops_result a = fakes::compute_hops(res(), h, 0);
  fakes::hyper_hops_result c = fakes::compute_hops(res(), h, 3);
  wbatch b;
  b.delete_hyperedge(2);
  b.insert_incidence(6, 1);  // {10, 11} + 1 joins 0 and 1
  const std::uint64_t before = h.version();
  (void)dyng::update(res(), h, b.view(), a, c);
  EXPECT_EQ(h.version(), before + 1);
  expect_equal(a, fakes::compute_hops(res(), h, 0), "source 0");
  expect_equal(c, fakes::compute_hops(res(), h, 3), "source 3");
}

TEST_P(HypergraphFixedPoint, TheCapRecomputesReportsOrThrows) {
  for (const hfp::on_limit at :
       {hfp::on_limit::fallback_recompute, hfp::on_limit::report, hfp::on_limit::error}) {
    whgraph h = make();
    fakes::hyper_hops_result r = fakes::compute_hops(res(), h, 0);
    fakes::hyper_hops_knobs knobs;
    knobs.policy.max_iterations = 0;  // the first push round is over the cap
    knobs.policy.at_limit = at;
    wbatch b;
    b.delete_hyperedge(5);  // one push round is needed
    if (at == hfp::on_limit::error) {
      EXPECT_THROW((void)fakes::update_hops(res(), h, b.view(), r, knobs), dyng::convergence_error);
      EXPECT_TRUE(r.poisoned);
      EXPECT_EQ(h.num_hyperedges(), 6);  // the commit happened: G_{t+1}
      continue;
    }
    const fakes::hyper_hops_stats s = fakes::update_hops(res(), h, b.view(), r, knobs);
    if (at == hfp::on_limit::fallback_recompute) {
      EXPECT_TRUE(s.fallback_used);
      expect_equal(r, fakes::compute_hops(res(), h, 0), "the recompute");
    } else {
      EXPECT_FALSE(s.converged);
      EXPECT_FALSE(s.fallback_used);
    }
  }
}

TEST_P(HypergraphFixedPoint, TheOwnBudgetEndsStepTwoThroughTheRecompute) {
  whgraph h = make();
  fakes::hyper_hops_result r = fakes::compute_hops(res(), h, 0);
  fakes::hyper_hops_knobs knobs;
  knobs.budget_after_loops = 1;
  dyng::profiler prof;
  resources rr = res();
  rr.attach_profiler(&prof);
  wbatch b;
  b.delete_hyperedge(5);
  const fakes::hyper_hops_stats s = fakes::update_hops(rr, h, b.view(), r, knobs);
  rr.attach_profiler(nullptr);
  EXPECT_TRUE(s.fallback_used);
  EXPECT_TRUE(s.converged);
  EXPECT_EQ(calls(prof, "test_hyper_hops.recompute"), 1);
  EXPECT_EQ(calls(prof, "test_hyper_hops.finalize"), 0);  // the recompute replaces it
  expect_equal(r, fakes::compute_hops(res(), h, 0), "over budget");
  // A batch whose Step 2 needs no loop call is never over the budget.
  wbatch none;
  none.insert_hyperedge({10}, 1);
  const fakes::hyper_hops_stats quiet = fakes::update_hops(res(), h, none.view(), r, knobs);
  EXPECT_FALSE(quiet.fallback_used);
  expect_equal(r, fakes::compute_hops(res(), h, 0), "no work");
}

TEST_P(HypergraphFixedPoint, AThrowingBudgetPoisonsTheResultAfterTheCommit) {
  whgraph h = make();
  fakes::hyper_hops_result r = fakes::compute_hops(res(), h, 0);
  fakes::hyper_hops_knobs knobs;
  knobs.budget_after_loops = 1;
  knobs.budget_throws = true;
  wbatch b;
  b.delete_hyperedge(5);
  const std::uint64_t before = h.version();
  EXPECT_THROW((void)fakes::update_hops(res(), h, b.view(), r, knobs), dyng::convergence_error);
  EXPECT_TRUE(r.poisoned);
  EXPECT_EQ(h.version(), before + 1);  // basic guarantee: the hypergraph holds G_{t+1}
  wbatch next;
  next.insert_hyperedge({1}, 1);
  EXPECT_THROW((void)fakes::update_hops(res(), h, next.view(), r), dyng::invalid_argument_error);
}

INSTANTIATE_TEST_SUITE_P(Backends, HypergraphFixedPoint,
                         ::testing::ValuesIn(dyng::test::suite_backends()), backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HypergraphFixedPoint);

}  // namespace
