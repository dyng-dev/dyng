// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file enactor_test.cpp
 * @brief The framework itself (PLAN Sections 4.5.1-4.5.5) with the two fake problems of
 *        fake_problems.hpp: the fixed hook order of both enactors and both families, one profiler
 *        stage per implemented hook, the old_view / new_view split (I1), the convergence cap and
 *        the on_limit policies (I3), Tier B, the device error check, the budget of the algorithm
 *        phase (I9), the participant adapter and run_update composition (one commit for several
 *        problems, equal to separate updates on copies of the graph).
 */
#include "core/budget_counters.hpp"
#include "fake_problems.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/policies.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fw = ::dyng::detail::framework;
using dyng::resources;
using dyng::test::framework_fakes::compute_levels;
using dyng::test::framework_fakes::compute_pairs;
using dyng::test::framework_fakes::edge;
using dyng::test::framework_fakes::graph_type;
using dyng::test::framework_fakes::hook_log;
using dyng::test::framework_fakes::levels_options;
using dyng::test::framework_fakes::levels_problem;
using dyng::test::framework_fakes::levels_result;
using dyng::test::framework_fakes::levels_stats;
using dyng::test::framework_fakes::make_batch;
using dyng::test::framework_fakes::make_graph;
using dyng::test::framework_fakes::pairs_problem;
using dyng::test::framework_fakes::pairs_result;
using dyng::test::framework_fakes::pairs_stats;
using strings = std::vector<std::string>;

/// A path 0 -> 1 -> ... -> n-1.
std::vector<edge> path(std::int32_t n) {
  std::vector<edge> e;
  for (std::int32_t v = 0; v + 1 < n; ++v) {
    e.emplace_back(v, v + 1);
  }
  return e;
}

/// BFS levels from `source` (the oracle of the levels fake).
std::vector<std::int32_t> bfs(const graph_type& g, std::int32_t source) {
  hook_log unused;
  return compute_levels(resources::sequential(), g, source, unused).level;
}

/// The single-result update of the levels fake: update_one through run_update.
levels_stats update_levels(const resources& res, graph_type& g,
                           const dyng::edge_batch<std::int32_t, dyng::unweighted>& b,
                           levels_result& r, hook_log& log, levels_options opt = {}) {
  return fw::update_one<levels_problem>(res, g, b.view(), r, log, opt);
}

/// The single-result update of the pairs fake.
pairs_stats update_pairs(const resources& res, graph_type& g,
                         const dyng::edge_batch<std::int32_t, dyng::unweighted>& b, pairs_result& r,
                         hook_log& log, dyng::test::framework_fakes::misbehaviour bad = {}) {
  return fw::update_one<pairs_problem>(res, g, b.view(), r, log, bad);
}

/// The stage names a profiler recorded, in order of first call, without the graph's own stages
/// (graph.build, graph.apply, ...).
strings stage_names(const dyng::profiler& p) {
  strings names;
  for (const auto& s : p.stages()) {
    if (s.name.rfind("graph.", 0) != 0) {
      names.push_back(s.name);
    }
  }
  return names;
}

/// The completed calls of one stage.
std::int64_t calls_of(const dyng::profiler& p, const std::string& name) {
  for (const auto& s : p.stages()) {
    if (s.name == name) {
      return s.calls;
    }
  }
  return 0;
}

// --- compute(): the static enactor ---------------------------------------------------------------

TEST(StaticEnactor, FixedPointRunsResetSeedLoopFinalize) {
  const resources res = resources::sequential();
  const graph_type g = make_graph(res, 5, path(5));
  hook_log log;
  const levels_result r = compute_levels(res, g, 0, log);
  EXPECT_EQ(r.level, (std::vector<std::int32_t>{0, 1, 2, 3, 4}));
  // Five loop rounds: four that discover a vertex and one that finds nothing (then empty).
  EXPECT_EQ(log.calls, (strings{"reset@0", "seed_static@0", "loop@0", "loop@0", "loop@0", "loop@0",
                                "loop@0", "finalize@0"}));
}

TEST(StaticEnactor, AggregateDeltaRunsResetCountFinalize) {
  const resources res = resources::sequential();
  const graph_type g = make_graph(res, 4, {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}});
  hook_log log;
  const pairs_result r = compute_pairs(res, g, log);
  EXPECT_EQ(r.pairs, 2);
  EXPECT_EQ(log.calls, (strings{"reset@0", "count@0", "finalize@0"}));
}

TEST(StaticEnactor, OneStagePerImplementedHook) {
  resources res = resources::sequential();
  dyng::profiler prof;
  res.attach_profiler(&prof);
  const graph_type g = make_graph(res, 3, path(3));
  hook_log log;
  (void)compute_levels(res, g, 0, log);
  (void)compute_pairs(res, g, log);
  res.attach_profiler(nullptr);
  // The loop is one stage however many rounds it runs.
  EXPECT_EQ(stage_names(prof), (strings{"test_levels.reset", "test_levels.seed", "test_levels.loop",
                                        "test_levels.finalize", "test_pairs.reset",
                                        "test_pairs.count", "test_pairs.finalize"}));
  EXPECT_EQ(calls_of(prof, "test_levels.loop"), 1);
}

TEST(StaticEnactor, TierBRunsComputeFusedInTheFusedStage) {
  resources res = resources::sequential();
  dyng::profiler prof;
  res.attach_profiler(&prof);
  const graph_type g = make_graph(res, 4, path(4));
  hook_log log;
  levels_options opt;
  opt.fused = true;
  const levels_result r = compute_levels(res, g, 1, log, opt);
  res.attach_profiler(nullptr);
  EXPECT_EQ(r.level, (std::vector<std::int32_t>{-1, 0, 1, 2}));
  EXPECT_EQ(log.calls.front(), "compute_fused@0");
  const strings names = stage_names(prof);
  EXPECT_NE(std::find(names.begin(), names.end(), "test_levels.enact_fused"), names.end());
  EXPECT_EQ(std::find(names.begin(), names.end(), "test_levels.loop"), names.end());
}

// --- update(): the hook order and the two halves ---------------------------------------------

TEST(UpdateEnactor, FixedPointHookOrderAndViews) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 5, path(4));  // vertex 4 is unreachable
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  log.calls.clear();
  const auto b = make_batch({{1, 2}}, {{0, 4}, {4, 2}});
  const levels_stats s = update_levels(res, g, b, r, log);
  EXPECT_EQ(g.version(), 1U);
  EXPECT_EQ(r.version, 1U);
  EXPECT_EQ(r.level, bfs(g, 0));
  EXPECT_EQ(r.level, (std::vector<std::int32_t>{0, 1, 2, 3, 1}));
  // Every hook before the commit sees version 0 (G_t), every hook after it version 1 (G_{t+1}).
  EXPECT_EQ(log.calls, (strings{"begin_update@0", "prepare@0", "before_apply@0", "resume@1",
                                "identify_affected@1", "seed@1", "loop@1", "loop@1", "loop@1",
                                "loop@1", "finalize@0", "end_update@1"}));
  EXPECT_EQ(s.iterations, 4);
  EXPECT_EQ(s.affected, 1);  // vertex 4 became reachable; vertex 2 keeps level 2 (now via 4)
  EXPECT_EQ(s.engine_used, dyng::engine::operators);
  EXPECT_TRUE(s.converged);
  EXPECT_FALSE(s.fallback_used);
  EXPECT_EQ(s.batch.deleted_edges, 1);  // the adapter copies the commit's summary
  EXPECT_EQ(s.batch.inserted_edges, 2);
}

TEST(UpdateEnactor, AggregateDeltaHookOrderSubtractsOnTheOldGraph) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}});
  hook_log log;
  pairs_result r = compute_pairs(res, g, log);
  ASSERT_EQ(r.pairs, 2);
  log.calls.clear();
  // Delete both edges of the pair {0, 1} (one pair, owned by the smaller id), add {2, 3}.
  const auto b = make_batch({{0, 1}, {1, 0}}, {{3, 2}});
  const pairs_stats s = update_pairs(res, g, b, r, log);
  EXPECT_EQ(log.calls,
            (strings{"begin_update@0", "normalize@0", "count_minus@0", "identify_affected@1",
                     "count_plus@1", "finalize@0", "end_update@1"}));
  EXPECT_EQ(s.removed, 1);  // exactly once (ownership::min_member), counted on G_t
  EXPECT_EQ(s.added, 1);
  EXPECT_EQ(r.pairs, 2);
  hook_log unused;
  EXPECT_EQ(r.pairs, compute_pairs(res, g, unused).pairs);
  EXPECT_EQ(s.engine_used, dyng::engine::operators);
}

TEST(UpdateEnactor, UpdateChainsEqualCompute) {
  // Random batch chains: every update equals compute on the new graph (the oracle of both fakes).
  std::mt19937_64 rng(20260929);
  const resources res = resources::sequential();
  for (int trial = 0; trial < 20; ++trial) {
    constexpr std::int32_t n = 9;
    std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
    std::vector<edge> edges;
    std::set<edge> present;
    for (int i = 0; i < 25; ++i) {
      const edge e{vertex(rng), vertex(rng)};
      if (e.first != e.second && present.insert(e).second) {
        edges.push_back(e);
      }
    }
    std::sort(edges.begin(), edges.end());
    graph_type g = make_graph(res, n, edges);
    hook_log log;
    levels_result levels = compute_levels(res, g, 0, log);
    pairs_result pairs = compute_pairs(res, g, log);
    for (int step = 0; step < 4; ++step) {
      std::vector<edge> del;
      std::vector<edge> ins;
      for (int i = 0; i < 6; ++i) {
        const edge e{vertex(rng), vertex(rng)};
        if (e.first != e.second) {
          ((i % 2 == 0) ? del : ins).push_back(e);
        }
      }
      const auto b = make_batch(del, ins);
      if (step % 2 == 0) {
        (void)update_levels(res, g, b, levels, log);
        pairs = compute_pairs(res, g, log);  // keep the other result current by recomputing
      } else {
        (void)update_pairs(res, g, b, pairs, log);
        levels = compute_levels(res, g, 0, log);
      }
      hook_log unused;
      EXPECT_EQ(levels.level, bfs(g, 0)) << trial << "/" << step;
      EXPECT_EQ(pairs.pairs, compute_pairs(res, g, unused).pairs) << trial << "/" << step;
    }
  }
}

TEST(UpdateEnactor, OneStagePerImplementedHook) {
  resources res = resources::sequential();
  graph_type g = make_graph(res, 4, {{0, 1}, {1, 0}, {1, 2}});
  hook_log log;
  levels_result levels = compute_levels(res, g, 0, log);
  dyng::profiler prof;
  res.attach_profiler(&prof);
  (void)update_levels(res, g, make_batch({}, {{2, 3}}), levels, log);
  res.attach_profiler(nullptr);
  pairs_result pairs = compute_pairs(res, g, log);  // on G_1, not profiled
  res.attach_profiler(&prof);
  (void)update_pairs(res, g, make_batch({{1, 0}}, {}), pairs, log);
  res.attach_profiler(nullptr);
  // Hooks a problem does not implement are neither called nor staged (test_levels has no
  // normalize or count hooks, test_pairs no prepare, before_apply, seed or loop).
  EXPECT_EQ(
      stage_names(prof),
      (strings{"test_levels.update", "test_levels.normalize", "test_levels.prepare",
               "test_levels.before_apply", "test_levels.commit", "test_levels.identify_affected",
               "test_levels.seed", "test_levels.loop", "test_levels.finalize", "test_pairs.update",
               "test_pairs.normalize", "test_pairs.count_minus", "test_pairs.commit",
               "test_pairs.identify_affected", "test_pairs.count_plus", "test_pairs.finalize"}));
  // test_levels.normalize is run_update's Step 0 of set semantics (once per update), not a hook;
  // test_pairs.normalize is that stage and the problem's own normalize hook.
  EXPECT_EQ(calls_of(prof, "test_levels.normalize"), 1);
  EXPECT_EQ(calls_of(prof, "test_pairs.normalize"), 2);
  EXPECT_EQ(calls_of(prof, "test_levels.loop"), 1);
}

// --- Tier B, engines ---------------------------------------------------------------------------

TEST(UpdateEnactor, TierBReplacesIdentifyAffectedToFinalize) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  log.calls.clear();
  levels_options opt;
  opt.fused = true;
  const levels_stats s = update_levels(res, g, make_batch({}, {{2, 3}}), r, log, opt);
  EXPECT_EQ(r.level, bfs(g, 0));
  // The framework still owns Step 0, before_apply and the commit.
  ASSERT_GE(log.calls.size(), 5U);
  EXPECT_EQ(
      (strings(log.calls.begin(), log.calls.begin() + 5)),
      (strings{"begin_update@0", "prepare@0", "before_apply@0", "resume@1", "enact_fused@1"}));
  EXPECT_EQ(log.calls.back(), "end_update@1");
  for (const std::string& call : log.calls) {
    EXPECT_NE(call, "identify_affected@1");
    EXPECT_NE(call, "finalize@0");
  }
  EXPECT_EQ(s.engine_used, dyng::engine::fused);
  EXPECT_EQ(s.affected, 1);
}

// --- convergence cap and on_limit (I3) -----------------------------------------------------------

TEST(UpdateEnactor, CapWithOnLimitErrorThrowsConvergenceError) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 6, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  levels_options opt;
  opt.policy.max_iterations = 2;  // the path to vertex 5 needs 5 rounds (+ 1 empty)
  opt.policy.at_limit = fw::on_limit::error;
  EXPECT_THROW((void)update_levels(res, g, make_batch({}, {{2, 3}, {3, 4}, {4, 5}}), r, log, opt),
               dyng::convergence_error);
  EXPECT_TRUE(r.poisoned);  // run_update poisons a participant that fails after the commit
  EXPECT_EQ(g.version(), 1U);
}

TEST(UpdateEnactor, CapWithOnLimitReportFinalizesThePartialResult) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 6, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  log.calls.clear();
  levels_options opt;
  opt.policy.max_iterations = 2;
  opt.policy.at_limit = fw::on_limit::report;
  const levels_stats s =
      update_levels(res, g, make_batch({}, {{2, 3}, {3, 4}, {4, 5}}), r, log, opt);
  EXPECT_FALSE(s.converged);
  EXPECT_EQ(s.iterations, 2);
  EXPECT_EQ(r.level, (std::vector<std::int32_t>{0, 1, 2, -1, -1, -1}));  // two rounds only
  EXPECT_EQ(log.calls[log.calls.size() - 2], "finalize@0");
}

TEST(UpdateEnactor, CapWithFallbackRecomputeRecomputes) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 6, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  log.calls.clear();
  levels_options opt;
  opt.policy.max_iterations = 1;
  opt.policy.at_limit = fw::on_limit::fallback_recompute;
  const levels_stats s =
      update_levels(res, g, make_batch({}, {{2, 3}, {3, 4}, {4, 5}}), r, log, opt);
  EXPECT_TRUE(s.fallback_used);
  EXPECT_TRUE(s.converged);
  EXPECT_EQ(r.level, bfs(g, 0));
  EXPECT_NE(std::find(log.calls.begin(), log.calls.end(), "recompute@1"), log.calls.end());
}

TEST(StaticEnactor, CapAppliesToComputeToo) {
  const resources res = resources::sequential();
  const graph_type g = make_graph(res, 6, path(6));
  hook_log log;
  levels_options opt;
  opt.policy.max_iterations = 2;
  opt.policy.at_limit = fw::on_limit::fallback_recompute;  // compute() cannot fall back: error
  EXPECT_THROW((void)compute_levels(res, g, 0, log, opt), dyng::convergence_error);
}

// --- device errors -------------------------------------------------------------------------------

TEST(UpdateEnactor, DeviceErrorBeforeTheCommitLeavesTheGraphUnchanged) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, {{0, 1}, {1, 0}});
  hook_log log;
  pairs_result r = compute_pairs(res, g, log);
  dyng::test::framework_fakes::misbehaviour bad;
  bad.device_error_before = dyng::detail::bits_of(dyng::detail::device_error::invalid_input);
  EXPECT_THROW((void)update_pairs(res, g, make_batch({{0, 1}}, {}), r, log, bad),
               dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0U);  // checked at the end of the phase, before the commit
  EXPECT_FALSE(r.poisoned);
  // The result is still usable.
  EXPECT_NO_THROW((void)update_pairs(res, g, make_batch({{0, 1}}, {}), r, log));
  EXPECT_EQ(r.pairs, 0);
}

TEST(UpdateEnactor, DeviceErrorAfterTheCommitThrowsAndPoisons) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, {{0, 1}, {1, 0}});
  hook_log log;
  pairs_result r = compute_pairs(res, g, log);
  dyng::test::framework_fakes::misbehaviour bad;
  bad.device_error_after = dyng::detail::bits_of(dyng::detail::device_error::capacity);
  try {
    (void)update_pairs(res, g, make_batch({}, {{2, 3}}), r, log, bad);
    FAIL() << "expected capacity_error";
  } catch (const dyng::capacity_error& e) {
    EXPECT_NE(std::string(e.what()).find("test_pairs::update"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("count(+) misbehaved"), std::string::npos) << e.what();
  }
  EXPECT_EQ(g.version(), 1U);
  EXPECT_TRUE(r.poisoned);
}

// --- invariant I1: an old_view after the commit -----------------------------------------------

TEST(Views, OldViewAfterTheCommitThrowsInDebugBuilds) {
#ifdef NDEBUG
  GTEST_SKIP() << "the view check is compiled in Debug builds only";
#else
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 3, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  levels_options opt;
  opt.bad.keep_old_view = true;
  EXPECT_THROW((void)update_levels(res, g, make_batch({}, {{2, 0}}), r, log, opt),
               dyng::internal_error);
#endif
}

TEST(Views, ReportTheirVersionAndCurrency) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 3, path(3));
  const fw::old_view<graph_type> before(g);
  EXPECT_TRUE(before.is_current());
  EXPECT_EQ(before.version(), 0U);
  EXPECT_EQ(before->num_vertices(), 3);
  (void)g.apply(res, make_batch({}, {{2, 0}}).view());
  EXPECT_FALSE(before.is_current());
  const fw::new_view<graph_type> after(g);
  EXPECT_TRUE(after.is_current());
  EXPECT_EQ(after.version(), 1U);
  EXPECT_EQ(after.get().num_edges(), 3);
#ifndef NDEBUG
  EXPECT_THROW((void)before.get(), dyng::internal_error);
#endif
}

// --- budgets (I9) ----------------------------------------------------------------------------------

TEST(Budgets, TheAlgorithmPhaseIsMeasured) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  levels_options opt;
  opt.limit = fw::budget::steady_state(0);
  (void)update_levels(res, g, make_batch({}, {{2, 3}}), r, log, opt);
  const fw::budget_report report = fw::last_budget_report();
  EXPECT_EQ(report.measured, dyng::detail::budgets_enabled());
  EXPECT_EQ(report.used.allocations, 0);
  EXPECT_EQ(report.used.host_syncs, 0);
  EXPECT_EQ(report.limit.allocations, 0);
}

TEST(Budgets, AnAllocationAfterTheCommitExceedsASteadyStateBudget) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  levels_options opt;
  opt.limit = fw::budget::steady_state(0);
  opt.bad.allocate_after_commit = true;
  if (dyng::detail::budgets_enabled()) {
    EXPECT_THROW((void)update_levels(res, g, make_batch({}, {{2, 3}}), r, log, opt),
                 dyng::internal_error);
    EXPECT_EQ(fw::last_budget_report().used.allocations, 1);
    EXPECT_EQ(fw::last_budget_report().used.allocated_bytes,
              16 * static_cast<std::int64_t>(sizeof(int)));
  } else {
    EXPECT_NO_THROW((void)update_levels(res, g, make_batch({}, {{2, 3}}), r, log, opt));
  }
}

// "Once reserved": a phase that notes a reservation may allocate (reported, not failed).
TEST(Budgets, AReservingPhaseMayAllocate) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  levels_options opt;
  opt.limit = fw::budget::steady_state(0);
  opt.bad.allocate_after_commit = true;
  opt.bad.reserve_after_commit = true;
  EXPECT_NO_THROW((void)update_levels(res, g, make_batch({}, {{2, 3}}), r, log, opt));
  const fw::budget_report report = fw::last_budget_report();
  EXPECT_EQ(report.reserving(), dyng::detail::budgets_enabled());
  EXPECT_EQ(report.used.allocations, dyng::detail::budgets_enabled() ? 1 : 0);
}

// The commit is measured separately (container growth, reported only): run_update() records it.
TEST(Budgets, TheCommitIsMeasuredSeparately) {
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 4, path(3));
  hook_log log;
  levels_result r = compute_levels(res, g, 0, log);
  dyng::detail::budget_counters marker;
  marker.host_syncs = -7;
  fw::record_commit_counts(marker);
  (void)update_levels(res, g, make_batch({}, {{2, 3}}), r, log);
  EXPECT_EQ(fw::last_commit_counts().host_syncs, 0);  // replaced by this update's commit
  EXPECT_EQ(fw::last_commit_counts().reservations, 0);
}

// Container work (container_scope) is counted, recorded as such, and not held against a budget;
// nested scopes count once.
TEST(Budgets, ContainerWorkIsNotHeldAgainstTheBudget) {
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  const resources res = resources::sequential();
  const fw::budget_scope scope;
  {
    const dyng::detail::container_scope outer;
    const dyng::buffer<int> a(res, 4);
    {
      const dyng::detail::container_scope inner;
      const dyng::buffer<int> b(res, 4);
    }
  }
  const dyng::buffer<int> own(res, 4);
  const dyng::detail::budget_counters used = scope.used();
  EXPECT_EQ(used.allocations, 3);
  EXPECT_EQ(used.container_allocations, 2);
  EXPECT_EQ(used.own_allocations(), 1);
  EXPECT_FALSE(fw::budget::steady_state(0).allows(used));
  dyng::detail::budget_counters container_only = used;
  container_only.allocations = 2;
  EXPECT_TRUE(fw::budget::steady_state(0).allows(container_only));
}

TEST(Budgets, UncheckedAllowsAnything) {
  const fw::budget unchecked = fw::budget::unchecked();
  dyng::detail::budget_counters used;
  used.allocations = 1000;
  used.host_syncs = 1000;
  EXPECT_TRUE(unchecked.allows(used));
  EXPECT_FALSE(fw::budget::steady_state(1).allows(used));
  used.allocations = 0;
  used.host_syncs = 1;
  EXPECT_TRUE(fw::budget::steady_state(1).allows(used));
  used.host_syncs = 2;
  EXPECT_FALSE(fw::budget::steady_state(1).allows(used));
  // A reserving phase: its allocations do not count against the budget, its syncs still do.
  used.allocations = 5;
  used.host_syncs = 1;
  used.reservations = 1;
  EXPECT_TRUE(fw::budget::steady_state(1).allows(used));
  used.host_syncs = 2;
  EXPECT_FALSE(fw::budget::steady_state(1).allows(used));
  EXPECT_TRUE(fw::budget::steady_state(0).bounded());
  EXPECT_FALSE(unchecked.bounded());
}

TEST(Budgets, CountersCountTheLibrarysAllocations) {
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  const resources res = resources::sequential();
  const fw::budget_scope scope;
  {
    const dyng::buffer<std::int64_t> a(res, 8);
    const dyng::buffer<std::int64_t> b(res, 4);
  }
  EXPECT_EQ(scope.used().allocations, 2);
  EXPECT_EQ(scope.used().allocated_bytes, 12 * static_cast<std::int64_t>(sizeof(std::int64_t)));
  EXPECT_EQ(scope.used().host_syncs, 0);
}

// --- composition: several problems, one commit (run_update) ----------------------------------

TEST(Composition, SeveralProblemsShareOneCommitAndEqualSeparateUpdates) {
  const resources res = resources::sequential();
  const std::vector<edge> edges{{0, 1}, {1, 0}, {1, 2}, {2, 3}, {3, 2}, {3, 4}};
  graph_type g = make_graph(res, 6, edges);
  hook_log levels_log;
  hook_log pairs_log;
  levels_result levels = compute_levels(res, g, 0, levels_log);
  pairs_result pairs = compute_pairs(res, g, pairs_log);
  levels_log.calls.clear();
  pairs_log.calls.clear();

  // The same batch applied separately to two copies of the graph.
  graph_type g_levels = g.clone(res);
  graph_type g_pairs = g.clone(res);
  levels_result levels_alone = levels;
  pairs_result pairs_alone = pairs;
  hook_log unused;
  const auto b = make_batch({{2, 3}, {1, 0}}, {{4, 3}, {4, 5}, {0, 5}});
  const levels_stats levels_alone_stats = update_levels(res, g_levels, b, levels_alone, unused);
  const pairs_stats pairs_alone_stats = update_pairs(res, g_pairs, b, pairs_alone, unused);

  // Both at once: every Step 0 / 1a on G_t, one commit, every Step 1b / 2 on G_{t+1}.
  levels_stats levels_stats_out;
  pairs_stats pairs_stats_out;
  fw::problem_participant<levels_problem> p1(levels_stats_out, levels, levels_log);
  fw::problem_participant<pairs_problem> p2(pairs_stats_out, pairs, pairs_log);
  dyng::detail::update_participant<std::int32_t, std::int64_t, dyng::unweighted>* list[] = {&p1,
                                                                                            &p2};
  (void)dyng::detail::run_update(res, g, b.view(), list, 2, "update.commit");

  EXPECT_EQ(g.version(), 1U);  // one commit for both
  EXPECT_EQ(levels.level, levels_alone.level);
  EXPECT_EQ(levels.level, bfs(g, 0));
  EXPECT_EQ(pairs.pairs, pairs_alone.pairs);
  EXPECT_EQ(levels_stats_out.affected, levels_alone_stats.affected);
  EXPECT_EQ(pairs_stats_out.removed, pairs_alone_stats.removed);
  EXPECT_EQ(pairs_stats_out.added, pairs_alone_stats.added);
  EXPECT_EQ(levels.version, 1U);
  EXPECT_EQ(pairs.version, 1U);
  // Every before-commit hook of both problems saw version 0, every after-commit hook version 1.
  for (const hook_log* log : {&levels_log, &pairs_log}) {
    bool committed = false;
    for (const std::string& call : log->calls) {
      if (call.rfind("resume", 0) == 0 || call.rfind("identify_affected", 0) == 0) {
        committed = true;
      }
      if (call.rfind("finalize", 0) == 0) {
        continue;  // finalize receives no view
      }
      EXPECT_EQ(call.back(), committed ? '1' : '0') << call;
    }
  }
}

TEST(Composition, APassedTwiceResultIsRejectedBeforeAnythingChanges) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 3, path(3));
  hook_log log;
  pairs_result pairs = compute_pairs(res, g, log);
  pairs_stats s1;
  pairs_stats s2;
  fw::problem_participant<pairs_problem> p1(s1, pairs, log);
  fw::problem_participant<pairs_problem> p2(s2, pairs, log);
  dyng::detail::update_participant<std::int32_t, std::int64_t, dyng::unweighted>* list[] = {&p1,
                                                                                            &p2};
  const auto b = make_batch({}, {{2, 0}});
  EXPECT_THROW((void)dyng::detail::run_update(res, g, b.view(), list, 2, "update.commit"),
               dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0U);
}

TEST(Composition, AStaleResultIsDetectedBeforeTheCommit) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 3, path(3));
  hook_log log;
  pairs_result pairs = compute_pairs(res, g, log);
  graph_type other = make_graph(res, 3, path(3));  // the same version, another graph
  pairs_result on_other = compute_pairs(res, other, log);
  EXPECT_THROW((void)update_pairs(res, g, make_batch({}, {{1, 0}}), on_other, log),
               dyng::stale_result_error);
  EXPECT_EQ(g.version(), 0U);
  (void)g.apply(res, make_batch({}, {{2, 0}}).view());  // the result is left behind
  EXPECT_THROW((void)update_pairs(res, g, make_batch({}, {{1, 0}}), pairs, log),
               dyng::stale_result_error);
  EXPECT_EQ(g.version(), 1U);
}

TEST(Composition, MakeParticipantBuildsTheAdapter) {
  const resources res = resources::sequential();
  graph_type g = make_graph(res, 3, {{0, 1}});
  hook_log log;
  pairs_result pairs = compute_pairs(res, g, log);
  pairs_stats s;
  auto participant = fw::make_participant<pairs_problem>(s, pairs, log);
  EXPECT_EQ(participant->target(), &pairs);
  EXPECT_FALSE(participant->reads_prepared_graph());
  auto* raw = participant.get();
  const auto b = make_batch({}, {{1, 0}});
  (void)dyng::detail::run_update(res, g, b.view(), &raw, 1, "update.commit");
  EXPECT_EQ(pairs.pairs, 1);
  EXPECT_EQ(s.added, 1);
}

TEST(Context, RecordsDeviceErrorsAndTheFirstDetail) {
  const resources res = resources::sequential();
  fw::context ctx(res, "test_levels");
  EXPECT_EQ(ctx.algorithm(), "test_levels");
  EXPECT_EQ(ctx.get_backend(), dyng::backend::sequential);
  EXPECT_FALSE(ctx.on_cuda());
  EXPECT_EQ(ctx.host_threads(), 1);
  ctx.raise_device_error(0, "ignored");
  EXPECT_EQ(ctx.device_errors(), 0U);
  ctx.raise_device_error(1, "first");
  ctx.raise_device_error(4, "second");
  EXPECT_EQ(ctx.device_errors(), 5U);
  EXPECT_EQ(ctx.device_error_detail(), "first");
  ctx.clear_device_errors();
  EXPECT_EQ(ctx.device_errors(), 0U);
}

TEST(StageNames, AreBuiltFromTheAlgorithmName) {
  const fw::hook_stages& s = fw::stages_of<levels_problem>();
  EXPECT_EQ(s.identify_affected, "test_levels.identify_affected");
  EXPECT_EQ(s.count_minus, "test_levels.count_minus");
  EXPECT_EQ(s.commit, "test_levels.commit");
  EXPECT_EQ(s.update_function, "test_levels::update");
  EXPECT_EQ(&s, &fw::stages_of<levels_problem>());  // built once
}

}  // namespace
