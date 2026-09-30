// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_test.cpp
 * @brief cycle_count: the API contract (options, result, errors, stale and poisoned results,
 *        profiler stages, workspace reuse, composition with sssp) and the hand cases of
 *        CycleEnumeration-GPU@0a976ad's static and histogram suites (johnson_static_test,
 *        openmp_johnson_test, sequential_parity_test, histogram_engine_test, histogram_test).
 */
#include "algorithms/cycle_count/problem.hpp"
#include "core/resources_access.hpp"
#include "support/cycle_count_support.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// Every cycle_count enumeration prints its enumerator's name.
TEST(CycleCountOptions, EnumerationsHaveTheirNames) {
  namespace cc = dyng::cycle_count;
  EXPECT_EQ(cc::to_string(cc::search_method::johnson), "johnson");
  EXPECT_EQ(cc::to_string(cc::cycle_mode::simple), "simple");
  EXPECT_EQ(cc::to_string(cc::cuda_scheduler::work_queue), "work_queue");
  EXPECT_EQ(cc::to_string(cc::cuda_scheduler::naive), "naive");
  EXPECT_EQ(cc::to_string(cc::cuda_work_items::automatic), "automatic");
  EXPECT_EQ(cc::to_string(cc::cuda_work_items::roots), "roots");
  EXPECT_EQ(cc::to_string(cc::cuda_work_items::edges), "edges");
  EXPECT_EQ(cc::to_string(cc::cuda_work_items::two_hop), "two_hop");
}

/// io::write_histogram_csv into a string.
[[maybe_unused]] std::string histogram_csv(dyng::array_view<const std::uint64_t> counts,
                                           bool include_total = true) {
  std::ostringstream out;
  dyng::io::write_histogram_csv(out, counts, include_total);
  return out.str();
}

using dyng::backend;
using dyng::resources;
using dyng::unweighted;
using dyng::test::cc_batch;
using dyng::test::cc_counts;
using dyng::test::cc_edge;
using dyng::test::cc_graph;
using dyng::test::cc_resources;
namespace cycle_count = dyng::cycle_count;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_w = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using hist = std::vector<std::uint64_t>;

cycle_count::options bound(int k) {
  cycle_count::options opt;
  opt.max_length = k;
  return opt;
}

/// 0->1, 1->0, 1->2, 2->0, 2->3, 3->1: one 2-cycle and two 3-cycles (the representative graph of
/// the original's static suites).
std::vector<cc_edge> overlapping_cycles() {
  return {{0, 1}, {1, 0}, {1, 2}, {2, 0}, {2, 3}, {3, 1}};
}

class CycleCountBackends : public ::testing::TestWithParam<backend> {
 protected:
  resources res_ = cc_resources(GetParam());
};

// --- JohnsonStaticTest (johnson_static_test.cpp), on every backend -----------------------------

TEST_P(CycleCountBackends, MatchesOracleOnAcyclicGraph) {
  const graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {0, 2}});
  const cycle_count::result r = cycle_count::compute(res_, g);
  EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, -1));
  EXPECT_EQ(r.total(), 0U);
}

TEST_P(CycleCountBackends, MatchesOracleOnSingleTriangle) {
  const graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {2, 0}});
  const cycle_count::result r = cycle_count::compute(res_, g);
  EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, -1));
  EXPECT_EQ(cc_counts(r), (hist{0, 0, 0, 1}));
}

TEST_P(CycleCountBackends, MatchesOracleOnOverlappingCycles) {
  const graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  const cycle_count::result r = cycle_count::compute(res_, g);
  EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, -1));
  EXPECT_EQ(r.count(2), 1U);
  EXPECT_EQ(r.count(3), 2U);
  EXPECT_EQ(r.count(4), 0U);
  EXPECT_EQ(r.total(), 3U);
  EXPECT_EQ(r.bound(), 4);
  EXPECT_EQ(r.counts().size(), 5U);
}

TEST_P(CycleCountBackends, HonorsMaximumCycleLength) {
  const graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 0}, {1, 2}, {2, 0}});
  const cycle_count::result r = cycle_count::compute(res_, g, bound(2));
  EXPECT_EQ(cc_counts(r), dyng::test::cc_brute(res_, g, 2));
  EXPECT_EQ(cc_counts(r), (hist{0, 0, 1}));
  EXPECT_EQ(r.bound(), 2);
  EXPECT_EQ(r.get_options().max_length, 2);
}

TEST_P(CycleCountBackends, RejectsInvalidMaximumCycleLength) {
  const graph_u g = cc_graph<graph_u>(res_, 0, {});
  for (const int k : {1, 0, -2, std::numeric_limits<int>::min()}) {
    EXPECT_THROW((void)cycle_count::compute(res_, g, bound(k)), dyng::invalid_argument_error) << k;
  }
  EXPECT_EQ(cc_counts(cycle_count::compute(res_, g)), (hist{0, 0, 0}));
  EXPECT_EQ(cc_counts(cycle_count::compute(res_, g, bound(4))), (hist{0, 0, 0, 0, 0}));
}

// Regression of the original: bounded Johnson must agree with the oracle on graphs dense enough to
// contain cycles both shorter and longer than the cap (a naive depth cutoff combined with
// Johnson's blocked lists silently drops shorter cycles).
TEST_P(CycleCountBackends, BoundedMatchesOracleOnRandomGraphs) {
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  for (int trial = 0; trial < 30; ++trial) {
    constexpr std::int32_t vertices = 8;
    std::vector<cc_edge> edges;
    for (std::int32_t u = 0; u < vertices; ++u) {
      for (std::int32_t v = 0; v < vertices; ++v) {
        if (u != v && coin(rng) < 0.4) {
          edges.emplace_back(u, v);
        }
      }
    }
    const graph_u g = cc_graph<graph_u>(res_, vertices, edges);
    for (int k = 2; k <= 7; ++k) {
      EXPECT_EQ(cc_counts(cycle_count::compute(res_, g, bound(k))),
                dyng::test::cc_brute(res_, g, k))
          << "trial " << trial << " max_length " << k;
    }
  }
}

// --- SequentialParityTest.StaticAlgorithmsAgreeOnRepresentativeGraphs ---------------------------

TEST_P(CycleCountBackends, StaticAgreesWithBruteForceOnRepresentativeGraphs) {
  const std::vector<std::pair<std::int64_t, std::vector<cc_edge>>> graphs = {
      {4, overlapping_cycles()}, {4, {{0, 1}, {1, 2}, {2, 3}}}};
  for (const auto& [n, edges] : graphs) {
    const graph_u g = cc_graph<graph_u>(res_, n, edges);
    EXPECT_EQ(cc_counts(cycle_count::compute(res_, g)), dyng::test::cc_brute(res_, g, -1));
  }
}

// --- CyclesThroughEdgeTest fixture graph, through the public update --------------------------------

TEST_P(CycleCountBackends, UpdateOfTheCyclesThroughEdgeFixture) {
  // 0->1->2->0 (triangle) and 1->0 (a 2-cycle 0<->1).
  graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {2, 0}, {1, 0}});
  cycle_count::result r = cycle_count::compute(res_, g, bound(8));
  EXPECT_EQ(cc_counts(r), (hist{0, 0, 1, 1, 0, 0, 0, 0, 0}));
  // Deleting 0->1 removes both cycles through it.
  const auto del = cc_batch<unweighted>({{0, 1}}, {});
  const cycle_count::stats st = cycle_count::update(res_, g, del.view(), r);
  EXPECT_EQ(r.total(), 0U);
  EXPECT_EQ(st.deletions, 1);
  EXPECT_EQ(st.insertions, 0);
  EXPECT_EQ(st.cycles_removed, 2U);
  EXPECT_EQ(st.cycles_added, 0U);
  EXPECT_EQ(st.affected, 2);
  EXPECT_EQ(st.batch.deleted_edges, 1);
  // Re-inserting it adds them back; deleting and re-inserting 2->0 is a no-op overall.
  const auto ins = cc_batch<unweighted>({{2, 0}}, {{0, 1}, {2, 0}});
  const cycle_count::stats st2 = cycle_count::update(res_, g, ins.view(), r);
  EXPECT_EQ(cc_counts(r), (hist{0, 0, 1, 1, 0, 0, 0, 0, 0}));
  EXPECT_EQ(st2.deletions, 1);
  EXPECT_EQ(st2.insertions, 2);
  EXPECT_EQ(st2.cycles_removed, 0U);  // 2->0 lies on no cycle once 0->1 is gone
  EXPECT_EQ(st2.cycles_added, 2U);
  EXPECT_EQ(st2.batch.cancelled_pairs, 1);
}

// --- HistogramEngineTest.CountMatchesSequentialRecompute --------------------------------------------

TEST_P(CycleCountBackends, CountMatchesSequentialRecompute) {
  const resources seq = resources::sequential();
  const graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {2, 0}, {1, 0}});
  const graph_u gs = cc_graph<graph_u>(seq, 3, {{0, 1}, {1, 2}, {2, 0}, {1, 0}});
  EXPECT_EQ(cc_counts(cycle_count::compute(res_, g)), cc_counts(cycle_count::compute(seq, gs)));
  EXPECT_EQ(cc_counts(cycle_count::compute(res_, g, bound(2))),
            cc_counts(cycle_count::compute(seq, gs, bound(2))));
}

TEST_P(CycleCountBackends, WeightedGraphsIgnoreWeights) {
  const graph_w gw = cc_graph<graph_w>(res_, 4, overlapping_cycles());
  const graph_u gu = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  EXPECT_EQ(cc_counts(cycle_count::compute(res_, gw, bound(4))),
            cc_counts(cycle_count::compute(res_, gu, bound(4))));
}

TEST_P(CycleCountBackends, UnboundedHistogramGrowsWithTheGraph) {
  graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {2, 0}});
  cycle_count::result r = cycle_count::compute(res_, g);
  EXPECT_EQ(r.bound(), 3);
  // Insertions through new vertices 3 and 4: the ring 0->1->2->3->4->0 of length 5.
  const auto b = cc_batch<unweighted>({{2, 0}}, {{2, 3}, {3, 4}, {4, 0}});
  (void)cycle_count::update(res_, g, b.view(), r);
  EXPECT_EQ(g.num_vertices(), 5);
  EXPECT_EQ(r.bound(), 5);
  EXPECT_EQ(cc_counts(r), (hist{0, 0, 0, 0, 0, 1}));
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g)));
}

TEST_P(CycleCountBackends, EmptyBatchLeavesHistogramUnchanged) {
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res_, g, bound(6));
  const hist before = cc_counts(r);
  const dyng::edge_batch<std::int32_t, unweighted> empty;
  const cycle_count::stats st = cycle_count::update(res_, g, empty.view(), r);
  EXPECT_EQ(cc_counts(r), before);
  EXPECT_EQ(st.affected, 0);
  EXPECT_EQ(st.cycles_added + st.cycles_removed, 0U);
  EXPECT_EQ(r.graph_version(), g.version());
}

TEST_P(CycleCountBackends, RejectsGraphsWithoutSortedSimpleRows) {
  const graph_u g =
      cc_graph<graph_u>(res_, 4, overlapping_cycles(), dyng::graph_properties::mosp_compatible());
  try {
    (void)cycle_count::compute(res_, g);
    FAIL() << "compute accepted a multigraph";
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_NE(std::string(e.what()).find("multi_edges::forbid"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("row_order::sorted"), std::string::npos) << e.what();
  }
  dyng::graph_properties props = dyng::graph_properties::cycle_enum_compatible();
  props.order = dyng::row_order::append;
  props.semantics = dyng::batch_semantics::upsert_last_wins();
  const graph_u g2 = cc_graph<graph_u>(res_, 4, overlapping_cycles(), props);
  EXPECT_THROW((void)cycle_count::compute(res_, g2), dyng::invalid_argument_error);
}

TEST_P(CycleCountBackends, StaleResultsAreRejected) {
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  const auto b = cc_batch<unweighted>({{0, 1}}, {});
  (void)g.apply(res_, b.view());  // the structure changes without the result
  EXPECT_THROW((void)cycle_count::update(res_, g, b.view(), r), dyng::stale_result_error);
  // A result of another graph with the same version.
  graph_u other = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r2 = cycle_count::compute(res_, other, bound(4));
  graph_u g3 = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  EXPECT_THROW((void)cycle_count::update(res_, g3, b.view(), r2), dyng::stale_result_error);
  EXPECT_EQ(g3.version(), 0U);  // nothing was applied
}

TEST_P(CycleCountBackends, NegativeBucketPoisonsTheResult) {
  graph_u g = cc_graph<graph_u>(res_, 3, {{0, 1}, {1, 2}, {2, 0}});
  cycle_count::result r = cycle_count::compute(res_, g, bound(3));
  // Corrupt the histogram: it claims no 3-cycle, then the batch deletes one.
  dyng::detail::cycle_count_access::state(r).counts[3] = 0;
  const auto b = cc_batch<unweighted>({{0, 1}}, {});
  EXPECT_THROW((void)cycle_count::update(res_, g, b.view(), r), dyng::internal_error);
  EXPECT_EQ(g.version(), 1U);  // the graph was updated
  EXPECT_THROW((void)r.counts(), dyng::stale_result_error);
  EXPECT_THROW((void)r.total(), dyng::stale_result_error);
  // Every later use but the noexcept queries throws (ADR 0006 point 5): the bound and the options
  // describe a histogram that no longer matches the graph.
  EXPECT_THROW((void)r.bound(), dyng::stale_result_error);
  EXPECT_THROW((void)r.get_options(), dyng::stale_result_error);
  EXPECT_THROW(r.set_options(bound(3)), dyng::stale_result_error);
  EXPECT_THROW((void)r.clone(res_), dyng::stale_result_error);
  EXPECT_EQ(r.graph_version(), 0U);
  EXPECT_THROW((void)cycle_count::update(res_, g, b.view(), r), dyng::stale_result_error);
  r = cycle_count::compute(res_, g, bound(3));
  EXPECT_EQ(r.total(), 0U);
}

// PLAN 5.1 / ADR 0006 point 2: the tunables change with set_options(); max_length, method and mode
// are fixed at compute().
TEST_P(CycleCountBackends, SetOptionsChangesOnlyTheTunables) {
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  cycle_count::options opt = r.get_options();
  opt.cuda_engine = dyng::engine::fused;
  opt.scheduler = cycle_count::cuda_scheduler::naive;
  opt.work_items = cycle_count::cuda_work_items::edges;
  r.set_options(opt);
  EXPECT_EQ(r.get_options().cuda_engine, dyng::engine::fused);
  EXPECT_EQ(r.get_options().scheduler, cycle_count::cuda_scheduler::naive);
  EXPECT_EQ(r.get_options().work_items, cycle_count::cuda_work_items::edges);
  EXPECT_EQ(r.get_options().max_length, 4);
  cycle_count::options longer = opt;
  longer.max_length = 5;
  EXPECT_THROW(r.set_options(longer), dyng::invalid_argument_error);
  cycle_count::options bad_items = opt;
  bad_items.work_items = static_cast<cycle_count::cuda_work_items>(9);
  EXPECT_THROW(r.set_options(bad_items), dyng::invalid_argument_error);
  EXPECT_EQ(r.get_options().work_items, cycle_count::cuda_work_items::edges);  // unchanged
  const auto b = cc_batch<unweighted>({{0, 1}}, {{3, 0}});
  (void)cycle_count::update(res_, g, b.view(), r);
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(4))));
}

// A graph without edges has no cycle, whatever its size and the bound: on cuda as well (the
// original's device counters return an empty histogram before they check the length bound).
TEST_P(CycleCountBackends, EdgelessGraphsOfAnySizeCount) {
  graph_u g = cc_graph<graph_u>(res_, 100, {});
  const cycle_count::result r = cycle_count::compute(res_, g, cycle_count::options{});
  EXPECT_EQ(r.bound(), 100);
  EXPECT_EQ(cc_counts(r), hist(101, 0));
  EXPECT_EQ(r.total(), 0U);
}

TEST_P(CycleCountBackends, InvalidBatchChangesNothing) {
  dyng::graph_properties props = dyng::graph_properties::cycle_enum_compatible();
  props.semantics.on_missing_delete = dyng::batch_semantics::missing_delete::error;
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles(), props);
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  const hist before = cc_counts(r);
  const auto bad = cc_batch<unweighted>({{3, 2}}, {{0, 3}});
  EXPECT_THROW((void)cycle_count::update(res_, g, bad.view(), r), dyng::invalid_argument_error);
  EXPECT_EQ(g.version(), 0U);
  EXPECT_EQ(cc_counts(r), before);
  const auto negative = cc_batch<unweighted>({}, {{-1, 2}});
  EXPECT_THROW((void)cycle_count::update(res_, g, negative.view(), r),
               dyng::invalid_argument_error);
  EXPECT_EQ(cc_counts(r), before);
  const auto good = cc_batch<unweighted>({{3, 1}}, {{0, 3}});
  (void)cycle_count::update(res_, g, good.view(), r);
  EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(4))));
}

TEST_P(CycleCountBackends, ProfilerStagesFollowTheHooks) {
  dyng::profiler prof;
  resources res = res_;
  res.attach_profiler(&prof);
  graph_u g = cc_graph<graph_u>(res, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res, g, bound(4));
  const auto b = cc_batch<unweighted>({{0, 1}}, {{3, 0}});
  (void)cycle_count::update(res, g, b.view(), r);
  res.attach_profiler(nullptr);
  std::set<std::string> names;
  for (const dyng::stage_record& s : prof.stages()) {
    names.insert(s.name);
  }
  for (const char* name :
       {"cycle_count.compute", "cycle_count.reset", "cycle_count.count", "cycle_count.finalize",
        "cycle_count.update", "cycle_count.normalize", "cycle_count.count_minus",
        "cycle_count.commit", "cycle_count.identify_affected", "cycle_count.count_plus",
        "graph.apply"}) {
    EXPECT_EQ(names.count(name), 1U) << name;
  }
  // The update does not build the in-edges (cycle_count reads the out-edges only).
  EXPECT_EQ(names.count("graph.transpose"), 0U);
  EXPECT_EQ(names.count("graph.transpose_device"), 0U);
  // On cuda the graph is uploaded once (by compute) and stays resident: the update merges the
  // batch on the device.
  EXPECT_EQ(names.count("graph.upload"), GetParam() == backend::cuda ? 1U : 0U);
}

/// The cycle_count stages of compute() and update() in call order (samples end with their
/// stage, so a stage follows the stages nested in it; graph.* stages depend on the graph's lazy
/// builds and are left out). The framework's enactors open exactly these stages
/// (parity/timed_regions/cycle_count.toml sums them; parity/results/M3.md records the migration).
TEST_P(CycleCountBackends, TheEnactorsOpenTheStagesOfTheTimedRegions) {
  // cuda is Tier B: enact_fused / compute_fused open the hooks' stages inside
  // cycle_count.enact_fused.
  const bool cuda = GetParam() == backend::cuda;
  for (const bool as_sets : {true, false}) {
    SCOPED_TRACE(as_sets ? "batch_semantics::set()" : "batch_semantics::upsert_last_wins()");
    dyng::graph_properties props = dyng::graph_properties::cycle_enum_compatible();
    if (!as_sets) {
      props.semantics = dyng::batch_semantics::upsert_last_wins();
    }
    dyng::profiler prof;
    resources res = res_;
    graph_u g = cc_graph<graph_u>(res, 4, overlapping_cycles(), props);
    res.attach_profiler(&prof);
    cycle_count::result r = cycle_count::compute(res, g, bound(4));
    const std::size_t after_compute = prof.samples().size();
    const auto b = cc_batch<unweighted>({{0, 1}}, {{3, 0}});
    (void)cycle_count::update(res, g, b.view(), r);
    res.attach_profiler(nullptr);
    std::vector<std::string> compute_names;
    std::vector<std::string> update_names;
    for (std::size_t i = 0; i < prof.samples().size(); ++i) {
      const std::string& name = prof.samples()[i].name;
      if (name.rfind("cycle_count.", 0) != 0) {
        continue;
      }
      (i < after_compute ? compute_names : update_names).push_back(name);
    }
    std::vector<std::string> compute_expected = {"cycle_count.reset", "cycle_count.count",
                                                 "cycle_count.finalize"};
    if (cuda) {
      compute_expected.emplace_back("cycle_count.enact_fused");
    }
    compute_expected.emplace_back("cycle_count.compute");
    // Under set semantics Step 0 runs once in run_update() (ADR 0020); the problem's normalize
    // hook then takes its lists.
    std::vector<std::string> update_expected = {"cycle_count.normalize"};
    if (as_sets) {
      update_expected.emplace_back("cycle_count.normalize");
    }
    for (const char* name :
         {"cycle_count.count_minus", "cycle_count.commit", "cycle_count.identify_affected",
          "cycle_count.count_plus", "cycle_count.finalize"}) {
      update_expected.emplace_back(name);
    }
    if (cuda) {
      update_expected.emplace_back("cycle_count.enact_fused");
    }
    update_expected.emplace_back("cycle_count.update");
    EXPECT_EQ(compute_names, compute_expected);
    EXPECT_EQ(update_names, update_expected);
    EXPECT_EQ(cc_counts(r), cc_counts(cycle_count::compute(res_, g, bound(4))));
  }
}

TEST_P(CycleCountBackends, SteadyStateUpdatesReuseOneWorkspace) {
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  auto& pool = dyng::detail::resources_access::workspaces(res_);
  for (int i = 0; i < 4; ++i) {
    const auto b = cc_batch<unweighted>({{0, 1}}, {});
    const auto back = cc_batch<unweighted>({}, {{0, 1}});
    (void)cycle_count::update(res_, g, b.view(), r);
    (void)cycle_count::update(res_, g, back.view(), r);
  }
  // One workspace per type: the normalized batch of the framework, cycle_count's host workspace
  // and, on cuda, its device workspace (the compute before leased the latter already).
  const bool cuda = GetParam() == backend::cuda;
  const std::uint64_t types = cuda ? 3U : 2U;
  const auto stats = pool.statistics();
  EXPECT_EQ(stats.created, types);
  EXPECT_EQ(stats.leased, 0U);
  EXPECT_EQ(stats.leases, 8U * types + (cuda ? 1U : 0U));
  EXPECT_GT(res_.workspace_bytes(), 0U);
}

TEST_P(CycleCountBackends, CloneAndMove) {
  graph_u g = cc_graph<graph_u>(res_, 4, overlapping_cycles());
  cycle_count::result r = cycle_count::compute(res_, g, bound(4));
  cycle_count::result copy = r.clone(res_);
  EXPECT_EQ(cc_counts(copy), cc_counts(r));
  EXPECT_EQ(copy.graph_version(), r.graph_version());
  EXPECT_EQ(copy.space(), dyng::memory_space::host);
  const auto b = cc_batch<unweighted>({{0, 1}}, {});
  (void)cycle_count::update(res_, g, b.view(), copy);  // the copy follows the graph
  EXPECT_NE(cc_counts(copy), cc_counts(r));
  cycle_count::result moved = std::move(r);
  EXPECT_EQ(moved.total(), 3U);
  EXPECT_THROW((void)r.counts(), dyng::invalid_argument_error);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(r.graph_version(), 0U);                              // NOLINT(bugprone-use-after-move)
}

TEST_P(CycleCountBackends, ComposesWithSssp) {
  // PLAN 5.3: a weighted graph with the default properties carries both results; one apply.
  const std::vector<cc_edge> edges = {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 1}, {1, 0}, {3, 4}};
  graph_w g = cc_graph<graph_w>(res_, 5, edges, dyng::graph_properties{});
  graph_w g_alone = cc_graph<graph_w>(res_, 5, edges, dyng::graph_properties{});
  auto tree = dyng::sssp::compute(res_, g, 0);
  cycle_count::result cyc = cycle_count::compute(res_, g, bound(4));
  cycle_count::result cyc_alone = cycle_count::compute(res_, g_alone, bound(4));
  const auto b = cc_batch<std::int32_t>({{2, 0}, {3, 1}}, {{4, 0}, {3, 2}, {1, 0}});
  auto [s1, s2] = dyng::update(res_, g, b.view(), tree, cyc);
  const cycle_count::stats alone = cycle_count::update(res_, g_alone, b.view(), cyc_alone);
  EXPECT_EQ(cc_counts(cyc), cc_counts(cyc_alone));
  EXPECT_EQ(cc_counts(cyc), cc_counts(cycle_count::compute(res_, g, bound(4))));
  EXPECT_EQ(s2.cycles_added, alone.cycles_added);
  EXPECT_EQ(s2.cycles_removed, alone.cycles_removed);
  EXPECT_EQ(s1.batch.inserted_edges, s2.batch.inserted_edges);
  const auto fresh = dyng::sssp::compute(res_, g, 0);
  EXPECT_EQ(dyng::to_vector(res_, tree.distances()), dyng::to_vector(res_, fresh.distances()));
  EXPECT_EQ(g.version(), 1U);
}

/// `m` distinct random edges on `n` vertices, without self-loops.
std::set<std::pair<std::int32_t, std::int32_t>> random_edge_set(std::mt19937& rng, std::int32_t n,
                                                                std::size_t m) {
  std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
  std::set<std::pair<std::int32_t, std::int32_t>> edges;
  while (edges.size() < m) {
    const std::int32_t u = vertex(rng);
    const std::int32_t v = vertex(rng);
    if (u != v) {
      edges.emplace(u, v);
    }
  }
  return edges;
}

/// The graph of an edge set; weights[i] is the weight of the i-th edge in set order.
graph_w weighted_graph(const resources& res, std::int32_t n,
                       const std::set<std::pair<std::int32_t, std::int32_t>>& edges,
                       const std::vector<std::int32_t>& weights, dyng::graph_properties props) {
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  std::size_t i = 0;
  for (const auto& [u, v] : edges) {
    list.add_edge(u, v, {weights[i++]});
  }
  props.num_weights = 1;
  return graph_w::from_edges(res, list.view(), props);
}

// PLAN 8.2 C10 on every backend: one dyng::update(res, g, batch, tree, hist) over a chain of
// batches equals sssp::update and cycle_count::update run alone on copies of the graph, with one
// commit per batch; under set semantics (Step 0 once, ADR 0020) and under upsert semantics; both
// orders of the results. A result left out of an update is stale afterwards (C11).
TEST_P(CycleCountBackends, OneUpdateOfSsspAndCycleCountEqualsSeparateUpdates) {
  constexpr std::int32_t n = 40;
  for (const bool as_sets : {true, false}) {
    SCOPED_TRACE(as_sets ? "batch_semantics::set()" : "batch_semantics::upsert_last_wins()");
    dyng::graph_properties props;  // sorted rows, no parallel edges (both algorithms' needs)
    if (as_sets) {
      props.semantics = dyng::batch_semantics::set();
    }
    std::mt19937 rng(as_sets ? 17U : 23U);
    std::set<std::pair<std::int32_t, std::int32_t>> edges = random_edge_set(rng, n, 150);
    std::uniform_int_distribution<std::int32_t> weight(1, 9);
    std::vector<std::int32_t> weights(edges.size());
    for (std::int32_t& w : weights) {
      w = weight(rng);
    }
    graph_w g = weighted_graph(res_, n, edges, weights, props);
    graph_w g_sssp = weighted_graph(res_, n, edges, weights, props);
    graph_w g_cycles = weighted_graph(res_, n, edges, weights, props);
    auto tree = dyng::sssp::compute(res_, g, 0);
    auto tree_alone = dyng::sssp::compute(res_, g_sssp, 0);
    cycle_count::result hist = cycle_count::compute(res_, g, bound(5));
    cycle_count::result hist_alone = cycle_count::compute(res_, g_cycles, bound(5));
    std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
    for (int step = 0; step < 4; ++step) {
      SCOPED_TRACE("batch " + std::to_string(step));
      // Deletions of existing edges and insertions of new ones (valid under both semantics).
      dyng::edge_batch<std::int32_t, std::int32_t> b;
      std::vector<std::pair<std::int32_t, std::int32_t>> existing(edges.begin(), edges.end());
      std::shuffle(existing.begin(), existing.end(), rng);
      for (std::size_t i = 0; i < 12; ++i) {
        b.delete_edge(existing[i].first, existing[i].second);
        edges.erase(existing[i]);
      }
      for (int inserted = 0; inserted < 12;) {
        const std::int32_t u = vertex(rng);
        const std::int32_t v = vertex(rng);
        if (u != v && edges.emplace(u, v).second) {
          b.insert_edge(u, v, {weight(rng)});
          ++inserted;
        }
      }
      dyng::profiler prof;
      resources res = res_;
      res.attach_profiler(&prof);
      dyng::sssp::stats s_sssp;
      cycle_count::stats s_cycles;
      if (step % 2 == 0) {
        std::tie(s_sssp, s_cycles) = dyng::update(res, g, b.view(), tree, hist);
      } else {
        std::tie(s_cycles, s_sssp) = dyng::update(res, g, b.view(), hist, tree);
      }
      res.attach_profiler(nullptr);
      const dyng::sssp::stats a_sssp = dyng::sssp::update(res_, g_sssp, b.view(), tree_alone);
      const cycle_count::stats a_cycles = cycle_count::update(res_, g_cycles, b.view(), hist_alone);
      // One commit for both results, and the stages of both algorithms around it.
      std::int64_t commits = 0;
      std::int64_t own_commits = 0;
      std::int64_t normalizations = 0;
      for (const dyng::stage_record& r : prof.stages()) {
        commits += r.name == "update.commit" ? r.calls : 0;
        own_commits += r.name == "sssp.commit" || r.name == "cycle_count.commit" ? r.calls : 0;
        normalizations += r.name == "update.normalize" ? r.calls : 0;
      }
      EXPECT_EQ(commits, 1);
      EXPECT_EQ(own_commits, 0);
      EXPECT_EQ(normalizations, as_sets ? 1 : 0);
      EXPECT_EQ(dyng::to_vector(res_, tree.distances()),
                dyng::to_vector(res_, tree_alone.distances()));
      EXPECT_EQ(dyng::to_vector(res_, tree.parents()), dyng::to_vector(res_, tree_alone.parents()));
      EXPECT_EQ(cc_counts(hist), cc_counts(hist_alone));
      EXPECT_EQ(s_sssp.affected, a_sssp.affected);
      EXPECT_EQ(s_sssp.invalidated, a_sssp.invalidated);
      EXPECT_EQ(s_sssp.engine_used, a_sssp.engine_used);
      EXPECT_EQ(s_cycles.affected, a_cycles.affected);
      EXPECT_EQ(s_cycles.cycles_added, a_cycles.cycles_added);
      EXPECT_EQ(s_cycles.cycles_removed, a_cycles.cycles_removed);
      EXPECT_EQ(s_cycles.deletions, a_cycles.deletions);
      EXPECT_EQ(s_cycles.insertions, a_cycles.insertions);
      EXPECT_EQ(s_cycles.engine_used, a_cycles.engine_used);
      EXPECT_EQ(s_sssp.batch.inserted_edges, a_cycles.batch.inserted_edges);
      EXPECT_EQ(s_cycles.batch.deleted_edges, a_sssp.batch.deleted_edges);
      EXPECT_EQ(tree.graph_version(), g.version());
      EXPECT_EQ(hist.graph_version(), g.version());
    }
    EXPECT_EQ(g.version(), 4U);
    EXPECT_EQ(cc_counts(hist), cc_counts(cycle_count::compute(res_, g, bound(5))));
    const auto fresh = dyng::sssp::compute(res_, g, 0);
    EXPECT_EQ(dyng::to_vector(res_, tree.distances()), dyng::to_vector(res_, fresh.distances()));
    // C11: an update of one result leaves the other stale.
    dyng::edge_batch<std::int32_t, std::int32_t> more;
    const auto first = *edges.begin();
    more.delete_edge(first.first, first.second);
    (void)dyng::sssp::update(res_, g, more.view(), tree);
    const dyng::edge_batch<std::int32_t, std::int32_t> empty;
    EXPECT_THROW((void)cycle_count::update(res_, g, empty.view(), hist), dyng::stale_result_error);
    EXPECT_THROW((void)dyng::update(res_, g, empty.view(), tree, hist), dyng::stale_result_error);
    EXPECT_EQ(g.version(), 5U);  // the failed updates changed nothing
    EXPECT_EQ(tree.graph_version(), g.version());
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, CycleCountBackends,
                         ::testing::ValuesIn(dyng::test::suite_backends()),
                         [](const ::testing::TestParamInfo<backend>& info) {
                           return dyng::test::cc_name(info.param);
                         });
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CycleCountBackends);

#if !(defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA)  // host-only cases (not repeated on cuda)

// --- OpenMPJohnsonTest (openmp_johnson_test.cpp) ---------------------------------------------------

TEST(CycleCountOpenmp, SingleThreadMatchesSequentialJohnson) {
  if (!dyng::backend_available(backend::openmp)) {
    EXPECT_THROW((void)resources::openmp(2), dyng::not_supported_error);
    GTEST_SKIP() << "OpenMP is not built";
  }
  for (const int threads : {1, 2, 3, 8}) {
    const resources omp = resources::openmp(threads);
    const resources seq = resources::sequential();
    const graph_u g = cc_graph<graph_u>(omp, 4, overlapping_cycles());
    const graph_u gs = cc_graph<graph_u>(seq, 4, overlapping_cycles());
    EXPECT_EQ(cc_counts(cycle_count::compute(omp, g)), cc_counts(cycle_count::compute(seq, gs)));
    EXPECT_EQ(cc_counts(cycle_count::compute(omp, g, bound(2))),
              cc_counts(cycle_count::compute(seq, gs, bound(2))));
  }
}

TEST(CycleCountOpenmp, RejectsInvalidConfiguration) {
  EXPECT_THROW((void)resources::openmp(-1), dyng::error);
  const resources seq = resources::sequential();
  const graph_u g = cc_graph<graph_u>(seq, 4, overlapping_cycles());
  EXPECT_THROW((void)cycle_count::compute(seq, g, bound(1)), dyng::invalid_argument_error);
}

TEST(CycleCountBackend, PlacementIsChecked) {
  const resources seq = resources::sequential();
  const graph_u g = cc_graph<graph_u>(seq, 4, overlapping_cycles());
  if (!dyng::backend_available(backend::cuda)) {
    GTEST_SKIP() << "no CUDA device or CUDA not built";
  }
  const resources cuda = resources::cuda(0);
  const graph_u gd = cc_graph<graph_u>(cuda, 4, overlapping_cycles());
  EXPECT_THROW((void)cycle_count::compute(seq, gd), dyng::invalid_argument_error);
}

// --- OptionsTest (options_test.cpp): the defaults -------------------------------------------------

TEST(CycleCountOptions, DefaultsToJohnsonSimpleCyclesWithoutBound) {
  const cycle_count::options opt;
  EXPECT_EQ(opt.max_length, -1);
  EXPECT_EQ(opt.method, cycle_count::search_method::johnson);
  EXPECT_EQ(opt.mode, cycle_count::cycle_mode::simple);
  // Without a bound every cycle is counted: a 5-ring and a 2-cycle.
  const resources seq = resources::sequential();
  const graph_u g =
      cc_graph<graph_u>(seq, 7, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {5, 6}, {6, 5}});
  const cycle_count::result r = cycle_count::compute(seq, g, opt);
  EXPECT_EQ(r.count(2), 1U);
  EXPECT_EQ(r.count(5), 1U);
  EXPECT_EQ(r.total(), 2U);
}

// --- HistogramTest (histogram_test.cpp): accessors, the CSV format, overflow ----------------------

TEST(CycleCountHistogram, AccessorsAndCsv) {
  const resources seq = resources::sequential();
  // Two disjoint triangles and a 2-cycle: counts 2:1 3:2.
  const graph_u g =
      cc_graph<graph_u>(seq, 8, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {6, 7}, {7, 6}});
  const cycle_count::result r = cycle_count::compute(seq, g, bound(5));
  EXPECT_EQ(r.count(0), 0U);
  EXPECT_EQ(r.count(1), 0U);
  EXPECT_EQ(r.count(2), 1U);
  EXPECT_EQ(r.count(3), 2U);
  EXPECT_EQ(r.count(6), 0U);  // beyond the bound
  EXPECT_EQ(r.count(-3), 0U);
  EXPECT_EQ(r.total(), 3U);
  EXPECT_EQ(histogram_csv(r.counts()), "# cycle_size, num_of_cycles\n2, 1\n3, 2\nTotal, 3\n");
  EXPECT_EQ(histogram_csv(r.counts(), false), "# cycle_size, num_of_cycles\n2, 1\n3, 2\n");
  // AllowsZeroIncrementWithoutCreatingEntry: zero counts print no line.
  const hist zero(6, 0);
  EXPECT_EQ(histogram_csv(dyng::host_view(zero), false), "# cycle_size, num_of_cycles\n");
  EXPECT_EQ(histogram_csv(dyng::host_view(zero)), "# cycle_size, num_of_cycles\nTotal, 0\n");
  // MergesWithDeterministicOrdering: lengths in increasing order.
  const hist merged = {0, 0, 1, 2, 8};
  EXPECT_EQ(histogram_csv(dyng::host_view(merged)),
            "# cycle_size, num_of_cycles\n2, 1\n3, 2\n4, 8\nTotal, 11\n");
}

TEST(CycleCountHistogram, DetectsCountOverflow) {
  std::uint64_t a = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(dyng::detail::cycle_count_checked_add(a, 1), dyng::capacity_error);
  std::uint64_t b = 5;
  dyng::detail::cycle_count_checked_add(b, 7);
  EXPECT_EQ(b, 12U);
  const hist big = {0, 0, std::numeric_limits<std::uint64_t>::max(), 1};
  EXPECT_THROW((void)histogram_csv(dyng::host_view(big)), dyng::capacity_error);
  std::ostringstream untouched;
  EXPECT_THROW(dyng::io::write_histogram_csv(untouched, dyng::host_view(big)),
               dyng::capacity_error);
  EXPECT_TRUE(untouched.str().empty());  // nothing is written on an error
  std::ostringstream failed;
  failed.setstate(std::ios::badbit);
  const hist one = {0, 0, 1};
  EXPECT_THROW(dyng::io::write_histogram_csv(failed, dyng::host_view(one)), dyng::io_error);
}

#endif  // host-only cases

}  // namespace
