// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file tutorial_test.cu
 * @brief The tutorial "Count a new hypergraph pattern (an Op)" as a test: the Ops of patterns.hpp
 *        counted by experimental::count_local_patterns on every backend (cuda, sequential,
 *        openmp) and both instantiations (hypergraph<int32_t, unweighted> and <int32_t, int32_t>),
 *        checked against the brute force of oracle.hpp and triad_count, through batches of
 *        hyperedge deletions and insertions under both id_reuse orders, with an empty, a
 *        deletion-only and an insertion-only batch among them: every Op is updated by
 *        dyng::update() and checked after every batch against the brute force and a fresh
 *        compute() ("baseline + deltas == recount"). Also the counter-example of the tutorial's
 *        section 6: an Op that is not symmetric in its ids agrees with compute() and is wrong
 *        after update().
 *
 * Built by nvcc in CUDA builds (cpp/tests/experimental/CMakeLists.txt: dyng_local_patterns_
 * tutorial_tests, label gpu; ci/gpu_local.sh runs it, also under compute-sanitizer). The
 * tutorial quotes it between its `// [tutorial: ...]` markers.
 */
#include "oracle.hpp"
#include "patterns.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/experimental/local_patterns.cuh>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace dx = dyng::experimental;

// [tutorial: host check]
// nvcc reports a __device__-only callback here (it would end a host backend's run with exit(1)).
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(tutorial::line_graph_triangles);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(tutorial::strong_pairs);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(tutorial::core_triads);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(tutorial::hub_pairs);
// [tutorial: host check end]

/// The resources of a backend (four OpenMP threads).
dyng::resources resources_of(dyng::backend b) {
  switch (b) {
    case dyng::backend::openmp:
      return dyng::resources::openmp(4);
    case dyng::backend::cuda:
      return dyng::resources::cuda();
    case dyng::backend::sequential:
      break;
  }
  return dyng::resources::sequential();
}

/// The backends of this build (cuda when a device is visible).
std::vector<dyng::backend> backends() {
  std::vector<dyng::backend> out = {dyng::backend::sequential};
  if (dyng::backend_available(dyng::backend::openmp)) out.push_back(dyng::backend::openmp);
  if (dyng::backend_available(dyng::backend::cuda)) out.push_back(dyng::backend::cuda);
  return out;
}

/// A random set of members of a hyperedge (1 to 6 of 30 vertices).
std::vector<std::int32_t> random_row(std::mt19937_64& rng) {
  std::set<std::int32_t> m;
  const auto size = static_cast<std::size_t>(1 + rng() % 6);
  while (m.size() < size) m.insert(static_cast<std::int32_t>(rng() % 30));
  return {m.begin(), m.end()};
}

/// The weight of the i-th hyperedge (ignored by the patterns; weights must be positive).
template <typename weight_t>
weight_t weight_of(std::size_t i) {
  if constexpr (dyng::is_unweighted_v<weight_t>) {
    (void)i;
    return {};
  } else {
    return static_cast<weight_t>(1 + i % 7);
  }
}

/// A random hypergraph of 40 hyperedges on 30 vertices, with its line graph.
template <typename graph_t>
graph_t random_hypergraph(const dyng::resources& res, std::uint64_t seed,
                          dyng::id_reuse reuse = dyng::id_reuse::erase_first) {
  std::mt19937_64 rng(seed);
  dyng::hyperedge_list<std::int32_t, typename graph_t::weight_type> list;
  list.num_vertices = 30;
  for (std::size_t i = 0; i < 40; ++i) {
    const std::vector<std::int32_t> row = random_row(rng);
    list.add_hyperedge(dyng::host_view(row), weight_of<typename graph_t::weight_type>(i));
  }
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  props.reuse = reuse;
  return graph_t::from_hyperedges(res, list.view(), props);
}

/// What a batch holds.
enum class batch_kind { empty, deletions, insertions, mixed };

/// A random batch of `kind`: a few deletions of ids below the bound (live or not), a few
/// insertions, both, or nothing.
template <typename graph_t>
dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> random_batch(
    const graph_t& h, std::mt19937_64& rng, batch_kind kind) {
  dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> batch;
  const auto bound = static_cast<std::uint64_t>(h.hyperedge_id_bound());
  if (kind == batch_kind::deletions || kind == batch_kind::mixed) {
    for (std::uint64_t i = 0, n = 1 + rng() % 8; i < n; ++i)
      batch.delete_hyperedge(static_cast<std::int64_t>(rng() % bound));
  }
  if (kind == batch_kind::insertions || kind == batch_kind::mixed) {
    for (std::size_t i = 0, n = 1 + rng() % 8; i < n; ++i) {
      const std::vector<std::int32_t> row = random_row(rng);
      batch.insert_hyperedge(dyng::host_view(row), weight_of<typename graph_t::weight_type>(i));
    }
  }
  return batch;
}

/// The batches of run_tutorial: an empty, a deletion-only and an insertion-only one, then mixed.
constexpr std::array<batch_kind, 6> batch_schedule = {batch_kind::empty,     batch_kind::deletions,
                                                      batch_kind::insertions, batch_kind::mixed,
                                                      batch_kind::mixed,      batch_kind::mixed};

/// The tutorial's sections 3-5 on one backend, one hypergraph type and one id_reuse order.
template <typename graph_t>
void run_tutorial(dyng::backend backend, std::uint64_t seed, dyng::id_reuse reuse) {
  SCOPED_TRACE(std::string("reuse ") + std::string(dyng::to_string(reuse)));
  // [tutorial: compute]
  const dyng::resources res = resources_of(backend);
  graph_t h = random_hypergraph<graph_t>(res, seed, reuse);

  using triangles_t = dx::count_local_patterns<tutorial::line_graph_triangles>;
  using pairs_t = dx::count_local_patterns<tutorial::strong_pairs>;
  triangles_t::result triangles = triangles_t::compute(res, h);
  pairs_t::result pairs = pairs_t::compute(res, h, tutorial::strong_pairs{2});
  // [tutorial: compute end]

  // The other two Ops and triad_count, updated in a second dyng::update() over a copy of the
  // hypergraph (the same state, so the same ids: the batches name ids of both).
  graph_t h2 = h.clone(res);
  using core_t = dx::count_local_patterns<tutorial::core_triads>;
  using hub_t = dx::count_local_patterns<tutorial::hub_pairs>;
  core_t::result cores = core_t::compute(res, h2);
  hub_t::result hubs = hub_t::compute(res, h2, tutorial::hub_pairs{7});
  dyng::triad_count::result motifs = dyng::triad_count::compute(res, h2);

  auto check = [&](const char* when) {
    SCOPED_TRACE(when);
    // [tutorial: check]
    const tutorial::member_rows rows = tutorial::live_rows(res, h);
    EXPECT_EQ(triangles.count(0), tutorial::brute_triangles(rows));
    EXPECT_EQ(triangles.total(), motifs.total());  // the 30 h-motifs of the same triangles
    EXPECT_EQ(pairs.count(0), tutorial::brute_strong_pairs(rows, 2));
    // [tutorial: check end]
    ASSERT_EQ(tutorial::live_rows(res, h2), rows);  // the copy is in the same state
    const std::array<std::uint64_t, 3> core_bins = tutorial::brute_core_triads(rows);
    for (int k = 0; k < 3; ++k) EXPECT_EQ(cores.count(k), core_bins[static_cast<std::size_t>(k)]);
    EXPECT_EQ(hubs.count(0), tutorial::brute_hub_pairs(rows, 7));
    // baseline + deltas == recount: every result was updated, never recomputed
    EXPECT_EQ(triangles.count(0), triangles_t::compute(res, h).count(0));
    EXPECT_EQ(pairs.count(0), pairs_t::compute(res, h, pairs.op()).count(0));
    const core_t::result core_recount = core_t::compute(res, h2);
    for (int k = 0; k < 3; ++k) EXPECT_EQ(cores.count(k), core_recount.count(k));
    EXPECT_EQ(hubs.count(0), hub_t::compute(res, h2, hubs.op()).count(0));
    const dyng::triad_count::result motif_recount = dyng::triad_count::compute(res, h2);
    for (int k = 0; k < motifs.num_bins(); ++k) EXPECT_EQ(motifs.count(k), motif_recount.count(k));
  };
  check("compute");

  std::mt19937_64 rng(seed * 31 + 7);
  for (std::size_t step = 0; step < batch_schedule.size(); ++step) {
    SCOPED_TRACE("batch " + std::to_string(step));
    const auto batch = random_batch(h, rng, batch_schedule[step]);
    core_t::result stale = core_t::compute(res, h);
    const std::uint64_t before = triangles.count(0);
    // [tutorial: update]
    const auto [ts, ps] = dyng::update(res, h, batch.view(), triangles, pairs);
    // ts.delta[0] is the change of the triangle count, ts.removed the triangles the deleted
    // hyperedges took away (counted on the hypergraph before the batch) and ts.added the ones
    // the inserted hyperedges brought (on the hypergraph after it).
    EXPECT_EQ(ts.delta[0], static_cast<std::int64_t>(ts.added - ts.removed));
    EXPECT_EQ(ts.inserted_frontier, static_cast<std::int64_t>(ts.batch.inserted_ids.size()));
    // [tutorial: update end]
    EXPECT_EQ(static_cast<std::int64_t>(triangles.count(0) - before), ts.delta[0]);
    EXPECT_EQ(ps.delta[0], static_cast<std::int64_t>(ps.added - ps.removed));
    EXPECT_EQ(ps.deleted_frontier, ts.deleted_frontier);
    EXPECT_EQ(triangles.graph_version(), h.version());
    // A result that was not part of that update is stale now.
    EXPECT_THROW((void)core_t::update(res, h, batch.view(), stale), dyng::stale_result_error);
    // The other Ops and triad_count, through the same batch on the copy.
    const auto [cs, hs, ms] = dyng::update(res, h2, batch.view(), cores, hubs, motifs);
    EXPECT_EQ(cs.batch.inserted_ids, ts.batch.inserted_ids);
    std::int64_t core_delta = 0;
    for (const std::int64_t d : cs.delta) core_delta += d;
    EXPECT_EQ(core_delta, static_cast<std::int64_t>(cs.added - cs.removed));
    EXPECT_EQ(hs.delta[0], static_cast<std::int64_t>(hs.added - hs.removed));
    EXPECT_EQ(cores.graph_version(), h2.version());
    (void)ms;
    check("after the batch");
  }
}

class Tutorial : public ::testing::TestWithParam<dyng::backend> {};

TEST_P(Tutorial, CountsMatchTheBruteForceAndTriadCountUnweighted) {
  for (const dyng::id_reuse reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      SCOPED_TRACE("seed " + std::to_string(seed));
      run_tutorial<dyng::hypergraph<std::int32_t, dyng::unweighted>>(GetParam(), seed, reuse);
    }
  }
}

TEST_P(Tutorial, CountsMatchTheBruteForceAndTriadCountWeighted) {
  for (const dyng::id_reuse reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    for (std::uint64_t seed = 4; seed <= 5; ++seed) {
      SCOPED_TRACE("seed " + std::to_string(seed));
      run_tutorial<dyng::hypergraph<std::int32_t, std::int32_t>>(GetParam(), seed, reuse);
    }
  }
}

// [tutorial: asymmetric]
/// NOT an Op to copy: it counts the pairs {a, b} with |a| > |b|, which depends on the order of
/// the ids. compute() happens to pass a < b, so it agrees with a brute force over every pair
/// i < j; a frontier phase of update() passes the changed hyperedge as `a`, so update() is wrong.
struct bigger_first_pairs {
  static constexpr int num_bins = 1;
  static constexpr dx::shape shape = dx::shape::pairs;
  std::uint64_t count = 0;

  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows& rows, int a, int b, bins_t& /*bins*/) {
    if (rows.size(a) > rows.size(b)) ++count;  // asymmetric in a and b: wrong under update()
  }

  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (count != 0) bins.add(0, count);
  }
};
// [tutorial: asymmetric end]

DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(bigger_first_pairs);

TEST_P(Tutorial, AnOpThatDependsOnTheOrderOfItsIdsIsWrongAfterAnUpdate) {
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  const dyng::resources res = resources_of(GetParam());
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 8;
  const std::vector<std::vector<std::int32_t>> rows = {{0, 1, 2}, {2, 3}, {3, 4, 5, 6}};
  for (const std::vector<std::int32_t>& row : rows) list.add_hyperedge(dyng::host_view(row));
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  graph_t h = graph_t::from_hyperedges(res, list.view(), props);
  using asym_t = dx::count_local_patterns<bigger_first_pairs>;
  asym_t::result r = asym_t::compute(res, h);
  EXPECT_EQ(r.count(0), 1U);  // {0, 1}: |0| > |1|; {1, 2}: |1| < |2|, so the brute force says 1
  dyng::hyperedge_batch<std::int32_t, dyng::unweighted> batch;
  batch.delete_hyperedge(2);
  (void)asym_t::update(res, h, batch.view(), r);
  // The deletion phase counted {1, 2} from the changed hyperedge 2, as pair(2, 1): |2| > |1|.
  EXPECT_EQ(r.count(0), 0U);
  EXPECT_EQ(asym_t::compute(res, h).count(0), 1U);  // the recount: update() went wrong
}

INSTANTIATE_TEST_SUITE_P(Backends, Tutorial, ::testing::ValuesIn(backends()),
                         [](const ::testing::TestParamInfo<dyng::backend>& info) {
                           return std::string(dyng::to_string(info.param));
                         });

TEST(Tutorial, TheSameOpOnEveryBackend) {
  // [tutorial: backends]
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  const graph_t h = random_hypergraph<graph_t>(dyng::resources::sequential(), 11);
  using pairs_t = dx::count_local_patterns<tutorial::strong_pairs>;
  const std::uint64_t on_host = pairs_t::compute(dyng::resources::sequential(), h).count(0);
  for (const dyng::backend b : backends()) {
    const dyng::resources res = resources_of(b);
    const graph_t copy = h.clone(res);  // the hypergraph moves to the backend of `res`
    EXPECT_EQ(pairs_t::compute(res, copy).count(0), on_host) << dyng::to_string(b);
  }
  // [tutorial: backends end]
}

}  // namespace
