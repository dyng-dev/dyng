// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file tutorial_test.cu
 * @brief The tutorial "Count a new hypergraph pattern (an Op)" as a test: the Ops of patterns.hpp
 *        counted by experimental::count_local_patterns on every backend (cuda, sequential,
 *        openmp) and both instantiations (hypergraph<int32_t, unweighted> and <int32_t, int32_t>),
 *        checked against the brute force of oracle.hpp and triad_count, through batches of
 *        hyperedge deletions and insertions ("baseline + deltas == recount").
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
graph_t random_hypergraph(const dyng::resources& res, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  dyng::hyperedge_list<std::int32_t, typename graph_t::weight_type> list;
  list.num_vertices = 30;
  for (std::size_t i = 0; i < 40; ++i) {
    const std::vector<std::int32_t> row = random_row(rng);
    list.add_hyperedge(dyng::host_view(row), weight_of<typename graph_t::weight_type>(i));
  }
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  return graph_t::from_hyperedges(res, list.view(), props);
}

/// A random batch: a few deletions of ids below the bound (live or not) and a few insertions.
template <typename graph_t>
dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> random_batch(
    const graph_t& h, std::mt19937_64& rng) {
  dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> batch;
  const auto bound = static_cast<std::uint64_t>(h.hyperedge_id_bound());
  for (std::uint64_t i = 0, n = rng() % 8; i < n; ++i)
    batch.delete_hyperedge(static_cast<std::int64_t>(rng() % bound));
  for (std::size_t i = 0, n = rng() % 8; i < n; ++i) {
    const std::vector<std::int32_t> row = random_row(rng);
    batch.insert_hyperedge(dyng::host_view(row), weight_of<typename graph_t::weight_type>(i));
  }
  return batch;
}

/// The tutorial's sections 3-5 on one backend and one hypergraph type.
template <typename graph_t>
void run_tutorial(dyng::backend backend, std::uint64_t seed) {
  // [tutorial: compute]
  const dyng::resources res = resources_of(backend);
  graph_t h = random_hypergraph<graph_t>(res, seed);

  using triangles_t = dx::count_local_patterns<tutorial::line_graph_triangles>;
  using pairs_t = dx::count_local_patterns<tutorial::strong_pairs>;
  triangles_t::result triangles = triangles_t::compute(res, h);
  pairs_t::result pairs = pairs_t::compute(res, h, tutorial::strong_pairs{2});
  // [tutorial: compute end]

  using core_t = dx::count_local_patterns<tutorial::core_triads>;
  using hub_t = dx::count_local_patterns<tutorial::hub_pairs>;
  core_t::result cores = core_t::compute(res, h);
  hub_t::result hubs = hub_t::compute(res, h, tutorial::hub_pairs{7});
  dyng::triad_count::result motifs = dyng::triad_count::compute(res, h);

  auto check = [&](const char* when) {
    SCOPED_TRACE(when);
    // [tutorial: check]
    const tutorial::member_rows rows = tutorial::live_rows(res, h);
    EXPECT_EQ(triangles.count(0), tutorial::brute_triangles(rows));
    EXPECT_EQ(triangles.total(), motifs.total());  // the 30 h-motifs of the same triangles
    EXPECT_EQ(pairs.count(0), tutorial::brute_strong_pairs(rows, 2));
    // [tutorial: check end]
    const std::array<std::uint64_t, 3> core_bins = tutorial::brute_core_triads(rows);
    for (int k = 0; k < 3; ++k) EXPECT_EQ(cores.count(k), core_bins[static_cast<std::size_t>(k)]);
    EXPECT_EQ(hubs.count(0), tutorial::brute_hub_pairs(rows, 7));
    // baseline + deltas == recount
    EXPECT_EQ(triangles.count(0), triangles_t::compute(res, h).count(0));
    EXPECT_EQ(pairs.count(0), pairs_t::compute(res, h, pairs.op()).count(0));
    EXPECT_EQ(hubs.count(0), hub_t::compute(res, h, hubs.op()).count(0));
  };
  check("compute");

  std::mt19937_64 rng(seed * 31 + 7);
  for (int step = 0; step < 4; ++step) {
    SCOPED_TRACE("batch " + std::to_string(step));
    const auto batch = random_batch(h, rng);
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
    // The other results were not part of that update: they are stale now. Recompute them.
    EXPECT_THROW((void)core_t::update(res, h, batch.view(), cores), dyng::stale_result_error);
    cores = core_t::compute(res, h);
    hubs = hub_t::compute(res, h, hubs.op());
    motifs = dyng::triad_count::compute(res, h);
    check("after the batch");
  }
}

class Tutorial : public ::testing::TestWithParam<dyng::backend> {};

TEST_P(Tutorial, CountsMatchTheBruteForceAndTriadCountUnweighted) {
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    run_tutorial<dyng::hypergraph<std::int32_t, dyng::unweighted>>(GetParam(), seed);
  }
}

TEST_P(Tutorial, CountsMatchTheBruteForceAndTriadCountWeighted) {
  for (std::uint64_t seed = 4; seed <= 5; ++seed) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    run_tutorial<dyng::hypergraph<std::int32_t, std::int32_t>>(GetParam(), seed);
  }
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
