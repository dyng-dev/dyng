// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file tutorial_host_test.cpp
 * @brief The host-only half of the tutorial "Count a new hypergraph pattern (an Op)": what a
 *        build without nvcc can check (hosted CPU CI). The Ops of patterns.hpp are host-compilable,
 *        so their callbacks run here on the rows experimental::access_rows() gives (every triple
 *        and pair of live hyperedges, enumerated directly, not by the engine), against the brute
 *        force of oracle.hpp; and the brute-force triangle count against triad_count's h_motif
 *        total on the host backends. The engine's runs of the same Ops need nvcc: tutorial_test.cu.
 */
#include "oracle.hpp"
#include "patterns.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace dx = dyng::experimental;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

/// A bins handle for driving an Op by hand: plain host bins.
struct plain_bins {
  std::array<std::uint64_t, 4> bins{};
  void add(int bin, std::uint64_t value) {
    bins[static_cast<std::size_t>(bin)] += value;
  }
  void overflow() {}
};

hgraph random_hypergraph(const dyng::resources& res, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 25;
  for (int i = 0; i < 35; ++i) {
    std::set<std::int32_t> m;
    const auto size = static_cast<std::size_t>(1 + rng() % 5);
    while (m.size() < size) m.insert(static_cast<std::int32_t>(rng() % 25));
    const std::vector<std::int32_t> row(m.begin(), m.end());
    list.add_hyperedge(dyng::host_view(row));
  }
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  return hgraph::from_hyperedges(res, list.view(), props);
}

/// The live ids of a state's rows.
std::vector<int> live_ids(const dx::row_access& a) {
  std::vector<int> out;
  for (std::int64_t id = 0; id < a.id_bound(); ++id)
    if (a.hyperedges().size(static_cast<int>(id)) > 0) out.push_back(static_cast<int>(id));
  return out;
}

/// An Op of shape pairs driven over every adjacent pair of live hyperedges.
template <typename op_t>
std::array<std::uint64_t, 4> run_pairs(const dx::row_access& a, op_t op) {
  const dx::hyperedge_rows rows = a.hyperedges();
  const std::vector<int> ids = live_ids(a);
  plain_bins bins;
  for (std::size_t i = 0; i < ids.size(); ++i)
    for (std::size_t j = i + 1; j < ids.size(); ++j)
      if (dx::intersect_size(rows.row(ids[i]), rows.row(ids[j])) > 0)
        op.pair(rows, ids[i], ids[j], bins);
  op.flush(bins);
  return bins.bins;
}

/// An Op of shape triangles driven over every triangle of the line graph.
template <typename op_t>
std::array<std::uint64_t, 4> run_triangles(const dx::row_access& a, op_t op) {
  const dx::hyperedge_rows rows = a.hyperedges();
  const std::vector<int> ids = live_ids(a);
  auto adjacent = [&](int x, int y) { return dx::intersect_size(rows.row(x), rows.row(y)) > 0; };
  plain_bins bins;
  for (std::size_t i = 0; i < ids.size(); ++i)
    for (std::size_t j = i + 1; j < ids.size(); ++j) {
      if (!adjacent(ids[i], ids[j])) continue;
      op.begin_wedge(rows, ids[i], ids[j]);
      for (std::size_t k = j + 1; k < ids.size(); ++k)
        if (adjacent(ids[i], ids[k]) && adjacent(ids[j], ids[k])) op.triangle(rows, ids[k], bins);
    }
  op.flush(bins);
  return bins.bins;
}

class TutorialHost : public ::testing::TestWithParam<dyng::backend> {};

TEST_P(TutorialHost, TheOpsCallbacksMatchTheBruteForce) {
  const dyng::resources res = GetParam() == dyng::backend::openmp ? dyng::resources::openmp(4)
                                                                  : dyng::resources::sequential();
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    const hgraph h = random_hypergraph(res, seed);
    const dx::row_access a = dx::access_rows(h);
    ASSERT_EQ(a.space(), dyng::memory_space::host);
    const tutorial::member_rows rows = tutorial::live_rows(res, h);
    EXPECT_EQ(run_triangles(a, tutorial::line_graph_triangles{})[0],
              tutorial::brute_triangles(rows));
    EXPECT_EQ(run_pairs(a, tutorial::strong_pairs{2})[0], tutorial::brute_strong_pairs(rows, 2));
    EXPECT_EQ(run_pairs(a, tutorial::strong_pairs{3})[0], tutorial::brute_strong_pairs(rows, 3));
    const std::array<std::uint64_t, 3> want = tutorial::brute_core_triads(rows);
    const std::array<std::uint64_t, 4> got = run_triangles(a, tutorial::core_triads{});
    for (std::size_t k = 0; k < 3; ++k) EXPECT_EQ(got[k], want[k]) << "core bin " << k;
    EXPECT_EQ(run_pairs(a, tutorial::hub_pairs{5})[0], tutorial::brute_hub_pairs(rows, 5));
    // The line graph's triangles are triad_count's triads.
    EXPECT_EQ(tutorial::brute_triangles(rows), dyng::triad_count::compute(res, h).total());
  }
}

std::vector<dyng::backend> host_backends() {
  std::vector<dyng::backend> out = {dyng::backend::sequential};
  if (dyng::backend_available(dyng::backend::openmp)) out.push_back(dyng::backend::openmp);
  return out;
}

INSTANTIATE_TEST_SUITE_P(Backends, TutorialHost, ::testing::ValuesIn(host_backends()),
                         [](const ::testing::TestParamInfo<dyng::backend>& info) {
                           return std::string(dyng::to_string(info.param));
                         });

}  // namespace
