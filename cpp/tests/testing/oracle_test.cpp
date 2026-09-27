// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file oracle_test.cpp
 * @brief dyng::testing: dijkstra() (lowest-id ties) and check_sssp_tree() finds what it must.
 */
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/dijkstra.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();

// 0->1 (1), 0->2 (1), 1->3 (2), 2->3 (2), 3->4 (1); vertex 5 is unreachable.
graph_t diamond(const dyng::resources& res) {
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 6;
  list.num_weights = 2;
  list.add_edge(0, 2, {1, 5});
  list.add_edge(0, 1, {1, 1});
  list.add_edge(2, 3, {2, 1});
  list.add_edge(1, 3, {2, 9});
  list.add_edge(3, 4, {1, 1});
  return graph_t::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
}

TEST(TestingDijkstra, LowestIdTiesAndUnreachableVertices) {
  const auto res = dyng::resources::sequential();
  const graph_t g = diamond(res);
  const auto t0 = dyng::testing::dijkstra(g, 0);
  EXPECT_EQ(t0.distances, (std::vector<std::int64_t>{0, 1, 1, 3, 4, inf}));
  EXPECT_EQ(t0.parents, (std::vector<std::int32_t>{-1, 0, 0, 1, 3, -1}));  // 3: 1 < 2
  const auto t1 = dyng::testing::dijkstra(g, 0, 1);
  EXPECT_EQ(t1.distances, (std::vector<std::int64_t>{0, 1, 5, 6, 7, inf}));
  EXPECT_EQ(t1.parents, (std::vector<std::int32_t>{-1, 0, 0, 2, 3, -1}));
  const auto none = dyng::testing::dijkstra(g, 17);
  EXPECT_EQ(none.distances, std::vector<std::int64_t>(6, inf));
  EXPECT_THROW((void)dyng::testing::dijkstra(g, 0, 2), dyng::invalid_argument_error);
}

TEST(TestingCheckSssp, FindsEveryKindOfError) {
  const auto res = dyng::resources::sequential();
  const graph_t g = diamond(res);
  const std::vector<std::int64_t> d{0, 1, 1, 3, 4, inf};
  std::vector<std::int32_t> p{-1, 0, 0, 1, 3, -1};
  auto check = [&](const std::vector<std::int64_t>& dd, const std::vector<std::int32_t>& pp,
                   bool canonical = true) {
    dyng::testing::check_sssp_options opt;
    opt.require_canonical = canonical;
    return dyng::testing::check_sssp_tree(g, 0, dyng::host_view(dd), dyng::host_view(pp), opt);
  };
  EXPECT_TRUE(check(d, p).ok());

  auto wrong_distance = d;
  wrong_distance[4] = 5;
  const auto c1 = check(wrong_distance, p);
  EXPECT_FALSE(c1.ok());
  EXPECT_EQ(c1.distance_mismatches, 1);
  EXPECT_EQ(c1.inconsistent_parents, 1);  // 3 -> 4 is not tight any more

  auto non_canonical = p;
  non_canonical[3] = 2;  // tight, but 1 < 2
  const auto c2 = check(d, non_canonical);
  EXPECT_FALSE(c2.ok());
  EXPECT_EQ(c2.non_canonical_parents, 1);
  EXPECT_EQ(c2.parent_mismatches, 1);
  EXPECT_TRUE(check(d, non_canonical, false).ok());  // consistent when ties are free

  auto not_an_edge = p;
  not_an_edge[4] = 0;
  EXPECT_EQ(check(d, not_an_edge, false).inconsistent_parents, 1);

  auto unreachable_with_parent = p;
  unreachable_with_parent[5] = 4;
  EXPECT_EQ(check(d, unreachable_with_parent, false).inconsistent_parents, 1);

  auto source_with_parent = p;
  source_with_parent[0] = 1;
  EXPECT_FALSE(check(d, source_with_parent, false).ok());

  EXPECT_THROW((void)check({0, 1}, {-1, 0}), dyng::invalid_argument_error);
  EXPECT_NE(c1.summary().find("distance mismatches=1"), std::string::npos);
}

}  // namespace
