// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_tutorial_test.cpp
 * @brief The C++ code of the "Shortest hyperpaths" tutorial (docs/tutorials/shortest_hyperpaths.md)
 *        beyond its example program: a batch built in code with hyperedge operations and incidence
 *        edits, the vertex distances, a shortest hyperpath walked back through the parents, the
 *        work budget read from stats::relaxations, and the check against the oracle
 *        dyng::testing::hyper_sssp_reference. The page quotes the marked ranges
 *        (ci/tutorial_quotes.py pins them); the expected values are the page's.
 */
#include <dyng/dyng.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using hypergraph_t = dyng::hypergraph<std::int32_t, std::int32_t>;
namespace hs = dyng::hyper_sssp;
constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();

hypergraph_t example_hypergraph(const dyng::resources& res) {
  const auto hes = dyng::io::read_hypergraph<std::int32_t, std::int32_t>(
      std::string(DYNG_EXAMPLES_DATA) + "/hyper_sssp.hg");
  dyng::hypergraph_properties hp;
  hp.storage = dyng::hypergraph_storage::slack_csr;
  hp.line_graph = dyng::line_graph_policy::slack_csr;
  return hypergraph_t::from_hyperedges(res, hes.view(), hp);
}

TEST(HyperSsspTutorial, BatchVertexDistancesPathBudgetAndOracle) {
  const dyng::resources res = dyng::resources::sequential();
  hypergraph_t hg = example_hypergraph(res);
  hs::result<std::int32_t> r = hs::compute(res, hg, /*source vertex=*/0);
  // [tutorial: batch]
  dyng::hyperedge_batch<std::int32_t, std::int32_t> batch;
  batch.delete_hyperedge(3);                      // -h 3
  batch.delete_incidence(5, 7);                   // -i 5 7: vertex 7 leaves hyperedge 5
  batch.insert_incidence(1, 6);                   // +i 1 6: vertex 6 joins hyperedge 1
  batch.insert_hyperedge({0, 11}, /*weight=*/6);  // +h 0 11 ; 6
  hs::stats st = hs::update(res, hg, batch.view(), r);
  // [tutorial: batch end]
  EXPECT_EQ(st.batch.deleted_incidences, 1);
  EXPECT_EQ(st.batch.inserted_incidences, 1);
  EXPECT_EQ(st.batch.inserted_ids, std::vector<std::int64_t>{3});
  // [tutorial: vertex distances]
  const dyng::array_view<const std::int64_t> vd = r.vertex_distances();  // host memory
  EXPECT_EQ(vd[0], 0);                                                   // the source
  EXPECT_EQ(vd[11], r.distance_to(11));  // the cheapest hyperedge holding vertex 11: id 3
  EXPECT_EQ(vd[11], 6);
  // [tutorial: vertex distances end]
  // [tutorial: path]
  const auto d = r.hyperedge_distances();  // host memory on the sequential backend
  const auto p = r.hyperedge_parents();
  std::vector<std::int32_t> path;  // the hyperedges from the source to hyperedge 6, reversed
  for (std::int32_t h = 6; h != -1; h = p[static_cast<std::size_t>(h)]) path.push_back(h);
  EXPECT_EQ(path, (std::vector<std::int32_t>{6, 7, 3}));  // 3 holds vertex 0, then 7, then 6
  EXPECT_EQ(d[6], 6 + 2 + 5);  // w(3) + w(7) + w(6): every hyperedge entered costs its weight
  // [tutorial: path end]
  // [tutorial: budget]
  hs::options o = r.get_options();
  o.work_budget = 0.0;  // recompute at the first push round or expansion
  r.set_options(o);
  dyng::hyperedge_batch<std::int32_t, std::int32_t> shortcut;
  shortcut.insert_hyperedge({0, 9}, /*weight=*/1);
  st = hs::update(res, hg, shortcut.view(), r);
  EXPECT_TRUE(st.fallback_used);
  EXPECT_EQ(st.fallback_reason, hs::fallback_reason::work_budget);
  EXPECT_GT(st.relaxations, 0);  // the work the budget counted: tune work_budget from it
  // [tutorial: budget end]
  // [tutorial: oracle]
  const auto rows = hg.h2v();  // the hyperedges by id
  const std::vector<std::int64_t> offsets = dyng::to_vector(res, rows.row_ptr);
  const std::vector<std::int32_t> members = dyng::to_vector(res, rows.col_ind);
  std::vector<std::int64_t> weights;
  for (const std::int32_t w : hg.weights()) weights.push_back(w);
  const auto [bad_distances, bad_parents] = dyng::testing::check_hyper_sssp_tree(
      offsets, members, hg.num_vertices(), weights, r.source(), /*source_is_vertex=*/true,
      dyng::to_vector(res, r.hyperedge_distances()), dyng::to_vector(res, r.hyperedge_parents()));
  EXPECT_EQ(bad_distances, 0);
  EXPECT_EQ(bad_parents, 0);
  // [tutorial: oracle end]
  EXPECT_LT(r.hyperedge_distances()[10], inf);
}

}  // namespace
