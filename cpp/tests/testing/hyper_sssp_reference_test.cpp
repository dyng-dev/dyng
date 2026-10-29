// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_reference_test.cpp
 * @brief dyng::testing::hyper_sssp_reference, check_hyper_sssp_tree and line_graph_delta (M11a
 *        criterion 5, ADR 0082): hand cases of both source kinds (MOSP_ESCHER's forced
 *        disconnection case among them), zero weights, dead rows, repeated members, the canonical
 *        (lowest-id tight) parents, the parent errors counted even where the distances differ,
 *        longer results, the line-graph delta, a brute force over random hypergraphs, and the
 *        refusal of malformed input.
 */
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace ts = dyng::testing;
using rows = std::vector<std::vector<std::int32_t>>;
using pairs = std::vector<std::pair<std::int64_t, std::int64_t>>;
constexpr std::int64_t inf = INT64_MAX / 4;

struct csr {
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> members;
};

csr to_csr(const rows& r) {
  csr c;
  for (const auto& row : r) {
    c.members.insert(c.members.end(), row.begin(), row.end());
    c.offsets.push_back(static_cast<std::int64_t>(c.members.size()));
  }
  return c;
}

// MOSP_ESCHER's forced-disconnection hypergraph (tests/unit/test_hsosp_matches_dijkstra.cu): the
// virtual source {0} and target {6} rows weigh 0.
const rows chain = {{0}, {0, 1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6}};
const std::vector<std::int64_t> chain_w = {0, 5, 7, 3, 11, 2, 0};

TEST(HyperSsspReference, HyperedgeSourceOnTheOriginalsChain) {
  const csr c = to_csr(chain);
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 7, chain_w, 0);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, 5, 12, 15, 26, 28, 28}));
  EXPECT_EQ(p, (std::vector<std::int32_t>{-1, 0, 1, 2, 3, 4, 5}));
}

TEST(HyperSsspReference, DeletingTheBridgeDisconnects) {
  rows r = chain;
  r[3].clear();  // the bridge hyperedge {3, 4} deleted (B id 4)
  const csr c = to_csr(r);
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 7, chain_w, 0);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, 5, 12, inf, inf, inf, inf}));
  EXPECT_EQ(p, (std::vector<std::int32_t>{-1, 0, 1, -1, -1, -1, -1}));
}

TEST(HyperSsspReference, VertexSourceStartsAtTheWeights) {
  const csr c = to_csr(chain);
  // From vertex 2: rows 1 {0,1,2} and 2 {2,3} start at their weights 5 and 7.
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 7, chain_w, 2, true);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0 + 5, 5, 7, 10, 21, 23, 23}));
  // Row 0 is reached from row 1 (D = 5 + 0); rows 1 and 2 from the source itself; row 3 from
  // row 2 (the only tight neighbour).
  EXPECT_EQ(p, (std::vector<std::int32_t>{1, -1, -1, 2, 3, 4, 5}));
}

TEST(HyperSsspReference, AVertexSourceBeatsALowerTightNeighbour) {
  // Row 1 holds the source vertex 0 and is tight both from the source and from row 0 (weight 0):
  // the source itself wins, as the original's virtual source row (the lowest id) does.
  const csr c = to_csr({{0}, {0, 1}});
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 2, {0, 4}, 0, true);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, 4}));
  EXPECT_EQ(p, (std::vector<std::int32_t>{-1, -1}));
}

TEST(HyperSsspReference, TheLowestIdTightNeighbourIsTheParent) {
  // 0 -> {1, 2} at equal cost -> 3 is tight from both 1 and 2: the parent is 1.
  const csr c = to_csr({{0, 1, 2}, {1, 3}, {2, 4}, {3, 4}});
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 5, {0, 3, 3, 1}, 0);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, 3, 3, 4}));
  EXPECT_EQ(p, (std::vector<std::int32_t>{-1, 0, 0, 1}));
}

TEST(HyperSsspReference, ZeroWeightsDeadRowsAndRepeatedMembers) {
  // Row 1 is dead (empty): it is nobody's neighbour although row 2 would share a vertex with it
  // if it were alive. Row 2 repeats a member. Row 3 is unreachable.
  const csr c = to_csr({{0, 1}, {}, {1, 1, 2}, {5}});
  const auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 6, {0, 9, 0, 0}, 0);
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, inf, 0, inf}));
  EXPECT_EQ(p, (std::vector<std::int32_t>{-1, -1, 0, -1}));
  // A dead source reaches nothing.
  const auto [d1, p1] = ts::hyper_sssp_reference(c.offsets, c.members, 6, {0, 9, 0, 0}, 1);
  EXPECT_EQ(d1, (std::vector<std::int64_t>{inf, inf, inf, inf}));
  EXPECT_EQ(p1, (std::vector<std::int32_t>{-1, -1, -1, -1}));
}

TEST(CheckHyperSsspTree, CountsDistancesAndParentsApart) {
  const csr c = to_csr(chain);
  auto [d, p] = ts::hyper_sssp_reference(c.offsets, c.members, 7, chain_w, 0);
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d, p),
            (std::pair<std::int64_t, std::int64_t>{0, 0}));
  // A wrong parent alone.
  auto p_bad = p;
  p_bad[4] = 2;
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d, p_bad),
            (std::pair<std::int64_t, std::int64_t>{0, 1}));
  // A wrong distance does not hide a wrong parent elsewhere (the original stops at distances).
  auto d_bad = d;
  d_bad[5] = 27;
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d_bad, p_bad),
            (std::pair<std::int64_t, std::int64_t>{1, 1}));
  // A wrong distance with its consistent parent: the parent is checked against the reference.
  d_bad = d;
  d_bad[6] = 30;
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d_bad, p),
            (std::pair<std::int64_t, std::int64_t>{1, 0}));
  // Extra ids must be dead or unreachable; missing ids count as unreachable.
  auto d_long = d;
  auto p_long = p;
  d_long.push_back(inf);
  p_long.push_back(-1);
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d_long, p_long),
            (std::pair<std::int64_t, std::int64_t>{0, 0}));
  d_long.back() = 3;
  p_long.back() = 0;
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d_long, p_long),
            (std::pair<std::int64_t, std::int64_t>{1, 1}));
  d.pop_back();
  p.pop_back();
  EXPECT_EQ(ts::check_hyper_sssp_tree(c.offsets, c.members, 7, chain_w, 0, false, d, p),
            (std::pair<std::int64_t, std::int64_t>{1, 1}));
}

TEST(LineGraphDelta, PairsOnlyBeforeAndOnlyAfter) {
  const csr pre = to_csr({{0, 1}, {1, 2}, {3}});
  // Row 0 loses vertex 1 (pair (0, 1) deleted), row 2 gains vertex 0 (pair (0, 2) inserted), a
  // new row 3 {2, 3} joins rows 1 and 2.
  const csr post = to_csr({{0}, {1, 2}, {3, 0}, {2, 3}});
  const auto [del, ins] =
      ts::line_graph_delta(pre.offsets, pre.members, post.offsets, post.members, 4);
  EXPECT_EQ(del, (pairs{{0, 1}}));
  EXPECT_EQ(ins, (pairs{{0, 2}, {1, 3}, {2, 3}}));
  const auto [del2, ins2] =
      ts::line_graph_delta(post.offsets, post.members, post.offsets, post.members, 4);
  EXPECT_TRUE(del2.empty());
  EXPECT_TRUE(ins2.empty());
  // A shorter post state: the missing ids are dead.
  const csr shorter = to_csr({{0}});
  const auto [del3, ins3] =
      ts::line_graph_delta(post.offsets, post.members, shorter.offsets, shorter.members, 4);
  EXPECT_EQ(del3, (pairs{{0, 2}, {1, 3}, {2, 3}}));
  EXPECT_TRUE(ins3.empty());
}

// Bellman-Ford by definition (no heap, no line graph): the independent check of the brute force.
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> brute(
    const rows& r, const std::vector<std::int64_t>& w, std::int64_t source, bool vertex) {
  const std::size_t m = r.size();
  const auto shares = [&](std::size_t a, std::size_t b) {
    if (a == b || r[a].empty() || r[b].empty()) return false;
    for (auto x : r[a])
      for (auto y : r[b])
        if (x == y) return true;
    return false;
  };
  const auto holds = [&](std::size_t a) {
    return std::find(r[a].begin(), r[a].end(), static_cast<std::int32_t>(source)) != r[a].end();
  };
  std::vector<std::int64_t> d(m, inf);
  if (vertex) {
    for (std::size_t a = 0; a < m; ++a)
      if (holds(a)) d[a] = w[a];
  } else if (!r[static_cast<std::size_t>(source)].empty()) {
    d[static_cast<std::size_t>(source)] = 0;
  }
  for (std::size_t round = 0; round < m; ++round)
    for (std::size_t a = 0; a < m; ++a)
      for (std::size_t b = 0; b < m; ++b)
        if (d[a] < inf && shares(a, b) && d[a] + w[b] < d[b]) d[b] = d[a] + w[b];
  std::vector<std::int32_t> p(m, -1);
  for (std::size_t b = 0; b < m; ++b) {
    if (d[b] >= inf) continue;
    if (!vertex && b == static_cast<std::size_t>(source)) continue;
    if (vertex && holds(b) && d[b] == w[b]) continue;
    for (std::size_t a = 0; a < m; ++a)
      if (shares(a, b) && d[a] < inf && d[a] + w[b] == d[b]) {
        p[b] = static_cast<std::int32_t>(a);
        break;
      }
  }
  return {d, p};
}

TEST(HyperSsspReference, AgreesWithABruteForce) {
  std::mt19937_64 rng(20261028);
  for (int trial = 0; trial < 300; ++trial) {
    const auto m = static_cast<std::size_t>(1 + rng() % 14);
    const auto n = static_cast<std::int32_t>(1 + rng() % 10);
    rows r(m);
    std::vector<std::int64_t> w(m);
    for (std::size_t a = 0; a < m; ++a) {
      const std::size_t k = rng() % 4 == 0 ? 0 : 1 + rng() % 4;  // a quarter of the rows dead
      for (std::size_t j = 0; j < k; ++j) r[a].push_back(static_cast<std::int32_t>(rng() % n));
      w[a] = static_cast<std::int64_t>(rng() % 4);  // weights 0..3: many ties
    }
    const csr c = to_csr(r);
    const bool vertex = trial % 2 == 1;
    const std::int64_t source =
        vertex ? static_cast<std::int64_t>(rng() % n) : static_cast<std::int64_t>(rng() % m);
    const auto got = ts::hyper_sssp_reference(c.offsets, c.members, n, w, source, vertex);
    const auto want = brute(r, w, source, vertex);
    ASSERT_EQ(got.first, want.first) << "trial " << trial;
    ASSERT_EQ(got.second, want.second) << "trial " << trial;
  }
}

TEST(LineGraphDelta, AgreesWithABruteForce) {
  std::mt19937_64 rng(7);
  for (int trial = 0; trial < 200; ++trial) {
    const auto n = static_cast<std::int32_t>(1 + rng() % 8);
    rows pre(1 + rng() % 10), post(1 + rng() % 10);
    for (auto* r : {&pre, &post})
      for (auto& row : *r)
        for (std::size_t j = 0, k = rng() % 4; j < k; ++j)
          row.push_back(static_cast<std::int32_t>(rng() % n));
    const auto adjacent = [](const rows& r) {
      std::set<std::pair<std::int64_t, std::int64_t>> s;
      for (std::size_t a = 0; a < r.size(); ++a)
        for (std::size_t b = a + 1; b < r.size(); ++b)
          for (auto x : r[a])
            if (std::find(r[b].begin(), r[b].end(), x) != r[b].end())
              s.insert({static_cast<std::int64_t>(a), static_cast<std::int64_t>(b)});
      return s;
    };
    const auto s0 = adjacent(pre);
    const auto s1 = adjacent(post);
    pairs del, ins;
    std::set_difference(s0.begin(), s0.end(), s1.begin(), s1.end(), std::back_inserter(del));
    std::set_difference(s1.begin(), s1.end(), s0.begin(), s0.end(), std::back_inserter(ins));
    const csr a = to_csr(pre);
    const csr b = to_csr(post);
    const auto got = ts::line_graph_delta(a.offsets, a.members, b.offsets, b.members, n);
    ASSERT_EQ(got.first, del) << "trial " << trial;
    ASSERT_EQ(got.second, ins) << "trial " << trial;
  }
}

TEST(HyperSsspReference, RefusesMalformedInput) {
  const csr c = to_csr(chain);
  const auto call = [&](const std::vector<std::int64_t>& off, const std::vector<std::int32_t>& mem,
                        std::int32_t n, const std::vector<std::int64_t>& w, std::int64_t s,
                        bool v) { return ts::hyper_sssp_reference(off, mem, n, w, s, v); };
  EXPECT_THROW((void)call({1, 2}, {0}, 7, {0}, 0, false), std::invalid_argument);
  EXPECT_THROW((void)call({0, 2, 1}, {0, 1}, 7, {0, 0}, 0, false), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, 6, chain_w, 0, false), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, 7, {0, 1}, 0, false), std::invalid_argument);
  auto neg = chain_w;
  neg[2] = -1;
  EXPECT_THROW((void)call(c.offsets, c.members, 7, neg, 0, false), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, 7, chain_w, 7, false), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, 7, chain_w, -1, false), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, 7, chain_w, 7, true), std::invalid_argument);
  EXPECT_THROW((void)call(c.offsets, c.members, -1, chain_w, 0, false), std::invalid_argument);
  auto huge = chain_w;
  huge[1] = inf / 2;
  huge[2] = inf / 2;
  EXPECT_THROW((void)call(c.offsets, c.members, 7, huge, 0, false), std::overflow_error);
  EXPECT_THROW((void)ts::line_graph_delta({0, 1}, {9}, c.offsets, c.members, 7),
               std::invalid_argument);
  EXPECT_THROW((void)ts::check_hyper_sssp_tree({0, 1}, {0}, 1, {}, 0, false, {}, {}),
               std::invalid_argument);
}

}  // namespace
