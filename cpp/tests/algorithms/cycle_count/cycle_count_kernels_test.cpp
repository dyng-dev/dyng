// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_kernels_test.cpp
 * @brief The private building blocks of cycle_count: count_cycles_through_edge with its ownership
 *        index (CycleEnumeration-GPU@0a976ad's cycles_through_edge_test), the workspace, and the
 *        fixed-capacity pruned search with the lower_bound closure (the search of the original's
 *        CUDA static kernels, ported for the host) against the oracles, with root, edge and
 *        two-hop prefixes as the work queue forms them.
 */
#include "algorithms/cycle_count/cycles_through_edge.hpp"
#include "algorithms/cycle_count/dfs.hpp"
#include "algorithms/cycle_count/problem.hpp"
#include "graph/graph_impl.hpp"
#include "support/cycle_count_support.hpp"

#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using dyng::resources;
using dyng::unweighted;
using dyng::test::cc_edge;
using dyng::test::cc_graph;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using engine_graph = dyng::detail::cycle_graph<std::int32_t, std::int64_t>;
using change = dyng::detail::edge_change<std::int32_t>;

engine_graph view_of(const graph_u& g) {
  const auto out = dyng::detail::graph_access::out_view(g);
  engine_graph cg;
  cg.vertex_count = static_cast<std::size_t>(out.num_vertices());
  cg.offsets = out.row_ptr.data();
  cg.neighbors = out.col_ind.data();
  return cg;
}

/// 0->1->2->0 (triangle) and 1->0 (a 2-cycle 0<->1): the original's fixture.
graph_u fixture(const resources& res) {
  return cc_graph<graph_u>(res, 3, {{0, 1}, {1, 2}, {2, 0}, {1, 0}});
}

std::vector<std::uint64_t> count_through(const graph_u& g, std::int32_t u, std::int32_t v,
                                         std::size_t owner_id,
                                         const dyng::detail::changed_edge_index& index,
                                         std::size_t max_len) {
  std::vector<std::uint64_t> counts(max_len + 1, 0);
  std::vector<char> visited(static_cast<std::size_t>(g.num_vertices()), 0);
  dyng::detail::count_cycles_through_edge(view_of(g), u, v, owner_id, index, max_len, visited,
                                          counts.data());
  for (const char mark : visited) {
    EXPECT_EQ(mark, 0);  // the search clears every mark it sets
  }
  return counts;
}

TEST(CyclesThroughEdge, CountsBothCyclesThroughAnEdge) {
  const resources res = resources::sequential();
  const graph_u g = fixture(res);
  const dyng::detail::changed_edge_index none;
  // Edge 0->1 is on the 2-cycle (0->1->0) and the 3-cycle (0->1->2->0).
  const std::vector<std::uint64_t> c = count_through(g, 0, 1, 0, none, 8);
  EXPECT_EQ(c[2], 1U);
  EXPECT_EQ(c[3], 1U);
}

TEST(CyclesThroughEdge, CountsEdgeOnlyOnLongerCycle) {
  const resources res = resources::sequential();
  const graph_u g = fixture(res);
  const dyng::detail::changed_edge_index none;
  // Edge 2->0 is only on the 3-cycle (2->0->1->2).
  const std::vector<std::uint64_t> c = count_through(g, 2, 0, 0, none, 8);
  EXPECT_EQ(c[2], 0U);
  EXPECT_EQ(c[3], 1U);
}

TEST(CyclesThroughEdge, RespectsMaxLength) {
  const resources res = resources::sequential();
  const graph_u g = fixture(res);
  const dyng::detail::changed_edge_index none;
  // Capping at length 2 keeps only the 2-cycle through 0->1.
  const std::vector<std::uint64_t> c = count_through(g, 0, 1, 0, none, 2);
  EXPECT_EQ(c[2], 1U);
  // A bound below 2 and edges outside the graph count nothing.
  EXPECT_EQ(count_through(g, 0, 1, 0, none, 1)[1], 0U);
  EXPECT_EQ(count_through(g, 0, 7, 0, none, 4), (std::vector<std::uint64_t>(5, 0)));
}

TEST(CyclesThroughEdge, OwnershipSkipsCyclesWithSmallerIdEdge) {
  const resources res = resources::sequential();
  const graph_u g = fixture(res);
  // Phase changes, sorted by (source, target): (0,1) has id 0, (1,0) has id 1.
  dyng::detail::changed_edge_index index;
  index.assign(std::vector<change>{{0, 1}, {1, 0}});
  EXPECT_EQ(index.size(), 2U);
  EXPECT_TRUE(index.forbidden_before(0, 1, 1));
  EXPECT_FALSE(index.forbidden_before(0, 1, 0));
  EXPECT_FALSE(index.forbidden_before(1, 0, 1));
  EXPECT_FALSE(index.forbidden_before(2, 0, 5));
  // Anchored on (1,0) as owner 1, the 2-cycle also uses (0,1) with id 0 < 1: owned by (0,1).
  EXPECT_EQ(count_through(g, 1, 0, 1, index, 8)[2], 0U);
  // Anchored on (0,1) as owner 0, both cycles through it are owned.
  const std::vector<std::uint64_t> c = count_through(g, 0, 1, 0, index, 8);
  EXPECT_EQ(c[2], 1U);
  EXPECT_EQ(c[3], 1U);
  // Reassigning reuses the table.
  index.assign(std::vector<change>{{2, 0}});
  EXPECT_EQ(index.size(), 1U);
  EXPECT_FALSE(index.forbidden_before(0, 1, 1));
}

TEST(ChangedEdgeIndex, AnswersAsAMapUnderCollisionsAndReuse) {
  // Many edges out of one vertex and into one vertex, then a smaller list in the larger table.
  std::vector<change> big;
  for (std::int32_t v = 0; v < 1000; ++v) {
    big.push_back({0, v});
  }
  for (std::int32_t u = 1; u < 1000; ++u) {
    big.push_back({u, 0});
  }
  std::sort(big.begin(), big.end(), [](const change& a, const change& b) {
    return a.source != b.source ? a.source < b.source : a.target < b.target;
  });
  dyng::detail::changed_edge_index index;
  index.assign(big);
  EXPECT_EQ(index.size(), big.size());
  for (std::size_t id = 0; id < big.size(); ++id) {
    EXPECT_FALSE(index.forbidden_before(big[id].source, big[id].target, id));
    EXPECT_TRUE(index.forbidden_before(big[id].source, big[id].target, id + 1));
  }
  EXPECT_FALSE(index.forbidden_before(1, 1, big.size()));
  EXPECT_FALSE(index.forbidden_before(2147483646, 2147483646, big.size()));
  const std::size_t bytes = index.bytes();
  index.assign(std::vector<change>{{5, 6}});
  EXPECT_EQ(index.bytes(), bytes);  // the arrays are reused
  EXPECT_TRUE(index.forbidden_before(5, 6, 1));
  EXPECT_FALSE(index.forbidden_before(0, 5, big.size()));  // the old entries are gone
  index.assign(std::vector<change>{});
  EXPECT_EQ(index.size(), 0U);
  EXPECT_FALSE(index.forbidden_before(5, 6, 1));
}

TEST(CycleCountWorkspace, ReserveKeepsZerosAndPadsThreads) {
  dyng::detail::cycle_count_workspace<std::int32_t> ws;
  ws.reserve(3, 10, 4);
  ASSERT_EQ(ws.visited.size(), 3U);
  EXPECT_EQ(ws.thread_visited(2).size(), 10U);  // sized on first use by its thread
  EXPECT_GE(ws.stride, 5U + 8U);
  EXPECT_EQ(ws.stride % 8, 0U);
  EXPECT_EQ(ws.counts.size(), 3 * ws.stride);
  ws.thread_counts(2)[4] = 9;
  ws.reserve(2, 20, 4);  // larger graph: marks grow, counters stay
  EXPECT_EQ(ws.thread_visited(0).size(), 20U);
  EXPECT_EQ(ws.thread_visited(2).size(), 20U);
  EXPECT_EQ(ws.thread_counts(2)[4], 9U);
  ws.reserve(4, 5, 30);  // more threads and a longer bound: counters are reallocated
  EXPECT_GE(ws.stride, 31U + 8U);
  EXPECT_EQ(ws.counts.size(), 4 * ws.stride);
  EXPECT_GT(ws.bytes(), 0U);
}

/// Counts by the fixed-capacity search with one kind of work item, as the CUDA work queue forms
/// them: roots (prefix 1), forward edges r -> v1 with v1 > r (prefix 2, the 2-cycle counted by
/// the caller), or two-hop paths r -> v1 -> v2 (prefix 3, the 3-cycle counted by the caller).
template <int cap>
std::vector<std::uint64_t> pruned_counts(const engine_graph& g, int max_length, int items) {
  std::uint64_t counts[cap + 1] = {};
  std::int32_t path[cap] = {};
  const auto n = static_cast<std::int32_t>(g.vertex_count);
  for (std::int32_t r = 0; r < n; ++r) {
    path[0] = r;
    if (items == 1) {
      dyng::detail::cycle_count_extend_prefix<cap>(g, path, 1, max_length, counts);
      continue;
    }
    for (auto e = g.offsets[r]; e < g.offsets[r + 1]; ++e) {
      const std::int32_t v1 = g.neighbors[e];
      if (v1 <= r) {
        continue;
      }
      std::size_t position = 0;
      std::size_t row_end = 0;
      path[1] = v1;
      if (items == 2) {
        if (dyng::detail::cycle_count_find_edge(g, v1, r, position, row_end) && max_length >= 2) {
          ++counts[2];
        }
        dyng::detail::cycle_count_extend_prefix<cap>(g, path, 2, max_length, counts);
        continue;
      }
      if (dyng::detail::cycle_count_find_edge(g, v1, r, position, row_end) && max_length >= 2) {
        ++counts[2];  // the 2-cycle closes the edge prefix
      }
      if (max_length < 3) {
        continue;
      }
      for (auto f = g.offsets[v1]; f < g.offsets[v1 + 1]; ++f) {
        const std::int32_t v2 = g.neighbors[f];
        if (v2 <= r || v2 == v1) {
          continue;
        }
        path[2] = v2;
        if (dyng::detail::cycle_count_find_edge(g, v2, r, position, row_end)) {
          ++counts[3];
        }
        dyng::detail::cycle_count_extend_prefix<cap>(g, path, 3, max_length, counts);
      }
    }
  }
  return std::vector<std::uint64_t>(counts, counts + max_length + 1);
}

TEST(CycleCountPrunedSearch, MatchesOracleWithEveryWorkItem) {
  const resources res = resources::sequential();
  std::mt19937_64 rng(4242);
  for (int trial = 0; trial < 150; ++trial) {
    const dyng::test::cc_spec spec = dyng::test::cc_random_spec(rng, 2, 12);
    const graph_u g =
        cc_graph<graph_u>(res, spec.vertex_count, dyng::test::cc_random_edges(spec, rng));
    const int k = 2 + trial % 7;
    const std::vector<std::uint64_t> expected = dyng::test::cc_oracle(res, g, k);
    SCOPED_TRACE(::testing::Message()
                 << "trial " << trial << " n=" << spec.vertex_count << " k=" << k);
    dyng::detail::cycle_count_dispatch_capacity(static_cast<std::size_t>(k), [&](auto cap) {
      constexpr int capacity = decltype(cap)::value;
      for (const int items : {1, 2, 3}) {
        EXPECT_EQ(pruned_counts<capacity>(view_of(g), k, items), expected) << "items " << items;
      }
    });
  }
}

TEST(CycleCountPrunedSearch, CapacityDispatch) {
  int seen = 0;
  for (const std::size_t k : {2U, 4U, 5U, 8U, 9U, 16U, 17U, 32U, 33U, 64U}) {
    dyng::detail::cycle_count_dispatch_capacity(k, [&](auto cap) { seen = decltype(cap)::value; });
    EXPECT_GE(static_cast<std::size_t>(seen), k);
    EXPECT_LT(static_cast<std::size_t>(seen), 2 * k + 4);
  }
  EXPECT_THROW(dyng::detail::cycle_count_dispatch_capacity(65, [](auto) {}),
               dyng::invalid_argument_error);
}

TEST(CycleCountPrunedSearch, LongRing) {
  // A ring of 40 vertices: one 40-cycle, found with capacity 64.
  const resources res = resources::sequential();
  std::vector<cc_edge> ring;
  for (std::int32_t v = 0; v < 40; ++v) {
    ring.emplace_back(v, (v + 1) % 40);
  }
  const graph_u g = cc_graph<graph_u>(res, 40, ring);
  const std::vector<std::uint64_t> counts = pruned_counts<64>(view_of(g), 40, 1);
  EXPECT_EQ(counts[40], 1U);
  EXPECT_EQ(pruned_counts<64>(view_of(g), 39, 2)[39], 0U);
}

}  // namespace
