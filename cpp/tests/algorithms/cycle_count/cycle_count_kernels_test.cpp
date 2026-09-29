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
#include "algorithms/cycle_count/work_queue.hpp"
#include "graph/graph_impl.hpp"
#include "support/cycle_count_support.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

// Invariant I9 (PLAN Section 4.7): the algorithm phase of a steady-state update allocates nothing.
// This executable counts the global operator new calls made while `counting_allocations` is set
// (not under the sanitizers, which bring their own allocator).
#if !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#define DYNG_TEST_COUNTS_ALLOCATIONS 1
namespace {
std::atomic<bool> counting_allocations{false};
std::atomic<long> allocations{0};
}  // namespace

// The replacement pair allocates with malloc and frees with free; GCC cannot see that they pair
// up once operator delete is inlined into a caller of the replaced operator new.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t size) {
  if (counting_allocations.load(std::memory_order_relaxed)) {
    allocations.fetch_add(1, std::memory_order_relaxed);
  }
  if (void* p = std::malloc(size == 0 ? 1 : size)) {
    return p;
  }
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

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
  dyng::detail::cycle_count_thread<std::int32_t> scratch;
  scratch.prepare(static_cast<std::size_t>(g.num_vertices()), 3);
  const std::size_t reached =
      dyng::detail::count_cycles_through_edge(view_of(g), u, v, owner_id, index, max_len, scratch);
  for (const char mark : scratch.visited) {
    EXPECT_EQ(mark, 0);  // the search clears every mark it sets
  }
  std::vector<std::uint64_t> counts = scratch.partial;
  EXPECT_LE(counts.size(), std::max<std::size_t>(max_len, 2) + 1);  // grown only as needed
  std::size_t longest = 0;
  for (std::size_t len = 0; len < counts.size(); ++len) {
    longest = counts[len] != 0 ? len : longest;
  }
  EXPECT_EQ(reached, longest);  // the longest length counted
  counts.resize(max_len + 1, 0);
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

TEST(CyclesThroughEdge, CountersGrowOnDemandUpToTheBound) {
  // A ring of 300 vertices closed by the edge (299, 0): one 300-cycle. The counters start small
  // and grow to the length found, not to the bound.
  const resources res = resources::sequential();
  std::vector<cc_edge> ring;
  for (std::int32_t v = 0; v < 300; ++v) {
    ring.emplace_back(v, (v + 1) % 300);
  }
  const graph_u g = cc_graph<graph_u>(res, 300, ring);
  const dyng::detail::changed_edge_index none;
  dyng::detail::cycle_count_thread<std::int32_t> scratch;
  scratch.prepare(300, 3);
  EXPECT_EQ(dyng::detail::count_cycles_through_edge(view_of(g), 299, 0, 0, none,
                                                    std::size_t{1} << 40, scratch),
            300U);
  EXPECT_EQ(scratch.partial.size(), 301U);
  EXPECT_EQ(scratch.partial[300], 1U);
  EXPECT_GE(scratch.stack.size(), 299U);  // the explicit stack grew with the path
  // A bound below the ring's length counts nothing and grows nothing.
  dyng::detail::cycle_count_thread<std::int32_t> small;
  small.prepare(300, 3);
  EXPECT_EQ(dyng::detail::count_cycles_through_edge(view_of(g), 299, 0, 0, none, 299, small), 0U);
  EXPECT_EQ(small.partial.size(), 3U);
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

TEST(CycleCountWorkspace, ReserveAndPrepareSizeEveryThread) {
  dyng::detail::cycle_count_workspace<std::int32_t> ws;
  ws.reserve(3, 10);
  ASSERT_EQ(ws.threads.size(), 3U);
  EXPECT_EQ(ws.marks_needed, 10U);
  EXPECT_TRUE(ws.threads[2].visited.empty());  // sized by its own thread in the phase
  ws.threads[2].prepare(ws.marks_needed, 5);
  EXPECT_EQ(ws.threads[2].visited.size(), 10U);
  EXPECT_EQ(ws.threads[2].partial.size(), 5U);
  EXPECT_FALSE(ws.threads[2].stack.empty());
  ws.threads[2].partial[4] = 9;
  ws.threads[2].visited[3] = 1;
  ws.reserve(2, 20);  // larger graph: the marks needed grow, the threads stay
  EXPECT_EQ(ws.threads.size(), 3U);
  EXPECT_EQ(ws.marks_needed, 20U);
  ws.threads[2].prepare(ws.marks_needed, 5);
  EXPECT_EQ(ws.threads[2].visited.size(), 20U);
  EXPECT_EQ(ws.threads[2].partial[4], 9U);  // prepare() keeps the counters
  ws.reset_threads();                       // after a failure: everything zero again
  EXPECT_EQ(ws.threads[2].partial[4], 0U);
  EXPECT_EQ(ws.threads[2].visited[3], 0);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&ws.threads[1]) % 64, 0U);  // one line per thread
  EXPECT_GT(ws.bytes(), 0U);
}

TEST(CycleCountWorkspace, SteadyStatePhasesAllocateNothing) {
#if !defined(DYNG_TEST_COUNTS_ALLOCATIONS)
  GTEST_SKIP() << "allocation counting is off under the sanitizers";
#else
  const resources res = resources::sequential();
  std::mt19937_64 rng(9);
  dyng::test::cc_spec spec;
  spec.vertex_count = 300;
  spec.edge_probability = 0.02;
  const std::vector<cc_edge> edges = dyng::test::cc_random_edges(spec, rng);
  const graph_u g = cc_graph<graph_u>(res, spec.vertex_count, edges);
  std::vector<change> changes;
  for (std::size_t i = 0; i < edges.size(); i += 7) {
    changes.push_back({edges[i].first, edges[i].second});
  }
  std::sort(changes.begin(), changes.end(), [](const change& a, const change& b) {
    return a.source != b.source ? a.source < b.source : a.target < b.target;
  });
  changes.erase(std::unique(changes.begin(), changes.end(),
                            [](const change& a, const change& b) {
                              return a.source == b.source && a.target == b.target;
                            }),
                changes.end());
  dyng::detail::cycle_count_workspace<std::int32_t> ws;
  std::vector<std::uint64_t> phase;
  const auto run = [&](int threads) {
    ws.reserve(threads, static_cast<std::size_t>(g.num_vertices()));
    ws.index.assign(changes);
    phase.clear();
    if (threads > 1) {
      dyng::detail::cycle_count_openmp_phase(view_of(g), changes, 6, threads, ws, phase);
    } else {
      dyng::detail::cycle_count_sequential_phase(view_of(g), changes, 6, ws, phase);
    }
  };
  std::vector<int> team_sizes{1};
#if DYNG_HAS_OPENMP
  team_sizes.push_back(4);
#endif
  for (const int threads : team_sizes) {
    run(threads);  // the first update of this size sizes the workspace
    const std::vector<std::uint64_t> first = phase;
    EXPECT_GT(first.size(), 3U);
    allocations = 0;
    counting_allocations = true;
    run(threads);
    counting_allocations = false;
    EXPECT_EQ(allocations.load(), 0) << "threads " << threads;
    EXPECT_EQ(phase, first);
  }
#endif
}

TEST(CycleCountWorkspace, PhasesDrainOnlyWhatTheyReached) {
  // Both phase engines leave every counter and mark zero, and fail loudly on an unreserved
  // workspace (the check a missing resize trips).
  const resources res = resources::sequential();
  const graph_u g = fixture(res);
  std::vector<change> changes{{0, 1}, {2, 0}};
  dyng::detail::cycle_count_workspace<std::int32_t> ws;
  ws.index.assign(changes);
  std::vector<std::uint64_t> phase;
  EXPECT_THROW(dyng::detail::cycle_count_sequential_phase(view_of(g), changes, 3, ws, phase),
               dyng::internal_error);
  ws.reserve(1, 3);
  dyng::detail::cycle_count_sequential_phase(view_of(g), changes, 3, ws, phase);
  EXPECT_EQ(phase, (std::vector<std::uint64_t>{0, 0, 1, 1}));  // 0->1->0 and 0->1->2->0
  for (const std::uint64_t c : ws.threads[0].partial) {
    EXPECT_EQ(c, 0U);
  }
#if DYNG_HAS_OPENMP
  std::vector<std::uint64_t> parallel;
  ws.reserve(4, 3);
  dyng::detail::cycle_count_openmp_phase(view_of(g), changes, 3, 4, ws, parallel);
  EXPECT_EQ(parallel, phase);
  for (const auto& t : ws.threads) {
    for (const std::uint64_t c : t.partial) {
      EXPECT_EQ(c, 0U);
    }
    for (const char m : t.visited) {
      EXPECT_EQ(m, 0);
    }
  }
  dyng::detail::cycle_count_workspace<std::int32_t> unreserved;
  unreserved.index.assign(changes);
  EXPECT_THROW(
      dyng::detail::cycle_count_openmp_phase(view_of(g), changes, 3, 4, unreserved, parallel),
      dyng::internal_error);
#endif
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

// ---- the host-side planning of the device counters ----------------------------------------------

TEST(CycleCountCudaPlanning, WorkQueueLaunchAndItems) {
  using dyng::detail::plan_work_queue_launch;
  using dyng::detail::resolve_work_items;
  using items = dyng::cycle_count::cuda_work_items;
  EXPECT_EQ(plan_work_queue_launch(0, 128, 4, 64).grid_blocks, 0U);
  EXPECT_EQ(plan_work_queue_launch(10, 128, 4, 64).grid_blocks, 10U);
  EXPECT_EQ(plan_work_queue_launch(1000000, 128, 4, 64).grid_blocks, 256U);
  EXPECT_EQ(plan_work_queue_launch(1000000, 128, 4, 64).block_size, 128U);
  EXPECT_THROW((void)plan_work_queue_launch(1, 0, 4, 64), dyng::invalid_argument_error);
  EXPECT_THROW((void)plan_work_queue_launch(1, 128, 0, 64), dyng::invalid_argument_error);
  EXPECT_THROW((void)plan_work_queue_launch(1, 128, 4, 0), dyng::invalid_argument_error);
  // The original's automatic choice (measured on DD, COLLAB, Twitch and GitHub).
  EXPECT_EQ(resolve_work_items(items::automatic, 2, 100, 1000), items::edges);
  EXPECT_EQ(resolve_work_items(items::automatic, 3, 100, 100000), items::edges);
  EXPECT_EQ(resolve_work_items(items::automatic, 4, 100, 1599), items::edges);
  EXPECT_EQ(resolve_work_items(items::automatic, 4, 100, 1600), items::two_hop);
  EXPECT_EQ(resolve_work_items(items::automatic, 5, 100, 10), items::two_hop);
  EXPECT_EQ(resolve_work_items(items::two_hop, 2, 100, 10), items::edges);
  EXPECT_EQ(resolve_work_items(items::roots, 7, 100, 10), items::roots);
  EXPECT_EQ(dyng::detail::cycle_count_effective_length(-1, 40), 40);
  EXPECT_EQ(dyng::detail::cycle_count_effective_length(-1, 1), 2);
  EXPECT_EQ(dyng::detail::cycle_count_effective_length(5, 3), 3);
  EXPECT_EQ(dyng::detail::cycle_count_effective_length(5, 100), 5);
}

}  // namespace
