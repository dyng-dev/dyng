// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_openmp_allocation_failure_test.cpp
 * @brief mosp's OpenMP compute and update under host allocation failures: the K sssp runs inside
 *        them use sssp's OpenMP engine, whose per-thread lists and gathers allocate inside
 *        parallel regions, so mosp inherits the fix of M11b (ADR 0094; the D13 class of ADR
 *        0061): out_of_memory_error with the guarantee mosp.hpp documents, never the end of the
 *        process.
 *
 * This executable replaces the global operator new, the aligned form included
 * (support/failing_allocator.hpp, DYNG_TEST_FAIL_ALIGNED_NEW), and fails every allocation of
 * mosp::compute and mosp::update in turn on openmp with 1, 8 and 56 threads, of the calling
 * thread and of any thread, under a deadlock watchdog: compute is strong (`g` unchanged);
 * update has its two outcomes (before the commit `g` and `r` unchanged and the update then
 * succeeds; after it `g` updated and `r` poisoned), both reached; and the call without a
 * failure gives the reference (the sequential backend's result, which is bit-exact with openmp).
 * Each call runs through a resources handle of its own, whose workspace pool is empty, so the
 * per-thread lists grow inside the call's regions. Not under the sanitizers (the tests skip).
 */
#define DYNG_TEST_FAIL_ALIGNED_NEW 1
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/mosp.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;

using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using result_t = dyng::mosp::result<std::int32_t>;
using batch_t = dyng::edge_batch<std::int32_t, std::int32_t>;

constexpr int num_objectives = 3;
constexpr int thread_counts[] = {1, 8, 56};
constexpr int watchdog_seconds = 900;
constexpr std::int32_t num_vertices = 600;
constexpr int num_random_edges = 2500;

/// A random three-weight graph with a spanning path, as mosp's randomized tests use.
graph_t random_graph(const dyng::resources& res) {
  std::mt19937 rng(5);
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = num_vertices;
  list.num_weights = num_objectives;
  for (std::int32_t v = 1; v < num_vertices; ++v) {
    list.add_edge(v - 1, v, {1 + static_cast<std::int32_t>(rng() % 20), 5, 1});
  }
  for (int i = 0; i < num_random_edges; ++i) {
    list.add_edge(
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(num_vertices)),
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(num_vertices)),
        {1 + static_cast<std::int32_t>(rng() % 20), 1 + static_cast<std::int32_t>(rng() % 3),
         1 + static_cast<std::int32_t>(rng() % 1000)});
  }
  dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
  props.num_weights = num_objectives;
  return graph_t::from_edges(res, list.view(), props);
}

/// Deletes the first out-edge of `count` random vertices and inserts as many random edges.
batch_t random_batch(const graph_t& g, int count) {
  std::mt19937 rng(17);
  const auto out = g.view().out;
  const std::int32_t n = g.num_vertices();
  batch_t b(num_objectives);
  for (int i = 0; i < count; ++i) {
    const auto u = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n));
    if (out.row_ptr[static_cast<std::size_t>(u) + 1] > out.row_ptr[static_cast<std::size_t>(u)]) {
      b.delete_edge(u, out.col_ind[static_cast<std::size_t>(out.row_ptr[u])]);
    }
    b.insert_edge(
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
        {1 + static_cast<std::int32_t>(rng() % 20), 1 + static_cast<std::int32_t>(rng() % 3),
         1 + static_cast<std::int32_t>(rng() % 1000)});
  }
  return b;
}

/// The K trees and the combined tree of a result.
struct snapshot {
  std::vector<std::vector<std::int64_t>> distances;
  std::vector<std::vector<std::int32_t>> parents;
  std::vector<std::int64_t> combined_distances;
  std::vector<std::int32_t> combined_parents;
  bool operator==(const snapshot& o) const {
    return distances == o.distances && parents == o.parents &&
           combined_distances == o.combined_distances && combined_parents == o.combined_parents;
  }
};

snapshot snapshot_of(const result_t& r) {
  snapshot s;
  for (int k = 0; k < r.num_objectives(); ++k) {
    const auto d = r.distances(k);
    const auto p = r.parents(k);
    s.distances.emplace_back(d.begin(), d.end());
    s.parents.emplace_back(p.begin(), p.end());
  }
  const auto cd = r.combined_distances();
  const auto cp = r.combined_parents();
  s.combined_distances.assign(cd.begin(), cd.end());
  s.combined_parents.assign(cp.begin(), cp.end());
  return s;
}

std::string label(const char* call, int threads, alloc::scope scope) {
  return std::string(call) + ", " + std::to_string(threads) + " threads, " +
         (scope == alloc::scope::calling_thread ? "calling thread" : "any thread");
}

class MospOpenmpAllocationFailure : public ::testing::Test {
 protected:
  void SetUp() override {
    if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
    if (!dyng::backend_available(dyng::backend::openmp)) {
      GTEST_SKIP() << "the openmp backend is not built";
    }
  }
};

TEST_F(MospOpenmpAllocationFailure, ComputeIsStrongUnderEveryAllocationFailure) {
  const dyng::resources seq = dyng::resources::sequential();
  const snapshot reference = snapshot_of(dyng::mosp::compute(seq, random_graph(seq), 0));
  for (const int threads : thread_counts) {
    for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
      const std::string what = label("compute", threads, scope);
      SCOPED_TRACE(what);
      std::optional<dyng::resources> res;
      std::optional<graph_t> g;
      std::optional<result_t> r;
      std::uint64_t version = 0;
      int failures = 0;
      alloc::within_seconds(watchdog_seconds, what.c_str(), [&] {
        failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
            scope,
            [&] {
              r.reset();
              g.reset();
              res.emplace(dyng::resources::openmp(threads));
              g.emplace(random_graph(*res));
              version = g->version();
            },
            [&] { r.emplace(dyng::mosp::compute(*res, *g, 0)); },
            [&](bool threw) {
              EXPECT_EQ(g->version(), version);
              if (!threw) {
                ASSERT_TRUE(r.has_value());
                EXPECT_TRUE(snapshot_of(*r) == reference);
              }
            });
      });
      EXPECT_GT(failures, 0);
    }
  }
}

TEST_F(MospOpenmpAllocationFailure, UpdateKeepsItsGuaranteeUnderEveryAllocationFailure) {
  const dyng::resources seq = dyng::resources::sequential();
  graph_t g0 = random_graph(seq);
  result_t r0 = dyng::mosp::compute(seq, g0, 0);
  const snapshot old_state = snapshot_of(r0);
  const batch_t batch = random_batch(g0, 150);
  (void)dyng::mosp::update(seq, g0, batch.view(), r0);
  const snapshot reference = snapshot_of(r0);
  for (const int threads : thread_counts) {
    for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
      const std::string what = label("update", threads, scope);
      SCOPED_TRACE(what);
      std::optional<dyng::resources> res;
      std::optional<graph_t> g;
      std::optional<result_t> r;
      std::uint64_t version = 0;
      int before_commit = 0;
      int after_commit = 0;
      alloc::within_seconds(watchdog_seconds, what.c_str(), [&] {
        (void)alloc::fail_each_allocation<dyng::out_of_memory_error>(
            scope,
            [&] {
              r.reset();
              g.reset();
              const dyng::resources setup = dyng::resources::openmp(threads);
              res.emplace(dyng::resources::openmp(threads));
              g.emplace(random_graph(setup));
              r.emplace(dyng::mosp::compute(setup, *g, 0));
              version = g->version();
            },
            [&] { (void)dyng::mosp::update(*res, *g, batch.view(), *r); },
            [&](bool threw) {
              if (!threw) {
                EXPECT_EQ(g->version(), version + 1);
                EXPECT_TRUE(snapshot_of(*r) == reference);
              } else if (g->version() == version) {
                ++before_commit;
                EXPECT_TRUE(snapshot_of(*r) == old_state);
                (void)dyng::mosp::update(*res, *g, batch.view(), *r);
                EXPECT_TRUE(snapshot_of(*r) == reference);
              } else {
                ++after_commit;
                EXPECT_EQ(g->version(), version + 1);
                EXPECT_THROW((void)r->distances(0), dyng::stale_result_error);
                EXPECT_THROW((void)dyng::mosp::update(*res, *g, batch.view(), *r),
                             dyng::stale_result_error);
              }
            });
      });
      EXPECT_GT(before_commit, 0);
      EXPECT_GT(after_commit, 0);
    }
  }
}

}  // namespace
