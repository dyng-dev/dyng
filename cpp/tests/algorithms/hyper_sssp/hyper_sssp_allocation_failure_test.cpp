// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_allocation_failure_test.cpp
 * @brief A host allocation failure inside hyper_sssp::compute(), update(), result::clone() or
 *        vertex_distances() is an out_of_memory_error with the guarantee the header documents
 *        (M11a step 4, robustness; ADR 0061, D13): no std::bad_alloc leaves the library and none
 *        ends the process.
 *
 * This executable replaces the global operator new (support/failing_allocator.hpp) and fails
 * every allocation of each call once, in turn: the calling thread's allocations, on the
 * sequential backend (hyper_sssp's host backend; the commit's slack_csr regions run on the
 * container's host team, ADR 0089 section 8, and are failed with every thread's allocations).
 * Not under the sanitizers, which replace operator new themselves (the tests skip).
 */
#include "algorithms/hyper_sssp/hyper_sssp_support.hpp"
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hyper_sssp.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;
namespace hs = dyng::hyper_sssp;
using dyng::test::hsssp::result_t;
using dyng::test::hsssp::wbatch;
using dyng::test::hsssp::whgraph;
using state_t = std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>;

whgraph make(const dyng::resources& res) {
  return dyng::test::hsssp::make(res, 9,
                                 {{0, 1, 2}, {2, 3}, {3, 4, 5}, {1, 5, 6}, {6, 7}, {7, 8}, {0, 8}},
                                 {3, 1, 4, 1, 5, 9, 2});
}

/// Deletes two hyperedges, inserts two, removes and adds an incidence.
wbatch make_batch() {
  wbatch b;
  b.delete_hyperedge(1);
  b.delete_hyperedge(4);
  b.insert_hyperedge({2, 4, 6}, 2);
  b.insert_hyperedge({5, 7}, 7);
  b.delete_incidence(3, 1);
  b.insert_incidence(5, 4);
  return b;
}

state_t state_of(const dyng::resources& res, const result_t& r) {
  return dyng::test::hsssp::host_of(res, r);
}

/// Whether the hypergraph is poisoned (check_integrity() refuses it).
bool poisoned(const dyng::resources& res, const whgraph& h) {
  try {
    h.check_integrity(res);
    return false;
  } catch (const dyng::invalid_argument_error&) {
    return true;
  }
}

TEST(HyperSsspAllocationFailure, ComputeReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const whgraph h = make(res);
  const state_t want = state_of(res, hs::compute(res, h, 0));
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread, [] {},
                [&] { EXPECT_EQ(state_of(res, hs::compute(res, h, 0)), want); }),
            0);
}

TEST(HyperSsspAllocationFailure, CloneAndVertexDistancesReportOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const whgraph h = make(res);
  const result_t r = hs::compute(res, h, 0);
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread, [] {}, [&] { (void)r.clone(res); }),
            0);
  const auto want = [&] {
    const result_t c = r.clone(res);
    const auto v = c.vertex_distances();
    return std::vector<std::int64_t>(v.begin(), v.end());
  }();
  std::optional<result_t> c;
  EXPECT_GT(
      alloc::fail_each_allocation<dyng::out_of_memory_error>(
          alloc::scope::calling_thread, [&] { c.emplace(r.clone(res)); },
          [&] {
            // Compared without allocating (the countdown is armed inside the call).
            const auto v = c->vertex_distances();
            EXPECT_TRUE(v.size() == want.size() && std::equal(v.begin(), v.end(), want.begin()));
          }),
      0);
}

// update(): every allocation of the call fails once (the commit's host-team regions among them,
// failed with every thread's allocations). Each outcome is one of the three the header documents:
// before the commit `h` and `r` unchanged (and both still usable); inside the commit `h` poisoned
// and `r` unchanged, not poisoned; after the commit `h` at the new version and `r` poisoned.
TEST(HyperSsspAllocationFailure, UpdateReportsOutOfMemoryWithTheDocumentedGuarantee) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const wbatch batch = make_batch();
  state_t want;
  {
    whgraph g = make(res);
    result_t r = hs::compute(res, g, 0);
    (void)hs::update(res, g, batch.view(), r);
    want = state_of(res, r);
    EXPECT_EQ(want, state_of(res, hs::compute(res, g, 0)));
  }
  for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
    SCOPED_TRACE(scope == alloc::scope::calling_thread ? "calling thread" : "any thread");
    std::optional<whgraph> h;
    std::optional<result_t> r;
    state_t before;
    std::uint64_t version = 0;
    int unchanged = 0, in_commit = 0, after_commit = 0;
    const int failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
        scope,
        [&] {
          h.emplace(make(res));
          r.emplace(hs::compute(res, *h, 0));
          before = state_of(res, *r);
          version = h->version();
        },
        [&] { (void)hs::update(res, *h, batch.view(), *r); },
        [&](bool threw) {
          if (!threw) {
            EXPECT_EQ(state_of(res, *r), want);
            return;
          }
          if (poisoned(res, *h)) {  // inside the commit
            ++in_commit;
            EXPECT_EQ(r->graph_version(), version);
            EXPECT_EQ(state_of(res, *r), before);  // not poisoned: the result before the batch
          } else if (h->version() == version) {    // before the commit
            ++unchanged;
            EXPECT_EQ(state_of(res, *r), before);
            (void)hs::update(res, *h, batch.view(), *r);  // both still usable
            EXPECT_EQ(state_of(res, *r), want);
          } else {  // after the commit
            ++after_commit;
            EXPECT_EQ(h->version(), version + 1);
            EXPECT_THROW((void)r->hyperedge_distances(), dyng::stale_result_error);
            EXPECT_EQ(state_of(res, hs::compute(res, *h, 0)), want);
          }
        });
    EXPECT_GT(failures, 0);
    if (scope == alloc::scope::calling_thread) {
      EXPECT_GT(unchanged, 0);
      EXPECT_GT(in_commit, 0);
    }
    (void)after_commit;  // reached where the update allocates after the commit
  }
}

}  // namespace
