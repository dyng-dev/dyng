// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_allocation_failure_test.cpp
 * @brief A host allocation failure inside triad_count::compute(), update() or result::clone() is
 *        an out_of_memory_error with the guarantee the header documents (the triad_count API
 *        review, ADR 0060; update() since the container's OpenMP regions rethrow, ADR 0061, D13):
 *        no std::bad_alloc leaves the library and none ends the process.
 *
 * This executable replaces the global operator new (support/failing_allocator.hpp) and fails
 * every allocation of each call once, in turn: the calling thread's allocations, on the host
 * backends (sequential, openmp with 1 and 4 threads). The oracle's OpenMP regions are failed with
 * every thread's allocations on 4 threads. Not under the sanitizers, which replace operator new
 * themselves (the tests skip).
 */
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/triad_reference.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;
namespace tc = dyng::triad_count;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using batch_t = dyng::hyperedge_batch<std::int32_t, dyng::unweighted>;
using counts_t = std::vector<std::uint64_t>;

constexpr tc::pattern all_patterns[] = {tc::pattern::h_motif, tc::pattern::stathyper_type1,
                                        tc::pattern::stathyper_type2, tc::pattern::stathyper_type3};

const std::vector<std::vector<std::int32_t>>& rows() {
  static const std::vector<std::vector<std::int32_t>> r = {
      {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 2, 4}, {4, 5, 6}, {1, 5, 6, 7}, {0, 7}};
  return r;
}

hgraph make(const dyng::resources& res,
            dyng::line_graph_policy lg = dyng::line_graph_policy::sorted_merge) {
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 8;
  for (const auto& r : rows()) list.add_hyperedge(dyng::host_view(r));
  dyng::hypergraph_properties props;
  props.line_graph = lg;  // sorted_merge: built here, not inside compute()
  return hgraph::from_hyperedges(res, list.view(), props);
}

/// Sequential, and openmp with one thread and with four.
std::vector<dyng::resources> host_resources() {
  std::vector<dyng::resources> out{dyng::resources::sequential()};
  if (dyng::backend_available(dyng::backend::openmp)) {
    out.push_back(dyng::resources::openmp(1));
    out.push_back(dyng::resources::openmp(4));
  }
  return out;
}

std::string name_of(const dyng::resources& res) {
  return std::string(dyng::to_string(res.get_backend())) + " x" + std::to_string(res.num_threads());
}

tc::options kind_of(tc::pattern p) {
  tc::options o;
  o.kind = p;
  return o;
}

counts_t counts_of(const tc::result& r) {
  const auto c = r.counts();
  return counts_t(c.begin(), c.end());
}

/// Deletes two hyperedges, inserts three.
batch_t make_batch() {
  batch_t b;
  b.delete_hyperedge(1);
  b.delete_hyperedge(4);
  const std::vector<std::int32_t> a = {1, 2, 3}, c = {3, 4, 5, 6}, d = {0, 6, 7};
  b.insert_hyperedge(dyng::host_view(a));
  b.insert_hyperedge(dyng::host_view(c));
  b.insert_hyperedge(dyng::host_view(d));
  return b;
}

/// Whether the hypergraph is poisoned (check_integrity() refuses it).
bool poisoned(const dyng::resources& res, const hgraph& h) {
  try {
    h.check_integrity(res);
    return false;
  } catch (const dyng::invalid_argument_error&) {
    return true;
  }
}

TEST(TriadCountAllocationFailure, ComputeReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  for (const dyng::resources& res : host_resources()) {
    for (const tc::pattern p : all_patterns) {
      for (const dyng::line_graph_policy lg :
           {dyng::line_graph_policy::sorted_merge, dyng::line_graph_policy::automatic}) {
        SCOPED_TRACE(name_of(res) + " " + std::string(tc::to_string(p)) + " " +
                     std::string(dyng::to_string(lg)));
        std::optional<hgraph> h;
        // automatic: a fresh hypergraph each time, so the line graph is built inside compute().
        EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                      alloc::scope::calling_thread, [&] { h.emplace(make(res, lg)); },
                      [&] { (void)tc::compute(res, *h, kind_of(p)); }),
                  0);
      }
    }
  }
}

TEST(TriadCountAllocationFailure, CloneReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const hgraph h = make(res);
  const tc::result r = tc::compute(res, h);
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread, [] {}, [&] { (void)r.clone(res); }),
            0);
}

// update(): every allocation of the call fails once (the commit's OpenMP regions among them,
// ADR 0061). Each outcome is one of the three the header documents: before the commit `h` and `r`
// unchanged; inside the commit `h` poisoned and `r` unchanged, not poisoned; after the commit `h`
// at the new version and `r` poisoned. The first two are reached for every pattern.
TEST(TriadCountAllocationFailure, UpdateReportsOutOfMemoryWithTheDocumentedGuarantee) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const batch_t batch = make_batch();
  for (const dyng::resources& res : host_resources()) {
    for (const tc::pattern p : all_patterns) {
      for (const dyng::line_graph_policy lg :
           {dyng::line_graph_policy::sorted_merge, dyng::line_graph_policy::automatic}) {
        SCOPED_TRACE(name_of(res) + " " + std::string(tc::to_string(p)) + " " +
                     std::string(dyng::to_string(lg)));
        counts_t want;
        {
          hgraph g = make(res, lg);
          tc::result r = tc::compute(res, g, kind_of(p));
          (void)tc::update(res, g, batch.view(), r);
          want = counts_of(r);
          EXPECT_EQ(want, counts_of(tc::compute(res, g, kind_of(p))));
        }
        std::optional<hgraph> h;
        std::optional<tc::result> r;
        counts_t before;
        std::uint64_t version = 0;
        int unchanged = 0, in_commit = 0, after_commit = 0;
        const int failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
            alloc::scope::calling_thread,
            [&] {
              h.emplace(make(res, lg));
              r.emplace(tc::compute(res, *h, kind_of(p)));
              before = counts_of(*r);
              version = h->version();
            },
            [&] { (void)tc::update(res, *h, batch.view(), *r); },
            [&](bool threw) {
              if (!threw) {
                EXPECT_EQ(counts_of(*r), want);
                return;
              }
              if (poisoned(res, *h)) {  // inside the commit
                ++in_commit;            // (a poisoned hypergraph's accessors are unspecified)
                EXPECT_EQ(r->graph_version(), version);
                EXPECT_EQ(counts_of(*r), before);    // not poisoned: the counts before the batch
              } else if (h->version() == version) {  // before the commit
                ++unchanged;
                EXPECT_EQ(counts_of(*r), before);
                (void)tc::update(res, *h, batch.view(), *r);  // both still usable
                EXPECT_EQ(counts_of(*r), want);
              } else {  // after the commit
                ++after_commit;
                EXPECT_EQ(h->version(), version + 1);
                EXPECT_THROW((void)r->counts(), dyng::stale_result_error);
                EXPECT_EQ(counts_of(tc::compute(res, *h, kind_of(p))), want);
              }
            });
        EXPECT_GT(failures, 0);
        EXPECT_GT(unchanged, 0);
        EXPECT_GT(in_commit, 0);  // the store phase, v2h_append's region among it
        (void)after_commit;       // reached where the update allocates after the commit
      }
    }
  }
}

// The oracle's OpenMP regions (its line graph's rows, the triangles' accumulators) on 4 threads,
// every thread's allocations failed in turn: the documented std::bad_alloc, on the calling thread.
TEST(TriadCountAllocationFailure, TheOracleRethrowsOnTheCallingThread) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  namespace tr = dyng::testing;
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int32_t> members;
  for (std::int32_t e = 0; e < 160; ++e) {
    for (const std::int32_t v : {e % 41, 41 + (e * 7) % 41, 82 + (e * 13) % 41})
      members.push_back(v);
    offsets.push_back(static_cast<std::int64_t>(members.size()));
  }
  for (const tr::triad_pattern p : {tr::triad_pattern::h_motif, tr::triad_pattern::stathyper_type2,
                                    tr::triad_pattern::stathyper_type3}) {
    SCOPED_TRACE(tr::triad_pattern_name(p));
    tr::triad_reference_options one;
    one.threads = 1;
    const counts_t want = tr::triad_reference(offsets, members, 123, p, one);
    for (const int threads : {1, 4}) {
      tr::triad_reference_options o;
      o.threads = threads;
      EXPECT_GT(alloc::fail_each_allocation<std::bad_alloc>(
                    threads > 1 ? alloc::scope::any_thread : alloc::scope::calling_thread, [] {},
                    [&] { EXPECT_EQ(tr::triad_reference(offsets, members, 123, p, o), want); }),
                0);
    }
  }
}

}  // namespace
