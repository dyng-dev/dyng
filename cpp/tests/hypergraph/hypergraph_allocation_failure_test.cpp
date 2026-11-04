// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_allocation_failure_test.cpp
 * @brief A host allocation failure inside the hypergraph container's OpenMP regions is the
 *        documented out_of_memory_error, with the documented guarantee (D13, ADR 0061).
 *
 * Before ADR 0061 a std::bad_alloc thrown inside `incidence_model::v2h_append` (the commit of
 * hypergraph::apply) or the line graph's row build (`flat_hyperedge_adjacency`: construction
 * with a line graph, the first h2h() under line_graph_policy::automatic) left an OpenMP region and
 * ended the process, on the sequential backend as on openmp and with any thread count (a region
 * whose `if` clause is false is still a region). This executable replaces the global operator
 * new (support/failing_allocator.hpp) and fails every allocation in turn:
 *   - of the public calls on the host backends (sequential, openmp with 1 and 4 threads; the
 *     calling thread's allocations): out_of_memory_error, and the hypergraph either unchanged or
 *     poisoned, as hypergraph::apply() documents;
 *   - of each region directly, with its loop run in parallel (4 threads, more than the regions'
 *     parallel thresholds, every thread's allocations): the std::bad_alloc comes out of the
 *     region on the calling thread, where the entry points translate it.
 * M11a step 3 adds the batch regions of slack_csr storage on its host team (Step 0's grouped
 * evaluation of the incidence edits, the H2V rows after them, the v2h edits by vertex).
 * M11a step 2 adds the regions of slack_csr storage: its v2h append, the slack CSR's build and
 * delta apply and the pair delta (operators::openmp_exec::for_each, ADR 0083), through the public
 * calls with incidence edits and, on 4 threads with every thread's allocations failing, directly.
 * Not under the sanitizers, which replace operator new themselves (the tests skip).
 */
#include "hypergraph/incidence_model.hpp"
#include "hypergraph/line_graph/host_line_graph.hpp"
#include "hypergraph/slack_csr/host_store.hpp"
#include "hypergraph/store.hpp"
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using batch_t = dyng::hyperedge_batch<std::int32_t, dyng::unweighted>;
using rows_t = std::vector<std::vector<std::int32_t>>;

/// The host backends of the sweeps: sequential, openmp with one thread and with four (their
/// small loops run on the calling thread: the regions' parallel thresholds are not reached).
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

const rows_t& small_rows() {
  static const rows_t rows = {{0, 1, 2}, {1, 2, 3},    {2, 3, 4}, {0, 2, 4},
                              {4, 5, 6}, {1, 5, 6, 7}, {0, 7}};
  return rows;
}

dyng::hyperedge_list<std::int32_t, dyng::unweighted> list_of(const rows_t& rows, std::int32_t n) {
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = n;
  for (const auto& r : rows) list.add_hyperedge(dyng::host_view(r));
  return list;
}

hgraph make(const dyng::resources& res, dyng::line_graph_policy lg,
            const dyng::hyperedge_list<std::int32_t, dyng::unweighted>& list) {
  dyng::hypergraph_properties props;
  props.line_graph = lg;
  return hgraph::from_hyperedges(res, list.view(), props);
}

hgraph make(const dyng::resources& res, dyng::line_graph_policy lg, const rows_t& rows,
            std::int32_t n) {
  return make(res, lg, list_of(rows, n));
}

/// Deletes two hyperedges, inserts three (one reuses a deleted id's members).
batch_t small_batch() {
  batch_t b;
  b.delete_hyperedge(1);
  b.delete_hyperedge(4);
  const std::vector<std::int32_t> a = {1, 2, 3}, c = {3, 4, 5, 6}, d = {0, 6, 7};
  b.insert_hyperedge(dyng::host_view(a));
  b.insert_hyperedge(dyng::host_view(c));
  b.insert_hyperedge(dyng::host_view(d));
  return b;
}

/// The rows of a hypergraph by id (its h2v view on the host).
rows_t rows_by_id(const hgraph& h) {
  const auto v = h.h2v();
  rows_t out(static_cast<std::size_t>(h.hyperedge_id_bound()));
  for (std::size_t id = 0; id < out.size(); ++id) {
    for (std::int64_t k = v.row_ptr[id]; k < v.row_ptr[id + 1]; ++k)
      out[id].push_back(v.col_ind[static_cast<std::size_t>(k)]);
  }
  return out;
}

/// Whether the hypergraph is poisoned (every call that reads its storage refuses it).
bool poisoned(const dyng::resources& res, const hgraph& h) {
  try {
    h.check_integrity(res);
    return false;
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_NE(std::string(e.what()).find("poisoned"), std::string::npos) << e.what();
    return true;
  }
}

// hypergraph::apply(): every allocation of the call fails once. Before the store changes the
// guarantee is strong (the version and the rows unchanged, the hypergraph usable); inside the
// store phase (v2h_append's region among it) the hypergraph is poisoned. Both outcomes happen.
TEST(HypergraphAllocationFailure, ApplyReportsOutOfMemoryOnTheHostBackends) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  for (const dyng::resources& res : host_resources()) {
    for (const dyng::line_graph_policy lg :
         {dyng::line_graph_policy::none, dyng::line_graph_policy::sorted_merge,
          dyng::line_graph_policy::automatic}) {
      SCOPED_TRACE(name_of(res) + " " + std::string(dyng::to_string(lg)));
      const batch_t batch = small_batch();
      rows_t after;
      {
        hgraph h = make(res, lg, small_rows(), 8);
        (void)h.apply(res, batch.view());
        after = rows_by_id(h);
      }
      std::optional<hgraph> h;
      rows_t before;
      std::uint64_t version = 0;
      int unchanged = 0, poisoned_count = 0;
      const int failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
          alloc::scope::calling_thread,
          [&] {
            h.emplace(make(res, lg, small_rows(), 8));
            before = rows_by_id(*h);
            version = h->version();
          },
          [&] { (void)h->apply(res, batch.view()); },
          [&](bool threw) {
            if (!threw) {
              EXPECT_EQ(h->version(), version + 1);
              EXPECT_EQ(rows_by_id(*h), after);
              h->check_integrity(res);
            } else if (poisoned(res, *h)) {
              ++poisoned_count;
              EXPECT_THROW((void)h->apply(res, batch.view()), dyng::invalid_argument_error);
            } else {
              ++unchanged;
              EXPECT_EQ(h->version(), version);
              EXPECT_EQ(rows_by_id(*h), before);
              (void)h->apply(res, batch.view());  // still usable: the batch applies
              EXPECT_EQ(rows_by_id(*h), after);
            }
          });
      EXPECT_GT(failures, 0);
      EXPECT_GT(unchanged, 0);
      EXPECT_GT(poisoned_count, 0);  // the store phase, v2h_append's region among it
    }
  }
}

// Construction with a line graph (sorted_merge builds it in from_hyperedges()) and the first
// h2h() under automatic: the line graph's row build is a region. A failed first h2h() keeps the
// hypergraph as it was (strong): the next h2h() builds the line graph.
TEST(HypergraphAllocationFailure, TheLineGraphBuildReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  for (const dyng::resources& res : host_resources()) {
    SCOPED_TRACE(name_of(res));
    const auto list = list_of(small_rows(), 8);
    EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                  alloc::scope::calling_thread, [] {},
                  [&] { (void)make(res, dyng::line_graph_policy::sorted_merge, list); }),
              0);
    std::optional<hgraph> h;
    std::vector<std::int64_t> want;
    {
      const hgraph full = make(res, dyng::line_graph_policy::sorted_merge, small_rows(), 8);
      const auto v = full.h2h();
      want.assign(v.col_ind.begin(), v.col_ind.end());
    }
    EXPECT_GT(
        alloc::fail_each_allocation<dyng::out_of_memory_error>(
            alloc::scope::calling_thread,
            [&] { h.emplace(make(res, dyng::line_graph_policy::automatic, small_rows(), 8)); },
            [&] { (void)h->h2h(); },
            [&](bool threw) {
              if (!threw) return;
              EXPECT_FALSE(poisoned(res, *h));
              const auto v = h->h2h();  // built now
              EXPECT_EQ(std::vector<std::int64_t>(v.col_ind.begin(), v.col_ind.end()), want);
            }),
        0);
  }
}

// ---- The regions themselves, run in parallel ---------------------------------------------------

/// A store of `n` vertices with `rows` (row i: hyperedge id i) and v2h.
dyng::detail::hyper::incidence_model make_store(const dyng::resources& res, std::int32_t n,
                                                const rows_t& rows) {
  dyng::detail::hyper::member_rows m;
  for (const auto& r : rows) {
    m.members.insert(m.members.end(), r.begin(), r.end());
    m.offsets.push_back(static_cast<std::int64_t>(m.members.size()));
  }
  return dyng::detail::hyper::incidence_model(res, n, m, dyng::capacity_policy{}, true);
}

// incidence_model::v2h_append: 1,200 groups (above its parallel threshold of 1,024) on 4
// threads, and on one thread (the region with a false `if` clause).
TEST(HypergraphAllocationFailure, TheV2hAppendRegionRethrowsOnTheCallingThread) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  constexpr std::int32_t n = 1200;
  rows_t rows;
  for (std::int32_t v = 0; v + 2 < n; v += 3) rows.push_back({v, v + 1, v + 2});
  dyng::detail::hyper::grouped_incidences g;
  for (std::int32_t v = 0; v < n; ++v) {
    g.vertices.push_back(v);
    g.ids.push_back(static_cast<std::int64_t>(rows.size()) + v / 3);
    g.ids.push_back(static_cast<std::int64_t>(rows.size()) + n / 3 + v);
    g.prefix.push_back(static_cast<std::int32_t>(g.ids.size()));
  }
  std::vector<dyng::resources> backends{dyng::resources::sequential()};
  if (dyng::backend_available(dyng::backend::openmp)) {
    backends.push_back(dyng::resources::openmp(4));
  }
  for (const dyng::resources& res : backends) {
    SCOPED_TRACE(name_of(res));
    std::optional<dyng::detail::hyper::incidence_model> store;
    const int failures = alloc::fail_each_allocation<std::bad_alloc>(
        res.get_backend() == dyng::backend::openmp ? alloc::scope::any_thread
                                                   : alloc::scope::calling_thread,
        [&] { store.emplace(make_store(res, n, rows)); }, [&] { store->v2h_append(g); });
    EXPECT_GE(failures, n);  // every v2h row grows once at least
  }
}

// The line graph's row build (flat_hyperedge_adjacency's gather region) on 4 threads and on one.
TEST(HypergraphAllocationFailure, TheLineGraphRowBuildRegionRethrowsOnTheCallingThread) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  constexpr std::int32_t n = 600;
  std::vector<std::int64_t> ptr{0};
  std::vector<std::int32_t> cols;
  for (std::int32_t e = 0; e < 1500; ++e) {
    for (const std::int32_t v : {e % n, (e * 7 + 1) % n, (e * 13 + 5) % n}) {
      if (std::find(cols.begin() + ptr.back(), cols.end(), v) == cols.end()) cols.push_back(v);
    }
    std::sort(cols.begin() + ptr.back(), cols.end());
    ptr.push_back(static_cast<std::int64_t>(cols.size()));
  }
  const auto keys = static_cast<std::int64_t>(ptr.size()) - 1;
  const dyng::detail::hyper::line_graph_layout want =
      dyng::detail::hyper::flat_hyperedge_adjacency(ptr, cols, n, keys, 1, INT64_MAX);
  for (const int threads : {1, 4}) {
    SCOPED_TRACE(std::to_string(threads) + " threads");
    const int failures = alloc::fail_each_allocation<std::bad_alloc>(
        threads > 1 ? alloc::scope::any_thread : alloc::scope::calling_thread, [] {},
        [&] {
          const auto got =
              dyng::detail::hyper::flat_hyperedge_adjacency(ptr, cols, n, keys, threads, INT64_MAX);
          EXPECT_EQ(got.payload, want.payload);
        });
    EXPECT_GT(failures, 0);
  }
}

// ---- slack_csr storage (M11a step 2; ADRs 0083, 0085, 0086) -------------------------------------

hgraph make_slack(const dyng::resources& res, dyng::line_graph_policy lg,
                  const dyng::hyperedge_list<std::int32_t, dyng::unweighted>& list) {
  dyng::hypergraph_properties props;
  props.storage = dyng::hypergraph_storage::slack_csr;
  props.store_v2h = true;
  props.line_graph = lg;
  return hgraph::from_hyperedges(res, list.view(), props);
}

hgraph make_slack(const dyng::resources& res, dyng::line_graph_policy lg, const rows_t& rows,
                  std::int32_t n) {
  return make_slack(res, lg, list_of(rows, n));
}

/// small_batch() with incidence edits (two removals, two additions, one skipped).
batch_t small_edit_batch() {
  batch_t b = small_batch();
  b.delete_incidence(0, 1);
  b.delete_incidence(5, 7);
  b.insert_incidence(2, 7);
  b.insert_incidence(3, 6);
  b.insert_incidence(6, 0);  // already a member: skipped
  return b;
}

// hypergraph::apply() on slack_csr storage, with incidence edits and every line-graph policy:
// strong before the store phase, poisoned inside it (the slack CSR apply and the pair delta among
// it), the documented error either way.
TEST(HypergraphAllocationFailure, ApplyOnSlackCsrReportsOutOfMemoryOnTheHostBackends) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  for (const dyng::resources& res : host_resources()) {
    for (const dyng::line_graph_policy lg :
         {dyng::line_graph_policy::none, dyng::line_graph_policy::slack_csr,
          dyng::line_graph_policy::automatic}) {
      SCOPED_TRACE(name_of(res) + " " + std::string(dyng::to_string(lg)));
      const batch_t batch = small_edit_batch();
      rows_t after;
      {
        hgraph h = make_slack(res, lg, small_rows(), 8);
        (void)h.apply(res, batch.view());
        after = rows_by_id(h);
      }
      std::optional<hgraph> h;
      rows_t before;
      std::uint64_t version = 0;
      int unchanged = 0, poisoned_count = 0;
      const int failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
          alloc::scope::calling_thread,
          [&] {
            h.emplace(make_slack(res, lg, small_rows(), 8));
            before = rows_by_id(*h);
            version = h->version();
          },
          [&] { (void)h->apply(res, batch.view()); },
          [&](bool threw) {
            if (!threw) {
              EXPECT_EQ(h->version(), version + 1);
              EXPECT_EQ(rows_by_id(*h), after);
              h->check_integrity(res);
            } else if (poisoned(res, *h)) {
              ++poisoned_count;
            } else {
              ++unchanged;
              EXPECT_EQ(h->version(), version);
              EXPECT_EQ(rows_by_id(*h), before);
              (void)h->apply(res, batch.view());
              EXPECT_EQ(rows_by_id(*h), after);
            }
          });
      EXPECT_GT(failures, 0);
      EXPECT_GT(unchanged, 0);
      EXPECT_GT(poisoned_count, 0);
    }
  }
}

// The slack CSR's build: at construction under line_graph_policy::slack_csr, and on the first
// h2h() under automatic, which keeps the hypergraph as it was when it fails.
TEST(HypergraphAllocationFailure, TheSlackLineGraphBuildReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  for (const dyng::resources& res : host_resources()) {
    SCOPED_TRACE(name_of(res));
    const auto list = list_of(small_rows(), 8);  // the test's own allocations outside the sweep
    EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                  alloc::scope::calling_thread, [] {},
                  [&] { (void)make_slack(res, dyng::line_graph_policy::slack_csr, list); }),
              0);
    std::vector<std::int64_t> want;
    {
      const hgraph full = make_slack(res, dyng::line_graph_policy::slack_csr, small_rows(), 8);
      const auto v = full.h2h();
      want.assign(v.col_ind.begin(), v.col_ind.end());
    }
    std::optional<hgraph> h;
    EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                  alloc::scope::calling_thread,
                  [&] {
                    h.emplace(make_slack(res, dyng::line_graph_policy::automatic, small_rows(), 8));
                  },
                  [&] { (void)h->h2h(); },
                  [&](bool threw) {
                    if (!threw) return;
                    EXPECT_FALSE(poisoned(res, *h));
                    const auto v = h->h2h();  // built now
                    EXPECT_EQ(std::vector<std::int64_t>(v.col_ind.begin(), v.col_ind.end()), want);
                  }),
              0);
  }
}

// The regions themselves on 4 threads, every thread's allocations failing: slack_host_store's v2h
// append (1,200 groups, above its parallel threshold) and a whole apply with incidence edits (the
// slack CSR's for_each regions and the pair delta's).
TEST(HypergraphAllocationFailure, TheSlackCsrRegionsRethrowOnTheCallingThread) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  if (!dyng::backend_available(dyng::backend::openmp)) GTEST_SKIP() << "no openmp backend";
  const dyng::resources res = dyng::resources::openmp(4);
  constexpr std::int32_t n = 1200;
  rows_t rows;
  for (std::int32_t v = 0; v + 2 < n; v += 3) rows.push_back({v, v + 1, v + 2});
  dyng::detail::hyper::member_rows m;
  for (const auto& r : rows) {
    m.members.insert(m.members.end(), r.begin(), r.end());
    m.offsets.push_back(static_cast<std::int64_t>(m.members.size()));
  }
  dyng::detail::hyper::grouped_incidences g;
  for (std::int32_t v = 0; v < n; ++v) {
    g.vertices.push_back(v);
    g.ids.push_back(static_cast<std::int64_t>(rows.size()) + v / 3);
    g.prefix.push_back(static_cast<std::int32_t>(g.ids.size()));
  }
  std::optional<dyng::detail::hyper::slack_host_store> store;
  EXPECT_GE(alloc::fail_each_allocation<std::bad_alloc>(
                alloc::scope::any_thread,
                [&] {
                  store.emplace(res, n, m, dyng::capacity_policy{},
                                dyng::detail::hyper::slack::knobs{});
                },
                [&] { store->v2h_append(g); }),
            n);  // every v2h row grows once at least
  // A whole apply: out_of_memory_error on the calling thread, whichever thread failed.
  batch_t batch;
  for (std::int32_t k = 0; k < 300; ++k) {
    batch.delete_incidence(k, 3 * k);
    batch.insert_incidence(k, (3 * k + 5) % n);
  }
  std::optional<hgraph> h;
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::any_thread,
                [&] { h.emplace(make_slack(res, dyng::line_graph_policy::slack_csr, rows, n)); },
                [&] { (void)h->apply(res, batch.view()); }),
            0);
}

// M11a step 3 (ADR 0089 section 8): the batch regions of slack_csr storage on the host team (at
// least 4096 items each, host_team.hpp), every thread's allocations failing: Step 0's evaluation
// of the incidence edits (a whole apply of 4,200 removals: its work rows), the H2V rows after the
// edits and the v2h edits grouped by vertex (4,200 each, 64 of them growing). Each failure leaves
// its region and is rethrown on the calling thread.
TEST(HypergraphAllocationFailure, TheSlackCsrBatchRegionsRethrowOnTheCallingThread) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  if (!dyng::backend_available(dyng::backend::openmp)) GTEST_SKIP() << "no openmp backend";
  const dyng::resources res = dyng::resources::openmp(4);
  constexpr std::int32_t groups = 4200;
  constexpr std::int32_t n = 3 * groups;
  rows_t rows;
  for (std::int32_t v = 0; v + 2 < n; v += 3) rows.push_back({v, v + 1, v + 2});
  // A whole apply: Step 0's regions allocate their work rows (the removals shrink every row, so
  // the store phase's H2V and v2h regions do not allocate here).
  batch_t batch;
  for (std::int32_t k = 0; k < groups; ++k) batch.delete_incidence(k, 3 * k + 1);
  std::optional<hgraph> h;
  rows_t before;
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::any_thread,
                [&] {
                  h.emplace(make_slack(res, dyng::line_graph_policy::none, rows, n));
                  before = rows_by_id(*h);
                },
                [&] { (void)h->apply(res, batch.view()); },
                [&](bool threw) {
                  if (threw && !poisoned(res, *h)) {
                    EXPECT_EQ(rows_by_id(*h), before);
                  }
                }),
            0);
  // The store's H2V rows after the edits and its v2h edits: 4,200 rows and vertices, of which
  // the first 64 grow (a member and a hyperedge more: an allocation each, inside the regions);
  // the others keep their size (a member replaced, a hyperedge removed).
  constexpr std::int32_t growing = 64;
  dyng::detail::hyper::member_rows m, post;
  dyng::detail::hyper::grouped_members removed, added;
  std::vector<std::int64_t> ids;
  for (std::int32_t g = 0; g < groups; ++g) {
    const auto& r = rows[static_cast<std::size_t>(g)];
    m.members.insert(m.members.end(), r.begin(), r.end());
    m.offsets.push_back(static_cast<std::int64_t>(m.members.size()));
    std::vector<std::int32_t> after = r;
    const std::int32_t extra = (3 * g + 3) % n;
    if (g >= growing) {
      removed.ids.push_back(g);
      removed.members.push_back(r.front());
      removed.prefix.push_back(static_cast<std::int32_t>(removed.members.size()));
      after.erase(after.begin());
    }
    after.push_back(extra);
    std::sort(after.begin(), after.end());
    post.members.insert(post.members.end(), after.begin(), after.end());
    post.offsets.push_back(static_cast<std::int64_t>(post.members.size()));
    ids.push_back(g);
    added.ids.push_back(g);
    added.members.push_back(extra);
    added.prefix.push_back(g + 1);
  }
  std::vector<std::uint64_t> by_vertex;
  std::vector<std::size_t> start;
  std::vector<std::int64_t> hyperedge;
  std::vector<std::int8_t> signs;
  for (std::int32_t v = 0; v < groups; ++v) {
    start.push_back(by_vertex.size());
    by_vertex.push_back(static_cast<std::uint64_t>(v) << 32 | static_cast<std::uint64_t>(v));
    hyperedge.push_back(v < growing ? groups + v : v / 3);  // vertex v is a member of v / 3
    signs.push_back(v < growing ? std::int8_t{+1} : std::int8_t{-1});
  }
  start.push_back(by_vertex.size());
  std::optional<dyng::detail::hyper::slack_host_store> store;
  const auto make_store = [&] {
    store.emplace(res, n, m, dyng::capacity_policy{}, dyng::detail::hyper::slack::knobs{});
  };
  EXPECT_GE(alloc::fail_each_allocation<std::bad_alloc>(
                alloc::scope::any_thread, make_store,
                [&] { store->h2v_edit(removed, added, ids, post); }),
            growing);
  std::vector<std::int32_t> lengths;
  EXPECT_GE(
      alloc::fail_each_allocation<std::bad_alloc>(
          alloc::scope::any_thread, make_store,
          [&] { store->v2h_apply(by_vertex, start, hyperedge.data(), signs.data(), lengths); }),
      growing);
}

}  // namespace
