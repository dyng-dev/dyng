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
 * Not under the sanitizers, which replace operator new themselves (the tests skip).
 */
#include "hypergraph/incidence_model.hpp"
#include "hypergraph/line_graph/host_line_graph.hpp"
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

}  // namespace
