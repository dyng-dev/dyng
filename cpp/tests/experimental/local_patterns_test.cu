// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file local_patterns_test.cu
 * @brief experimental::count_local_patterns (M9b, ADR 0058) on every backend (cuda, sequential,
 *        openmp) and both instantiations: the four built-in Ops equal triad_count bit for bit
 *        (counts, deltas, removed / added, frontiers) in one dyng::update() with it; the callbacks
 *        see 0-based live ids and member codes that decode to the hypergraph's members (no CBST
 *        key, +1 or sentinel leaks), checked against a brute force; the contract (stale and
 *        moved-from results, clone, op(), the requirements, an invalid bin, the largest num_bins,
 *        compute() and update() of a result that dyng::update() composes), and access_rows().
 *
 * nvcc only (the extension header); built in CUDA builds as dyng_local_patterns_tests (label
 * gpu), whose host backends run on the machine's CPU.
 */
#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/experimental/local_patterns.cuh>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace dx = dyng::experimental;
namespace tc = dyng::triad_count;
namespace triad = dyng::detail::triad;
using dyng::backend;

constexpr std::int32_t num_vertices = 40;

dyng::resources resources_of(backend b) {
  switch (b) {
    case backend::openmp:
      return dyng::resources::openmp(4);
    case backend::cuda:
      return dyng::resources::cuda();
    case backend::sequential:
      break;
  }
  return dyng::resources::sequential();
}

std::vector<backend> backends() {
  std::vector<backend> out = {backend::sequential};
  if (dyng::backend_available(backend::openmp)) out.push_back(backend::openmp);
  if (dyng::backend_available(backend::cuda)) out.push_back(backend::cuda);
  return out;
}

std::vector<std::int32_t> random_row(std::mt19937_64& rng) {
  std::set<std::int32_t> m;
  const auto size = static_cast<std::size_t>(1 + rng() % 6);
  while (m.size() < size) m.insert(static_cast<std::int32_t>(rng() % num_vertices));
  return {m.begin(), m.end()};
}

template <typename weight_t>
weight_t weight_of(std::size_t i) {
  if constexpr (dyng::is_unweighted_v<weight_t>) {
    (void)i;
    return {};
  } else {
    return static_cast<weight_t>(1 + i % 5);
  }
}

template <typename graph_t>
graph_t random_hypergraph(const dyng::resources& res, std::uint64_t seed, int hyperedges = 45,
                          dyng::line_graph_policy lg = dyng::line_graph_policy::sorted_merge) {
  std::mt19937_64 rng(seed);
  dyng::hyperedge_list<std::int32_t, typename graph_t::weight_type> list;
  list.num_vertices = num_vertices;
  for (int i = 0; i < hyperedges; ++i) {
    const std::vector<std::int32_t> row = random_row(rng);
    list.add_hyperedge(dyng::host_view(row),
                       weight_of<typename graph_t::weight_type>(static_cast<std::size_t>(i)));
  }
  dyng::hypergraph_properties props;
  props.line_graph = lg;
  return graph_t::from_hyperedges(res, list.view(), props);
}

/// A random batch: deletions of ids below the bound (live, dead, repeated), insertions of random
/// rows and of a copy of a live row.
template <typename graph_t>
dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> random_batch(
    const dyng::resources& res, const graph_t& h, std::mt19937_64& rng) {
  dyng::hyperedge_batch<std::int32_t, typename graph_t::weight_type> batch;
  const auto bound = static_cast<std::uint64_t>(h.hyperedge_id_bound());
  for (std::uint64_t i = 0, n = rng() % 9; i < n; ++i)
    batch.delete_hyperedge(static_cast<std::int64_t>(rng() % (bound + 2)));
  for (std::size_t i = 0, n = rng() % 9; i < n; ++i) {
    const std::vector<std::int32_t> row = random_row(rng);
    batch.insert_hyperedge(dyng::host_view(row), weight_of<typename graph_t::weight_type>(i));
  }
  const auto list = h.to_hyperedge_list(res);
  if (list.num_hyperedges() > 0) {
    const auto e =
        static_cast<std::size_t>(rng() % static_cast<std::uint64_t>(list.num_hyperedges()));
    const std::vector<std::int32_t> copy(list.members.begin() + list.offsets[e],
                                         list.members.begin() + list.offsets[e + 1]);
    batch.insert_hyperedge(dyng::host_view(copy), weight_of<typename graph_t::weight_type>(e));
  }
  return batch;
}

template <typename result_t>
std::vector<std::uint64_t> counts_of(const result_t& r) {
  const auto c = r.counts();
  return {c.begin(), c.end()};
}

// ------------------------------------------------------------------------------------------------
// The four built-in Ops are triad_count's patterns, bit for bit
// ------------------------------------------------------------------------------------------------

using motif_t = dx::count_local_patterns<triad::motif_op>;
using type1_t = dx::count_local_patterns<triad::choose3_op>;
using type2_t = dx::count_local_patterns<triad::type2_op>;
using type3_t = dx::count_local_patterns<triad::type3_op>;

tc::options kind(tc::pattern p) {
  tc::options o;
  o.kind = p;
  return o;
}

template <typename local_stats_t>
void expect_same_update(const tc::stats& t, const local_stats_t& l, int num_bins) {
  ASSERT_EQ(l.delta.size(), static_cast<std::size_t>(num_bins));
  for (int k = 0; k < num_bins; ++k)
    EXPECT_EQ(l.delta[static_cast<std::size_t>(k)], t.delta[static_cast<std::size_t>(k)])
        << "bin " << k;
  EXPECT_EQ(l.removed, t.triads_removed);
  EXPECT_EQ(l.added, t.triads_added);
  EXPECT_EQ(l.deleted_frontier, t.deleted_frontier);
  EXPECT_EQ(l.inserted_frontier, t.inserted_frontier);
  EXPECT_EQ(l.frontier_visits, t.frontier_visits);
  EXPECT_EQ(l.affected, t.affected);
  EXPECT_EQ(l.engine_used, t.engine_used);
  EXPECT_EQ(l.batch.inserted_ids, t.batch.inserted_ids);
}

template <typename graph_t>
void builtin_ops_equal_triad_count(backend b, std::uint64_t seed) {
  const dyng::resources res = resources_of(b);
  graph_t h = random_hypergraph<graph_t>(res, seed);
  tc::result t0 = tc::compute(res, h, kind(tc::pattern::h_motif));
  tc::result t1 = tc::compute(res, h, kind(tc::pattern::stathyper_type1));
  tc::result t2 = tc::compute(res, h, kind(tc::pattern::stathyper_type2));
  tc::result t3 = tc::compute(res, h, kind(tc::pattern::stathyper_type3));
  motif_t::result l0 = motif_t::compute(res, h);
  type1_t::result l1 = type1_t::compute(res, h);
  type2_t::result l2 = type2_t::compute(res, h);
  type3_t::result l3 = type3_t::compute(res, h);
  auto expect_counts = [&] {
    EXPECT_EQ(counts_of(l0), counts_of(t0));
    EXPECT_EQ(counts_of(l1), counts_of(t1));
    EXPECT_EQ(counts_of(l2), counts_of(t2));
    EXPECT_EQ(counts_of(l3), counts_of(t3));
    EXPECT_EQ(l0.graph_version(), h.version());
  };
  expect_counts();
  EXPECT_EQ(l0.num_bins(), 30);
  EXPECT_EQ(l1.num_bins(), 1);
  std::mt19937_64 rng(seed + 101);
  for (int step = 0; step < 4; ++step) {
    SCOPED_TRACE("batch " + std::to_string(step));
    const auto batch = random_batch(res, h, rng);
    // One commit for the eight results (dyng::update composes them).
    const auto st = dyng::update(res, h, batch.view(), t0, t1, t2, t3, l0, l1, l2, l3);
    expect_same_update(std::get<0>(st), std::get<4>(st), 30);
    expect_same_update(std::get<1>(st), std::get<5>(st), 1);
    expect_same_update(std::get<2>(st), std::get<6>(st), 1);
    expect_same_update(std::get<3>(st), std::get<7>(st), 1);
    expect_counts();
    // baseline + deltas == recount
    EXPECT_EQ(counts_of(l0), counts_of(motif_t::compute(res, h)));
    EXPECT_EQ(counts_of(l2), counts_of(type2_t::compute(res, h)));
  }
  // And alone, through count_local_patterns::update().
  const auto batch = random_batch(res, h, rng);
  const tc::stats ts = tc::update(res, h, batch.view(), t3);
  EXPECT_THROW((void)type3_t::update(res, h, batch.view(), l3), dyng::stale_result_error);
  l3 = type3_t::compute(res, h);
  EXPECT_EQ(counts_of(l3), counts_of(t3));
  (void)ts;
}

class LocalPatterns : public ::testing::TestWithParam<backend> {
 protected:
  [[nodiscard]] dyng::resources res() const {
    return resources_of(GetParam());
  }
};

TEST_P(LocalPatterns, TheBuiltInOpsEqualTriadCountUnweighted) {
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    builtin_ops_equal_triad_count<dyng::hypergraph<std::int32_t, dyng::unweighted>>(GetParam(),
                                                                                    seed);
  }
}

TEST_P(LocalPatterns, TheBuiltInOpsEqualTriadCountWeighted) {
  builtin_ops_equal_triad_count<dyng::hypergraph<std::int32_t, std::int32_t>>(GetParam(), 4);
}

// ------------------------------------------------------------------------------------------------
// No CBST key, +1 or sentinel reaches a callback
// ------------------------------------------------------------------------------------------------

/// Checks one hyperedge as a callback sees it: a live id (non-empty row) and codes that are
/// ascending and decode into [0, num_vertices). Returns whether something is wrong.
DYNG_HD bool bad_row(const dx::hyperedge_rows& rows, int id) {
  if (id < 0 || rows.size(id) <= 0) return true;
  const dx::member_span r = rows.row(id);
  for (int i = 0; i < r.size(); ++i) {
    const std::int32_t v = rows.decode(r[i]);
    if (v < 0 || v >= num_vertices || (i > 0 && r[i] <= r[i - 1])) return true;
  }
  return false;
}

/// The sum of a hyperedge's decoded members + 1 (a checksum the brute force recomputes).
DYNG_HD std::uint64_t member_sum(const dx::hyperedge_rows& rows, int id) {
  std::uint64_t s = 0;
  for (const dx::member_code c : rows.row(id)) s += static_cast<std::uint64_t>(rows.decode(c)) + 1;
  return s;
}

/// Bins: 0 the structures, 1 the callbacks that saw something wrong, 2 the sum over the
/// structures of their decoded members + 1. Triangles: begin_wedge's ids checked too.
struct probe_triangles {
  static constexpr int num_bins = 3;
  static constexpr dx::shape shape = dx::shape::triangles;
  int a = 0, b = 0;
  std::uint64_t bad = 0;
  DYNG_HD void begin_wedge(const dx::hyperedge_rows& rows, int wa, int wb) {
    a = wa;
    b = wb;
    if (bad_row(rows, wa) || bad_row(rows, wb) || wa == wb) ++bad;
  }
  template <class bins_t>
  DYNG_HD void triangle(const dx::hyperedge_rows& rows, int c, bins_t& bins) {
    if (bad_row(rows, c) || c == a || c == b) ++bad;
    bins.add(0, 1);
    bins.add(2, member_sum(rows, a) + member_sum(rows, b) + member_sum(rows, c));
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    if (bad != 0) bins.add(1, bad);
  }
};

struct probe_pairs {
  static constexpr int num_bins = 3;
  static constexpr dx::shape shape = dx::shape::pairs;
  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows& rows, int a, int b, bins_t& bins) {
    bins.add(1, bad_row(rows, a) || bad_row(rows, b) || a == b ? 1 : 0);
    bins.add(0, 1);
    bins.add(2, member_sum(rows, a) + member_sum(rows, b));
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t&) {}
};

struct probe_keys {
  static constexpr int num_bins = 3;
  static constexpr dx::shape shape = dx::shape::keys;
  template <class bins_t>
  DYNG_HD void key(const dx::hyperedge_rows& rows, int a, bins_t& bins) {
    bins.add(1, bad_row(rows, a) ? 1 : 0);
    bins.add(0, 1);
    bins.add(2, member_sum(rows, a));
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t&) {}
};

using rows_t = std::vector<std::vector<std::int32_t>>;

template <typename graph_t>
rows_t live_rows(const dyng::resources& res, const graph_t& h) {
  const auto list = h.to_hyperedge_list(res);
  rows_t out;
  for (std::size_t e = 0; e + 1 < list.offsets.size(); ++e)
    out.emplace_back(list.members.begin() + list.offsets[e],
                     list.members.begin() + list.offsets[e + 1]);
  for (auto& r : out) std::sort(r.begin(), r.end());
  return out;
}

bool adjacent(const std::vector<std::int32_t>& x, const std::vector<std::int32_t>& y) {
  std::vector<std::int32_t> both;
  std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(both));
  return !both.empty();
}

std::uint64_t sum_of(const std::vector<std::int32_t>& r) {
  std::uint64_t s = 0;
  for (const std::int32_t v : r) s += static_cast<std::uint64_t>(v) + 1;
  return s;
}

/// The brute force of the probes: {structures, 0, checksum} per shape.
std::array<std::array<std::uint64_t, 3>, 3> brute_probes(const rows_t& rows) {
  std::array<std::array<std::uint64_t, 3>, 3> out{};
  const std::size_t m = rows.size();
  for (std::size_t i = 0; i < m; ++i) {
    out[2][0] += 1;
    out[2][2] += sum_of(rows[i]);
    for (std::size_t j = i + 1; j < m; ++j) {
      if (!adjacent(rows[i], rows[j])) continue;
      out[1][0] += 1;
      out[1][2] += sum_of(rows[i]) + sum_of(rows[j]);
      for (std::size_t k = j + 1; k < m; ++k) {
        if (!adjacent(rows[i], rows[k]) || !adjacent(rows[j], rows[k])) continue;
        out[0][0] += 1;
        out[0][2] += sum_of(rows[i]) + sum_of(rows[j]) + sum_of(rows[k]);
      }
    }
  }
  return out;
}

TEST_P(LocalPatterns, CallbacksSeeZeroBasedLiveIdsAndCodesThatDecodeToTheMembers) {
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  using tri_t = dx::count_local_patterns<probe_triangles>;
  using pair_t = dx::count_local_patterns<probe_pairs>;
  using key_t = dx::count_local_patterns<probe_keys>;
  graph_t h = random_hypergraph<graph_t>(res(), 9);
  tri_t::result tri = tri_t::compute(res(), h);
  pair_t::result pairs = pair_t::compute(res(), h);
  key_t::result keys = key_t::compute(res(), h);
  std::mt19937_64 rng(77);
  for (int step = 0; step <= 4; ++step) {
    SCOPED_TRACE("after " + std::to_string(step) + " batches");
    const auto want = brute_probes(live_rows(res(), h));
    EXPECT_EQ(counts_of(tri), (std::vector<std::uint64_t>{want[0][0], 0, want[0][2]}));
    EXPECT_EQ(counts_of(pairs), (std::vector<std::uint64_t>{want[1][0], 0, want[1][2]}));
    EXPECT_EQ(counts_of(keys), (std::vector<std::uint64_t>{want[2][0], 0, want[2][2]}));
    if (step == 4) break;
    // Batches with deletions leave dead ids below the bound: the keys shape skips them.
    const auto batch = random_batch(res(), h, rng);
    (void)dyng::update(res(), h, batch.view(), tri, pairs, keys);
  }
}

// ------------------------------------------------------------------------------------------------
// The contract
// ------------------------------------------------------------------------------------------------

/// Pairs by the size of their overlap, one bin per size up to max_pattern_bins - 1 (the largest
/// num_bins).
struct overlap_histogram {
  static constexpr int num_bins = dx::max_pattern_bins;
  static constexpr dx::shape shape = dx::shape::pairs;
  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows& rows, int a, int b, bins_t& bins) {
    const int n = dx::intersect_size(rows.row(a), rows.row(b));
    bins.add(n < num_bins ? n : num_bins - 1, 1);
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t&) {}
};

/// An Op that names bin `wrong` once per pair (an invalid bin for wrong < 0 or >= num_bins).
struct wrong_bin {
  static constexpr int num_bins = 2;
  static constexpr dx::shape shape = dx::shape::pairs;
  int wrong = 2;
  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows&, int, int, bins_t& bins) {
    bins.add(wrong, 1);
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t&) {}
};

/// Hyperedges by size: bin min(|e|, 3) - 1 (keys: no line graph needed).
struct sizes {
  static constexpr int num_bins = 3;
  static constexpr dx::shape shape = dx::shape::keys;
  template <class bins_t>
  DYNG_HD void key(const dx::hyperedge_rows& rows, int a, bins_t& bins) {
    const int n = rows.size(a);
    bins.add((n < 3 ? n : 3) - 1, 1);
  }
  template <class bins_t>
  DYNG_HD void flush(bins_t&) {}
};

// Every shape through the host-callback check (ADR 0058): it compiles for DYNG_HD callbacks and
// adds nothing that runs (the compile-failure test device_only shows nvcc reporting the others).
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(probe_triangles);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(probe_pairs);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(probe_keys);
DYNG_EXPERIMENTAL_CHECK_HOST_CALLBACKS(sizes);

TEST_P(LocalPatterns, TheLargestNumBinsAndTheOpsValue) {
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  using hist_t = dx::count_local_patterns<overlap_histogram>;
  graph_t h = random_hypergraph<graph_t>(res(), 3);
  hist_t::result r = hist_t::compute(res(), h);
  EXPECT_EQ(r.num_bins(), dx::max_pattern_bins);
  const rows_t rows = live_rows(res(), h);
  std::vector<std::uint64_t> want(static_cast<std::size_t>(dx::max_pattern_bins), 0);
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t j = i + 1; j < rows.size(); ++j) {
      std::vector<std::int32_t> both;
      std::set_intersection(rows[i].begin(), rows[i].end(), rows[j].begin(), rows[j].end(),
                            std::back_inserter(both));
      if (both.empty()) continue;
      ++want[both.size()];
      ++total;
    }
  EXPECT_EQ(counts_of(r), want);
  EXPECT_EQ(r.total(), total);
  using pairs_t = dx::count_local_patterns<wrong_bin>;
  const pairs_t::result fine = pairs_t::compute(res(), h, wrong_bin{1});
  EXPECT_EQ(fine.op().wrong, 1);
  EXPECT_EQ(fine.count(1), total);
}

TEST_P(LocalPatterns, AnInvalidBinIsAnInvalidArgumentAndPoisonsAnUpdatedResult) {
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  using op_t = dx::count_local_patterns<wrong_bin>;
  graph_t h = random_hypergraph<graph_t>(res(), 5);
  EXPECT_THROW((void)op_t::compute(res(), h, wrong_bin{2}), dyng::invalid_argument_error);
  EXPECT_THROW((void)op_t::compute(res(), h, wrong_bin{-1}), dyng::invalid_argument_error);
  // A result whose Op is fine until an update: the bad Op cannot be swapped in, so an update of a
  // fine result stays fine; the poisoning path is triad_count's (tested there).
  op_t::result r = op_t::compute(res(), h, wrong_bin{0});
  std::mt19937_64 rng(5);
  const auto batch = random_batch(res(), h, rng);
  (void)op_t::update(res(), h, batch.view(), r);
  EXPECT_EQ(r.count(0), op_t::compute(res(), h, wrong_bin{0}).count(0));
}

TEST_P(LocalPatterns, TheContractOfTheResult) {
  using graph_t = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  using op_t = dx::count_local_patterns<sizes>;
  // keys reads no line graph: a hypergraph without one is fine.
  graph_t h = random_hypergraph<graph_t>(res(), 2, 30, dyng::line_graph_policy::none);
  op_t::result r = op_t::compute(res(), h);
  EXPECT_EQ(r.space(), dyng::memory_space::host);
  EXPECT_EQ(r.total(), static_cast<std::uint64_t>(h.num_hyperedges()));
  EXPECT_THROW((void)r.count(3), dyng::invalid_argument_error);
  EXPECT_THROW((void)r.count(-1), dyng::invalid_argument_error);
  const op_t::result copy = r.clone(res());
  EXPECT_EQ(counts_of(copy), counts_of(r));
  // A shape that reads the line graph refuses a hypergraph without one.
  EXPECT_THROW((void)dx::count_local_patterns<probe_pairs>::compute(res(), h),
               dyng::invalid_argument_error);
  // Stale: an update of the hypergraph without the result.
  std::mt19937_64 rng(2);
  const auto batch = random_batch(res(), h, rng);
  (void)h.apply(res(), batch.view());
  EXPECT_THROW((void)op_t::update(res(), h, batch.view(), r), dyng::stale_result_error);
  EXPECT_EQ(copy.graph_version() + 1, h.version());
  // Moved from.
  op_t::result moved = std::move(r);
  EXPECT_THROW((void)r.counts(), dyng::invalid_argument_error);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(r.graph_version(), 0U);                              // NOLINT(bugprone-use-after-move)
  EXPECT_THROW((void)op_t::update(res(), h, batch.view(), r), dyng::invalid_argument_error);
  // A hypergraph of another backend (the host backends share their hypergraphs).
  if (GetParam() == backend::cuda) {
    EXPECT_THROW((void)op_t::compute(dyng::resources::sequential(), h),
                 dyng::invalid_argument_error);
  }
}

TEST_P(LocalPatterns, AccessRowsGivesTheHyperedgesMembers) {
  using graph_t = dyng::hypergraph<std::int32_t, std::int32_t>;
  const graph_t h = random_hypergraph<graph_t>(res(), 8);
  const dx::row_access a = dx::access_rows(h);
  EXPECT_EQ(a.id_bound(), h.hyperedge_id_bound());
  EXPECT_TRUE(a.has_line_graph());
  if (GetParam() == backend::cuda) {
    EXPECT_EQ(a.space(), dyng::memory_space::device);
    EXPECT_GE(a.device(), 0);
    return;
  }
  EXPECT_EQ(a.space(), dyng::memory_space::host);
  EXPECT_EQ(a.device(), -1);
  const auto list = h.to_hyperedge_list(res());
  // Every live id's row decodes to its members (to_hyperedge_list lists the live ids in order).
  std::size_t e = 0;
  for (std::int64_t id = 0; id < a.id_bound(); ++id) {
    const dx::member_span row = a.hyperedges().row(static_cast<int>(id));
    if (row.empty()) continue;
    ASSERT_LT(e + 1, list.offsets.size());
    std::vector<std::int32_t> got;
    for (const dx::member_code c : row) got.push_back(dx::hyperedge_rows::decode(c));
    std::vector<std::int32_t> want(list.members.begin() + list.offsets[e],
                                   list.members.begin() + list.offsets[e + 1]);
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want) << "id " << id;
    ++e;
  }
  EXPECT_EQ(e + 1, list.offsets.size());
}

INSTANTIATE_TEST_SUITE_P(Backends, LocalPatterns, ::testing::ValuesIn(backends()),
                         [](const ::testing::TestParamInfo<backend>& info) {
                           return std::string(dyng::to_string(info.param));
                         });

TEST(LocalPatternsNames, ShapesAndTheConceptChecks) {
  EXPECT_EQ(dx::to_string(dx::shape::triangles), "triangles");
  EXPECT_EQ(dx::to_string(dx::shape::pairs), "pairs");
  EXPECT_EQ(dx::to_string(dx::shape::keys), "keys");
  static_assert(dx::count_local_patterns<triad::motif_op>::valid);
  static_assert(dx::count_local_patterns<probe_triangles>::valid);
  static_assert(dyng::detail::local_patterns::builtin_op<triad::motif_op>::value);
  static_assert(!dyng::detail::local_patterns::builtin_op<probe_triangles>::value);
}

}  // namespace
