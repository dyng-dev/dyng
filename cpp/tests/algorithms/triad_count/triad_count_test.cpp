// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_test.cpp
 * @brief Hand cases and property tests of triad_count against the independent oracle
 *        (dyng::testing::triad_reference), on the host backends (dyng_triad_count_tests) and, with
 *        DYNG_TEST_CUDA=1, on cuda (dyng_triad_count_cuda_tests; M9a criteria 5, 6, 9, 10).
 *
 * Every pattern is checked after compute() and after every batch: the counts equal the oracle's,
 * equal a fresh compute() ("baseline + deltas == recount"), and the stats agree: delta is the
 * counts after minus before, the frontier sizes are the batch's, and triads_removed /
 * triads_added are the oracle's removed() / added() of the deleted and the ASSIGNED inserted ids
 * (every backend runs an incremental engine: the wedge engine on cuda, the host engine on the
 * host backends, M9a step 4). The batches have duplicates, copies of live hyperedges,
 * single-member hyperedges and hyperedges of more than 1,024 members, reused ids and ids deleted
 * and reinserted in one batch; the hyperedge id bound goes down in one case; random multi-batch
 * sequences run under both id_reuse orders. The launch counters show every (mode, shape) and
 * both early outs on every backend; OpenMP gives the same counts on 1, 8 and 56 threads.
 */
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"
#include "framework/fake_hyper_problems.hpp"
#include "operators/count_delta.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/triad_reference.hpp>
#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace tc = dyng::triad_count;
namespace tr = dyng::testing;
using dyng::backend;
using dyng::resources;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using wgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using batch_t = dyng::hyperedge_batch<std::int32_t, dyng::unweighted>;
using rows = std::vector<std::vector<std::int32_t>>;
using counts = std::vector<std::uint64_t>;

constexpr std::array<tc::pattern, 4> all_patterns = {
    tc::pattern::h_motif, tc::pattern::stathyper_type1, tc::pattern::stathyper_type2,
    tc::pattern::stathyper_type3};

tr::triad_pattern oracle_pattern(tc::pattern p) {
  switch (p) {
    case tc::pattern::h_motif:
      return tr::triad_pattern::h_motif;
    case tc::pattern::stathyper_type1:
      return tr::triad_pattern::stathyper_type1;
    case tc::pattern::stathyper_type2:
      return tr::triad_pattern::stathyper_type2;
    case tc::pattern::stathyper_type3:
      return tr::triad_pattern::stathyper_type3;
  }
  return tr::triad_pattern::h_motif;
}

counts counts_of(const tc::result& r) {
  const auto c = r.counts();
  return counts(c.data(), c.data() + c.size());
}

std::uint64_t sum(const counts& c) {
  return std::accumulate(c.begin(), c.end(), std::uint64_t{0});
}

/// The rows of a hypergraph by id (dead ids: empty rows), on the host: the oracle's input.
struct id_rows {
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> members;
  std::int32_t num_vertices = 0;
};

template <typename graph_t>
id_rows rows_by_id(const graph_t& h) {
  const auto view = h.h2v();
  id_rows out;
  out.offsets = dyng::test::host_copy(view.row_ptr);
  out.members = dyng::test::host_copy(view.col_ind);
  if (out.offsets.empty()) out.offsets.assign(1, 0);
  out.num_vertices = h.num_vertices();
  return out;
}

counts oracle(const id_rows& g, tc::pattern p) {
  return tr::triad_reference(g.offsets, g.members, g.num_vertices, oracle_pattern(p));
}

hgraph make(const resources& res, std::int32_t n, const rows& rs,
            dyng::line_graph_policy lg = dyng::line_graph_policy::sorted_merge,
            bool store_v2h = true) {
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = n;
  for (const auto& r : rs) list.add_hyperedge(dyng::host_view(r));
  dyng::hypergraph_properties props;
  props.line_graph = lg;
  props.store_v2h = store_v2h;
  return hgraph::from_hyperedges(res, list.view(), props);
}

tc::options kind_of(tc::pattern p) {
  tc::options o;
  o.kind = p;
  return o;
}

/// The four results of a hypergraph, one per pattern.
template <typename graph_t>
std::vector<tc::result> compute_all(const resources& res, const graph_t& h) {
  std::vector<tc::result> out;
  for (const tc::pattern p : all_patterns) out.push_back(tc::compute(res, h, kind_of(p)));
  return out;
}

class TriadCount : public ::testing::TestWithParam<backend> {
 protected:
  [[nodiscard]] resources res() const {
    return dyng::test::make_resources(GetParam(), 4);
  }

  /// Every result equals the oracle and a fresh compute() of h.
  template <typename graph_t>
  void expect_exact(const graph_t& h, const std::vector<tc::result>& rs) {
    const id_rows g = rows_by_id(h);
    for (const tc::result& r : rs) {
      SCOPED_TRACE(std::string(tc::to_string(r.kind())));
      EXPECT_EQ(counts_of(r), oracle(g, r.kind())) << "!= the oracle";
      EXPECT_EQ(counts_of(r), counts_of(tc::compute(res(), h, kind_of(r.kind()))))
          << "baseline + deltas != recount";
      EXPECT_EQ(r.graph_version(), h.version());
    }
  }

  /// One update of all four results through one dyng::update(), checked against the oracle and
  /// with the stats checked.
  template <typename graph_t, typename batch_view_t>
  void step(graph_t& h, std::vector<tc::result>& rs, const batch_view_t& b) {
    const id_rows before = rows_by_id(h);
    std::vector<counts> old;
    for (const tc::result& r : rs) old.push_back(counts_of(r));
    const auto st = dyng::update(res(), h, b, rs[0], rs[1], rs[2], rs[3]);
    const std::array<tc::stats, 4> stats = {std::get<0>(st), std::get<1>(st), std::get<2>(st),
                                            std::get<3>(st)};
    const id_rows after = rows_by_id(h);
    const std::vector<std::int64_t>& inserted = stats[0].batch.inserted_ids;
    std::vector<std::int64_t> deleted;
    for (std::int64_t id = 0; id + 1 < static_cast<std::int64_t>(before.offsets.size()); ++id) {
      const auto i = static_cast<std::size_t>(id);
      const bool live_before = before.offsets[i + 1] > before.offsets[i];
      const bool live_after = id + 1 < static_cast<std::int64_t>(after.offsets.size()) &&
                              after.offsets[i + 1] > after.offsets[i];
      const bool reinserted = std::find(inserted.begin(), inserted.end(), id) != inserted.end();
      if (live_before && (!live_after || reinserted)) deleted.push_back(id);
    }
    expect_exact(h, rs);
    for (std::size_t i = 0; i < rs.size(); ++i) {
      const tc::pattern p = rs[i].kind();
      SCOPED_TRACE(std::string(tc::to_string(p)));
      const tc::stats& s = stats[i];
      const counts now = counts_of(rs[i]);
      EXPECT_EQ(s.deleted_frontier, static_cast<std::int64_t>(deleted.size()));
      EXPECT_EQ(s.deleted_frontier, s.batch.deleted_hyperedges);
      EXPECT_EQ(s.inserted_frontier, static_cast<std::int64_t>(inserted.size()));
      EXPECT_EQ(s.frontier_visits, s.deleted_frontier + s.inserted_frontier);
      EXPECT_EQ(s.iterations, 0);
      EXPECT_TRUE(s.converged);
      std::int64_t affected = 0;
      for (int k = 0; k < tc::max_bins; ++k) {
        const auto u = static_cast<std::size_t>(k);
        const std::int64_t want =
            k < rs[i].num_bins() ? static_cast<std::int64_t>(now[u] - old[i][u]) : 0;
        EXPECT_EQ(s.delta[u], want) << "bin " << k;
        affected += want != 0 ? 1 : 0;
      }
      EXPECT_EQ(s.affected, affected);
      EXPECT_FALSE(s.fallback_used) << "every backend is incremental";
      EXPECT_EQ(s.engine_used,
                GetParam() == backend::cuda ? dyng::engine::fused : dyng::engine::operators);
      // The exactly-once phases: the oracle's removed(G_t, deleted) and added(G_{t+1}, inserted).
      const counts removed = tr::triad_reference_removed(
          before.offsets, before.members, before.num_vertices, deleted, oracle_pattern(p));
      const counts added = tr::triad_reference_added(
          after.offsets, after.members, after.num_vertices, inserted, oracle_pattern(p));
      EXPECT_EQ(s.triads_removed, sum(removed));
      EXPECT_EQ(s.triads_added, sum(added));
      for (std::size_t k = 0; k < removed.size(); ++k) {
        EXPECT_EQ(s.delta[k], static_cast<std::int64_t>(added[k] - removed[k])) << "bin " << k;
      }
    }
    last_inserted = inserted;
    last_deleted = deleted;
  }

  std::vector<std::int64_t> last_inserted;  ///< the ASSIGNED ids of the last step()
  std::vector<std::int64_t> last_deleted;   ///< the live ids the last step() deleted
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

INSTANTIATE_TEST_SUITE_P(Backends, TriadCount, ::testing::ValuesIn(dyng::test::suite_backends()),
                         backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TriadCount);

/// Hyperedges with every kind of overlap: a common vertex of three, pairwise-only overlaps, a
/// duplicate, a single-member hyperedge, an isolated one.
const rows hand_rows = {{0, 1, 2},       {1, 2, 3}, {2, 3, 4},    {0, 4, 5},       {5, 6},
                        {0, 1, 2},       {7},       {2, 5, 8, 9}, {1, 3, 5, 7, 9}, {10, 11, 12},
                        {0, 2, 4, 6, 8}, {3, 6, 9}};

TEST_P(TriadCount, ComputeMatchesTheOracle) {
  const hgraph h = make(res(), 13, hand_rows);
  const std::vector<tc::result> rs = compute_all(res(), h);
  expect_exact(h, rs);
  const counts motif = counts_of(rs[0]);
  ASSERT_EQ(motif.size(), 30U);
  for (int k = 20; k <= 25; ++k) EXPECT_EQ(motif[static_cast<std::size_t>(k)], 0U) << "open triad";
  EXPECT_GT(sum(motif), 0U);
  EXPECT_EQ(rs[0].num_bins(), 30);
  EXPECT_EQ(rs[1].num_bins(), 1);
  EXPECT_EQ(rs[0].kind(), tc::pattern::h_motif);
  EXPECT_EQ(rs[3].kind(), tc::pattern::stathyper_type3);
  EXPECT_EQ(rs[0].total(), sum(motif));
  // Type 1: sum of C(|e|, 3) over the hyperedges.
  std::uint64_t type1 = 0;
  for (const auto& r : hand_rows) {
    const auto n = static_cast<std::uint64_t>(r.size());
    if (n >= 3) type1 += n * (n - 1) * (n - 2) / 6;
  }
  EXPECT_EQ(rs[1].count(0), type1);
  EXPECT_EQ(rs[0].space(), dyng::memory_space::host);
}

TEST_P(TriadCount, ASingleTriadPerClassLandsInItsBin) {
  // One triple per reachable class (one vertex per region of its representative Venn id).
  const auto reps = tr::triad_class_representatives();
  for (std::size_t cls = 0; cls < reps.size(); ++cls) {
    const int id = reps[cls];
    rows triple(3);
    std::int32_t v = 0;
    for (int region = 0; region < 7; ++region) {
      if (((id >> region) & 1) == 0) continue;
      const std::array<std::array<int, 3>, 7> in = {
          {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}};
      for (int e = 0; e < 3; ++e)
        if (in[static_cast<std::size_t>(region)][static_cast<std::size_t>(e)] != 0)
          triple[static_cast<std::size_t>(e)].push_back(v);
      ++v;
    }
    if (std::any_of(triple.begin(), triple.end(), [](const auto& r) { return r.empty(); })) {
      continue;
    }
    SCOPED_TRACE("class " + std::to_string(cls + 1) + ", Venn id " + std::to_string(id));
    const hgraph h = make(res(), v, triple);
    const tc::result r = tc::compute(res(), h);
    const counts c = counts_of(r);
    const bool closed = cls < 20 || cls >= 26;  // ids 1-20 and 27-30: a triangle of the line graph
    for (std::size_t k = 0; k < c.size(); ++k) {
      EXPECT_EQ(c[k], closed && k == cls ? 1U : 0U) << "bin " << k;
    }
  }
}

TEST_P(TriadCount, UpdatesMatchTheOracleAndTheirStats) {
  // A random hypergraph on 1,300 vertices (small hyperedges on the first 60, so that they overlap,
  // and one of 1,100 members), then 5 batches: random deletions (with repeated and dead ids),
  // insertions of random rows, of copies of live rows, of single members and of a hyperedge of
  // 1,050 members.
  std::mt19937_64 rng(4242);
  const std::int32_t n = 1300;
  auto random_row = [&](int max_card) {
    std::uniform_int_distribution<int> card(1, max_card);
    std::uniform_int_distribution<std::int32_t> vertex(0, 59);
    std::set<std::int32_t> s;
    const int k = card(rng);
    while (static_cast<int>(s.size()) < k) s.insert(vertex(rng));
    return std::vector<std::int32_t>(s.begin(), s.end());
  };
  rows initial;
  for (int i = 0; i < 45; ++i) initial.push_back(random_row(6));
  std::vector<std::int32_t> big(1100);
  std::iota(big.begin(), big.end(), 30);
  initial.push_back(big);
  hgraph h = make(res(), n, initial);
  std::vector<tc::result> rs = compute_all(res(), h);
  expect_exact(h, rs);
  for (int b = 0; b < 5; ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    const id_rows g = rows_by_id(h);
    std::vector<std::int64_t> live;
    for (std::size_t i = 0; i + 1 < g.offsets.size(); ++i)
      if (g.offsets[i + 1] > g.offsets[i]) live.push_back(static_cast<std::int64_t>(i));
    batch_t batch;
    std::shuffle(live.begin(), live.end(), rng);
    const std::size_t deletions = std::min<std::size_t>(live.size(), 4 + 3 * b);
    for (std::size_t i = 0; i < deletions; ++i) batch.delete_hyperedge(live[i]);
    if (deletions > 0) batch.delete_hyperedge(live[0]);                       // repeated
    batch.delete_hyperedge(static_cast<std::int64_t>(g.offsets.size()) + 5);  // out of range
    for (int i = 0; i < 6; ++i) batch.insert_hyperedge(dyng::host_view(random_row(7)));
    if (deletions < live.size()) {  // a copy of a live hyperedge
      const auto id = static_cast<std::size_t>(live[deletions]);
      batch.insert_hyperedge(dyng::host_view(std::vector<std::int32_t>(
          g.members.begin() + g.offsets[id], g.members.begin() + g.offsets[id + 1])));
    }
    batch.insert_hyperedge({static_cast<std::int32_t>(b)});  // a single member
    if (b == 2) {
      std::vector<std::int32_t> wide(1050);
      std::iota(wide.begin(), wide.end(), 10);
      batch.insert_hyperedge(dyng::host_view(wide));
    }
    step(h, rs, batch.view());
    h.check_integrity(res());
  }
}

TEST_P(TriadCount, TheIdBoundGoingDownKeepsTheCountsExact) {
  // Deleting the top ids leaves dead slots; a surplus insertion (one row larger than every dead
  // slot) drops them, so hyperedge_id_bound() goes down while the line graph keeps its keys (the
  // largest bound so far): every h2v-indexed loop must stop at the h2v view's bound.
  const rows initial = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 2, 4}, {1, 3, 4}, {0, 1, 4}};
  hgraph h = make(res(), 40, initial);
  std::vector<tc::result> rs = compute_all(res(), h);
  batch_t first;
  first.delete_hyperedge(4);
  first.delete_hyperedge(5);
  step(h, rs, first.view());
  EXPECT_EQ(h.hyperedge_id_bound(), 6);
  batch_t second;
  second.insert_hyperedge({0, 1, 2, 3, 4, 10, 11, 12, 13, 14});
  step(h, rs, second.view());
  EXPECT_EQ(h.hyperedge_id_bound(), 5) << "the bound went down";
  batch_t third;
  third.delete_hyperedge(0);
  third.insert_hyperedge({1, 2, 4});
  third.insert_hyperedge({0, 3});
  step(h, rs, third.view());
  h.check_integrity(res());
  // The bound goes down in the batch that deletes the top ids: the rows above the new bound held
  // live hyperedges of three members until this batch (a type-1 count past the h2v view's bound
  // would find them).
  hgraph g = make(res(), 40, initial);
  std::vector<tc::result> gs = compute_all(res(), g);
  batch_t at_once;
  at_once.delete_hyperedge(4);
  at_once.delete_hyperedge(5);
  at_once.insert_hyperedge({0, 1, 2, 3, 4, 10, 11, 12, 13, 14});
  step(g, gs, at_once.view());
  EXPECT_LT(g.hyperedge_id_bound(), 6) << "the bound went down in the same batch";
  g.check_integrity(res());
}

TEST_P(TriadCount, SpecialBatches) {
  hgraph h = make(res(), 13, hand_rows);
  std::vector<tc::result> rs = compute_all(res(), h);
  {
    SCOPED_TRACE("empty");
    step(h, rs, batch_t{}.view());
  }
  {
    SCOPED_TRACE("insertions only");
    batch_t b;
    b.insert_hyperedge({0, 1, 2, 3});
    b.insert_hyperedge({2, 3, 4, 5});
    step(h, rs, b.view());
  }
  {
    SCOPED_TRACE("deletions only");
    batch_t b;
    b.delete_hyperedge(1);
    b.delete_hyperedge(8);
    step(h, rs, b.view());
  }
  {
    SCOPED_TRACE("a batch larger than the hypergraph");
    batch_t b;
    for (std::int32_t i = 0; i < 30; ++i) b.insert_hyperedge({i % 13, (i + 1) % 13, (i * 5) % 13});
    b.delete_hyperedge(0);
    step(h, rs, b.view());
  }
  {
    SCOPED_TRACE("delete everything");
    batch_t b;
    for (std::int64_t id = 0; id < h.hyperedge_id_bound(); ++id) b.delete_hyperedge(id);
    step(h, rs, b.view());
    EXPECT_EQ(h.num_hyperedges(), 0);
    for (const tc::result& r : rs) EXPECT_EQ(r.total(), 0U);
  }
  {
    SCOPED_TRACE("insert into the empty hypergraph");
    batch_t b;
    for (const auto& r : hand_rows) b.insert_hyperedge(dyng::host_view(r));
    step(h, rs, b.view());
  }
}

TEST_P(TriadCount, RequirementsAreCheckedPerPattern) {
  const hgraph none = make(res(), 13, hand_rows, dyng::line_graph_policy::none);
  for (const tc::pattern p :
       {tc::pattern::h_motif, tc::pattern::stathyper_type2, tc::pattern::stathyper_type3}) {
    SCOPED_TRACE(std::string(tc::to_string(p)));
    try {
      (void)tc::compute(res(), none, kind_of(p));
      ADD_FAILURE() << "accepted a hypergraph without a line graph";
    } catch (const dyng::invalid_argument_error& e) {
      EXPECT_NE(std::string(e.what()).find("line_graph_policy::none"), std::string::npos)
          << e.what();
      EXPECT_NE(std::string(e.what()).find(std::string(tc::to_string(p))), std::string::npos);
    }
  }
  // Type 1 reads the hyperedges' sizes only: no line graph, and updates keep none.
  hgraph h = make(res(), 13, hand_rows, dyng::line_graph_policy::none);
  tc::result r1 = tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1));
  EXPECT_EQ(counts_of(r1), oracle(rows_by_id(h), tc::pattern::stathyper_type1));
  batch_t b;
  b.delete_hyperedge(2);
  b.insert_hyperedge({4, 5, 6, 7});
  (void)tc::update(res(), h, b.view(), r1);
  EXPECT_EQ(counts_of(r1), oracle(rows_by_id(h), tc::pattern::stathyper_type1));
  // Without v2h the line graph cannot be kept: h_motif names store_v2h.
  const hgraph no_v2h = make(res(), 13, hand_rows, dyng::line_graph_policy::automatic, false);
  try {
    (void)tc::compute(res(), no_v2h);
    ADD_FAILURE() << "accepted a hypergraph without v2h";
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_NE(std::string(e.what()).find("store_v2h"), std::string::npos) << e.what();
  }
  // Under automatic, compute() builds the line graph once and the counts are exact.
  const hgraph automatic = make(res(), 13, hand_rows, dyng::line_graph_policy::automatic);
  expect_exact(automatic, compute_all(res(), automatic));
}

TEST_P(TriadCount, VertexIdsAtTheInt32LimitAreRefusedWhereTheMergeCannotCountThem) {
  // The rows hold vertex + 1 and the classifying merge ends a row at INT32_MAX: with
  // 2,147,483,647 vertices the vertex 2^31 - 2 would read as the end of its row. h_motif and type
  // 3 refuse such a hypergraph (before they need its line graph); type 1 counts it.
  constexpr std::int32_t n = std::numeric_limits<std::int32_t>::max();
  const hgraph h =
      make(res(), n, {{0, 5, n - 1}, {5, n - 1}}, dyng::line_graph_policy::none, false);
  for (const tc::pattern p : {tc::pattern::h_motif, tc::pattern::stathyper_type3}) {
    try {
      (void)tc::compute(res(), h, kind_of(p));
      ADD_FAILURE() << "accepted vertex ids up to 2^31 - 2";
    } catch (const dyng::invalid_argument_error& e) {
      EXPECT_NE(std::string(e.what()).find("2^31 - 3"), std::string::npos) << e.what();
    }
  }
  EXPECT_EQ(tc::compute(res(), h, kind_of(tc::pattern::stathyper_type1)).count(0), 1U);
  // One vertex fewer is accepted by the limit (and fails only for the missing line graph).
  const hgraph below = make(res(), n - 1, {{0, 5, n - 2}}, dyng::line_graph_policy::none, false);
  try {
    (void)tc::compute(res(), below);
    ADD_FAILURE() << "accepted a hypergraph without a line graph";
  } catch (const dyng::invalid_argument_error& e) {
    EXPECT_EQ(std::string(e.what()).find("2^31 - 3"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("line graph"), std::string::npos) << e.what();
  }
}

TEST_P(TriadCount, OptionsAndTheResult) {
  hgraph h = make(res(), 13, hand_rows);
  tc::result r = tc::compute(res(), h);
  EXPECT_EQ(tc::to_string(tc::pattern::h_motif), "h_motif");
  EXPECT_EQ(tc::to_string(tc::pattern::stathyper_type1), "stathyper_type1");
  EXPECT_EQ(tc::to_string(tc::pattern::stathyper_type2), "stathyper_type2");
  EXPECT_EQ(tc::to_string(tc::pattern::stathyper_type3), "stathyper_type3");
  EXPECT_THROW((void)r.count(-1), dyng::invalid_argument_error);
  EXPECT_THROW((void)r.count(30), dyng::invalid_argument_error);
  tc::options o = r.get_options();
  o.kind = tc::pattern::stathyper_type1;
  EXPECT_THROW(r.set_options(o), dyng::invalid_argument_error);  // kind is fixed
  o.kind = tc::pattern::h_motif;
  o.cuda_engine = dyng::engine::fused;
  r.set_options(o);
  EXPECT_EQ(r.get_options().cuda_engine, dyng::engine::fused);
  const tc::result copy = r.clone(dyng::resources::sequential());
  EXPECT_EQ(counts_of(copy), counts_of(r));
  EXPECT_EQ(copy.graph_version(), r.graph_version());
  tc::options op;
  op.cuda_engine = dyng::engine::operators;
  if (GetParam() == backend::cuda) {
    EXPECT_THROW((void)tc::compute(res(), h, op), dyng::not_supported_error);
    r.set_options(op);
    batch_t b;
    b.delete_hyperedge(0);
    const std::uint64_t version = h.version();
    EXPECT_THROW((void)tc::update(res(), h, b.view(), r), dyng::not_supported_error);
    EXPECT_EQ(h.version(), version) << "nothing changes";
  } else {
    EXPECT_EQ(counts_of(tc::compute(res(), h, op)), counts_of(r));  // ignored on the host
  }
  tc::result moved = std::move(r);
  EXPECT_THROW((void)r.counts(), dyng::invalid_argument_error);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved.num_bins(), 30);
  // A result is stale after a separate apply.
  batch_t b;
  b.delete_hyperedge(3);
  (void)h.apply(res(), b.view());
  moved.set_options(tc::options{});
  EXPECT_THROW((void)tc::update(res(), h, batch_t{}.view(), moved), dyng::stale_result_error);
}

TEST_P(TriadCount, ABinThatWouldGoNegativePoisonsTheResult) {
  hgraph h = make(res(), 13, hand_rows);
  tc::result r = tc::compute(res(), h);
  // A corrupted result (no triad) cannot lose the triads the batch destroys: the signed delta of
  // the subtract phase would take a bin below zero (every backend is incremental).
  dyng::detail::triad_count_access::state(r).counts.fill(0);
  batch_t b;
  b.delete_hyperedge(0);
  b.delete_hyperedge(1);
  const std::uint64_t version = h.version();
  EXPECT_THROW((void)tc::update(res(), h, b.view(), r), dyng::internal_error);
  EXPECT_EQ(h.version(), version + 1) << "the batch was applied";
  EXPECT_THROW((void)r.counts(), dyng::stale_result_error) << "the result is poisoned";
}

TEST_P(TriadCount, TwoPatternsAndAFakePartnerShareOneCommit) {
  namespace fakes = dyng::test::framework_fakes;
  hgraph joint = make(res(), 13, hand_rows);
  hgraph alone = make(res(), 13, hand_rows);
  hgraph alone1 = make(res(), 13, hand_rows);
  hgraph alone_fake = make(res(), 13, hand_rows);
  tc::result motif = tc::compute(res(), joint);
  tc::result type1 = tc::compute(res(), joint, kind_of(tc::pattern::stathyper_type1));
  fakes::hyper_pairs_result pairs = fakes::compute_pairs(res(), joint);
  tc::result motif_alone = tc::compute(res(), alone);
  tc::result type1_alone = tc::compute(res(), alone1, kind_of(tc::pattern::stathyper_type1));
  fakes::hyper_pairs_result pairs_alone = fakes::compute_pairs(res(), alone_fake);
  for (int step = 0; step < 3; ++step) {
    batch_t b;
    b.delete_hyperedge(step);
    b.insert_hyperedge({step, step + 3, step + 6});
    b.insert_hyperedge({1, 2, 3, step + 9});
    const auto [s_motif, s_type1, s_pairs] =
        dyng::update(res(), joint, b.view(), motif, type1, pairs);
    const auto a_motif = tc::update(res(), alone, b.view(), motif_alone);
    const auto a_type1 = tc::update(res(), alone1, b.view(), type1_alone);
    (void)fakes::update_pairs(res(), alone_fake, b.view(), pairs_alone);
    EXPECT_EQ(counts_of(motif), counts_of(motif_alone));
    EXPECT_EQ(counts_of(type1), counts_of(type1_alone));
    EXPECT_EQ(pairs.value, pairs_alone.value);
    EXPECT_EQ(s_motif.delta, a_motif.delta);
    EXPECT_EQ(s_type1.delta, a_type1.delta);
    EXPECT_EQ(s_motif.batch.inserted_ids, a_motif.batch.inserted_ids);
    EXPECT_EQ(s_motif.triads_removed, a_motif.triads_removed);
    EXPECT_EQ(s_motif.triads_added, a_motif.triads_added);
    (void)s_pairs;
  }
  std::vector<tc::result> both;
  both.push_back(std::move(motif));
  both.push_back(std::move(type1));
  expect_exact(joint, both);
}

/// The three member rows of one triple of a Venn id (one vertex per region of the id); empty when
/// a member would be empty (the id needs every hyperedge in some region).
rows triple_of(int id, std::int32_t& vertices) {
  rows triple(3);
  std::int32_t v = 0;
  const std::array<std::array<int, 3>, 7> in = {
      {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}}};
  for (int region = 0; region < 7; ++region) {
    if (((id >> region) & 1) == 0) continue;
    for (int e = 0; e < 3; ++e) {
      if (in[static_cast<std::size_t>(region)][static_cast<std::size_t>(e)] != 0) {
        triple[static_cast<std::size_t>(e)].push_back(v);
      }
    }
    ++v;
  }
  vertices = v;
  if (std::any_of(triple.begin(), triple.end(), [](const auto& r) { return r.empty(); })) {
    return {};
  }
  return triple;
}

TEST_P(TriadCount, EveryClassIsAddedAndRemovedByUpdates) {
  // For each reachable class: from an empty hypergraph, insert its triple in one batch (one
  // frontier of three: the triad is counted once, at its smallest member), delete one member,
  // reinsert it, then insert the three again as three batches of one; the closed classes (bins
  // 0-19 and 26-29) move by one in their bin, the open ones (20-25) never move.
  const auto reps = tr::triad_class_representatives();
  for (std::size_t cls = 0; cls < reps.size(); ++cls) {
    std::int32_t v = 0;
    const rows triple = triple_of(reps[cls], v);
    if (triple.empty()) continue;
    SCOPED_TRACE("class " + std::to_string(cls + 1));
    hgraph h = make(res(), 7, {});
    std::vector<tc::result> rs = compute_all(res(), h);
    const bool closed = cls < 20 || cls >= 26;
    batch_t all;
    for (const auto& r : triple) all.insert_hyperedge(dyng::host_view(r));
    {
      step(h, rs, all.view());
      const counts c = counts_of(rs[0]);
      for (std::size_t k = 0; k < c.size(); ++k) {
        EXPECT_EQ(c[k], closed && k == cls ? 1U : 0U) << "bin " << k;
      }
    }
    batch_t one;
    one.delete_hyperedge(last_inserted.empty() ? 0 : last_inserted[1]);
    step(h, rs, one.view());
    EXPECT_EQ(rs[0].total(), 0U);
    batch_t back;
    back.insert_hyperedge(dyng::host_view(triple[1]));
    step(h, rs, back.view());
    EXPECT_EQ(rs[0].total(), closed ? 1U : 0U);
    batch_t none;
    for (std::int64_t id = 0; id < h.hyperedge_id_bound(); ++id) none.delete_hyperedge(id);
    step(h, rs, none.view());
    for (const auto& r : triple) {
      batch_t b;
      b.insert_hyperedge(dyng::host_view(r));
      step(h, rs, b.view());
    }
    EXPECT_EQ(rs[0].total(), closed ? 1U : 0U);
    const counts c = counts_of(rs[0]);
    for (int k = 20; k <= 25; ++k) EXPECT_EQ(c[static_cast<std::size_t>(k)], 0U) << "open triad";
    h.check_integrity(res());
  }
}

TEST_P(TriadCount, ReusedIdsAndAnInsertedIdEqualToADeletedId) {
  // Under erase_first a slot freed by the batch is reused in the same batch: the add phase's
  // frontier holds an id the subtract phase's frontier held too (with another row). Under
  // insert_first the new ids are never the batch's deleted ones.
  for (const dyng::id_reuse reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    SCOPED_TRACE(std::string(dyng::to_string(reuse)));
    dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
    list.num_vertices = 13;
    for (const auto& r : hand_rows) list.add_hyperedge(dyng::host_view(r));
    dyng::hypergraph_properties props;
    props.line_graph = dyng::line_graph_policy::sorted_merge;
    props.reuse = reuse;
    hgraph h = hgraph::from_hyperedges(res(), list.view(), props);
    std::vector<tc::result> rs = compute_all(res(), h);
    for (int round = 0; round < 3; ++round) {
      batch_t b;
      b.delete_hyperedge(2 + round);
      b.delete_hyperedge(7);
      b.insert_hyperedge({2, 3, 4});  // the same members as the deleted id 2 (round 0)
      b.insert_hyperedge({0, 1, 2, 5, 9});
      b.insert_hyperedge({4, 5, 6, 7});
      step(h, rs, b.view());
      bool reused = false;
      for (const std::int64_t id : last_inserted) {
        reused =
            reused || std::find(last_deleted.begin(), last_deleted.end(), id) != last_deleted.end();
      }
      if (reuse == dyng::id_reuse::insert_first) {
        EXPECT_FALSE(reused) << "insert_first never reuses a slot of the batch";
      } else if (round == 0) {
        EXPECT_TRUE(reused) << "erase_first reuses the freed slots";
      }
      h.check_integrity(res());
    }
  }
}

TEST_P(TriadCount, DuplicatesSingleMembersAndWideHyperedges) {
  hgraph h = make(res(), 1200, hand_rows);
  std::vector<tc::result> rs = compute_all(res(), h);
  std::vector<std::int32_t> wide(1100);
  std::iota(wide.begin(), wide.end(), 0);
  std::vector<std::int32_t> wide2(1030);
  std::iota(wide2.begin(), wide2.end(), 50);
  {
    SCOPED_TRACE("duplicates of each other and of live rows, single members, two wide rows");
    batch_t b;
    b.insert_hyperedge({0, 1, 2});  // a third copy of hand_rows[0]
    b.insert_hyperedge({0, 1, 2});
    b.insert_hyperedge({5});
    b.insert_hyperedge({5});
    b.insert_hyperedge({11});
    b.insert_hyperedge(dyng::host_view(wide));
    b.insert_hyperedge(dyng::host_view(wide2));
    b.insert_hyperedge(dyng::host_view(wide2));
    step(h, rs, b.view());
    h.check_integrity(res());
  }
  {
    SCOPED_TRACE("delete one copy of each duplicate and one wide row, repeated deletions");
    batch_t b;
    b.delete_hyperedge(0);
    b.delete_hyperedge(0);
    b.delete_hyperedge(last_inserted[0]);
    b.delete_hyperedge(last_inserted[2]);
    b.delete_hyperedge(last_inserted[6]);
    b.insert_hyperedge({1, 2, 3, 4, 5, 6});
    step(h, rs, b.view());
    h.check_integrity(res());
  }
}

TEST_P(TriadCount, RandomMultiBatchSequencesMatchTheOracle) {
  // The property test: random hypergraphs and 4 consecutive random batches (deletions of random
  // live, dead and repeated ids; insertions of random rows, copies of live rows, single members
  // and now and then a hyperedge of more than 1,024 members) under both id_reuse orders; after
  // every batch the counts and stats of all four patterns equal the oracle's (step()).
  for (const dyng::id_reuse reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      SCOPED_TRACE(std::string(dyng::to_string(reuse)) + " seed " + std::to_string(seed));
      std::mt19937_64 rng(seed * 7919 + (reuse == dyng::id_reuse::erase_first ? 0 : 1));
      std::uniform_int_distribution<std::int32_t> vertex(0, 39);
      std::uniform_int_distribution<int> card(1, 6);
      auto random_row = [&] {
        std::set<std::int32_t> m;
        const int k = card(rng);
        while (static_cast<int>(m.size()) < k) m.insert(vertex(rng));
        return std::vector<std::int32_t>(m.begin(), m.end());
      };
      dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
      list.num_vertices = 1100;
      for (int i = 0; i < 30; ++i) list.add_hyperedge(dyng::host_view(random_row()));
      dyng::hypergraph_properties props;
      props.line_graph = dyng::line_graph_policy::sorted_merge;
      props.reuse = reuse;
      hgraph h = hgraph::from_hyperedges(res(), list.view(), props);
      std::vector<tc::result> rs = compute_all(res(), h);
      for (int b = 0; b < 4; ++b) {
        SCOPED_TRACE("batch " + std::to_string(b));
        const id_rows g = rows_by_id(h);
        std::vector<std::int64_t> live;
        for (std::size_t i = 0; i + 1 < g.offsets.size(); ++i)
          if (g.offsets[i + 1] > g.offsets[i]) live.push_back(static_cast<std::int64_t>(i));
        std::shuffle(live.begin(), live.end(), rng);
        batch_t batch;
        const std::size_t deletions = std::min<std::size_t>(live.size(), rng() % 9);
        for (std::size_t i = 0; i < deletions; ++i) batch.delete_hyperedge(live[i]);
        if (deletions > 0) batch.delete_hyperedge(live[0]);
        batch.delete_hyperedge(h.hyperedge_id_bound() + 3);
        const int insertions = static_cast<int>(rng() % 9);
        for (int i = 0; i < insertions; ++i) batch.insert_hyperedge(dyng::host_view(random_row()));
        if (deletions < live.size()) {
          const auto id = static_cast<std::size_t>(live[deletions]);
          batch.insert_hyperedge(dyng::host_view(std::vector<std::int32_t>(
              g.members.begin() + g.offsets[id], g.members.begin() + g.offsets[id + 1])));
        }
        batch.insert_hyperedge({vertex(rng)});
        if (b == 1) {
          std::vector<std::int32_t> wide(1050);
          std::iota(wide.begin(), wide.end(), static_cast<std::int32_t>(seed));
          batch.insert_hyperedge(dyng::host_view(wide));
        }
        step(h, rs, batch.view());
        h.check_integrity(res());
      }
    }
  }
}

TEST_P(TriadCount, OpenMPThreadCountsGiveIdenticalCounts) {
  if (GetParam() != backend::openmp) {
    GTEST_SKIP() << "the OpenMP backend's thread counts";
  }
  // The host engine's static partition and per-thread bins added in thread order: 1, 8 and 56
  // threads give the same counts and stats on the same batches.
  std::vector<std::vector<counts>> runs;
  std::vector<std::vector<std::array<std::int64_t, 30>>> deltas;
  for (const int threads : {1, 8, 56}) {
    const resources r = resources::openmp(threads);
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<std::int32_t> vertex(0, 299);
    dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
    list.num_vertices = 300;
    for (int i = 0; i < 1500; ++i) {
      std::set<std::int32_t> m;
      const int k = 1 + static_cast<int>(rng() % 6);
      while (static_cast<int>(m.size()) < k) m.insert(vertex(rng));
      list.add_hyperedge(dyng::host_view(std::vector<std::int32_t>(m.begin(), m.end())));
    }
    dyng::hypergraph_properties props;
    props.line_graph = dyng::line_graph_policy::sorted_merge;
    hgraph h = hgraph::from_hyperedges(r, list.view(), props);
    std::vector<tc::result> rs = compute_all(r, h);
    std::vector<counts> got;
    std::vector<std::array<std::int64_t, 30>> d;
    for (const auto& x : rs) got.push_back(counts_of(x));
    for (int b = 0; b < 3; ++b) {
      batch_t batch;
      for (int i = 0; i < 200; ++i) batch.delete_hyperedge(static_cast<std::int64_t>(rng() % 1500));
      for (int i = 0; i < 200; ++i) {
        batch.insert_hyperedge({vertex(rng), vertex(rng) / 2 + 150, vertex(rng) / 3});
      }
      const auto st = dyng::update(r, h, batch.view(), rs[0], rs[1], rs[2], rs[3]);
      d.push_back(std::get<0>(st).delta);
      d.push_back(std::get<1>(st).delta);
      d.push_back(std::get<2>(st).delta);
      d.push_back(std::get<3>(st).delta);
      for (const auto& x : rs) got.push_back(counts_of(x));
    }
    runs.push_back(got);
    deltas.push_back(d);
  }
  EXPECT_EQ(runs[1], runs[0]) << "8 threads";
  EXPECT_EQ(runs[2], runs[0]) << "56 threads";
  EXPECT_EQ(deltas[1], deltas[0]) << "8 threads";
  EXPECT_EQ(deltas[2], deltas[0]) << "56 threads";
}

TEST(TriadCountHostEngine, EveryVennPatternGetsItsClassOrIsReportedAndNotWritten) {
  // The host engine's bins handle (operators::count_delta's): motif_op's class of each of the 128
  // Venn-region patterns lands in its bin; a pattern of no class (bin -1) sets count_invalid_bin
  // and writes nothing (the device engine's error word, triad_count_kernels_test.cu).
  namespace ops = dyng::detail::operators;
  namespace triad = dyng::detail::triad;
  const std::array<std::int8_t, 128> map = tr::triad_motif_class_map();
  int classified = 0;
  for (int id = 0; id < 128; ++id) {
    SCOPED_TRACE("Venn id " + std::to_string(id));
    const int a = (id >> 0) & 1, b = (id >> 1) & 1, c = (id >> 2) & 1;
    const int d = (id >> 3) & 1, e = (id >> 4) & 1, f = (id >> 5) & 1, g = (id >> 6) & 1;
    std::array<std::uint64_t, 32> guarded{};  // bins[-1], the 30 bins, bins[30]
    std::uint32_t status = 0;
    ops::host_count_bins<30> bins{guarded.data() + 1, &status};
    bins.add(triad::compute_motif_index(a + d + f + g, b + d + e + g, c + e + f + g, d + g, e + g,
                                        f + g, g),
             1);
    EXPECT_EQ(guarded[0], 0U) << "bins[-1] was written";
    EXPECT_EQ(guarded[31], 0U) << "bins[30] was written";
    const int expected = map[static_cast<std::size_t>(id)] - 1;
    if (expected < 0) {
      EXPECT_EQ(status, ops::count_invalid_bin);
      for (int k = 1; k <= 30; ++k) EXPECT_EQ(guarded[static_cast<std::size_t>(k)], 0U);
    } else {
      ++classified;
      EXPECT_EQ(status, 0U);
      for (int k = 1; k <= 30; ++k) {
        EXPECT_EQ(guarded[static_cast<std::size_t>(k)], k - 1 == expected ? 1U : 0U) << k - 1;
      }
    }
  }
  EXPECT_EQ(classified, 96);
}

/// The launch counters of a result (internal).
const dyng::detail::triad_launch_counters& launches(const tc::result& r) {
  return dyng::detail::triad_count_access::find(r)->launches;
}

TEST_P(TriadCount, KernelPathsAndEarlyOuts) {
  // mode 0 full, 1 frontier; shape 0 triangles, 1 pairs, 2 keys.
  hgraph h = make(res(), 13, hand_rows);
  std::vector<tc::result> rs = compute_all(res(), h);
  const int shape[4] = {0, 2, 1, 0};
  for (std::size_t i = 0; i < rs.size(); ++i) {
    EXPECT_EQ(launches(rs[i]).kernels[0][shape[i]], 1) << "the full count";
  }
  {  // insertions only: the subtract phase has no sources (the first early out)
    batch_t b;
    b.insert_hyperedge({0, 1, 2, 3});
    step(h, rs, b.view());
    for (std::size_t i = 0; i < rs.size(); ++i) {
      EXPECT_EQ(launches(rs[i]).no_sources[1][shape[i]], 1);
      EXPECT_EQ(launches(rs[i]).kernels[1][shape[i]], 1);
    }
  }
  {  // deleting an isolated hyperedge ({10, 11, 12}): a source without wedges (the second early
     // out; type 1 counts keys, not wedges, and launches its key kernel)
    batch_t b;
    b.delete_hyperedge(9);
    step(h, rs, b.view());
    for (std::size_t i = 0; i < rs.size(); ++i) {
      if (shape[i] == 2) {
        EXPECT_EQ(launches(rs[i]).kernels[1][2], 2);
      } else {
        EXPECT_EQ(launches(rs[i]).no_wedges[1][shape[i]], 1);
      }
      EXPECT_EQ(launches(rs[i]).no_sources[1][shape[i]], 2);  // and nothing inserted
    }
  }
  // Every (mode, shape) ran a kernel (a pass of the host engine).
  for (int m = 0; m < 2; ++m) {
    for (std::size_t i = 0; i < rs.size(); ++i) {
      EXPECT_GE(launches(rs[i]).kernels[m][shape[i]], 1) << "mode " << m << " pattern " << i;
    }
  }
}

TEST_P(TriadCount, EqualShapeBatchesUnderAGrowingIdBoundReserveRarely) {
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  // 50 batches of 10 deletions and 20 insertions: the id bound, and with it the line graph's keys
  // and the frontier flags, grows by about 10 per batch. With headroom (scratch_buffer::
  // reserve_with_headroom) the scratch grows a few times, not in every update (M9a review: the
  // exact reservation freed and reallocated the flags in nearly every batch).
  const resources rs = res();  // one handle: its workspace pool and memory resource
  for (const tc::pattern p : all_patterns) {
    SCOPED_TRACE(std::string(tc::to_string(p)));
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<std::int32_t> vertex(0, 299);
    rows initial;
    for (int i = 0; i < 400; ++i) {
      std::vector<std::int32_t> r = {vertex(rng), vertex(rng), vertex(rng)};
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
      initial.push_back(r);
    }
    hgraph h = make(rs, 300, initial);
    tc::result r = tc::compute(rs, h, kind_of(p));
    int reserving = 0;
    std::int64_t next_deleted = 0;
    for (int b = 0; b < 50; ++b) {
      batch_t batch;
      for (int i = 0; i < 10; ++i) batch.delete_hyperedge(next_deleted++);
      for (int i = 0; i < 20; ++i) batch.insert_hyperedge({vertex(rng), 300 - 1 - i});
      const dyng::detail::framework::budget_scope scope;
      (void)tc::update(rs, h, batch.view(), r);
      reserving += scope.used().reservations > 0 ? 1 : 0;
    }
    EXPECT_LE(reserving, 5) << "updates that grew the triad scratch";
    EXPECT_EQ(counts_of(r), counts_of(tc::compute(rs, h, kind_of(p))));
  }
}

#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA

TEST_P(TriadCount, AnUpdateSynchronizesAtTheEnginesReadPointsOnly) {
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  for (const tc::pattern p : all_patterns) {
    SCOPED_TRACE(std::string(tc::to_string(p)));
    hgraph h = make(res(), 13, hand_rows);
    tc::result r = tc::compute(res(), h, kind_of(p));
    batch_t b;
    b.delete_hyperedge(2);
    b.insert_hyperedge({1, 2, 3, 4});
    (void)tc::update(res(), h, b.view(), r);
    const auto report = dyng::detail::framework::last_budget_report();
    ASSERT_TRUE(report.measured);
    // The wedge count of each phase and the bins (type 1: the bins); ESCHER-GPU's update made
    // 90-96 blocking calls per batch (M9a step 2), its counting part at least 6.
    EXPECT_EQ(report.used.own_host_syncs(), p == tc::pattern::stathyper_type1 ? 1 : 3);
    EXPECT_EQ(report.limit.host_syncs, p == tc::pattern::stathyper_type1 ? 1 : 3);
  }
}

TEST_P(TriadCount, RepeatedRunsAreIdentical) {
  std::vector<counts> first;
  for (int run = 0; run < 3; ++run) {
    std::mt19937_64 rng(77);
    std::uniform_int_distribution<std::int32_t> vertex(0, 199);
    rows initial;
    for (int i = 0; i < 400; ++i) initial.push_back({vertex(rng), vertex(rng), vertex(rng)});
    for (auto& r : initial) {
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
    }
    hgraph h = make(res(), 200, initial);
    std::vector<tc::result> rs = compute_all(res(), h);
    std::vector<counts> got;
    for (int b = 0; b < 3; ++b) {
      batch_t batch;
      for (int i = 0; i < 40; ++i) batch.delete_hyperedge(static_cast<std::int64_t>(i * 7 + b));
      for (int i = 0; i < 40; ++i) batch.insert_hyperedge({vertex(rng) / 2, 100 + vertex(rng) / 2});
      (void)dyng::update(res(), h, batch.view(), rs[0], rs[1], rs[2], rs[3]);
      for (const auto& r : rs) got.push_back(counts_of(r));
    }
    if (run == 0) {
      first = got;
    } else {
      EXPECT_EQ(got, first) << "run " << run;
    }
  }
}
#endif

}  // namespace
