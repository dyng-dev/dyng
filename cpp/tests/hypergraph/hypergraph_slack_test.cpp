// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_slack_test.cpp
 * @brief slack_csr storage and the incidence edits (M11a steps 2 and 3; ADRs 0085, 0086, 0089):
 *        the property matrix, MOSP_ESCHER's skip rules (one test per rule), the phase order, the
 *        applied record, the slack line graph against the oracle after every batch, the
 *        order-free pair delta, the forced CSR-tail rebuild and node-capacity growth, the
 *        candidate limit, clones across the backends, the views; escher storage keeps refusing
 *        incidence edits. Compiled into the host test executable (sequential, openmp) and, with
 *        DYNG_TEST_CUDA=1, into the CUDA one (the CUDA store of M11a step 3).
 */
#include "hypergraph/capacity.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using dyng::backend;
using dyng::capacity_policy;
using dyng::hypergraph_properties;
using dyng::hypergraph_storage;
using dyng::id_reuse;
using dyng::line_graph_policy;
using dyng::resources;
using dyng::unweighted;
using dyng::detail::hypergraph_access;
using dyng::test::hyper::h2v_rows;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using dyng::test::hyper::v2h_rows;
using hgraph = dyng::hypergraph<std::int32_t, unweighted>;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using batch_t = dyng::hyperedge_batch<std::int32_t, unweighted>;
using wbatch_t = dyng::hyperedge_batch<std::int32_t, std::int32_t>;
using pair_list = std::vector<std::pair<std::int64_t, std::int64_t>>;

hypergraph_properties slack_props(id_reuse reuse = id_reuse::insert_first,
                                  line_graph_policy lg = line_graph_policy::slack_csr,
                                  capacity_policy c = capacity_policy::mosp_escher(256)) {
  hypergraph_properties p;
  p.storage = hypergraph_storage::slack_csr;
  p.line_graph = lg;
  p.reuse = reuse;
  p.capacity = c;
  return p;
}

class SlackBackends : public ::testing::TestWithParam<backend> {
 protected:
  resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
  bool cuda() const {
    return GetParam() == backend::cuda;
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

// The flat rows of a hypergraph by id (the oracle's input).
template <typename weight_t>
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> flat(
    const dyng::hypergraph<std::int32_t, weight_t>& h) {
  const auto v = h.h2v();
  return {dyng::test::hyper::host_vector(v.row_ptr), dyng::test::hyper::host_vector(v.col_ind)};
}

pair_list zip(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
  pair_list out;
  for (std::size_t i = 0; i < a.size(); ++i) out.emplace_back(a[i], b[i]);
  return out;
}

// Applies a batch and checks everything a batch must give on slack_csr storage: the pair delta
// against testing::line_graph_delta of the states, h2h() against rebuild_line_graph, the store's
// integrity (its slack CSR against the incidence lists), the record's rows and lengths.
template <typename graph_t, typename batch_view_t>
dyng::hyper_apply_summary apply_checked(const resources& res, graph_t& h, const batch_view_t& b) {
  const auto pre = flat(h);
  const auto s = h.apply(res, b);
  const auto post = flat(h);
  const auto& a = hypergraph_access::applied(h);
  const auto& pairs = hypergraph_access::pairs(h);
  EXPECT_TRUE(a.has_slack_record);
  if (h.properties().line_graph != line_graph_policy::none &&
      hypergraph_access::store(h).line_graph_enabled()) {
    const auto want = dyng::testing::line_graph_delta(pre.first, pre.second, post.first,
                                                      post.second, h.num_vertices());
    EXPECT_TRUE(a.has_line_graph_delta);
    EXPECT_EQ(zip(pairs.deleted_a, pairs.deleted_b), want.first);
    EXPECT_EQ(zip(pairs.inserted_a, pairs.inserted_b), want.second);
    EXPECT_EQ(dyng::test::hyper::check_line_graph(h), "");
    const auto* slack = hypergraph_access::slack(h);
    EXPECT_NE(slack, nullptr);
    if (slack != nullptr && !h.h2h().col_ind.empty()) {
      EXPECT_EQ(slack->rows().pair_count * 2, h.h2h().num_entries());
    }
  }
  // The touched rows before and after, the touched vertices' lengths after.
  const rows now = h2v_rows(h);
  for (std::size_t i = 0; i < a.touched_hyperedges.size(); ++i) {
    const std::int64_t id = a.touched_hyperedges[i];
    std::vector<std::int32_t> post_row(
        a.touched_post_rows.members.begin() + a.touched_post_rows.offsets[i],
        a.touched_post_rows.members.begin() + a.touched_post_rows.offsets[i + 1]);
    const std::vector<std::int32_t> want = id < static_cast<std::int64_t>(now.size())
                                               ? now[static_cast<std::size_t>(id)]
                                               : std::vector<std::int32_t>{};
    EXPECT_EQ(post_row, want) << "id " << id;
    std::vector<std::int32_t> pre_row(
        a.touched_pre_rows.members.begin() + a.touched_pre_rows.offsets[i],
        a.touched_pre_rows.members.begin() + a.touched_pre_rows.offsets[i + 1]);
    std::vector<std::int32_t> pre_want;
    if (id + 1 < static_cast<std::int64_t>(pre.first.size()))
      pre_want.assign(pre.second.begin() + pre.first[static_cast<std::size_t>(id)],
                      pre.second.begin() + pre.first[static_cast<std::size_t>(id) + 1]);
    EXPECT_EQ(pre_row, pre_want) << "id " << id;
  }
  const auto v2h = v2h_rows(h);
  for (std::size_t i = 0; i < a.touched_vertices.size(); ++i)
    EXPECT_EQ(
        a.touched_vertex_lengths[i],
        static_cast<std::int32_t>(v2h[static_cast<std::size_t>(a.touched_vertices[i])].size()));
  h.check_integrity(res);
  return s;
}

// ---- The property matrix (HD9) ------------------------------------------------------------

TEST_P(SlackBackends, ThePropertyMatrix) {
  const auto list = list_of(4, rows{{0, 1}, {1, 2}});
  const auto expect_not_supported = [&](hypergraph_properties p, const std::string& names) {
    try {
      (void)hgraph::from_hyperedges(res(), list.view(), p);
      ADD_FAILURE() << "accepted";
    } catch (const dyng::not_supported_error& e) {
      EXPECT_NE(std::string(e.what()).find(names), std::string::npos) << e.what();
    }
  };
  // escher storage: none, automatic (sorted_merge) and sorted_merge; slack_csr's line graph is
  // refused, naming sorted_merge and slack_csr storage.
  hypergraph_properties escher;
  for (const auto lg :
       {line_graph_policy::none, line_graph_policy::automatic, line_graph_policy::sorted_merge}) {
    escher.line_graph = lg;
    EXPECT_NO_THROW((void)hgraph::from_hyperedges(res(), list.view(), escher));
  }
  escher.line_graph = line_graph_policy::slack_csr;
  expect_not_supported(escher, "line_graph_policy::sorted_merge");
  expect_not_supported(escher, "hypergraph_storage::slack_csr");
  // slack_csr storage: none, automatic (slack_csr) and slack_csr; sorted_merge is refused,
  // naming slack_csr; v2h is required.
  auto slack = slack_props();
  slack.line_graph = line_graph_policy::sorted_merge;
  expect_not_supported(slack, "line_graph_policy::slack_csr");
  slack.line_graph = line_graph_policy::slack_csr;
  slack.store_v2h = false;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list.view(), slack),
               dyng::invalid_argument_error);
  slack.store_v2h = true;
  for (const auto lg :
       {line_graph_policy::none, line_graph_policy::automatic, line_graph_policy::slack_csr}) {
    slack.line_graph = lg;
    const hgraph h = hgraph::from_hyperedges(res(), list.view(), slack);
    EXPECT_EQ(hypergraph_access::store(h).line_graph_enabled(), lg == line_graph_policy::slack_csr);
    h.check_integrity(res());
  }
}

// slack_csr storage on every backend (the CUDA store of M11a step 3, ADR 0089): a hypergraph moves
// between the host backends and cuda with its ids, rows, weights and line graph, and the same
// later batch gives the same ids, rows and pairs on both.
TEST_P(SlackBackends, SlackCsrMovesBetweenTheHostAndCuda) {
  if (!dyng::backend_available(backend::cuda)) GTEST_SKIP() << "no CUDA in this build";
  const auto list = list_of(8, rows{{0, 1, 2}, {2, 3}, {4}, {5, 6}});
  const resources device = resources::cuda(dyng::test::hyper::resources_of(backend::cuda).device());
  const resources host = dyng::test::hyper::resources_of(backend::sequential);
  hgraph a = hgraph::from_hyperedges(res(), list.view(), slack_props());
  batch_t b;
  b.delete_hyperedge(2);
  b.insert_incidence(0, 7);
  b.delete_incidence(1, 3);
  b.insert_hyperedge({3, 4, 6});
  (void)a.apply(res(), b.view());
  for (const resources& to : {device, host}) {
    hgraph c = a.to_backend(to);
    EXPECT_EQ(h2v_rows(c), h2v_rows(a));
    EXPECT_EQ(dyng::test::hyper::h2h_rows(c), dyng::test::hyper::h2h_rows(a));
    c.check_integrity(to);
    hgraph d = a.clone(res());
    batch_t next;
    next.delete_incidence(0, 7);
    next.insert_hyperedge({1, 4});
    next.insert_incidence(3, 2);
    const auto s1 = apply_checked(to, c, next.view());
    const auto s2 = apply_checked(res(), d, next.view());
    EXPECT_EQ(s1.inserted_ids, s2.inserted_ids);
    EXPECT_EQ(h2v_rows(c), h2v_rows(d));
    EXPECT_EQ(hypergraph_access::pairs(c).inserted_a, hypergraph_access::pairs(d).inserted_a);
    EXPECT_EQ(hypergraph_access::pairs(c).deleted_b, hypergraph_access::pairs(d).deleted_b);
  }
}

// ---- escher storage keeps refusing incidence edits (strong guarantee) ----------------------

TEST_P(SlackBackends, EscherStorageRefusesIncidenceEdits) {
  hypergraph_properties p;
  p.line_graph = line_graph_policy::none;
  hgraph h = hgraph::from_hyperedges(res(), list_of(4, rows{{0, 1}, {1, 2}}).view(), p);
  const rows before = h2v_rows(h);
  for (int kind = 0; kind < 2; ++kind) {
    batch_t b;
    b.delete_hyperedge(0);
    if (kind == 0) b.insert_incidence(1, 3);
    if (kind == 1) b.delete_incidence(1, 2);
    try {
      (void)h.apply(res(), b.view());
      ADD_FAILURE() << "an incidence edit was accepted under escher storage";
    } catch (const dyng::not_supported_error& e) {
      EXPECT_NE(std::string(e.what()).find("hypergraph_storage::slack_csr"), std::string::npos)
          << e.what();
    }
    EXPECT_EQ(h.version(), 0U);
    EXPECT_EQ(h2v_rows(h), before);
    h.check_integrity(res());
  }
}

// ---- MOSP_ESCHER's skip rules, one test each (ADR 0041 section 8; ADR 0085) -----------------

class SlackEdits : public SlackBackends {
 protected:
  // {0,1,2} (id 0), {2,3} (id 1), {4} (id 2), {5,6} (id 3) over 8 vertices.
  hgraph make() const {
    return hgraph::from_hyperedges(res(), list_of(8, rows{{0, 1, 2}, {2, 3}, {4}, {5, 6}}).view(),
                                   slack_props());
  }
  // Applies b and checks the counts: applied removals and additions and skipped edits.
  void expect_counts(hgraph& h, const batch_t& b, std::int64_t removed, std::int64_t added,
                     std::int64_t skipped) const {
    const auto s = apply_checked(res(), h, b.view());
    EXPECT_EQ(s.deleted_incidences, removed);
    EXPECT_EQ(s.inserted_incidences, added);
    EXPECT_EQ(s.skipped_ops - s.ignored_deletions - s.dropped_members, skipped);
    // C9: applied + skipped = requested.
    EXPECT_EQ(s.deleted_incidences + s.inserted_incidences +
                  (s.skipped_ops - s.ignored_deletions - s.dropped_members),
              static_cast<std::int64_t>(b.delete_incidence_hyperedge().size() +
                                        b.insert_incidence_hyperedge().size()));
  }
};

TEST_P(SlackEdits, AnEditOfADeadOrOutOfRangeIdIsSkipped) {
  hgraph h = make();
  batch_t first;
  first.delete_hyperedge(2);
  (void)apply_checked(res(), h, first.view());
  batch_t b;
  b.insert_incidence(2, 0);   // dead
  b.delete_incidence(2, 4);   // dead
  b.insert_incidence(-1, 0);  // negative
  b.insert_incidence(4, 0);   // at the bound
  b.delete_incidence(99, 0);  // far above
  expect_counts(h, b, 0, 0, 5);
  EXPECT_EQ(h2v_rows(h)[0], (std::vector<std::int32_t>{0, 1, 2}));
}

TEST_P(SlackEdits, AnEditOfAnIdTheBatchDeletesOrInsertsIsSkipped) {
  hgraph h = make();
  batch_t b;
  b.delete_hyperedge(1);
  b.insert_incidence(1, 7);    // deleted earlier in the batch (phase 1)
  b.delete_incidence(1, 2);    // the same
  b.insert_hyperedge({6, 7});  // gets id 4 under insert_first (no deleted slot before the batch)
  b.insert_incidence(4, 0);    // the id the batch inserts: out of range on G_t
  expect_counts(h, b, 0, 0, 3);
  const rows now = h2v_rows(h);
  EXPECT_EQ(now[4], (std::vector<std::int32_t>{6, 7}));
}

TEST_P(SlackEdits, AnOutOfRangeVertexIsSkippedAnOutOfRangeMemberIsRejected) {
  hgraph h = make();
  batch_t b;
  b.insert_incidence(0, 8);  // out of range: skipped
  b.delete_incidence(0, -1);
  b.insert_incidence(0, 7);  // applied
  expect_counts(h, b, 0, 1, 2);
  EXPECT_EQ(h2v_rows(h)[0], (std::vector<std::int32_t>{0, 1, 2, 7}));
  // An inserted hyperedge with a member out of range has no G_t state to be checked against: the
  // batch is invalid (invalid_argument_error, strong guarantee).
  const std::uint64_t version = h.version();
  batch_t bad;
  bad.insert_incidence(0, 3);
  bad.insert_hyperedge({8});
  EXPECT_THROW((void)h.apply(res(), bad.view()), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), version);
  EXPECT_EQ(h2v_rows(h)[0], (std::vector<std::int32_t>{0, 1, 2, 7}));
  h.check_integrity(res());
}

TEST_P(SlackEdits, RemovingANonMemberIsSkipped) {
  hgraph h = make();
  batch_t b;
  b.delete_incidence(0, 3);  // not a member
  b.delete_incidence(0, 1);  // applied
  b.delete_incidence(0, 1);  // not a member any more
  expect_counts(h, b, 1, 0, 2);
  EXPECT_EQ(h2v_rows(h)[0], (std::vector<std::int32_t>{0, 2}));
}

TEST_P(SlackEdits, ARemovalThatWouldEmptyItsHyperedgeIsSkipped) {
  hgraph h = make();
  batch_t b;
  b.delete_incidence(1, 2);  // {2,3} -> {3}
  b.delete_incidence(1, 3);  // the second of two removals on a 2-vertex hyperedge: skipped
  b.delete_incidence(2, 4);  // the only member of {4}: skipped
  expect_counts(h, b, 1, 0, 2);
  const rows now = h2v_rows(h);
  EXPECT_EQ(now[1], (std::vector<std::int32_t>{3}));
  EXPECT_EQ(now[2], (std::vector<std::int32_t>{4}));
  EXPECT_EQ(h.num_hyperedges(), 4);
}

TEST_P(SlackEdits, AddingAnExistingMemberIsSkipped) {
  hgraph h = make();
  batch_t b;
  b.insert_incidence(3, 5);  // a member
  b.insert_incidence(3, 7);  // applied
  b.insert_incidence(3, 7);  // a repeat: a member by then
  expect_counts(h, b, 0, 1, 2);
  EXPECT_EQ(h2v_rows(h)[3], (std::vector<std::int32_t>{5, 6, 7}));
}

// B's phase order: hyperedge deletions, then incidence removals, then additions, then hyperedge
// insertions (whatever the order in which the batch lists them).
TEST_P(SlackEdits, ThePhaseOrderIsDeletionsRemovalsAdditionsInsertions) {
  hgraph h = make();
  batch_t b;
  b.insert_incidence(0, 1);  // listed first, but runs after the removal: 1 is no member then
  b.delete_incidence(0, 1);
  b.insert_incidence(3, 4);  // an addition before the removal of the same pair: the removal
  b.delete_incidence(3, 4);  // runs first (not a member: skipped), the addition applies
  expect_counts(h, b, 1, 2, 1);
  const rows now = h2v_rows(h);
  EXPECT_EQ(now[0], (std::vector<std::int32_t>{0, 1, 2}));  // removed, then added back
  EXPECT_EQ(now[3], (std::vector<std::int32_t>{4, 5, 6}));
  // The net change of {0,1,2} is nothing: the pair delta has none of it.
  EXPECT_EQ(hypergraph_access::applied(h).edited_ids, (std::vector<std::int64_t>{0, 3}));
  const auto& a = hypergraph_access::pairs(h);
  EXPECT_EQ(zip(a.inserted_a, a.inserted_b), (pair_list{{2, 3}}));  // {4} and {4,5,6}
  EXPECT_TRUE(a.deleted_a.empty());
}

// The record of a batch with every kind of operation: the (vertex, id, sign) changes, the new,
// recreated and dead ids, the counts.
TEST_P(SlackEdits, TheAppliedRecord) {
  for (const id_reuse reuse : {id_reuse::insert_first, id_reuse::erase_first}) {
    SCOPED_TRACE(std::string(dyng::to_string(reuse)));
    hgraph h = hgraph::from_hyperedges(
        res(), list_of(8, rows{{0, 1, 2}, {2, 3}, {4}, {5, 6}}).view(), slack_props(reuse));
    batch_t b;
    b.delete_hyperedge(2);
    b.delete_incidence(0, 0);
    b.insert_incidence(1, 7);
    b.insert_hyperedge({4, 7});
    const auto s = apply_checked(res(), h, b.view());
    const auto& a = hypergraph_access::applied(h);
    ASSERT_EQ(s.inserted_ids.size(), 1U);
    const std::int64_t id = s.inserted_ids[0];
    // insert_first: no deleted slot before the batch, the row goes after the last key (id 4);
    // erase_first: the slot of id 2 (a single, capacity 3) fits {4,7} (id 2 recreated).
    EXPECT_EQ(id, reuse == id_reuse::insert_first ? 4 : 2);
    EXPECT_EQ(a.new_ids, reuse == id_reuse::insert_first ? std::vector<std::int64_t>{4}
                                                         : std::vector<std::int64_t>{});
    EXPECT_EQ(a.recreated_ids, reuse == id_reuse::insert_first ? std::vector<std::int64_t>{}
                                                               : std::vector<std::int64_t>{2});
    EXPECT_EQ(a.dead_ids, reuse == id_reuse::insert_first ? std::vector<std::int64_t>{2}
                                                          : std::vector<std::int64_t>{});
    EXPECT_EQ(a.deleted_incidences, 1);
    EXPECT_EQ(a.inserted_incidences, 1);
    EXPECT_EQ(a.edited_ids, (std::vector<std::int64_t>{0, 1}));
    // Every applied change, whole hyperedges included: (4, 2, -), (0, 0, -), (7, 1, +), (4, id, +),
    // (7, id, +).
    std::vector<std::tuple<std::int32_t, std::int64_t, int>> got;
    for (std::size_t i = 0; i < a.incidence_delta.vertices.size(); ++i)
      got.emplace_back(a.incidence_delta.vertices[i], a.incidence_delta.ids[i],
                       a.incidence_delta.signs[i]);
    std::sort(got.begin(), got.end());
    std::vector<std::tuple<std::int32_t, std::int64_t, int>> want = {
        {4, 2, -1}, {0, 0, -1}, {7, 1, 1}, {4, id, 1}, {7, id, 1}};
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want);
    EXPECT_EQ(s.skipped_ops, 0);
  }
}

// ---- The order-free pair delta ------------------------------------------------------------

// Permuting the operations of each phase (distinct incidences, no removal that could empty a
// row, distinct deletions) gives the same rows, ids and pairs; one thread or eight give the same.
TEST_P(SlackEdits, PermutingAPhasesOperationsGivesTheSamePairs) {
  std::mt19937_64 rng(3);
  rows initial;
  for (int i = 0; i < 60; ++i) {
    std::vector<std::int32_t> r;
    for (int k = 0; k < 2 + static_cast<int>(rng() % 5); ++k)
      r.push_back(static_cast<std::int32_t>(rng() % 40));
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    initial.push_back(r);
  }
  const auto list = list_of(40, initial);
  // One batch: 6 deletions, removals of the second member of 10 rows with 3 or more members,
  // additions of a non-member to 10 other rows.
  std::vector<std::int64_t> deletions;
  std::vector<std::pair<std::int64_t, std::int32_t>> removals, additions;
  for (std::int64_t id = 0; id < 60 && deletions.size() < 6; id += 7) deletions.push_back(id);
  for (std::int64_t id = 1; id < 60 && removals.size() < 10; id += 3) {
    if (std::find(deletions.begin(), deletions.end(), id) != deletions.end()) continue;
    if (initial[static_cast<std::size_t>(id)].size() >= 3)
      removals.emplace_back(id, initial[static_cast<std::size_t>(id)][1]);
  }
  for (std::int64_t id = 2; id < 60 && additions.size() < 10; id += 5) {
    if (std::find(deletions.begin(), deletions.end(), id) != deletions.end()) continue;
    const auto& r = initial[static_cast<std::size_t>(id)];
    for (std::int32_t v = 0; v < 40; ++v)
      if (!std::binary_search(r.begin(), r.end(), v)) {
        additions.emplace_back(id, v);
        break;
      }
  }
  pair_list first_deleted, first_inserted;
  rows first_rows;
  for (int trial = 0; trial < 6; ++trial) {
    std::shuffle(deletions.begin(), deletions.end(), rng);
    std::shuffle(removals.begin(), removals.end(), rng);
    std::shuffle(additions.begin(), additions.end(), rng);
    const resources r = GetParam() == backend::openmp
                            ? dyng::test::hyper::resources_of(backend::openmp, trial % 2 ? 8 : 1)
                            : res();
    hgraph h = hgraph::from_hyperedges(r, list.view(), slack_props());
    batch_t b;
    for (const auto id : deletions) b.delete_hyperedge(id);
    for (const auto& [id, v] : removals) b.delete_incidence(id, v);
    for (const auto& [id, v] : additions) b.insert_incidence(id, v);
    b.insert_hyperedge({1, 2, 3});
    (void)apply_checked(r, h, b.view());
    const auto& a = hypergraph_access::pairs(h);
    if (trial == 0) {
      first_deleted = zip(a.deleted_a, a.deleted_b);
      first_inserted = zip(a.inserted_a, a.inserted_b);
      first_rows = h2v_rows(h);
      EXPECT_FALSE(first_deleted.empty());
      EXPECT_FALSE(first_inserted.empty());
    } else {
      EXPECT_EQ(zip(a.deleted_a, a.deleted_b), first_deleted) << trial;
      EXPECT_EQ(zip(a.inserted_a, a.inserted_b), first_inserted) << trial;
      EXPECT_EQ(h2v_rows(h), first_rows) << trial;
    }
  }
}

// Random batches of every kind under both orders and the presets: after every batch the pair
// delta equals the oracle's, h2h() the rebuilt line graph, the store its integrity checks; the
// sequential and openmp backends give the same ids, rows and pairs.
TEST_P(SlackEdits, RandomBatchesKeepTheLineGraphExact) {
  for (const id_reuse reuse : {id_reuse::insert_first, id_reuse::erase_first}) {
    for (const auto& capacity :
         {capacity_policy::mosp_escher(512), capacity_policy::escher_gpu(), capacity_policy{}}) {
      SCOPED_TRACE(std::string(dyng::to_string(reuse)));
      std::mt19937_64 rng(11);
      const std::int32_t n = 50;
      rows initial;
      std::vector<std::int32_t> w0;
      for (int i = 0; i < 80; ++i) {
        std::vector<std::int32_t> r;
        for (int k = 0; k < 1 + static_cast<int>(rng() % 6); ++k)
          r.push_back(static_cast<std::int32_t>(rng() % n));
        initial.push_back(r);
        w0.push_back(1 + static_cast<std::int32_t>(rng() % 100));
      }
      whgraph h =
          whgraph::from_hyperedges(res(), list_of<std::int32_t>(n, initial, w0).view(),
                                   slack_props(reuse, line_graph_policy::slack_csr, capacity));
      whgraph twin = whgraph::from_hyperedges(
          resources::sequential(), list_of<std::int32_t>(n, initial, w0).view(),
          slack_props(reuse, line_graph_policy::slack_csr, capacity));
      for (int batch = 0; batch < 40; ++batch) {
        wbatch_t b;
        const std::int64_t bound = h.hyperedge_id_bound();
        for (int k = 0; k < static_cast<int>(rng() % 6); ++k)
          b.delete_hyperedge(
              static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound + 2)));
        for (int k = 0; k < static_cast<int>(rng() % 12); ++k)
          b.delete_incidence(
              static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound + 1)),
              static_cast<std::int32_t>(rng() % (n + 1)));
        for (int k = 0; k < static_cast<int>(rng() % 12); ++k)
          b.insert_incidence(
              static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound + 1)),
              static_cast<std::int32_t>(rng() % (n + 1)));
        for (int k = 0; k < static_cast<int>(rng() % 5); ++k) {
          std::vector<std::int32_t> r;
          for (int m = 0; m < 1 + static_cast<int>(rng() % 7); ++m)
            r.push_back(static_cast<std::int32_t>(rng() % n));
          b.insert_hyperedge(dyng::host_view(r), 1 + static_cast<std::int32_t>(rng() % 100));
        }
        const auto s = apply_checked(res(), h, b.view());
        const auto t = twin.apply(resources::sequential(), b.view());
        ASSERT_EQ(s.inserted_ids, t.inserted_ids) << "batch " << batch;
        ASSERT_EQ(s.skipped_ops, t.skipped_ops) << "batch " << batch;
        ASSERT_EQ(h2v_rows(h), h2v_rows(twin)) << "batch " << batch;
        const auto& a = hypergraph_access::pairs(h);
        const auto& at = hypergraph_access::pairs(twin);
        ASSERT_EQ(a.inserted_a, at.inserted_a) << "batch " << batch;
        ASSERT_EQ(a.deleted_b, at.deleted_b) << "batch " << batch;
        // C9.
        EXPECT_EQ(s.deleted_incidences + s.inserted_incidences +
                      (s.skipped_ops - s.ignored_deletions - s.dropped_members),
                  static_cast<std::int64_t>(b.delete_incidence_hyperedge().size() +
                                            b.insert_incidence_hyperedge().size()));
        EXPECT_EQ(s.deleted_hyperedges + s.ignored_deletions,
                  static_cast<std::int64_t>(b.delete_hyperedges().size()));
      }
    }
  }
}

// ---- The slack CSR's paths: the tail rebuild and node-capacity growth ------------------------

TEST_P(SlackEdits, ForcedTailRebuildAndNodeGrowth) {
  // Headroom 1.0: the entries fit the initial layout plus 4096 (MOSP_ESCHER's test_h2h_delta
  // pattern on every third configuration); a node capacity of the initial bound: the first new
  // id grows the arrays.
  dyng::detail::hyper::slack::knobs knobs;
  knobs.entry_headroom = 1.0;
  knobs.node_capacity = 30;
  rows initial;
  for (int i = 0; i < 30; ++i) initial.push_back({i % 10, 10 + i % 7, 20});  // every row shares 20
  hgraph h = hypergraph_access::build(
      res(), list_of(40, initial).view(),
      slack_props(id_reuse::insert_first, line_graph_policy::slack_csr, capacity_policy{}), knobs);
  const auto* slack = hypergraph_access::slack(h);
  ASSERT_NE(slack, nullptr);
  EXPECT_EQ(slack->rows().node_capacity, 30);
  for (int batch = 0; batch < 12; ++batch) {
    batch_t b;
    for (int k = 0; k < 20; ++k) b.insert_hyperedge({20, 21 + (batch * 20 + k) % 19});
    (void)apply_checked(res(), h, b.view());
  }
  const auto c = slack->counters();
  EXPECT_GT(c.csr_rebuilds, 0);
  // Rows moved to the tail (the host store counts them; the CUDA store counts the tail's slots).
  EXPECT_GT(cuda() ? c.tail_entries : c.relocations, 0);
  EXPECT_GT(c.node_growths, 0);
  EXPECT_GT(slack->rows().node_capacity, 30);
  // The headroom knob refuses less than 1 (and NaN), as MOSP_ESCHER's buildCsr.
  knobs.entry_headroom = 0.5;
  EXPECT_THROW(
      (void)hypergraph_access::build(res(), list_of(40, initial).view(), slack_props(), knobs),
      dyng::invalid_argument_error);
}

// More candidates than the limit (INT_MAX, MOSP_ESCHER's; planted lower here) is a capacity_error
// in the store phase: the hypergraph is poisoned (basic guarantee).
TEST_P(SlackEdits, MoreCandidatesThanTheLimitIsACapacityError) {
  dyng::detail::hyper::slack::knobs knobs;
  knobs.max_candidates = 10;
  rows initial;
  for (int i = 0; i < 20; ++i) initial.push_back({0, i + 1});
  hgraph h = hypergraph_access::build(res(), list_of(30, initial).view(), slack_props(), knobs);
  batch_t b;
  b.insert_hyperedge({0, 25});  // 20 neighbours through vertex 0: more than 10 candidates
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);  // poisoned
  // Within the limit nothing happens.
  knobs.max_candidates = 1000;
  hgraph g = hypergraph_access::build(res(), list_of(30, initial).view(), slack_props(), knobs);
  (void)apply_checked(res(), g, b.view());
}

// ---- Views, policies, clones -----------------------------------------------------------------

TEST_P(SlackEdits, AutomaticBuildsTheSlackCsrOnFirstReadAndNoneKeepsNone) {
  const auto list = list_of(8, rows{{0, 1, 2}, {2, 3}, {4}, {5, 6}});
  hgraph a = hgraph::from_hyperedges(
      res(), list.view(), slack_props(id_reuse::insert_first, line_graph_policy::automatic));
  EXPECT_FALSE(hypergraph_access::store(a).line_graph_enabled());
  batch_t b;
  b.insert_incidence(2, 3);
  (void)a.apply(res(), b.view());
  EXPECT_FALSE(hypergraph_access::applied(a).has_line_graph_delta);
  EXPECT_EQ(dyng::test::hyper::check_line_graph(a), "");  // the first h2h() builds it
  EXPECT_TRUE(hypergraph_access::store(a).line_graph_enabled());
  batch_t c;
  c.delete_incidence(2, 3);
  (void)apply_checked(res(), a, c.view());
  EXPECT_EQ(hypergraph_access::store(a).builds().line_graph_builds, 1);  // maintained since
  // none: no line graph, the edits still apply.
  hgraph n = hgraph::from_hyperedges(res(), list.view(),
                                     slack_props(id_reuse::insert_first, line_graph_policy::none));
  (void)n.apply(res(), b.view());
  EXPECT_EQ(h2v_rows(n)[2], (std::vector<std::int32_t>{3, 4}));
  EXPECT_THROW((void)n.h2h(), dyng::invalid_argument_error);
  n.check_integrity(res());
}

TEST_P(SlackEdits, TheH2hViewIsSortedOnRequestAndCounted) {
  hgraph h = make();
  batch_t b;
  b.insert_incidence(2, 0);
  b.insert_incidence(3, 2);
  (void)h.apply(res(), b.view());
  const auto before = h.store_stats().view_bytes;
  const auto v = h.h2h();
  EXPECT_TRUE(v.overlaps.empty());  // HD9: overlaps stay empty under slack_csr
  EXPECT_GT(h.store_stats().view_bytes, before);
  EXPECT_EQ(dyng::test::hyper::h2h_rows(h)[0], (std::vector<std::int64_t>{1, 2, 3}));
  EXPECT_EQ(hypergraph_access::store(h).builds().h2h, 1);
  (void)h.h2h();
  EXPECT_EQ(hypergraph_access::store(h).builds().h2h, 1);  // once per version
}

TEST_P(SlackEdits, ClonesKeepIdsRowsTheAllocatorAndTheLineGraph) {
  whgraph h = whgraph::from_hyperedges(
      res(), list_of<std::int32_t>(8, rows{{0, 1, 2}, {2, 3}, {4}, {5, 6}}, {3, 4, 5, 6}).view(),
      slack_props(id_reuse::insert_first, line_graph_policy::slack_csr,
                  capacity_policy::mosp_escher(16)));
  wbatch_t b;
  b.delete_hyperedge(2);
  b.insert_incidence(0, 7);
  b.insert_incidence(0, 6);
  b.insert_incidence(0, 5);  // overflows the row's tail: an overflow segment
  (void)apply_checked(res(), h, b.view());
  for (const backend to : {backend::sequential, backend::openmp}) {
    if (!dyng::backend_available(to)) continue;
    const resources r = dyng::test::hyper::resources_of(to);
    whgraph c = h.to_backend(r);
    EXPECT_EQ(h2v_rows(c), h2v_rows(h));
    EXPECT_EQ(dyng::test::hyper::h2h_rows(c), dyng::test::hyper::h2h_rows(h));
    EXPECT_EQ(dyng::test::host_copy(c.weights()), dyng::test::host_copy(h.weights()));
    c.check_integrity(r);
    // The same later batch gives the same ids and rows (the allocator state, tails included).
    wbatch_t next;
    next.delete_incidence(0, 7);
    next.insert_hyperedge({1, 4}, 9);
    next.insert_hyperedge({0, 1, 2, 3, 4, 5}, 8);
    whgraph d = h.clone(res());
    const auto s1 = apply_checked(r, c, next.view());
    const auto s2 = apply_checked(res(), d, next.view());
    EXPECT_EQ(s1.inserted_ids, s2.inserted_ids);
    EXPECT_EQ(h2v_rows(c), h2v_rows(d));
  }
}

TEST_P(SlackEdits, TheStrongGuaranteeOfMalformedEdits) {
  hgraph h = make();
  const rows before = h2v_rows(h);
  std::vector<std::int64_t> ids{0, 1};
  std::vector<std::int32_t> vertices{5};
  dyng::hyperedge_batch_view<std::int32_t, unweighted> v;
  v.insert_incidence_hyperedge = dyng::host_view(ids);
  v.insert_incidence_vertex = dyng::host_view(vertices);  // one vertex for two additions
  EXPECT_THROW((void)h.apply(res(), v), dyng::invalid_argument_error);
  dyng::hyperedge_batch_view<std::int32_t, unweighted> w;
  w.delete_incidence_hyperedge = dyng::host_view(ids);
  w.delete_incidence_vertex = dyng::host_view(vertices);
  EXPECT_THROW((void)h.apply(res(), w), dyng::invalid_argument_error);
  EXPECT_EQ(h.version(), 0U);
  EXPECT_EQ(h2v_rows(h), before);
  h.check_integrity(res());
}

TEST_P(SlackEdits, TheStagesOfTheCommit) {
  hgraph h = make();
  dyng::profiler prof;
  resources r = res();
  r.attach_profiler(&prof);
  batch_t b;
  b.delete_hyperedge(0);
  b.insert_incidence(1, 7);
  b.insert_hyperedge({0, 7});
  (void)h.apply(r, b.view());
  std::vector<std::string> names;
  for (const auto& st : prof.stages()) names.push_back(st.name);
  for (const char* want :
       {"hypergraph.apply", "hypergraph.apply.normalize", "hypergraph.apply.ids",
        "hypergraph.apply.incidence", "hypergraph.apply.translate", "hypergraph.apply.line_graph"})
    EXPECT_NE(std::find(names.begin(), names.end(), want), names.end()) << want;
}

// ---- The host team (M11a step 3, ADR 0089 section 8) ---------------------------------------

// Batches large enough for the parallel host regions (host_team.hpp: Step 0's groups, the H2V
// rows, the v2h vertices, at least 4096 each) give the state, the ids, the skips, the record and
// the pairs of the sequential backend, on 1 and 16 host threads (cuda: the store's host model).
TEST_P(SlackEdits, LargeBatchesAreTheSameOnEveryHostTeam) {
  std::mt19937_64 rng(29);
  const std::int32_t n = 6000;
  rows initial;
  std::vector<std::int32_t> w0;
  for (int i = 0; i < 9000; ++i) {
    std::vector<std::int32_t> r;
    for (int k = 0; k < 1 + static_cast<int>(rng() % 5); ++k)
      r.push_back(static_cast<std::int32_t>(rng() % n));
    initial.push_back(r);
    w0.push_back(1 + static_cast<std::int32_t>(rng() % 100));
  }
  const auto list = list_of<std::int32_t>(n, initial, w0);
  for (const id_reuse reuse : {id_reuse::insert_first, id_reuse::erase_first}) {
    SCOPED_TRACE(std::string(dyng::to_string(reuse)));
    const auto props =
        slack_props(reuse, line_graph_policy::slack_csr, capacity_policy::mosp_escher(1 << 20));
    std::vector<resources> teams;
    if (GetParam() == backend::openmp) {
      teams = {resources::openmp(1), resources::openmp(16)};
    } else if (GetParam() == backend::cuda) {
      teams = {resources::cuda(0, {}, 1), resources::cuda(0, {}, 16)};
    } else {
      teams = {resources::sequential()};
    }
    std::vector<whgraph> hs;
    for (const resources& r : teams) hs.push_back(whgraph::from_hyperedges(r, list.view(), props));
    whgraph twin = whgraph::from_hyperedges(resources::sequential(), list.view(), props);
    for (int batch = 0; batch < 2; ++batch) {
      wbatch_t b;
      const std::int64_t bound = twin.hyperedge_id_bound();
      for (int k = 0; k < 4500; ++k)
        b.delete_hyperedge(static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound)));
      for (int k = 0; k < 5000; ++k)
        b.delete_incidence(static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound + 1)),
                           static_cast<std::int32_t>(rng() % (n + 1)));
      for (int k = 0; k < 5000; ++k)
        b.insert_incidence(static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(bound + 1)),
                           static_cast<std::int32_t>(rng() % (n + 1)));
      for (int k = 0; k < 4500; ++k) {
        std::vector<std::int32_t> r;
        for (int m = 0; m < 1 + static_cast<int>(rng() % 6); ++m)
          r.push_back(static_cast<std::int32_t>(rng() % n));
        b.insert_hyperedge(dyng::host_view(r), 1 + static_cast<std::int32_t>(rng() % 100));
      }
      const auto t = twin.apply(resources::sequential(), b.view());
      const auto& at = hypergraph_access::applied(twin);
      const auto& pt = hypergraph_access::pairs(twin);
      for (std::size_t i = 0; i < hs.size(); ++i) {
        SCOPED_TRACE("team " + std::to_string(i) + ", batch " + std::to_string(batch));
        const auto s = hs[i].apply(teams[i], b.view());
        ASSERT_EQ(s.inserted_ids, t.inserted_ids);
        ASSERT_EQ(s.skipped_ops, t.skipped_ops);
        ASSERT_EQ(s.deleted_incidences, t.deleted_incidences);
        ASSERT_EQ(s.inserted_incidences, t.inserted_incidences);
        ASSERT_EQ(h2v_rows(hs[i]), h2v_rows(twin));
        ASSERT_EQ(v2h_rows(hs[i]), v2h_rows(twin));
        const auto& a = hypergraph_access::applied(hs[i]);
        EXPECT_EQ(a.edited_ids, at.edited_ids);
        EXPECT_EQ(a.touched_hyperedges, at.touched_hyperedges);
        EXPECT_EQ(a.touched_pre_rows.members, at.touched_pre_rows.members);
        EXPECT_EQ(a.touched_post_rows.offsets, at.touched_post_rows.offsets);
        EXPECT_EQ(a.touched_vertices, at.touched_vertices);
        EXPECT_EQ(a.touched_vertex_lengths, at.touched_vertex_lengths);
        EXPECT_EQ(a.dead_ids, at.dead_ids);
        EXPECT_EQ(a.incidence_delta.ids, at.incidence_delta.ids);
        const auto& p = hypergraph_access::pairs(hs[i]);
        EXPECT_EQ(p.deleted_a, pt.deleted_a);
        EXPECT_EQ(p.deleted_b, pt.deleted_b);
        EXPECT_EQ(p.inserted_a, pt.inserted_a);
        EXPECT_EQ(p.inserted_b, pt.inserted_b);
        hs[i].check_integrity(teams[i]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, SlackBackends, ::testing::ValuesIn(dyng::test::suite_backends()),
                         backend_name);
INSTANTIATE_TEST_SUITE_P(Backends, SlackEdits, ::testing::ValuesIn(dyng::test::suite_backends()),
                         backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SlackBackends);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SlackEdits);

}  // namespace
