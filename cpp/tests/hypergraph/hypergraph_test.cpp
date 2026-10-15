// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_test.cpp
 * @brief The hypergraph container (M8b, ADR 0041): construction, apply semantics under both
 *        id_reuse orders, skipped ops, the strong guarantee and poisoning, placement, clone and
 *        to_backend, weights, the views and the counters. Compiled into the host test executable
 *        (sequential, openmp) and, with DYNG_TEST_CUDA=1, into the CUDA one (cuda).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::hypergraph;
using dyng::hypergraph_properties;
using dyng::id_reuse;
using dyng::resources;
using dyng::unweighted;
using dyng::test::hyper::h2v_rows;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using dyng::test::hyper::v2h_rows;
using hgraph = hypergraph<std::int32_t, unweighted>;
using whgraph = hypergraph<std::int32_t, std::int32_t>;

class HypergraphBackends : public ::testing::TestWithParam<backend> {
 protected:
  resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

hypergraph_properties props_of(id_reuse reuse,
                               dyng::capacity_policy c = dyng::capacity_policy::escher_gpu()) {
  hypergraph_properties p;
  p.line_graph = dyng::line_graph_policy::none;
  p.reuse = reuse;
  p.capacity = c;
  return p;
}

TEST_P(HypergraphBackends, FromHyperedgesGivesIdsInListOrder) {
  const auto list = list_of(6, rows{{2, 0, 1}, {4}, {5, 3, 3}});
  const hgraph h = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  EXPECT_EQ(h.num_vertices(), 6);
  EXPECT_EQ(h.num_hyperedges(), 3);
  EXPECT_EQ(h.hyperedge_id_bound(), 3);
  EXPECT_EQ(h.version(), 0U);
  EXPECT_EQ(h.space(),
            GetParam() == backend::cuda ? dyng::memory_space::device : dyng::memory_space::host);
  EXPECT_EQ(h2v_rows(h), (rows{{0, 1, 2}, {4}, {3, 5}}));
  EXPECT_EQ(v2h_rows(h), (std::vector<std::vector<std::int64_t>>{{0}, {0}, {0}, {2}, {1}, {2}}));
  EXPECT_EQ(h.h2v().version, 0U);
  h.check_integrity(res());
  const auto back = h.to_hyperedge_list(res());
  EXPECT_EQ(back.offsets, (std::vector<std::int64_t>{0, 3, 4, 6}));
  EXPECT_EQ(back.members, (std::vector<std::int32_t>{0, 1, 2, 4, 3, 5}));
}

TEST_P(HypergraphBackends, EmptyListAndVerticesWithoutHyperedges) {
  dyng::hyperedge_list<std::int32_t> list;
  list.num_vertices = 4;
  hgraph h = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  EXPECT_EQ(h.num_hyperedges(), 0);
  EXPECT_EQ(h.hyperedge_id_bound(), 0);
  EXPECT_EQ(v2h_rows(h).size(), 4U);
  h.check_integrity(res());
  // The first batch of an empty hypergraph numbers its rows from 0.
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  b.insert_hyperedge({1, 2});
  b.insert_hyperedge({3});
  const auto s = h.apply(res(), b.view());
  EXPECT_EQ(s.inserted_ids, (std::vector<std::int64_t>{0, 1}));
  h.check_integrity(res());
  // Zero vertices, zero hyperedges.
  dyng::hyperedge_list<std::int32_t> none;
  const hgraph z = hgraph::from_hyperedges(res(), none.view(), props_of(id_reuse::erase_first));
  EXPECT_EQ(z.num_vertices(), 0);
  z.check_integrity(res());
}

TEST_P(HypergraphBackends, ConstructionRejectsInvalidInput) {
  const auto p = props_of(id_reuse::erase_first);
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0}, {}}).view(), p),
               dyng::invalid_argument_error);
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0, 3}}).view(), p),
               dyng::invalid_argument_error);
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{-1}}).view(), p),
               dyng::invalid_argument_error);
  auto bad_offsets = list_of(3, rows{{0, 1}});
  bad_offsets.offsets.back() = 1;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), bad_offsets.view(), p),
               dyng::invalid_argument_error);
  EXPECT_THROW(
      (void)whgraph::from_hyperedges(res(), list_of<std::int32_t>(3, rows{{0}}, {0}).view(), p),
      dyng::invalid_argument_error);
  auto slack = p;
  slack.storage = dyng::hypergraph_storage::slack_csr;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), slack),
               dyng::not_supported_error);
  auto merged = p;
  merged.line_graph = dyng::line_graph_policy::sorted_merge;
  merged.store_v2h = false;  // the merge reads v2h
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), merged),
               dyng::invalid_argument_error);
  auto slack_h2h = p;
  slack_h2h.line_graph = dyng::line_graph_policy::slack_csr;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), slack_h2h),
               dyng::not_supported_error);
  auto grow = p;
  grow.capacity.grow_factor = 2.0;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), grow),
               dyng::not_supported_error);
}

// ADR 0041, section 9: three hyperedges {0,1,2,3} (a slot of 8), {4} and {5,6} (slots of 4); a
// batch deletes id 1 and inserts {7,8}. erase_first reuses the slot of id 1; insert_first has no
// deleted slot when it inserts, so the row goes after the last live key.
TEST_P(HypergraphBackends, IdReuseOrdersGiveDifferentIds) {
  const auto list = list_of(9, rows{{0, 1, 2, 3}, {4}, {5, 6}});
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({7, 8});
  hgraph a = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  hgraph c = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::insert_first));
  const auto sa = a.apply(res(), b.view());
  const auto sc = c.apply(res(), b.view());
  EXPECT_EQ(sa.inserted_ids, (std::vector<std::int64_t>{1}));
  EXPECT_EQ(sc.inserted_ids, (std::vector<std::int64_t>{3}));
  EXPECT_EQ(h2v_rows(a), (rows{{0, 1, 2, 3}, {7, 8}, {5, 6}}));
  EXPECT_EQ(h2v_rows(c), (rows{{0, 1, 2, 3}, {}, {5, 6}, {7, 8}}));
  EXPECT_EQ(sa.id_bound_after, 3);
  EXPECT_EQ(sc.id_bound_after, 4);
  a.check_integrity(res());
  c.check_integrity(res());
}

// The surplus rule: a deleted slot that is not reused is dropped, the bound can go down, and an
// id that was dead before the batch can be given out again.
TEST_P(HypergraphBackends, SurplusRuleDropsDeadSlotsAndReissuesIds) {
  const auto list = list_of(40, rows{{0}, {1}, {2}, {3}, {4}});
  hgraph h = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  dyng::hyperedge_batch<std::int32_t, unweighted> d;
  d.delete_hyperedge(3);
  d.delete_hyperedge(4);
  (void)h.apply(res(), d.view());
  EXPECT_EQ(h.hyperedge_id_bound(), 5);  // dead slots stay in the tree
  EXPECT_EQ(h.num_hyperedges(), 3);
  // One item of 10 members fits no slot of 4: it is surplus, the dead slots 3 and 4 are dropped
  // and the row gets the key after the last live one: id 3 again; the bound goes down to 4.
  dyng::hyperedge_batch<std::int32_t, unweighted> i;
  i.insert_hyperedge({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
  const auto s = h.apply(res(), i.view());
  EXPECT_EQ(s.inserted_ids, (std::vector<std::int64_t>{3}));
  EXPECT_EQ(h.hyperedge_id_bound(), 4);
  h.check_integrity(res());
}

TEST_P(HypergraphBackends, SkippedOpsCountDeadOutOfRangeRepeatedAndDuplicates) {
  const auto list = list_of(10, rows{{0, 1}, {2, 3}, {4, 5}});
  for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
    hgraph h = hgraph::from_hyperedges(res(), list.view(), props_of(reuse));
    dyng::hyperedge_batch<std::int32_t, unweighted> first;
    first.delete_hyperedge(2);
    (void)h.apply(res(), first.view());
    dyng::hyperedge_batch<std::int32_t, unweighted> b;
    b.delete_hyperedge(2);             // dead
    b.delete_hyperedge(-1);            // out of range
    b.delete_hyperedge(99);            // out of range
    b.delete_hyperedge(0);             // live
    b.delete_hyperedge(0);             // repeated
    b.insert_hyperedge({7, 7, 8, 7});  // two dropped members
    const auto s = h.apply(res(), b.view());
    EXPECT_EQ(s.deleted_hyperedges, 1);
    EXPECT_EQ(s.ignored_deletions, 4);
    EXPECT_EQ(s.dropped_members, 2);
    EXPECT_EQ(s.skipped_ops, 6);
    EXPECT_EQ(s.inserted_hyperedges, 1);
    EXPECT_EQ(s.num_hyperedges_after, 2);
    const rows got = h2v_rows(h);
    EXPECT_EQ(got[static_cast<std::size_t>(s.inserted_ids[0])], (std::vector<std::int32_t>{7, 8}));
    h.check_integrity(res());
  }
}

// Review fix (M8b): Step 0 removed repeated deletion ids with a sorted-vector insert, quadratic
// in the batch (one repeated id in a 400K batch cost about 20 s). The rule is unchanged: the first
// deletion of an id counts, the later ones are skipped, the batch order is kept.
TEST_P(HypergraphBackends, RepeatedDeletionsInALargeShuffledBatchKeepTheFirstOccurrence) {
  constexpr std::int32_t n = 200000;
  dyng::hyperedge_list<std::int32_t, unweighted> list;
  list.num_vertices = n;
  for (std::int32_t v = 0; v < n; ++v) list.add_hyperedge({v});
  hgraph h = hgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  std::vector<std::int64_t> order(static_cast<std::size_t>(n));
  for (std::int32_t i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::uint64_t x = 12345;  // a fixed LCG shuffle (std::shuffle's draws are library-specific)
  for (std::size_t i = order.size() - 1; i > 0; --i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    std::swap(order[i], order[static_cast<std::size_t>((x >> 33) % (i + 1))]);
  }
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  for (const std::int64_t id : order) b.delete_hyperedge(id);
  b.delete_hyperedge(order[5]);  // repeated
  b.delete_hyperedge(order[0]);  // repeated
  b.delete_hyperedge(order[5]);  // repeated twice
  b.delete_hyperedge(n + 7);     // out of range
  const auto s = h.apply(res(), b.view());
  EXPECT_EQ(s.deleted_hyperedges, n);
  EXPECT_EQ(s.ignored_deletions, 4);
  EXPECT_EQ(s.skipped_ops, 4);
  EXPECT_EQ(s.num_hyperedges_after, 0);
  EXPECT_EQ(dyng::detail::hypergraph_access::applied(h).deleted_ids, order);
  h.check_integrity(res());
}

TEST(HypergraphTouchedMarks, GiveSortedUniqueValuesOnBothPathsAndClearTheMarks) {
  dyng::detail::hyper::touched_marks<std::int32_t> marks;
  std::vector<std::int32_t> out;
  marks.begin(1000);  // few values: sorted
  for (const std::int32_t v : {7, 3, 7, 999, 0, 3}) marks.mark(v);
  marks.take_sorted(out);
  EXPECT_EQ(out, (std::vector<std::int32_t>{0, 3, 7, 999}));
  marks.begin(10);  // most of the range: scanned
  for (const std::int32_t v : {9, 1, 4, 1, 0, 2, 5, 9}) marks.mark(v);
  marks.take_sorted(out);
  EXPECT_EQ(out, (std::vector<std::int32_t>{0, 1, 2, 4, 5, 9}));
  marks.begin(1000);  // the marks were cleared
  marks.mark(5);
  marks.take_sorted(out);
  EXPECT_EQ(out, (std::vector<std::int32_t>{5}));
}

TEST_P(HypergraphBackends, ValidationErrorsLeaveTheHypergraphUnchanged) {
  const auto list = list_of<std::int32_t>(6, rows{{0, 1}, {2, 3}}, {5, 7});
  whgraph h = whgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  const std::uint64_t state = dyng::detail::hypergraph_access::state_id(h);
  const auto before = h.to_hyperedge_list(res());
  const auto expect_unchanged = [&]() {
    EXPECT_EQ(h.version(), 0U);
    EXPECT_EQ(dyng::detail::hypergraph_access::state_id(h), state);
    const auto now = h.to_hyperedge_list(res());
    EXPECT_EQ(now.offsets, before.offsets);
    EXPECT_EQ(now.members, before.members);
    EXPECT_EQ(now.weights, before.weights);
    h.check_integrity(res());
  };
  {  // a vertex out of range (after a valid deletion: nothing is deleted either)
    dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
    b.delete_hyperedge(0);
    b.insert_hyperedge({1, 6}, 3);
    EXPECT_THROW((void)h.apply(res(), b.view()), dyng::invalid_argument_error);
    expect_unchanged();
  }
  {  // an empty hyperedge
    dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
    b.delete_hyperedge(1);
    b.insert_hyperedge(dyng::array_view<const std::int32_t>(), 3);
    EXPECT_THROW((void)h.apply(res(), b.view()), dyng::invalid_argument_error);
    expect_unchanged();
  }
  {  // a bad weight
    dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
    b.delete_hyperedge(1);
    b.insert_hyperedge({1, 2}, 0);
    EXPECT_THROW((void)h.apply(res(), b.view()), dyng::invalid_argument_error);
    expect_unchanged();
  }
  {  // missing weights
    std::vector<std::int64_t> offsets{0, 1};
    std::vector<std::int32_t> members{1};
    dyng::hyperedge_batch_view<std::int32_t, std::int32_t> v;
    v.insert_offsets = dyng::host_view(offsets);
    v.insert_members = dyng::host_view(members);
    EXPECT_THROW((void)h.apply(res(), v), dyng::invalid_argument_error);
    expect_unchanged();
  }
  {  // incidence edits (0.4)
    dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
    b.delete_hyperedge(1);
    b.insert_incidence(0, 4);
    try {
      (void)h.apply(res(), b.view());
      ADD_FAILURE() << "an incidence edit was accepted";
    } catch (const dyng::not_supported_error& e) {
      EXPECT_NE(std::string(e.what()).find("0.4"), std::string::npos) << e.what();
    }
    dyng::hyperedge_batch<std::int32_t, std::int32_t> c;
    c.delete_incidence(0, 0);
    EXPECT_THROW((void)h.apply(res(), c.view()), dyng::not_supported_error);
    expect_unchanged();
    // The owning batch reports its edits as the view does (the four-op batch of 0.4).
    EXPECT_EQ(b.insert_incidence_hyperedge(), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(b.insert_incidence_vertex(), (std::vector<std::int32_t>{4}));
    EXPECT_EQ(c.delete_incidence_hyperedge(), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(c.delete_incidence_vertex(), (std::vector<std::int32_t>{0}));
    EXPECT_TRUE(b.delete_incidence_hyperedge().empty() && c.insert_incidence_vertex().empty());
    EXPECT_EQ(c.view().delete_incidence_vertex.size(), 1U);
  }
  {  // malformed offsets
    std::vector<std::int64_t> offsets{0, 2, 1};
    std::vector<std::int32_t> members{1, 2};
    std::vector<std::int32_t> w{1, 1};
    dyng::hyperedge_batch_view<std::int32_t, std::int32_t> v;
    v.insert_offsets = dyng::host_view(offsets);
    v.insert_members = dyng::host_view(members);
    v.insert_weights = dyng::host_view(w);
    EXPECT_THROW((void)h.apply(res(), v), dyng::invalid_argument_error);
    expect_unchanged();
  }
}

// A store operation that fails poisons the hypergraph: MOSP_ESCHER's fixed capacity (no extra
// entries) cannot hold a large inserted row.
TEST_P(HypergraphBackends, CapacityErrorPoisons) {
  auto p = props_of(id_reuse::erase_first, dyng::capacity_policy::mosp_escher(0));
  hgraph h = hgraph::from_hyperedges(res(), list_of(2000, rows{{0, 1}, {2, 3}}).view(), p);
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  std::vector<std::int32_t> big;
  for (std::int32_t v = 0; v < 1500; ++v) big.push_back(v);
  b.insert_hyperedge(dyng::host_view(big));
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::invalid_argument_error);
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);
  EXPECT_THROW(h.check_integrity(res()), dyng::invalid_argument_error);
  EXPECT_THROW((void)h.clone(res()), dyng::invalid_argument_error);
  EXPECT_THROW((void)h.to_hyperedge_list(res()), dyng::invalid_argument_error);
  EXPECT_THROW((void)h.weights(), dyng::invalid_argument_error);  // M8b review
}

TEST_P(HypergraphBackends, VersionAndStateIdentity) {
  hgraph h = hgraph::from_hyperedges(res(), list_of(5, rows{{0}, {1}}).view(),
                                     props_of(id_reuse::erase_first));
  const std::uint64_t s0 = dyng::detail::hypergraph_access::state_id(h);
  dyng::hyperedge_batch<std::int32_t, unweighted> empty;
  const auto s = h.apply(res(), empty.view());
  EXPECT_EQ(h.version(), 1U);
  EXPECT_EQ(s.skipped_ops, 0);
  const std::uint64_t s1 = dyng::detail::hypergraph_access::state_id(h);
  EXPECT_NE(s0, s1);
  const hgraph c = h.clone(res());
  EXPECT_EQ(c.version(), 1U);
  EXPECT_EQ(dyng::detail::hypergraph_access::state_id(c), s1);
  const auto& applied = dyng::detail::hypergraph_access::applied(h);
  EXPECT_EQ(applied.version_before, 0U);
  EXPECT_EQ(applied.version_after, 1U);
  EXPECT_EQ(applied.state_before, s0);
  EXPECT_EQ(applied.state_after, s1);
}

TEST_P(HypergraphBackends, AppliedRecord) {
  hgraph h = hgraph::from_hyperedges(res(), list_of(8, rows{{0, 1}, {2, 3}, {4}}).view(),
                                     props_of(id_reuse::erase_first));
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({5, 6});
  const auto s = dyng::detail::hypergraph_access::apply(res(), h, b.view(), true);
  const auto& a = dyng::detail::hypergraph_access::applied(h);
  EXPECT_EQ(a.inserted_ids, s.inserted_ids);
  EXPECT_EQ(a.deleted_ids, (std::vector<std::int64_t>{1}));
  EXPECT_EQ(a.deleted_members.members, (std::vector<std::int32_t>{2, 3}));
  EXPECT_EQ(a.touched_hyperedges, (std::vector<std::int64_t>{1}));
  EXPECT_EQ(a.touched_vertices, (std::vector<std::int32_t>{2, 3, 5, 6}));
  ASSERT_TRUE(a.has_incidence_delta);
  EXPECT_EQ(a.incidence_delta.signs, (std::vector<std::int8_t>{-1, -1, 1, 1}));
  if (GetParam() == backend::cuda) {
    ASSERT_EQ(a.inserted_ids_device.size(), 1U);
    EXPECT_EQ(dyng::to_vector(res(), a.inserted_ids_device.view()), s.inserted_ids);
  } else {
    EXPECT_EQ(a.inserted_ids_device.size(), 0U);
  }
}

TEST_P(HypergraphBackends, WeightsById) {
  const auto list = list_of<std::int32_t>(6, rows{{0, 1}, {2, 3}, {4}}, {10, 20, 30});
  whgraph h = whgraph::from_hyperedges(res(), list.view(), props_of(id_reuse::erase_first));
  EXPECT_EQ(dyng::test::host_copy(h.weights()), (std::vector<std::int32_t>{10, 20, 30}));
  dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({5}, 99);
  const auto s = h.apply(res(), b.view());
  std::vector<std::int32_t> want(static_cast<std::size_t>(h.hyperedge_id_bound()), 0);
  want[1] = 20;
  want[2] = 30;
  want[static_cast<std::size_t>(s.inserted_ids[0])] = 99;
  EXPECT_EQ(dyng::test::host_copy(h.weights()), want);
  h.check_integrity(res());
  const auto list_back = h.to_hyperedge_list(res());
  EXPECT_EQ(list_back.weights.size(), 3U);
  // An unweighted hypergraph takes no weights.
  hgraph u =
      hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), props_of(id_reuse::erase_first));
  EXPECT_TRUE(u.weights().empty());
}

TEST_P(HypergraphBackends, LineGraphPolicies) {
  auto p = props_of(id_reuse::erase_first);
  const hgraph none = hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), p);
  EXPECT_THROW((void)none.h2h(), dyng::invalid_argument_error);
  p.line_graph = dyng::line_graph_policy::automatic;
  const hgraph automatic = hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), p);
  EXPECT_EQ(automatic.h2h().num_rows(), 1);  // built on first read (hypergraph_line_graph_test)
  p.store_v2h = false;
  p.line_graph = dyng::line_graph_policy::none;
  const hgraph no_v2h = hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), p);
  EXPECT_THROW((void)no_v2h.v2h(), dyng::invalid_argument_error);
  no_v2h.check_integrity(res());
}

TEST_P(HypergraphBackends, CloneAndToBackendKeepIdsAndLaterIds) {
  const auto list = list_of(30, rows{{0, 1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11}});
  auto p = props_of(id_reuse::erase_first);
  hgraph h = hgraph::from_hyperedges(res(), list.view(), p);
  dyng::hyperedge_batch<std::int32_t, unweighted> d;
  d.delete_hyperedge(0);
  d.delete_hyperedge(3);
  (void)h.apply(res(), d.view());
  dyng::hyperedge_batch<std::int32_t, unweighted> next;
  next.insert_hyperedge({20, 21, 22, 23, 24, 25, 26});
  next.insert_hyperedge({27});
  next.insert_hyperedge({28, 29});
  for (const backend target : dyng::test::comparison_backends()) {
    const resources tres = dyng::test::hyper::resources_of(target);
    hgraph c = h.to_backend(tres);
    EXPECT_EQ(c.version(), h.version());
    EXPECT_EQ(c.hyperedge_id_bound(), h.hyperedge_id_bound());
    EXPECT_EQ(h2v_rows(c), h2v_rows(h)) << dyng::to_string(target);
    c.check_integrity(tres);
    // Back to the original backend: the ids, and the ids of a later batch, survive.
    hgraph back = c.to_backend(res());
    hgraph original = h.clone(res());
    const auto s1 = back.apply(res(), next.view());
    const auto s2 = original.apply(res(), next.view());
    const auto s3 = c.apply(tres, next.view());
    EXPECT_EQ(s1.inserted_ids, s2.inserted_ids) << dyng::to_string(target);
    EXPECT_EQ(s3.inserted_ids, s2.inserted_ids) << dyng::to_string(target);
    EXPECT_EQ(h2v_rows(back), h2v_rows(original));
    // The payload accounting moves with the ids (ADR 0041, Consequences).
    EXPECT_EQ(back.store_stats().payload_used, original.store_stats().payload_used);
    EXPECT_EQ(back.store_stats().payload_capacity, original.store_stats().payload_capacity);
  }
}

TEST_P(HypergraphBackends, PlacementMismatchThrows) {
  hgraph h =
      hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), props_of(id_reuse::erase_first));
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  if (GetParam() == backend::cuda) {
    EXPECT_THROW((void)h.apply(resources::sequential(), b.view()), dyng::invalid_argument_error);
    EXPECT_THROW(h.check_integrity(resources::sequential()), dyng::invalid_argument_error);
  } else if (dyng::backend_available(backend::cuda)) {
    EXPECT_THROW((void)h.apply(resources::cuda(0), b.view()), dyng::invalid_argument_error);
    EXPECT_THROW((void)h.to_hyperedge_list(resources::cuda(0)), dyng::invalid_argument_error);
  }
  EXPECT_EQ(h.version(), 0U);
}

TEST_P(HypergraphBackends, MovedFromThrows) {
  hgraph h =
      hgraph::from_hyperedges(res(), list_of(3, rows{{0}}).view(), props_of(id_reuse::erase_first));
  hgraph other = std::move(h);
  EXPECT_EQ(h.num_hyperedges(), 0);  // NOLINT(bugprone-use-after-move): the documented state
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(other.num_hyperedges(), 1);
}

TEST_P(HypergraphBackends, StoreStats) {
  hgraph h = hgraph::from_hyperedges(res(), list_of(3, rows{{0, 1}, {2}}).view(),
                                     props_of(id_reuse::erase_first));
  const auto st = h.store_stats();
  EXPECT_GT(st.payload_capacity, 0);
  EXPECT_GT(st.payload_used, 0);
  if (GetParam() == backend::cuda) {
    EXPECT_GT(st.device_bytes, 0);
  } else {
    EXPECT_EQ(st.device_bytes, 0);
    EXPECT_GT(st.host_bytes, 0);
  }
  (void)h.v2h();
  EXPECT_GT(h.store_stats().view_bytes, 0);
}

// The profiler stages of parity/timed_regions/hypergraph.toml (PLAN 4.7.5).
TEST_P(HypergraphBackends, ProfilerStages) {
  dyng::profiler prof;
  resources r = res();
  r.attach_profiler(&prof);
  hgraph h = hgraph::from_hyperedges(r, list_of(5, rows{{0, 1}, {2}}).view(),
                                     props_of(id_reuse::erase_first));
  dyng::hyperedge_batch<std::int32_t, unweighted> b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({3, 4});
  (void)h.apply(r, b.view());
  r.attach_profiler(nullptr);
  std::vector<std::string> names;
  for (const auto& st : prof.stages()) names.push_back(st.name);
  const auto has = [&](const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
  };
  for (const char* n : {"hypergraph.build", "hypergraph.apply", "hypergraph.apply.delete",
                        "hypergraph.apply.insert"}) {
    EXPECT_TRUE(has(n)) << n;
  }
  if (GetParam() == backend::cuda) {
    EXPECT_TRUE(has("hypergraph.upload"));
    EXPECT_TRUE(has("hypergraph.apply.views"));
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, HypergraphBackends,
                         ::testing::ValuesIn(dyng::test::suite_backends()), backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HypergraphBackends);

TEST(HypergraphNames, EnumerationsHaveNames) {
  EXPECT_EQ(dyng::to_string(dyng::hypergraph_storage::escher), "escher");
  EXPECT_EQ(dyng::to_string(dyng::hypergraph_storage::slack_csr), "slack_csr");
  EXPECT_EQ(dyng::to_string(dyng::line_graph_policy::none), "none");
  EXPECT_EQ(dyng::to_string(dyng::line_graph_policy::automatic), "automatic");
  EXPECT_EQ(dyng::to_string(dyng::line_graph_policy::sorted_merge), "sorted_merge");
  EXPECT_EQ(dyng::to_string(dyng::line_graph_policy::slack_csr), "slack_csr");
  EXPECT_EQ(dyng::to_string(id_reuse::erase_first), "erase_first");
  EXPECT_EQ(dyng::to_string(id_reuse::insert_first), "insert_first");
}

TEST(HypergraphDefaults, PlanDefaults) {
  const hypergraph_properties p;
  EXPECT_EQ(p.storage, dyng::hypergraph_storage::escher);
  EXPECT_EQ(p.line_graph, dyng::line_graph_policy::automatic);
  EXPECT_EQ(p.reuse, id_reuse::erase_first);  // O16
  EXPECT_TRUE(p.store_v2h);
  const dyng::capacity_policy c;
  EXPECT_EQ(c.initial_headroom, 0.125);
  EXPECT_EQ(c.grow_factor, 1.5);
  EXPECT_TRUE(c.compact_before_grow);
  EXPECT_EQ(c.max_payload, 0);
}

TEST(HyperedgeBatch, BuilderAndView) {
  dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
  EXPECT_TRUE(b.empty());
  b.delete_hyperedge(4);
  b.insert_hyperedge({1, 2}, 5);
  b.insert_hyperedge({3}, 6);
  const auto v = b.view();
  EXPECT_EQ(v.num_insertions(), 2U);
  EXPECT_EQ(v.num_deletions(), 1U);
  EXPECT_FALSE(v.empty());
  EXPECT_EQ(b.insert_offsets(), (std::vector<std::int64_t>{0, 2, 3}));
  EXPECT_EQ(b.insert_weights(), (std::vector<std::int32_t>{5, 6}));
  b.clear();
  EXPECT_TRUE(b.empty());
  EXPECT_TRUE(b.view().empty());
  // M8b review: a view with only an incidence vertex array (unpaired, rejected by apply()) is not
  // empty.
  const std::vector<std::int32_t> vertex{3};
  dyng::hyperedge_batch_view<std::int32_t, std::int32_t> only_vertex;
  only_vertex.delete_incidence_vertex = dyng::host_view(vertex);
  EXPECT_FALSE(only_vertex.empty());
  only_vertex = {};
  only_vertex.insert_incidence_vertex = dyng::host_view(vertex);
  EXPECT_FALSE(only_vertex.empty());
  dyng::hyperedge_list<std::int32_t> list;
  list.add_hyperedge({1, 2, 3});
  EXPECT_EQ(list.num_hyperedges(), 1);
  EXPECT_EQ(list.view().num_hyperedges(), 1);
}

}  // namespace
