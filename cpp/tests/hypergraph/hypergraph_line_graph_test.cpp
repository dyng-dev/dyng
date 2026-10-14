// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_line_graph_test.cpp
 * @brief The line graph of the hypergraph container (M8b step 2, ADR 0042): the policies, h2h
 *        against dyng::testing::rebuild_line_graph after every batch (deleted and reinserted keys
 *        in one batch, hubs, rows long enough for the block paths, delete-all, empty batches), the
 *        views (out offsets, memory space, the internal row views, how often each view is built),
 *        clone and to_backend, the capacity limit and the profiler stage. Compiled into the host
 *        test executable (sequential, openmp) and, with DYNG_TEST_CUDA=1, the CUDA one (cuda).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/line_graph_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::hypergraph_properties;
using dyng::id_reuse;
using dyng::line_graph_policy;
using dyng::resources;
using dyng::unweighted;
using dyng::test::hyper::check_line_graph;
using dyng::test::hyper::h2h_rows;
using dyng::test::hyper::id_rows;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using hgraph = dyng::hypergraph<std::int32_t, unweighted>;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using batch = dyng::hyperedge_batch<std::int32_t, unweighted>;

class HypergraphLineGraph : public ::testing::TestWithParam<backend> {
 protected:
  resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

hypergraph_properties merged(id_reuse reuse = id_reuse::erase_first,
                             dyng::capacity_policy c = dyng::capacity_policy::escher_gpu()) {
  hypergraph_properties p;
  p.line_graph = line_graph_policy::sorted_merge;
  p.reuse = reuse;
  p.capacity = c;
  return p;
}

// The internal state of a hypergraph's store (the view build counters, the line graph layout).
template <typename weight_t>
const dyng::detail::hyper::incidence_store& store_of(
    const dyng::hypergraph<std::int32_t, weight_t>& h) {
  return dyng::detail::hypergraph_access::store(h);
}

TEST_P(HypergraphLineGraph, Policies) {
  auto p = merged();
  p.line_graph = line_graph_policy::none;
  const auto list = list_of(4, rows{{0, 1}, {1, 2}, {3}});
  const hgraph none = hgraph::from_hyperedges(res(), list.view(), p);
  EXPECT_THROW((void)none.h2h(), dyng::invalid_argument_error);
  EXPECT_FALSE(store_of(none).line_graph_enabled());

  // automatic: built by the first read, then maintained.
  p.line_graph = line_graph_policy::automatic;
  hgraph automatic = hgraph::from_hyperedges(res(), list.view(), p);
  EXPECT_FALSE(store_of(automatic).line_graph_enabled());
  EXPECT_EQ(h2h_rows(automatic), (id_rows{{1}, {0}, {}}));
  EXPECT_TRUE(store_of(automatic).line_graph_enabled());
  EXPECT_EQ(store_of(automatic).builds().line_graph_builds, 1);
  batch b;
  b.insert_hyperedge({2, 3});
  (void)automatic.apply(res(), b.view());
  EXPECT_EQ(check_line_graph(automatic), "");
  EXPECT_EQ(store_of(automatic).builds().line_graph_builds, 1);  // maintained, not rebuilt

  // sorted_merge: from the start.
  const hgraph sorted = hgraph::from_hyperedges(res(), list.view(), merged());
  EXPECT_TRUE(store_of(sorted).line_graph_enabled());
  EXPECT_EQ(check_line_graph(sorted), "");

  // slack_csr: planned for 0.4.
  p.line_graph = line_graph_policy::slack_csr;
  try {
    (void)hgraph::from_hyperedges(res(), list.view(), p);
    ADD_FAILURE() << "slack_csr accepted";
  } catch (const dyng::not_supported_error& e) {
    EXPECT_NE(std::string(e.what()).find("0.4"), std::string::npos) << e.what();
  }

  // The line graph needs v2h.
  auto no_v2h = merged();
  no_v2h.store_v2h = false;
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list.view(), no_v2h),
               dyng::invalid_argument_error);
  no_v2h.line_graph = line_graph_policy::automatic;
  const hgraph lazy = hgraph::from_hyperedges(res(), list.view(), no_v2h);
  EXPECT_THROW((void)lazy.h2h(), dyng::invalid_argument_error);
}

// Every batch of a mixed stream, both orders: h2h equals the oracle after every batch, and the
// row view's out offsets are the first entries above the row's id.
TEST_P(HypergraphLineGraph, EqualsTheOracleAfterEveryBatch) {
  for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
    SCOPED_TRACE(std::string(dyng::to_string(reuse)));
    std::mt19937_64 rng(reuse == id_reuse::erase_first ? 11 : 12);
    const std::int32_t n = 300;
    const rows initial = dyng::test::hyper::random_rows(rng, n, 80, 7);
    hgraph h = hgraph::from_hyperedges(res(), list_of(n, initial).view(), merged(reuse));
    ASSERT_EQ(check_line_graph(h), "");
    using dyng::test::hyper::stream_mix;
    for (const stream_mix mix :
         {stream_mix::sparse, stream_mix::delete_reinsert, stream_mix::dense,
          stream_mix::equal_sizes, stream_mix::reissue, stream_mix::limits, stream_mix::larger_than,
          stream_mix::delete_all, stream_mix::skewed}) {
      SCOPED_TRACE(dyng::test::hyper::mix_name(mix));
      rows current;
      const auto live = dyng::test::hyper::live_ids(h, current);
      const auto tb = dyng::test::hyper::next_batch(rng, mix, std::min<std::int32_t>(n, 300), live,
                                                    h.hyperedge_id_bound(), current);
      (void)h.apply(res(), dyng::test::hyper::batch_of(tb).view());
      ASSERT_EQ(check_line_graph(h), "");
      h.check_integrity(res());
    }
  }
}

// A key deleted and reinserted in the same batch (erase_first reuses its slot), a hub vertex in
// every hyperedge, a row of more than 1,024 neighbours (the warp and block paths of the merge),
// an empty batch, a delete-all batch and a batch larger than the hypergraph.
TEST_P(HypergraphLineGraph, SpecialBatches) {
  const std::int32_t n = 3000;
  rows initial;
  for (std::int32_t i = 0; i < 1100; ++i) initial.push_back({0, 1 + i, 1 + (i * 7) % 2999});
  hgraph h = hgraph::from_hyperedges(res(), list_of(n, initial).view(), merged());
  ASSERT_EQ(check_line_graph(h), "");
  {
    // Delete id 3 and insert its row again: the same slot, the same id.
    batch b;
    b.delete_hyperedge(3);
    b.insert_hyperedge(dyng::host_view(initial[3]));
    const auto s = h.apply(res(), b.view());
    EXPECT_EQ(s.inserted_ids, (std::vector<std::int64_t>{3}));
    ASSERT_EQ(check_line_graph(h), "");
  }
  {
    // An empty batch.
    batch b;
    (void)h.apply(res(), b.view());
    ASSERT_EQ(check_line_graph(h), "");
  }
  {
    // A hub hyperedge of 1,200 members, and deletions of hub rows.
    batch b;
    std::vector<std::int32_t> big;
    for (std::int32_t v = 0; v < 1200; ++v) big.push_back(v * 2);
    b.insert_hyperedge(dyng::host_view(big));
    for (std::int64_t id = 10; id < 40; ++id) b.delete_hyperedge(id);
    (void)h.apply(res(), b.view());
    ASSERT_EQ(check_line_graph(h), "");
  }
  {
    // Every live hyperedge deleted; the same rows inserted again (a batch larger than the
    // hypergraph that is left: none).
    rows current;
    const auto live = dyng::test::hyper::live_ids(h, current);
    batch b;
    for (const std::int64_t id : live) b.delete_hyperedge(id);
    (void)h.apply(res(), b.view());
    EXPECT_EQ(h.num_hyperedges(), 0);
    ASSERT_EQ(check_line_graph(h), "");
    batch again;
    for (const std::int64_t id : live)
      again.insert_hyperedge(dyng::host_view(current[static_cast<std::size_t>(id)]));
    for (int i = 0; i < 50; ++i) again.insert_hyperedge({0, i + 1});
    (void)h.apply(res(), again.view());
    ASSERT_EQ(check_line_graph(h), "");
  }
  h.check_integrity(res());
}

// The internal row views (what M9's kernels read): A's encoding (keys = ids + 1, values + 1),
// rebuilt once per commit; the public h2v, v2h and h2h views built at most once per version and
// never by apply().
TEST_P(HypergraphLineGraph, ViewsAndHowOftenTheyAreBuilt) {
  const auto list = list_of(6, rows{{0, 1}, {1, 2}, {3, 4}, {4, 5}, {0, 5}});
  hgraph h = hgraph::from_hyperedges(res(), list.view(), merged());
  const auto& store = store_of(h);
  const auto before = store.builds();
  batch b;
  b.delete_hyperedge(1);
  b.insert_hyperedge({2, 3});
  (void)h.apply(res(), b.view());
  (void)h.apply(res(), batch().view());
  auto after = store.builds();
  EXPECT_EQ(after.row_views, before.row_views + 2);  // once per commit
  EXPECT_EQ(after.h2v, before.h2v);                  // apply builds no public view
  EXPECT_EQ(after.v2h, before.v2h);
  EXPECT_EQ(after.h2h, before.h2h);
  (void)h.v2h();
  (void)h.v2h();
  (void)h.h2h();
  (void)h.h2h();
  after = store.builds();
  EXPECT_EQ(after.v2h, before.v2h + 1);  // at most once per version
  EXPECT_EQ(after.h2h, before.h2h + 1);
  (void)h.apply(res(), batch().view());
  (void)h.v2h();
  EXPECT_EQ(store.builds().v2h, before.v2h + 2);

  // The internal views, decoded, equal the public ones.
  const auto views = dyng::detail::hypergraph_access::views(h);
  EXPECT_TRUE(views.has_h2h);
  EXPECT_EQ(views.space, h.space());
  EXPECT_EQ(views.id_bound, h.hyperedge_id_bound());
  const auto& lg = store.line_graph();
  EXPECT_EQ(views.h2h.max_key, lg.num_keys());
  // Host copies of the internal arrays.
  const auto copy = [&](const int* p, std::size_t count) {
    return dyng::test::hyper::host_vector(dyng::array_view<const int>(
        p, count, h.space(), h.space() == dyng::memory_space::device ? 0 : -1));
  };
  const auto bound = static_cast<std::size_t>(views.id_bound);
  const auto h2v_loc = copy(views.h2v.loc, bound + 1), h2v_len = copy(views.h2v.len, bound + 1);
  const auto h2h_loc = copy(views.h2h.loc, static_cast<std::size_t>(views.h2h.max_key) + 1);
  const auto h2h_len = copy(views.h2h.len, static_cast<std::size_t>(views.h2h.max_key) + 1);
  const auto h2h_out = copy(views.h2h.out, static_cast<std::size_t>(views.h2h.max_key) + 1);
  const rows h2v = dyng::test::hyper::h2v_rows(h);
  const id_rows h2h = h2h_rows(h);
  for (std::size_t id = 0; id < bound; ++id) {
    SCOPED_TRACE("id " + std::to_string(id));
    ASSERT_EQ(static_cast<std::size_t>(h2v_len[id + 1]), h2v[id].size());
    const auto members = copy(views.h2v.flat + h2v_loc[id + 1], h2v[id].size());
    for (std::size_t k = 0; k < h2v[id].size(); ++k) EXPECT_EQ(members[k], h2v[id][k] + 1);
    ASSERT_EQ(static_cast<std::size_t>(h2h_len[id + 1]), h2h[id].size());
    const auto neighbours = copy(views.h2h.flat + h2h_loc[id + 1], h2h[id].size());
    for (std::size_t k = 0; k < h2h[id].size(); ++k) EXPECT_EQ(neighbours[k], h2h[id][k] + 1);
    const auto above =
        std::upper_bound(h2h[id].begin(), h2h[id].end(), static_cast<std::int64_t>(id)) -
        h2h[id].begin();
    EXPECT_EQ(h2h_out[id + 1] - h2h_loc[id + 1], above);
  }
}

// clone and to_backend keep the line graph (the same layout, entry by entry) and later batches
// keep it equal to the oracle.
TEST_P(HypergraphLineGraph, CloneAndToBackendKeepTheLineGraph) {
  std::mt19937_64 rng(5);
  const rows initial = dyng::test::hyper::random_rows(rng, 200, 120, 6);
  hgraph h = hgraph::from_hyperedges(res(), list_of(200, initial).view(), merged());
  batch b;
  for (std::int64_t id = 100; id < 120; ++id) b.delete_hyperedge(id);
  b.insert_hyperedge({1, 2, 3});
  (void)h.apply(res(), b.view());
  const auto layout = store_of(h).line_graph();
  std::vector<resources> targets = {res(), resources::sequential()};
  if (dyng::backend_available(backend::cuda)) targets.push_back(resources::cuda(0));
  for (const resources& r : targets) {
    SCOPED_TRACE(std::string(dyng::to_string(r.get_backend())));
    hgraph copy = h.clone(r);
    const auto got = store_of(copy).line_graph();
    EXPECT_EQ(got.starts, layout.starts);
    EXPECT_EQ(got.counts, layout.counts);
    EXPECT_EQ(got.payload, layout.payload);
    batch next;
    next.delete_hyperedge(3);
    next.insert_hyperedge({4, 5, 199});
    (void)copy.apply(r, next.view());
    EXPECT_EQ(check_line_graph(copy), "");
    copy.check_integrity(r);
  }
}

// A line graph larger than the policy's max_payload: capacity_error at construction, and a batch
// that makes it too large poisons the hypergraph.
TEST_P(HypergraphLineGraph, CapacityLimit) {
  dyng::capacity_policy c = dyng::capacity_policy::escher_gpu();
  c.max_payload = 120;
  rows hub;
  for (int i = 0; i < 9; ++i) hub.push_back({0, i + 1});  // h2h: 9 rows of 8 = 108 entries
  EXPECT_NO_THROW((void)hgraph::from_hyperedges(res(), list_of(12, hub).view(),
                                                merged(id_reuse::erase_first, c)));
  hub.push_back({0, 11});  // h2h: 11 rows of 10 entries, 12 slots each = 132 > 120
  hub.push_back({0, 10});
  EXPECT_THROW((void)hgraph::from_hyperedges(res(), list_of(12, hub).view(),
                                             merged(id_reuse::erase_first, c)),
               dyng::capacity_error);
  hub.resize(9);
  hgraph h =
      hgraph::from_hyperedges(res(), list_of(12, hub).view(), merged(id_reuse::erase_first, c));
  batch b;
  b.insert_hyperedge({0, 10});
  b.insert_hyperedge({0, 11});
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.h2h(), dyng::invalid_argument_error);  // poisoned
}

// The stage of the line-graph phase (parity/timed_regions/hypergraph.toml, maintain.line_graph).
TEST_P(HypergraphLineGraph, ProfilerStage) {
  dyng::profiler prof;
  resources r = res();
  r.attach_profiler(&prof);
  hgraph h = hgraph::from_hyperedges(r, list_of(5, rows{{0, 1}, {1, 2}}).view(), merged());
  batch b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({3, 2});
  (void)h.apply(r, b.view());
  r.attach_profiler(nullptr);
  std::vector<std::string> names;
  for (const auto& st : prof.stages()) names.push_back(st.name);
  for (const char* n : {"hypergraph.build", "hypergraph.apply", "hypergraph.apply.delete",
                        "hypergraph.apply.insert", "hypergraph.apply.line_graph"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), n), names.end()) << n;
  }
}

// Weighted hypergraphs: the h2h view carries the weights by id (host memory).
TEST_P(HypergraphLineGraph, WeightedView) {
  const auto list = list_of<std::int32_t>(4, rows{{0, 1}, {1, 2}}, {7, 9});
  whgraph h = whgraph::from_hyperedges(res(), list.view(), merged());
  const auto v = h.h2h();
  EXPECT_EQ(dyng::test::host_copy(v.weights), (std::vector<std::int32_t>{7, 9}));
  EXPECT_EQ(check_line_graph(h), "");
}

INSTANTIATE_TEST_SUITE_P(Backends, HypergraphLineGraph,
                         ::testing::ValuesIn(dyng::test::suite_backends()), backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HypergraphLineGraph);

// The oracle itself on a small case and on malformed input.
TEST(LineGraphOracle, SmallCaseAndErrors) {
  // ids 0: {0,1}, 1: dead, 2: {1,2}, 3: {3}, 4: {0,3,3}
  const std::vector<std::int64_t> ptr = {0, 2, 2, 4, 5, 8};
  const std::vector<std::int32_t> cols = {0, 1, 1, 2, 3, 0, 3, 3};
  const auto lg = dyng::testing::rebuild_line_graph(ptr, cols, 4);
  ASSERT_EQ(lg.num_ids(), 5);
  EXPECT_EQ(lg.row(0), (std::vector<std::int64_t>{2, 4}));
  EXPECT_TRUE(lg.row(1).empty());
  EXPECT_EQ(lg.row(2), (std::vector<std::int64_t>{0}));
  EXPECT_EQ(lg.row(3), (std::vector<std::int64_t>{4}));
  EXPECT_EQ(lg.row(4), (std::vector<std::int64_t>{0, 3}));
  EXPECT_THROW((void)dyng::testing::rebuild_line_graph({0, 1}, {4}, 4), std::invalid_argument);
  EXPECT_THROW((void)dyng::testing::rebuild_line_graph({1, 1}, {0}, 4), std::invalid_argument);
  EXPECT_THROW((void)dyng::testing::rebuild_line_graph({0, 2, 1}, {0, 1}, 4),
               std::invalid_argument);
}

}  // namespace
