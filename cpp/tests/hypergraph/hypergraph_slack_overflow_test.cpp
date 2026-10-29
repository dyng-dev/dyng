// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_slack_overflow_test.cpp
 * @brief The overflow paths and the limits of slack_csr storage (M11a step 3; ADRs 0086, 0089;
 *        M11a criteria 10 and 17), on every backend: MOSP_ESCHER@4b86159's test_h2h_delta pattern
 *        (headroom 1.0 on every third configuration) with the CSR-tail rebuild and, on the CUDA
 *        store, the re-uploads of the incidence mirror, each asserted through the counters; a row
 *        that relocates batch after batch; a rebuild and a re-upload in one batch; node-capacity
 *        growth; and the over-limit inputs, each a capacity_error (never undefined behaviour or a
 *        silent wrap): more candidates than the limit, ids past the int32 node ids, entries past
 *        the int32 counts (through the internal limits of slack::knobs).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
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
using dyng::detail::hypergraph_access;
using dyng::test::hyper::list_of;
using dyng::test::hyper::rows;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using wbatch_t = dyng::hyperedge_batch<std::int32_t, std::int32_t>;
using knobs_t = dyng::detail::hyper::slack::knobs;
using pair_list = std::vector<std::pair<std::int64_t, std::int64_t>>;

hypergraph_properties slack_props(capacity_policy c = capacity_policy::mosp_escher(4096)) {
  hypergraph_properties p;
  p.storage = hypergraph_storage::slack_csr;
  p.line_graph = line_graph_policy::slack_csr;
  p.reuse = id_reuse::insert_first;
  p.capacity = c;
  return p;
}

class SlackOverflow : public ::testing::TestWithParam<backend> {
 protected:
  [[nodiscard]] resources res() const {
    return dyng::test::hyper::resources_of(GetParam());
  }
  [[nodiscard]] bool cuda() const {
    return GetParam() == backend::cuda;
  }
};

std::string backend_name(const ::testing::TestParamInfo<backend>& info) {
  return std::string(dyng::to_string(info.param));
}

std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> flat(const resources& res,
                                                                     const whgraph& h) {
  const auto v = h.h2v();
  return {dyng::to_vector(res, v.row_ptr), dyng::to_vector(res, v.col_ind)};
}

// A batch and every check of a slack_csr batch: the pairs against testing::line_graph_delta of
// the two states, check_integrity() (on cuda: the mirror and the device CSR against the host model,
// the device weights, the capacity invariants), h2h() against rebuild_line_graph.
void apply_checked(const resources& res, whgraph& h, const wbatch_t& b) {
  const auto pre = flat(res, h);
  (void)h.apply(res, b.view());
  const auto post = flat(res, h);
  const auto want = dyng::testing::line_graph_delta(pre.first, pre.second, post.first, post.second,
                                                    h.num_vertices());
  const auto& d = hypergraph_access::pairs(h);
  pair_list del, ins;
  for (std::size_t i = 0; i < d.deleted_a.size(); ++i)
    del.emplace_back(d.deleted_a[i], d.deleted_b[i]);
  for (std::size_t i = 0; i < d.inserted_a.size(); ++i)
    ins.emplace_back(d.inserted_a[i], d.inserted_b[i]);
  EXPECT_EQ(del, want.first);
  EXPECT_EQ(ins, want.second);
  h.check_integrity(res);
  EXPECT_EQ(dyng::test::hyper::check_line_graph(h), "");
}

// MOSP_ESCHER's test_h2h_delta (tests/unit/test_h2h_delta.cu:48-53, 84-88): random configurations,
// every third with no spare capacity (headroom 1.0: the CSR's entries are its initial layout plus
// 4096, the mirror's values their initial size plus 64), six random batches of either kind each.
TEST_P(SlackOverflow, MospEscherTestH2hDeltaPattern) {
  std::mt19937_64 meta(20260725);
  std::int64_t rebuilds = 0, reuploads = 0, tail_batches = 0, batches = 0;
  for (int cfg = 0; cfg < 24; ++cfg) {
    const auto n = static_cast<std::int32_t>(60 + meta() % 400);
    const int m = 80 + static_cast<int>(meta() % 500);
    const auto pool = static_cast<std::int32_t>(16 + meta() % 64);
    const int c_max = 2 + static_cast<int>(meta() % 7);
    rows initial;
    std::vector<std::int32_t> w;
    std::mt19937_64 gen(meta());
    for (int i = 0; i < m; ++i) {
      std::vector<std::int32_t> r;
      const int c = 1 + static_cast<int>(gen() % static_cast<std::uint64_t>(c_max));
      for (int k = 0; k < c; ++k)
        r.push_back(static_cast<std::int32_t>(gen() % 10 == 0 ? gen() % n : gen() % pool));
      initial.push_back(r);
      w.push_back(1 + static_cast<std::int32_t>(gen() % 100));
    }
    knobs_t knobs;
    knobs.entry_headroom = cfg % 3 == 0 ? 1.0 : 1.3;
    knobs.node_capacity = m + 2048;
    whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(n, initial, w).view(),
                                         slack_props(), knobs);
    auto* slack = hypergraph_access::slack(h);
    ASSERT_NE(slack, nullptr);
    for (int bi = 0; bi < 6; ++bi) {
      const int size = 5 + static_cast<int>(meta() % 80);
      const auto del_pct = static_cast<int>(meta() % 101);
      const bool vertex_kind = meta() % 2 == 0;
      wbatch_t b;
      const std::int64_t bound = h.hyperedge_id_bound();
      const int deletions = size * del_pct / 100;
      for (int k = 0; k < size; ++k) {
        const auto id = static_cast<std::int64_t>(gen() % static_cast<std::uint64_t>(bound));
        const auto v = static_cast<std::int32_t>(gen() % 3 == 0 ? gen() % n : gen() % pool);
        if (vertex_kind) {
          if (k < deletions) {
            b.delete_incidence(id, v);
          } else {
            b.insert_incidence(id, v);
          }
        } else if (k < deletions) {
          b.delete_hyperedge(id);
        } else {
          std::vector<std::int32_t> r{v};
          for (int j = 0; j < static_cast<int>(gen() % 4); ++j)
            r.push_back(static_cast<std::int32_t>(gen() % pool));
          b.insert_hyperedge(dyng::host_view(r), 1 + static_cast<std::int32_t>(gen() % 100));
        }
      }
      const auto before = slack->counters();
      apply_checked(res(), h, b);
      const auto after = slack->counters();
      rebuilds += after.csr_rebuilds - before.csr_rebuilds;
      reuploads += after.mirror_reuploads - before.mirror_reuploads;
      tail_batches += after.tail_entries > before.tail_entries ? 1 : 0;
      ++batches;
    }
  }
  // Both overflow paths ran (the mirror's on the CUDA store only: the host stores keep no mirror),
  // and rows moved to the tail in many batches.
  EXPECT_GT(rebuilds, 0);
  EXPECT_GT(tail_batches, batches / 4);
  if (cuda()) {
    EXPECT_GT(reuploads, 0);
  } else {
    EXPECT_EQ(reuploads, 0);
  }
}

// One row relocates batch after batch: the hyperedge 0 = {0 .. 299} gains twenty neighbours per
// batch (each new hyperedge {x, 1000 + x} meets it alone), so its slot (32, then 64, ...)
// overflows again and again; the tail has room (no rebuild), and every relocation keeps the row
// exact.
TEST_P(SlackOverflow, ARowRelocatesRepeatedly) {
  knobs_t knobs;
  knobs.entry_headroom = 300.0;
  knobs.node_capacity = 4096;
  std::vector<std::int32_t> big(300);
  for (int v = 0; v < 300; ++v) big[static_cast<std::size_t>(v)] = v;
  whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(4000, rows{big}, {1}).view(),
                                       slack_props(), knobs);
  const auto* slack = hypergraph_access::slack(h);
  std::int64_t moves = 0;
  std::int64_t last_cap = 0;
  for (int batch = 0; batch < 12; ++batch) {
    wbatch_t b;
    for (int k = 0; k < 20; ++k) {
      const auto v = static_cast<std::int32_t>(batch * 20 + k);
      b.insert_hyperedge({v, 1000 + v}, 1 + k);
    }
    apply_checked(res(), h, b);
    EXPECT_EQ(slack->counters().csr_rebuilds, 0);
    // The slot of row 0 (host or device memory).
    std::int32_t cap = 0;
    const auto rows_view = slack->rows();
    if (rows_view.space == dyng::memory_space::host) {
      cap = rows_view.cap[0];
    } else {
      dyng::copy<std::int32_t>(
          res(),
          dyng::array_view<const std::int32_t>(rows_view.cap, 1, rows_view.space, rows_view.device),
          dyng::host_view(&cap, 1));
      res().synchronize();
    }
    moves += cap > last_cap && batch > 0 ? 1 : 0;
    last_cap = cap;
  }
  EXPECT_GE(moves, 3);  // the row moved to a larger slot in several batches
  const auto h2h = dyng::test::hyper::h2h_rows(h);
  EXPECT_EQ(h2h[0].size(), 240U);
}

// One batch that overflows the CSR's tail (a rebuild) and, on the CUDA store, the vertex mirror's
// tail (a re-upload): headroom 1.0, thirty rows on vertex 20, two hundred insertions on it. The
// pairs of the record and the device weights are right after the rebuild.
TEST_P(SlackOverflow, ARebuildAndAReuploadInOneBatch) {
  knobs_t knobs;
  knobs.entry_headroom = 1.0;
  knobs.node_capacity = 1024;
  rows initial;
  std::vector<std::int32_t> w;
  for (int i = 0; i < 30; ++i) {
    initial.push_back({i % 10, 20});
    w.push_back(1 + i);
  }
  whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(300, initial, w).view(),
                                       slack_props(), knobs);
  const auto* slack = hypergraph_access::slack(h);
  wbatch_t b;
  for (int k = 0; k < 200; ++k) b.insert_hyperedge({20, 30 + k}, 7 + k % 50);
  b.delete_hyperedge(3);
  apply_checked(res(), h, b);
  const auto c = slack->counters();
  EXPECT_EQ(c.csr_rebuilds, 1);
  EXPECT_EQ(c.mirror_reuploads, cuda() ? 2 : 0);  // the vertex rows and the hyperedge rows
  // The weights of the new ids (on the device: check_integrity compared them) and their rows.
  const auto weights = dyng::to_vector(res(), h.weights());
  EXPECT_EQ(weights[30], 7);
  EXPECT_EQ(weights[3], 0);  // deleted
  // A later batch on the rebuilt CSR.
  wbatch_t next;
  next.delete_incidence(31, 20);
  next.insert_incidence(31, 5);
  apply_checked(res(), h, next);
}

// The node arrays grow (by half) when an id passes the node capacity, with the rows, weights and
// the mirror kept.
TEST_P(SlackOverflow, TheNodeCapacityGrows) {
  knobs_t knobs;
  knobs.node_capacity = 8;
  whgraph h = hypergraph_access::build(
      res(), list_of<std::int32_t>(40, rows{{0, 1}, {1, 2}, {2, 3}}, {5, 6, 7}).view(),
      slack_props(), knobs);
  const auto* slack = hypergraph_access::slack(h);
  EXPECT_EQ(slack->rows().node_capacity, 8);
  for (int batch = 0; batch < 4; ++batch) {
    wbatch_t b;
    for (int k = 0; k < 5; ++k) b.insert_hyperedge({3 + batch * 5 + k, 4 + batch * 5 + k}, 9);
    apply_checked(res(), h, b);
  }
  EXPECT_GT(slack->counters().node_growths, 0);
  EXPECT_GE(slack->rows().node_capacity, h.hyperedge_id_bound());
}

// ---- Over-limit inputs (M11a criterion 17): capacity_error, never undefined behaviour ----------

// (a) More candidate pairs than the limit (MOSP_ESCHER's INT_MAX; planted lower): a capacity_error
// in the store phase, the hypergraph poisoned (the basic guarantee); within the limit nothing
// happens.
TEST_P(SlackOverflow, MoreCandidatesThanTheLimit) {
  knobs_t knobs;
  knobs.max_candidates = 10;
  rows initial;
  std::vector<std::int32_t> w;
  for (int i = 0; i < 20; ++i) {
    initial.push_back({0, i + 1});
    w.push_back(1);
  }
  whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(30, initial, w).view(),
                                       slack_props(), knobs);
  wbatch_t b;
  b.insert_hyperedge({0, 25}, 3);  // 20 neighbours through vertex 0: more than 10 candidates
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);  // poisoned
  knobs.max_candidates = 1000;
  whgraph g = hypergraph_access::build(res(), list_of<std::int32_t>(30, initial, w).view(),
                                       slack_props(), knobs);
  apply_checked(res(), g, b);
}

// (b) Ids past the int32 node ids of the line graph (planted: max_ids 12): a batch whose new ids
// would pass it is a capacity_error (poisoned), never a wrap.
TEST_P(SlackOverflow, IdsPastTheNodeIdLimit) {
  knobs_t knobs;
  knobs.max_ids = 12;
  rows initial;
  std::vector<std::int32_t> w;
  for (int i = 0; i < 10; ++i) {
    initial.push_back({i});
    w.push_back(1);
  }
  whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(20, initial, w).view(),
                                       slack_props(), knobs);
  wbatch_t fits;
  fits.insert_hyperedge({11}, 2);
  fits.insert_hyperedge({12}, 2);
  apply_checked(res(), h, fits);  // ids 10 and 11: the bound is 12
  wbatch_t over;
  over.insert_hyperedge({13}, 2);  // id 12: the bound would be 13
  EXPECT_THROW((void)h.apply(res(), over.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);  // poisoned
  // At construction: more rows than the limit.
  knobs.max_ids = 5;
  EXPECT_THROW((void)hypergraph_access::build(res(), list_of<std::int32_t>(20, initial, w).view(),
                                              slack_props(), knobs),
               dyng::capacity_error);
}

// (c) Entries past the int32 counts (planted: max_entries): the line graph's entries at the build
// (nothing is built) and at a rebuild inside a batch (poisoned); the incidence mirror's values.
TEST_P(SlackOverflow, EntriesPastTheLimit) {
  rows initial;
  std::vector<std::int32_t> w;
  for (int i = 0; i < 30; ++i) {
    initial.push_back({i % 10, 20});
    w.push_back(1);
  }
  knobs_t knobs;
  knobs.max_entries = 1000;  // the build needs 30 slots of 64 (+ 4096)
  EXPECT_THROW((void)hypergraph_access::build(res(), list_of<std::int32_t>(300, initial, w).view(),
                                              slack_props(), knobs),
               dyng::capacity_error);
  // A rebuild over the limit inside a batch.
  knobs.entry_headroom = 1.0;
  knobs.max_entries = 30 * 64 + 4096 + 100;
  whgraph h = hypergraph_access::build(res(), list_of<std::int32_t>(300, initial, w).view(),
                                       slack_props(), knobs);
  wbatch_t b;
  for (int k = 0; k < 200; ++k) b.insert_hyperedge({20, 30 + k}, 1);
  EXPECT_THROW((void)h.apply(res(), b.view()), dyng::capacity_error);
  EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);  // poisoned
}

INSTANTIATE_TEST_SUITE_P(Backends, SlackOverflow, ::testing::ValuesIn(dyng::test::suite_backends()),
                         backend_name);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SlackOverflow);

}  // namespace
