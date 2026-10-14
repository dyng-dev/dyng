// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_random_test.cpp
 * @brief Random batch streams on every backend, in lockstep with the host model of the ESCHER
 *        store (M8b, criteria 5 and 6): after every batch the inserted ids, h2v rows by id and
 *        v2h rows equal the model's, h2h equals dyng::testing::rebuild_line_graph (line_graph_policy::
 *        sorted_merge), check_integrity passes, and every backend gives the same ids, line graphs,
 *        rows and summaries (C3). The mixes: sparse, dense, skewed (cardinality >= 1,024), many
 *        equal sizes, delete and re-insert in one batch, delete-all and re-insert, a batch larger
 *        than the hypergraph, deletions of ids at the limits; under ESCHER-GPU's and
 *        MOSP_ESCHER's presets and dynG's default (with a small slack, so that it compacts and
 *        grows), and both id_reuse orders.
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::capacity_policy;
using dyng::id_reuse;
using dyng::test::hyper::rows;
using dyng::test::hyper::stream_mix;

struct preset_case {
  std::string name;   ///< for the trace
  std::string model;  ///< the model's preset
  capacity_policy capacity;
  std::vector<stream_mix> mixes;
};

std::vector<stream_mix> all_mixes() {
  return {stream_mix::sparse,      stream_mix::dense,           stream_mix::skewed,
          stream_mix::equal_sizes, stream_mix::delete_reinsert, stream_mix::delete_all,
          stream_mix::larger_than, stream_mix::limits,          stream_mix::reissue};
}

std::vector<preset_case> cases() {
  capacity_policy small_default;  // dynG's default with a small slack: compacts and grows
  small_default.initial_slack = 16;
  return {
      {"escher_gpu", "escher_gpu", capacity_policy::escher_gpu(), all_mixes()},
      {"mosp_escher", "mosp_escher", capacity_policy::mosp_escher(1 << 14), all_mixes()},
      // MOSP_ESCHER with a small fixed payload: H2V compacts at its trigger point (and may run
      // out, which every backend reports the same way). Low-churn mixes only: the host store has
      // no V2H payload limit (ADR 0041, section 5).
      {"mosp_escher_small",
       "mosp_escher",
       capacity_policy::mosp_escher(48),
       {stream_mix::sparse, stream_mix::equal_sizes, stream_mix::delete_reinsert,
        stream_mix::limits, stream_mix::reissue}},
      {"dyng_default_small", "dyng_default", small_default, all_mixes()},
  };
}

// One stream on one backend: per batch the summary's ids and counts and the h2v rows.
struct stream_record {
  std::vector<std::vector<std::int64_t>> ids;
  std::vector<std::vector<std::int64_t>> counts;
  std::vector<rows> h2v;
  std::vector<dyng::test::hyper::id_rows> h2h;
  std::vector<std::vector<std::int32_t>> h2h_payload;  // the line graph's layout (ESCHER-GPU's)
  bool failed = false;
};

template <typename weight_t>
stream_record run_stream(backend b, const preset_case& pc, id_reuse reuse, stream_mix mix,
                         std::uint64_t seed) {
  using hgraph = dyng::hypergraph<std::int32_t, weight_t>;
  const dyng::resources res = dyng::test::hyper::resources_of(b);
  std::mt19937_64 rng(seed);
  const std::int32_t n = 1200;
  const rows initial = dyng::test::hyper::random_rows(rng, n, 60, 8);
  std::vector<std::int32_t> w;
  for (std::size_t i = 0; i < initial.size(); ++i) w.push_back(static_cast<std::int32_t>(i + 1));
  dyng::hypergraph_properties props;
  // The line graph is maintained by every batch and checked against the oracle (criterion 6).
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  props.reuse = reuse;
  props.capacity = pc.capacity;
  std::vector<weight_t> ww;
  if constexpr (!dyng::is_unweighted_v<weight_t>) ww.assign(w.begin(), w.end());
  const auto list = dyng::test::hyper::list_of<weight_t>(n, initial, ww);
  hgraph h = hgraph::from_hyperedges(res, list.view(), props);
  dyng::test::hyper::lockstep model(n, initial, props, pc.model);
  EXPECT_TRUE(model.problems().empty());
  EXPECT_EQ(model.compare(h), "");
  stream_record rec;
  // 6 batches per stream (3 under compute-sanitizer, DYNG_CBST_REDUCED).
  const int batches = dyng::test::hyper::reduced_sizes() ? 3 : 6;
  for (int batch = 0; batch < batches; ++batch) {
    SCOPED_TRACE("batch " + std::to_string(batch));
    rows current;
    const auto live = dyng::test::hyper::live_ids(h, current);
    const auto tb =
        dyng::test::hyper::next_batch(rng, mix, n, live, h.hyperedge_id_bound(), current);
    const auto expected = model.apply(tb);
    const auto bv = dyng::test::hyper::batch_of<weight_t>(tb);
    if (!expected.ok) {
      // The store's payload is exhausted: every backend fails the same way and is poisoned.
      EXPECT_THROW((void)h.apply(res, bv.view()), dyng::capacity_error);
      EXPECT_THROW((void)h.h2v(), dyng::invalid_argument_error);
      rec.failed = true;
      return rec;
    }
    const auto s = h.apply(res, bv.view());
    EXPECT_EQ(s.inserted_ids, expected.ids);
    EXPECT_EQ(s.ignored_deletions, expected.ignored);
    EXPECT_EQ(s.dropped_members, expected.dropped);
    EXPECT_EQ(model.compare(h), "");
    EXPECT_EQ(dyng::test::hyper::check_line_graph(h), "");
    h.check_integrity(res);
    {  // The applied record's touched sets, by brute force (they are built from the batch).
      const auto& a = dyng::detail::hypergraph_access::applied(h);
      std::vector<std::int64_t> touched_ids = a.deleted_ids;
      touched_ids.insert(touched_ids.end(), s.inserted_ids.begin(), s.inserted_ids.end());
      std::sort(touched_ids.begin(), touched_ids.end());
      touched_ids.erase(std::unique(touched_ids.begin(), touched_ids.end()), touched_ids.end());
      EXPECT_EQ(a.touched_hyperedges, touched_ids);
      std::vector<std::int32_t> touched = a.deleted_members.members;
      const rows now = dyng::test::hyper::h2v_rows(h);
      for (const std::int64_t id : s.inserted_ids) {
        const auto& r = now[static_cast<std::size_t>(id)];
        touched.insert(touched.end(), r.begin(), r.end());
      }
      std::sort(touched.begin(), touched.end());
      touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
      EXPECT_EQ(a.touched_vertices, touched);
    }
    rec.ids.push_back(s.inserted_ids);
    rec.h2h.push_back(dyng::test::hyper::h2h_rows(h));
    rec.h2h_payload.push_back(dyng::detail::hypergraph_access::store(h).line_graph().payload);
    rec.counts.push_back({s.inserted_hyperedges, s.deleted_hyperedges, s.skipped_ops,
                          s.ignored_deletions, s.dropped_members, s.num_hyperedges_after,
                          s.id_bound_after});
    rec.h2v.push_back(dyng::test::hyper::h2v_rows(h));
    if constexpr (!dyng::is_unweighted_v<weight_t>) {
      // Weights by id: the inserted ones, 0 for dead ids.
      const auto weights = dyng::test::host_copy(h.weights());
      for (std::size_t i = 0; i < s.inserted_ids.size(); ++i)
        EXPECT_EQ(weights[static_cast<std::size_t>(s.inserted_ids[i])], tb.weights[i]);
    }
    if (::testing::Test::HasFailure()) return rec;
  }
  return rec;
}

template <typename weight_t>
void run_all(std::uint64_t seed_base) {
  const auto backends = dyng::test::comparison_backends();
  std::uint64_t seed = seed_base;
  for (const preset_case& pc : cases()) {
    for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
      for (const stream_mix mix : pc.mixes) {
        ++seed;
        stream_record reference;
        bool first = true;
        for (const backend b : backends) {
          SCOPED_TRACE(pc.name + " " + std::string(dyng::to_string(reuse)) + " " +
                       dyng::test::hyper::mix_name(mix) + " " + std::string(dyng::to_string(b)) +
                       " seed " + std::to_string(seed));
          const stream_record r = run_stream<weight_t>(b, pc, reuse, mix, seed);
          if (::testing::Test::HasFatalFailure()) return;
          if (first) {
            reference = r;
            first = false;
          } else {
            // C3: identical inserted ids, rows and summaries on every backend.
            EXPECT_EQ(r.failed, reference.failed);
            EXPECT_EQ(r.ids, reference.ids);
            EXPECT_EQ(r.counts, reference.counts);
            EXPECT_EQ(r.h2v, reference.h2v);
            EXPECT_EQ(r.h2h, reference.h2h);
            EXPECT_EQ(r.h2h_payload, reference.h2h_payload);
          }
          if (::testing::Test::HasFailure()) return;
        }
      }
    }
  }
}

TEST(HypergraphRandom, UnweightedStreamsMatchTheModelOnEveryBackend) {
  run_all<dyng::unweighted>(100);
}

TEST(HypergraphRandom, WeightedStreamsMatchTheModelOnEveryBackend) {
  run_all<std::int32_t>(500);
}

// The openmp backend gives the same ids and rows with 1 and 8 threads.
TEST(HypergraphRandom, OpenmpThreadCountsAgree) {
  if (!dyng::backend_available(backend::openmp)) GTEST_SKIP() << "no openmp backend";
  using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  std::vector<std::vector<std::int64_t>> ids[2];
  std::vector<rows> h2v[2];
  for (int t = 0; t < 2; ++t) {
    const dyng::resources res = dyng::resources::openmp(t == 0 ? 1 : 8);
    std::mt19937_64 rng(77);
    const rows initial = dyng::test::hyper::random_rows(rng, 3000, 2000, 12);
    dyng::hypergraph_properties props;
    props.line_graph = dyng::line_graph_policy::none;
    hgraph h =
        hgraph::from_hyperedges(res, dyng::test::hyper::list_of(3000, initial).view(), props);
    for (int batch = 0; batch < 4; ++batch) {
      rows current;
      const auto live = dyng::test::hyper::live_ids(h, current);
      const auto tb = dyng::test::hyper::next_batch(rng, stream_mix::dense, 3000, live,
                                                    h.hyperedge_id_bound(), current);
      ids[t].push_back(h.apply(res, dyng::test::hyper::batch_of(tb).view()).inserted_ids);
      h.check_integrity(res);
      h2v[t].push_back(dyng::test::hyper::h2v_rows(h));
    }
  }
  EXPECT_EQ(ids[0], ids[1]);
  EXPECT_EQ(h2v[0], h2v[1]);
}

}  // namespace
