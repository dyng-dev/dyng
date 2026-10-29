// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file vertex_arrival_test.cpp
 * @brief generators::legacy::dynlp_vertex_arrival_stream() (ADR 0067): the libstdc++ 12 twins of
 *        std::shuffle (32-bit engine), std::generate_canonical<float, 24> and
 *        std::uniform_real_distribution<float> against std:: (libstdc++ builds; this file is
 *        compiled with -ffp-contract=off, like the generator), the options, the stream's shape
 *        and its independence of the thread count and of the offset type. The byte identity with
 *        LabelPropagation-CUDA@a276a3a is checked by dyng_lp_replay_tests (the committed 5k5 and
 *        50k digests) and over the whole corpus by parity/compare.py label_propagation generator.
 */
#include "generators/dynlp_stream.hpp"
#include "util/rng.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/generators/vertex_arrival.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace {

namespace gen = dyng::generators;
namespace legacy = dyng::generators::legacy;

/// A 32-bit "engine" that returns a fixed sequence (to reach the >= 1 correction).
struct fixed_engine {
  using result_type = std::uint_fast32_t;
  static constexpr result_type min() {
    return 0;
  }
  static constexpr result_type max() {
    return 0xffffffffU;
  }
  std::vector<result_type> values;
  std::size_t next = 0;
  result_type operator()() {
    return values[next++ % values.size()];
  }
};

#if defined(__GLIBCXX__)
TEST(DynlpDraws, ShuffleMatchesLibstdcxx) {
  for (const std::uint32_t seed : {1U, 42U, 12387U, 4294967295U}) {
    for (const std::size_t n :
         {0UL, 1UL, 2UL, 3UL, 4UL, 1000UL, 5000UL, 65534UL, 65535UL, 65536UL, 65537UL, 100001UL}) {
      SCOPED_TRACE(std::to_string(seed) + " " + std::to_string(n));
      std::mt19937 ours(seed);
      std::mt19937 theirs(seed);
      std::vector<float> a(n);
      for (std::size_t i = 0; i < n; ++i) {
        a[i] = i < n / 2 ? 0.0F : 1.0F;
      }
      std::vector<float> b = a;
      dyng::detail::legacy_shuffle32(a, ours);
      std::shuffle(b.begin(), b.end(), theirs);
      ASSERT_EQ(a, b);
      std::vector<std::int32_t> c(n);
      std::iota(c.begin(), c.end(), 0);
      std::vector<std::int32_t> d = c;
      dyng::detail::legacy_shuffle32(c, ours);
      std::shuffle(d.begin(), d.end(), theirs);
      ASSERT_EQ(c, d);
      ASSERT_EQ(ours(), theirs());  // the same number of engine calls
    }
  }
}

TEST(DynlpDraws, UniformRealFloatMatchesLibstdcxx) {
  for (const std::uint32_t seed : {1U, 42U, 12387U}) {
    std::mt19937 ours(seed);
    std::mt19937 theirs(seed);
    std::uniform_real_distribution<float> same(0.5F, 1.0F);
    std::uniform_real_distribution<float> cross(0.01F, 0.5F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    for (int i = 0; i < 200000; ++i) {
      ASSERT_EQ(dyng::detail::legacy_uniform_real_float(ours, 0.5F, 1.0F), same(theirs));
      ASSERT_EQ(dyng::detail::legacy_uniform_real_float(ours, 0.01F, 0.5F), cross(theirs));
      ASSERT_EQ(dyng::detail::legacy_uniform_real_float(ours, 0.0F, 1.0F), unit(theirs));
      ASSERT_EQ(dyng::detail::legacy_generate_canonical_float(ours),
                (std::generate_canonical<float, 24>(theirs)));
    }
  }
}

TEST(DynlpDraws, CanonicalCorrectionAtTheTop) {
  // Engine values that round to 2^32 in float: libstdc++ returns nextafter(1.0f, 0.0f).
  for (const std::uint32_t x :
       {0xffffffffU, 0xffffff80U, 0xffffff7fU, 0xfffffe00U, 0U, 1U, 0x80000000U}) {
    fixed_engine ours{{x}};
    fixed_engine theirs{{x}};
    const float a = dyng::detail::legacy_generate_canonical_float(ours);
    const float b = std::generate_canonical<float, 24>(theirs);
    EXPECT_EQ(a, b) << x;
    EXPECT_LT(a, 1.0F);
  }
}

TEST(DynlpDraws, StandardLibraryCheckIncludesDynlp) {
  EXPECT_TRUE(dyng::detail::dynlp_draws_match_standard_library());
  EXPECT_TRUE(legacy::standard_library_matches_legacy_streams());
}
#else
TEST(DynlpDraws, MatchLibstdcxx) {
  // The twins reproduce libstdc++; other standard libraries draw differently (ADR 0067).
  EXPECT_FALSE(dyng::detail::dynlp_draws_match_standard_library());
  GTEST_SKIP() << "not a libstdc++ build: the twins are compared with std:: on libstdc++ only";
}
#endif

TEST(VertexArrivalStream, RejectsInvalidOptions) {
  const auto res = dyng::resources::sequential();
  const auto bad = [&](auto mutate) {
    gen::vertex_arrival_options o;
    o.total_vertices = 100;
    mutate(o);
    EXPECT_THROW((void)legacy::dynlp_vertex_arrival_stream(res, o), dyng::invalid_argument_error);
  };
  bad([](gen::vertex_arrival_options& o) { o.total_vertices = 1; });
  bad([](gen::vertex_arrival_options& o) { o.total_vertices = std::int64_t{1} << 31; });
  bad([](gen::vertex_arrival_options& o) { o.average_degree = 0; });
  bad([](gen::vertex_arrival_options& o) { o.num_batches = 0; });
  bad([](gen::vertex_arrival_options& o) { o.ground_truth_fraction = -0.1F; });
  bad([](gen::vertex_arrival_options& o) { o.deleted_fraction = 1.5F; });
  // The original's 32-bit offsets: 2 * target edges > 2^31 - 1.
  bad([](gen::vertex_arrival_options& o) {
    o.total_vertices = 1000000000;
    o.average_degree = 5;
  });
}

gen::vertex_arrival_options small_options() {
  gen::vertex_arrival_options o;
  o.total_vertices = 3001;
  o.average_degree = 5;
  o.num_batches = 4;
  o.seed = 7;
  return o;
}

TEST(VertexArrivalStream, ShapeOfTheStream) {
  const auto s =
      legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), small_options());
  const auto& g = s.graph;
  const std::int32_t n = g.num_vertices();
  ASSERT_EQ(n, 3001);
  ASSERT_EQ(g.num_weights, 1);
  ASSERT_EQ(g.weights.size(), g.col_ind.size());
  ASSERT_EQ(s.true_labels.size(), 3001U);
  ASSERT_EQ(s.row_capacity.size(), 3001U);
  // Balanced classes; rows sorted and duplicate-free; symmetric with equal weights.
  EXPECT_EQ(std::count(s.true_labels.begin(), s.true_labels.end(), 0), 1500);
  std::set<std::pair<std::int32_t, std::int32_t>> edges;
  double sum = 0.0;
  for (std::int32_t u = 0; u < n; ++u) {
    EXPECT_EQ(s.row_capacity[static_cast<std::size_t>(u)], g.row_ptr[u + 1] - g.row_ptr[u]);
    for (std::int32_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
      if (e > g.row_ptr[u]) {
        EXPECT_LT(g.col_ind[e - 1], g.col_ind[e]);
      }
      const std::int32_t v = g.col_ind[e];
      const float w = g.weights[e];
      sum += w;
      EXPECT_NE(u, v);
      const bool same =
          s.true_labels[static_cast<std::size_t>(u)] == s.true_labels[static_cast<std::size_t>(v)];
      EXPECT_TRUE(same ? (w >= 0.5F && w <= 1.0F) : (w >= 0.01F && w <= 0.5F)) << w;
      if (u < v) {
        edges.insert({u, v});
      }
    }
  }
  EXPECT_EQ(edges.size() * 2, g.col_ind.size());
  EXPECT_EQ(edges.size(), 3001U * 5 / 2);
  EXPECT_EQ(s.similarity_threshold,
            static_cast<float>(sum / static_cast<double>(g.col_ind.size())));
  // Batches: every vertex arrives once; b0 is balanced ground truth; edges u < v of the full
  // graph whose later endpoint arrives in the batch; departures are earlier unlabelled arrivals.
  ASSERT_EQ(s.batches.size(), 4U);
  std::vector<int> arrival(static_cast<std::size_t>(n), -1);
  std::size_t max_v = 0;
  std::size_t max_e = 0;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    const auto& batch = s.batches[b];
    ASSERT_EQ(batch.insert_vertex_labels.size(), batch.insert_vertices.size());
    for (std::size_t i = 0; i < batch.insert_vertices.size(); ++i) {
      const std::int32_t v = batch.insert_vertices[i];
      ASSERT_EQ(arrival[static_cast<std::size_t>(v)], -1);
      arrival[static_cast<std::size_t>(v)] = static_cast<int>(b);
      const std::int8_t l = batch.insert_vertex_labels[i];
      ASSERT_TRUE(l == -1 || l == s.true_labels[static_cast<std::size_t>(v)]);
      if (b == 0) {
        ASSERT_NE(l, -1);
      }
    }
    max_v = std::max(max_v, batch.insert_vertices.size());
    max_e = std::max(max_e, batch.edge_sources.size());
  }
  EXPECT_EQ(std::count(arrival.begin(), arrival.end(), -1), 0);
  EXPECT_EQ(s.batches[0].insert_vertices.size(), 30U);  // max(2, int(3001 * 0.01f))
  std::size_t total_edges = 0;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    const auto& batch = s.batches[b];
    ASSERT_EQ(batch.edge_destinations.size(), batch.edge_sources.size());
    ASSERT_EQ(batch.edge_weights.size(), batch.edge_sources.size());
    for (std::size_t e = 0; e < batch.edge_sources.size(); ++e) {
      const std::int32_t u = batch.edge_sources[e];
      const std::int32_t v = batch.edge_destinations[e];
      ASSERT_LT(u, v);
      ASSERT_TRUE(edges.count({u, v}) == 1);
      ASSERT_EQ(
          std::max(arrival[static_cast<std::size_t>(u)], arrival[static_cast<std::size_t>(v)]),
          static_cast<int>(b));
    }
    total_edges += batch.edge_sources.size();
    for (const std::int32_t v : batch.delete_vertices) {
      ASSERT_LE(arrival[static_cast<std::size_t>(v)], static_cast<int>(b));
    }
    if (b > 0) {
      // int(arrivals * 0.09f) departures.
      EXPECT_EQ(batch.delete_vertices.size(),
                static_cast<std::size_t>(static_cast<float>(batch.insert_vertices.size()) * 0.09F));
    }
  }
  EXPECT_EQ(total_edges, edges.size());
  EXPECT_EQ(s.max_batch_vertices, max_v);
  EXPECT_EQ(s.max_batch_edges, max_e);
}

template <typename a_t, typename b_t>
void expect_same_stream(const gen::vertex_arrival_stream<std::int32_t, a_t>& a,
                        const gen::vertex_arrival_stream<std::int32_t, b_t>& b) {
  ASSERT_EQ(a.graph.row_ptr.size(), b.graph.row_ptr.size());
  for (std::size_t i = 0; i < a.graph.row_ptr.size(); ++i) {
    ASSERT_EQ(static_cast<std::int64_t>(a.graph.row_ptr[i]),
              static_cast<std::int64_t>(b.graph.row_ptr[i]));
  }
  EXPECT_EQ(a.graph.col_ind, b.graph.col_ind);
  EXPECT_EQ(a.graph.weights, b.graph.weights);
  EXPECT_EQ(a.true_labels, b.true_labels);
  EXPECT_EQ(a.similarity_threshold, b.similarity_threshold);
  ASSERT_EQ(a.batches.size(), b.batches.size());
  for (std::size_t k = 0; k < a.batches.size(); ++k) {
    EXPECT_EQ(a.batches[k].insert_vertices, b.batches[k].insert_vertices);
    EXPECT_EQ(a.batches[k].insert_vertex_labels, b.batches[k].insert_vertex_labels);
    EXPECT_EQ(a.batches[k].edge_sources, b.batches[k].edge_sources);
    EXPECT_EQ(a.batches[k].edge_destinations, b.batches[k].edge_destinations);
    EXPECT_EQ(a.batches[k].edge_weights, b.batches[k].edge_weights);
    EXPECT_EQ(a.batches[k].delete_vertices, b.batches[k].delete_vertices);
  }
}

TEST(VertexArrivalStream, IndependentOfThreadsAndOffsetType) {
  auto o = small_options();
  o.total_vertices = 40000;  // more rows than the 1024 chunks of the edge lists
  o.num_batches = 7;
  const auto seq = legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  const auto wide = legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int64_t>(
      dyng::resources::sequential(), o);
  expect_same_stream(seq, wide);
#if DYNG_HAS_OPENMP
  for (const int t : {2, 8}) {
    const auto par = legacy::dynlp_vertex_arrival_stream(dyng::resources::openmp(t), o);
    expect_same_stream(seq, par);
  }
#endif
  // The internal arrays agree with the public stream.
  const auto raw = dyng::detail::dynlp_legacy_generate(o, 1);
  ASSERT_EQ(raw.batches.size(), seq.batches.size());
  EXPECT_EQ(raw.tau, seq.similarity_threshold);
  for (std::size_t b = 0; b < raw.batches.size(); ++b) {
    EXPECT_EQ(raw.batches[b].inserted, seq.batches[b].insert_vertices);
    EXPECT_EQ(raw.batches[b].deleted, seq.batches[b].delete_vertices);
    std::size_t labelled = 0;
    for (const std::int8_t l : seq.batches[b].insert_vertex_labels) {
      labelled += l >= 0 ? 1 : 0;
    }
    EXPECT_EQ(labelled, raw.batches[b].gt_ids.size());
  }
}

TEST(VertexArrivalStream, OneBatchAndSmallGraphs) {
  // With one batch only b0 exists: the remaining vertices never arrive (as in the original).
  gen::vertex_arrival_options o;
  o.total_vertices = 500;
  o.num_batches = 1;
  const auto s = legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  ASSERT_EQ(s.batches.size(), 1U);
  EXPECT_EQ(s.batches[0].insert_vertices.size(),
            4U);  // max(2, int(500 * 0.01f)) = 5 -> 2 per class
  // Two vertices: the spanning tree is the only edge.
  o.total_vertices = 2;
  o.num_batches = 3;
  const auto t = legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  EXPECT_EQ(t.graph.col_ind.size(), 2U);
}

}  // namespace
