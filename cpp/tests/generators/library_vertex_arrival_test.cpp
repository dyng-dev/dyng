// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file library_vertex_arrival_test.cpp
 * @brief generators::generate_vertex_arrival_stream() (ADR 0107): the stream's shape, its
 *        committed platform-independent digests (the same on every CI toolchain: this file runs in
 *        the CPU suites of the local GCC 12.2 build and the hosted gcc-13 and clang-18 jobs), its
 *        independence of the thread count and the offset type, and that it is not the legacy
 *        stream (whose digests stay those of LabelPropagation-CUDA, dyng_lp_replay_tests).
 */
#include "support/sha256.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace gen = dyng::generators;

/// A canonical serialization of a stream: every array in a fixed order, the offsets as int64 (so
/// the digest does not depend on edge_t), the floats as their bits, each array preceded by its
/// length.
template <typename edge_t>
std::string digest(const gen::vertex_arrival_stream<std::int32_t, edge_t>& s) {
  dyng::test::sha256 h;
  const auto put_size = [&](std::size_t n) {
    const auto v = static_cast<std::uint64_t>(n);
    h.update(&v, sizeof(v));
  };
  const auto put = [&](const auto& v) {
    put_size(v.size());
    if (!v.empty()) {
      h.update(v.data(), v.size() * sizeof(v[0]));
    }
  };
  std::vector<std::int64_t> row_ptr(s.graph.row_ptr.begin(), s.graph.row_ptr.end());
  std::vector<std::int64_t> capacity(s.row_capacity.begin(), s.row_capacity.end());
  put(row_ptr);
  put(s.graph.col_ind);
  put(s.graph.weights);
  put(s.true_labels);
  put(capacity);
  put_size(s.batches.size());
  for (const auto& b : s.batches) {
    put(b.insert_vertices);
    put(b.insert_vertex_labels);
    put(b.edge_sources);
    put(b.edge_destinations);
    put(b.edge_weights);
    put(b.delete_vertices);
  }
  std::uint32_t tau = 0;
  std::memcpy(&tau, &s.similarity_threshold, sizeof(tau));
  h.update(&tau, sizeof(tau));
  return h.hex();
}

gen::vertex_arrival_options options(std::int64_t n, std::int32_t batches, std::uint32_t seed,
                                    std::int32_t degree = 5) {
  gen::vertex_arrival_options o;
  o.total_vertices = n;
  o.num_batches = batches;
  o.seed = seed;
  o.average_degree = degree;
  return o;
}

/// The committed digests (ADR 0107): a change of a draw, of its order or of the structure changes
/// them, and is a change of the stream for the library's major version.
struct golden {
  const char* name;
  gen::vertex_arrival_options opt;
  const char* sha256;
};

const golden goldens[] = {
    {"5k5_s42", options(5000, 5, 42),
     "af4e0342a312b161ebba6f172df790de7a6e314f262fab6c31e38aa3918b6a72"},
    {"5k5_s1_d3", options(5000, 5, 1, 3),
     "31c2e8bb92c7f3e2122e2ebb4fadfb2bf7114a1bbcab71657d766d78a1106e3f"},
    {"50k_s42", options(50000, 10, 42),
     "fd3b595743d1018b91792ae7692b1d44d66ee4f61a8cb690a402f15682fc9dae"},
    {"3001x4_s7_d7", options(3001, 4, 7, 7),
     "9a307ebb8d9437892d29f633b4ad201025bca26e844696a55f6794528d0db8c1"},
};

TEST(LibraryVertexArrivalStream, CommittedDigests) {
  for (const golden& g : goldens) {
    SCOPED_TRACE(g.name);
    const auto s = gen::generate_vertex_arrival_stream(dyng::resources::sequential(), g.opt);
    EXPECT_EQ(digest(s), g.sha256);
  }
}

TEST(LibraryVertexArrivalStream, IndependentOfThreadsAndOffsetType) {
  const auto o = options(40000, 7, 3);  // more rows than the 1024 chunks of the edge lists
  const std::string seq =
      digest(gen::generate_vertex_arrival_stream(dyng::resources::sequential(), o));
  EXPECT_EQ(digest(gen::generate_vertex_arrival_stream<std::int32_t, std::int64_t>(
                dyng::resources::sequential(), o)),
            seq);
#if DYNG_HAS_OPENMP
  for (const int t : {2, 8, 13}) {
    SCOPED_TRACE(t);
    EXPECT_EQ(digest(gen::generate_vertex_arrival_stream(dyng::resources::openmp(t), o)), seq);
  }
#endif
}

TEST(LibraryVertexArrivalStream, ShapeOfTheStream) {
  const auto o = options(3001, 4, 11);
  const auto s = gen::generate_vertex_arrival_stream(dyng::resources::sequential(), o);
  const auto& g = s.graph;
  const std::int32_t n = g.num_vertices();
  ASSERT_EQ(n, 3001);
  ASSERT_EQ(s.true_labels.size(), 3001U);
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
  ASSERT_EQ(s.batches.size(), 4U);
  std::vector<int> arrival(static_cast<std::size_t>(n), -1);
  std::size_t total_edges = 0;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    const auto& batch = s.batches[b];
    ASSERT_EQ(batch.insert_vertex_labels.size(), batch.insert_vertices.size());
    std::size_t labelled = 0;
    for (std::size_t i = 0; i < batch.insert_vertices.size(); ++i) {
      const std::int32_t v = batch.insert_vertices[i];
      ASSERT_EQ(arrival[static_cast<std::size_t>(v)], -1);
      arrival[static_cast<std::size_t>(v)] = static_cast<int>(b);
      const std::int8_t l = batch.insert_vertex_labels[i];
      ASSERT_TRUE(l == -1 || l == s.true_labels[static_cast<std::size_t>(v)]);
      labelled += l >= 0 ? 1 : 0;
    }
    if (b == 0) {
      EXPECT_EQ(batch.insert_vertices.size(), 30U);  // max(2, int(3001 * 0.01f)), both classes
      EXPECT_EQ(labelled, 30U);
    } else {
      EXPECT_EQ(labelled, std::max<std::size_t>(
                              1, static_cast<std::size_t>(
                                     static_cast<float>(batch.insert_vertices.size()) * 0.01F)));
      EXPECT_EQ(batch.delete_vertices.size(),
                static_cast<std::size_t>(static_cast<float>(batch.insert_vertices.size()) * 0.09F));
    }
  }
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    const auto& batch = s.batches[b];
    for (std::size_t e = 0; e < batch.edge_sources.size(); ++e) {
      const std::int32_t u = batch.edge_sources[e];
      const std::int32_t v = batch.edge_destinations[e];
      ASSERT_LT(u, v);
      ASSERT_EQ(edges.count({u, v}), 1U);
      ASSERT_EQ(
          std::max(arrival[static_cast<std::size_t>(u)], arrival[static_cast<std::size_t>(v)]),
          static_cast<int>(b));
    }
    total_edges += batch.edge_sources.size();
    for (const std::int32_t v : batch.delete_vertices) {
      ASSERT_LE(arrival[static_cast<std::size_t>(v)], static_cast<int>(b));
    }
  }
  EXPECT_EQ(std::count(arrival.begin(), arrival.end(), -1), 0);
  EXPECT_EQ(total_edges, edges.size());
}

TEST(LibraryVertexArrivalStream, IsNotTheLegacyStream) {
  const auto o = options(5000, 5, 42);
  const auto lib = gen::generate_vertex_arrival_stream(dyng::resources::sequential(), o);
  const auto old = gen::legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  EXPECT_NE(digest(lib), digest(old));
  EXPECT_EQ(lib.graph.col_ind.size(), old.graph.col_ind.size());  // the same model
  // Different seeds give different streams.
  EXPECT_NE(digest(gen::generate_vertex_arrival_stream(dyng::resources::sequential(),
                                                       options(5000, 5, 43))),
            digest(lib));
}

TEST(LibraryVertexArrivalStream, RejectsInvalidOptions) {
  const auto res = dyng::resources::sequential();
  for (auto o : {options(1, 5, 1), options(5000, 0, 1), options(5000, 5, 1, 0)}) {
    EXPECT_THROW((void)gen::generate_vertex_arrival_stream(res, o), dyng::invalid_argument_error);
  }
  auto o = options(5000, 5, 1);
  o.ground_truth_fraction = 1.5F;
  EXPECT_THROW((void)gen::generate_vertex_arrival_stream(res, o), dyng::invalid_argument_error);
  o = options(5000, 5, 1);
  o.deleted_fraction = -0.1F;
  EXPECT_THROW((void)gen::generate_vertex_arrival_stream(res, o), dyng::invalid_argument_error);
}

}  // namespace
