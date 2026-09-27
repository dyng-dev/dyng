// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_apply_fixture_test.cpp
 * @brief Byte parity of graph apply and the MOSP readers/writers with MOSP-OpenMP@c352151.
 *
 * The fixtures in cpp/tests/data/mosp_graph_io were produced by the pinned original
 * (parity/fixtures/graph_io/make_graph_io_fixtures.sh): applyChangeBatch() + writeCsrGraph() for
 * the updated CSR, its weightIncreaseMask, transposeCsrGraph() for the in-edges, updateGraphCSR()
 * for the 10 generateTestCases cases, and `mospPrep mtx2csr` for the seeded weights.
 */
#include "graph/graph_impl.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/matrix_market.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace {

using dyng::test::data_path;
using dyng::test::read_lines;
using dyng::test::read_text;

const std::string fixture_dir = "mosp_graph_io/";

/// Expects the three triplet files at `got` and `expected` to be byte-identical.
void expect_same_triplet(const std::string& got, const std::string& expected) {
  for (const char* suffix : {"RowPtr.txt", "ColInd.txt", "Values.txt"}) {
    const std::string a = read_text(got + suffix);
    const std::string b = read_text(expected + suffix);
    EXPECT_FALSE(b.empty() && std::string(suffix) == "RowPtr.txt") << expected << suffix;
    EXPECT_TRUE(a == b) << got << suffix << " differs from " << expected << suffix;
  }
}

template <typename graph_t>
class MospApplyFixture : public ::testing::Test {};

using graph_types = ::testing::Types<dyng::graph<std::int32_t, std::int32_t, std::int32_t>,
                                     dyng::graph<std::int32_t, std::int64_t, std::int32_t>,
                                     dyng::graph<std::int64_t, std::int64_t, std::int32_t>>;
TYPED_TEST_SUITE(MospApplyFixture, graph_types);

TYPED_TEST(MospApplyFixture, UpdatedCsrIsByteEqualToApplyChangeBatch) {
  using graph_t = TypeParam;
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  const auto res = dyng::resources::sequential();
  const std::vector<std::string> cases = read_lines(data_path(fixture_dir + "cases.txt"));
  ASSERT_EQ(cases.size(), 27u);
  dyng::test::temp_dir tmp;
  for (const std::string& name : cases) {
    SCOPED_TRACE(name);
    const std::string dir = data_path(fixture_dir + name + "/");
    const auto original = dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(dir + "graphCsr");
    const int K = original.num_weights;
    auto g = graph_t::from_csr(res, original.view(), dyng::graph_properties::mosp_compatible());
    // The reader + writer reproduce the input files.
    dyng::io::write_csr_triplet(tmp.path(name + "/original/graphCsr"), g.view().out);
    expect_same_triplet(tmp.path(name + "/original/graphCsr"), dir + "graphCsr");

    dyng::io::legacy_batch_options options;
    options.num_weights = K;
    options.num_vertices = g.num_vertices();
    const auto batch = dyng::io::read_legacy_batch<vertex_t, weight_t>(dir + "insert.txt",
                                                                       dir + "delete.txt", options);
    if (name != "h4") {  // h4 has blank lines and trailing blanks on purpose
      dyng::io::write_legacy_batch(tmp.path(name + "/insert.txt"), tmp.path(name + "/delete.txt"),
                                   batch.view());
      EXPECT_EQ(read_text(tmp.path(name + "/insert.txt")), read_text(dir + "insert.txt"));
      EXPECT_EQ(read_text(tmp.path(name + "/delete.txt")), read_text(dir + "delete.txt"));
    }

    dyng::detail::apply_delta<vertex_t> delta;
    (void)dyng::detail::graph_access::apply(res, g, batch.view(), &delta);
    g.check_integrity(res);
    EXPECT_EQ(g.version(), 1u);

    // The updated CSR, byte for byte (edge order and parallel edges included).
    dyng::io::write_csr_triplet(tmp.path(name + "/applied/graphCsr"), g.view().out);
    expect_same_triplet(tmp.path(name + "/applied/graphCsr"), dir + "applied/graphCsr");
    // The in-edges, byte for byte with transposeCsrGraph().
    dyng::io::write_csr_triplet(tmp.path(name + "/applied/graphCsrTransposed"), g.view().in);
    expect_same_triplet(tmp.path(name + "/applied/graphCsrTransposed"),
                        dir + "applied/graphCsrTransposed");
    // The weight-increase classification (weightIncreaseMask bits).
    const auto lines = read_lines(dir + "applied/graphCsrWeightIncrease.txt");
    ASSERT_EQ(lines.size(), delta.insert_src.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
      std::string flags;
      for (int k = 0; k < K; ++k) {
        flags += std::to_string(
            delta.weight_increased[i * static_cast<std::size_t>(K) + static_cast<std::size_t>(k)]);
        if (k + 1 < K) {
          flags += ' ';
        }
      }
      EXPECT_EQ(flags, lines[i]) << "insertion " << i;
    }
    EXPECT_EQ(delta.delete_src.size(), batch.num_deletions());
  }
}

TYPED_TEST(MospApplyFixture, TestCasesMatchUpdateGraphCsrAsEdgeSets) {
  using graph_t = TypeParam;
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  const auto res = dyng::resources::sequential();
  for (int i = 0; i < 10; ++i) {
    const std::string name = "testCase" + std::to_string(i);
    SCOPED_TRACE(name);
    const std::string dir = data_path(fixture_dir + name + "/");
    const auto original = dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(dir + "graphCsr");
    auto g = graph_t::from_csr(res, original.view(), dyng::graph_properties::mosp_compatible());
    dyng::io::legacy_batch_options options;
    options.num_weights = original.num_weights;
    options.num_vertices = g.num_vertices();
    const auto batch = dyng::io::read_legacy_batch<vertex_t, weight_t>(dir + "insert.txt",
                                                                       dir + "delete.txt", options);
    (void)g.apply(res, batch.view());
    // updateGraphCSR() writes sorted rows, so it agrees with applyChangeBatch() as an edge set
    // (the comparison mospTest's "apply" group makes).
    const auto theirs =
        dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(dir + "updateGraphCSR/graphCsr");
    const auto mine = g.to_csr(res);
    const auto multiset = [](const dyng::csr<vertex_t, edge_t, weight_t>& c) {
      std::vector<std::tuple<std::int64_t, std::int64_t, std::vector<std::int64_t>>> out;
      for (std::size_t u = 0; u + 1 < c.row_ptr.size(); ++u) {
        for (auto e = static_cast<std::size_t>(c.row_ptr[u]);
             e < static_cast<std::size_t>(c.row_ptr[u + 1]); ++e) {
          std::vector<std::int64_t> w;
          for (int k = 0; k < c.num_weights; ++k) {
            w.push_back(c.weight(static_cast<edge_t>(e), k));
          }
          out.emplace_back(static_cast<std::int64_t>(u), static_cast<std::int64_t>(c.col_ind[e]),
                           w);
        }
      }
      std::sort(out.begin(), out.end());
      return out;
    };
    EXPECT_EQ(multiset(mine), multiset(theirs));
    EXPECT_EQ(mine.row_ptr, theirs.row_ptr);
  }
}

TEST(MospPrepFixture, Mtx2CsrWeightsAreBitExact) {
  const auto res = dyng::resources::sequential();
  struct prep_case {
    const char* mtx;
    const char* expected;
    int k;
    std::int64_t wmax;
    std::uint32_t seed;
  };
  const prep_case cases[] = {
      {"m0_symmetric.mtx", "m0_k3_seed12345_", 3, 100, 12345},
      {"m1_general.mtx", "m1_k1_seed12345_", 1, 100, 12345},
      {"m1_general.mtx", "m1_k4_seed7_", 4, 2147483647, 7},
  };
  dyng::test::temp_dir tmp;
  for (const auto& c : cases) {
    SCOPED_TRACE(c.expected);
    dyng::io::matrix_market_options options;
    options.weights = dyng::io::matrix_market_weights::random;
    options.random.num_weights = c.k;
    options.random.min = 1;
    options.random.max = c.wmax;
    options.random.seed = c.seed;
    const auto edges = dyng::io::read_matrix_market<std::int32_t, std::int32_t>(
        data_path(fixture_dir + "mtx/" + c.mtx), options);
    const auto g = dyng::graph<std::int32_t, std::int32_t, std::int32_t>::from_edges(
        res, edges.view(), dyng::graph_properties::mosp_compatible());
    dyng::io::write_csr_triplet(tmp.path(c.expected), g.view().out);
    expect_same_triplet(tmp.path(c.expected), data_path(fixture_dir + "mtx/" + c.expected));
  }
}

}  // namespace
