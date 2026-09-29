// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_oracle_test.cpp
 * @brief dyng::testing cycle oracles: equal to CycleEnumeration-GPU@0a976ad's subset-DP oracle and
 *        brute force on the committed fixtures (the graphs before and after each batch), closed
 *        forms, the two oracles against each other on random graphs with and without a length
 *        bound, and the edge-set recount.
 */
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/testing/cycle_oracle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using dyng::unweighted;
using dyng::test::data_path;
using csr_u = dyng::csr<std::int32_t, std::int64_t, unweighted>;
using dyng::testing::brute_force_simple_cycles;
using dyng::testing::oracle_simple_cycles;

/// A CSR on n vertices from (u, v) pairs, rows in the given order.
csr_u make_csr(std::int64_t n, std::vector<std::pair<std::int32_t, std::int32_t>> edges) {
  std::stable_sort(edges.begin(), edges.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });
  csr_u g;
  g.row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  for (const auto& [u, v] : edges) {
    ++g.row_ptr[static_cast<std::size_t>(u) + 1];
    g.col_ind.push_back(v);
  }
  for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
    g.row_ptr[u + 1] += g.row_ptr[u];
  }
  return g;
}

TEST(CycleOracle, ClosedForms) {
  // The complete digraph on 5 vertices: C(5, k) (k - 1)! cycles of length k.
  std::vector<std::pair<std::int32_t, std::int32_t>> complete;
  for (std::int32_t u = 0; u < 5; ++u) {
    for (std::int32_t v = 0; v < 5; ++v) {
      if (u != v) {
        complete.emplace_back(u, v);
      }
    }
  }
  const csr_u k5 = make_csr(5, complete);
  const dyng::testing::cycle_histogram expected = {0, 0, 10, 20, 30, 24};
  EXPECT_EQ(oracle_simple_cycles(k5.view()), expected);
  EXPECT_EQ(brute_force_simple_cycles(k5.view()), expected);
  EXPECT_EQ(oracle_simple_cycles(k5.view(), 3), (dyng::testing::cycle_histogram{0, 0, 10, 20}));
  EXPECT_EQ(brute_force_simple_cycles(k5.view(), 3),
            (dyng::testing::cycle_histogram{0, 0, 10, 20}));
  EXPECT_EQ(oracle_simple_cycles(k5.view(), 7),
            (dyng::testing::cycle_histogram{0, 0, 10, 20, 30, 24, 0, 0}));
  // A directed triangle with a self-loop and a repeated edge: one 3-cycle for the oracle (it
  // ignores both); the brute force follows the parallel edge twice.
  const csr_u t = make_csr(3, {{0, 1}, {0, 1}, {1, 2}, {2, 0}, {1, 1}});
  EXPECT_EQ(oracle_simple_cycles(t.view()), (dyng::testing::cycle_histogram{0, 0, 0, 1}));
  EXPECT_EQ(brute_force_simple_cycles(t.view()), (dyng::testing::cycle_histogram{0, 0, 0, 2}));
  // Tiny graphs.
  const csr_u empty = make_csr(0, {});
  EXPECT_EQ(oracle_simple_cycles(empty.view()), (dyng::testing::cycle_histogram{0, 0, 0}));
  EXPECT_EQ(brute_force_simple_cycles(empty.view()), (dyng::testing::cycle_histogram{0, 0, 0}));
}

TEST(CycleOracle, Errors) {
  const csr_u g = make_csr(17, {{0, 1}});
  EXPECT_THROW((void)oracle_simple_cycles(g.view()), dyng::invalid_argument_error);
  EXPECT_NO_THROW((void)brute_force_simple_cycles(g.view()));
  const csr_u h = make_csr(3, {{0, 1}});
  EXPECT_THROW((void)oracle_simple_cycles(h.view(), 1), dyng::invalid_argument_error);
  EXPECT_THROW((void)brute_force_simple_cycles(h.view(), 0), dyng::invalid_argument_error);
  dyng::edge_batch<std::int32_t, unweighted> b;
  b.insert_edge(-1, 0);
  EXPECT_THROW((void)dyng::testing::edge_set_after_batch(h.view(), b.view()),
               dyng::invalid_argument_error);
}

// The original's oracle and brute force on the fixture graphs before and after each batch.
TEST(CycleOracle, FixturesMatchTheOriginalOracles) {
  const std::vector<std::string> names = dyng::test::cycle_enum_case_names();
  ASSERT_EQ(names.size(), 80u);
  int with_cycles = 0;
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const auto c = dyng::test::read_cycle_enum_case(data_path("cycle_enum/cases/" + name + ".txt"));
    const std::string expected =
        dyng::test::read_text(data_path("cycle_enum/cases/" + name + ".expected"));
    std::vector<std::pair<std::int32_t, std::int32_t>> edges;
    for (std::size_t i = 0; i < c.src.size(); ++i) {
      edges.emplace_back(static_cast<std::int32_t>(c.src[i]), static_cast<std::int32_t>(c.dst[i]));
    }
    std::sort(edges.begin(), edges.end());
    const csr_u before = make_csr(c.n, edges);
    dyng::edge_batch<std::int32_t, unweighted> b;
    for (std::size_t j = 0; j < c.del_src.size(); ++j) {
      b.delete_edge(static_cast<std::int32_t>(c.del_src[j]),
                    static_cast<std::int32_t>(c.del_dst[j]));
    }
    for (std::size_t i = 0; i < c.ins_src.size(); ++i) {
      b.insert_edge(static_cast<std::int32_t>(c.ins_src[i]),
                    static_cast<std::int32_t>(c.ins_dst[i]));
    }
    const csr_u after = dyng::testing::edge_set_after_batch(before.view(), b.view());
    EXPECT_EQ(dyng::test::csr_text(after.view()),
              dyng::test::section(expected, "after", "oracle_before"));
    const std::string got =
        dyng::test::histogram_text("oracle_before", oracle_simple_cycles(before.view())) +
        dyng::test::histogram_text("oracle_after", oracle_simple_cycles(after.view())) +
        dyng::test::histogram_text("brute_before", brute_force_simple_cycles(before.view())) +
        dyng::test::histogram_text("brute_after", brute_force_simple_cycles(after.view()));
    EXPECT_EQ(got, expected.substr(expected.find("oracle_before")));
    with_cycles += got.find(':') != std::string::npos;
  }
  EXPECT_GT(with_cycles, 40);
}

TEST(CycleOracle, SubsetDpEqualsBruteForceOnRandomGraphs) {
  std::mt19937_64 rng(2718);
  for (int trial = 0; trial < 300; ++trial) {
    const std::int64_t n = 1 + static_cast<std::int64_t>(rng() % 11);
    const double p = 0.1 + 0.1 * static_cast<double>(rng() % 7);
    std::vector<std::pair<std::int32_t, std::int32_t>> edges;
    for (std::int32_t u = 0; u < n; ++u) {
      for (std::int32_t v = 0; v < n; ++v) {
        if (u != v && std::uniform_real_distribution<double>(0, 1)(rng) < p) {
          edges.emplace_back(u, v);
        }
      }
    }
    const csr_u g = make_csr(n, edges);
    const int bound = trial % 3 == 0 ? -1 : 2 + trial % 7;
    SCOPED_TRACE(::testing::Message() << "trial " << trial << " n=" << n << " k=" << bound);
    EXPECT_EQ(oracle_simple_cycles(g.view(), bound), brute_force_simple_cycles(g.view(), bound));
  }
}

}  // namespace
