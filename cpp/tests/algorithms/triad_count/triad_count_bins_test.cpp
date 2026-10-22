// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_bins_test.cpp
 * @brief The triad-bins JSON (io::write_triad_bins_json(), io::read_triad_bins_json()) with
 *        triad_count: the io module's names of the patterns are triad_count::to_string()'s, and a
 *        result's bins make the round trip (the io-only tests: cpp/tests/io/triad_bins_test.cpp).
 */
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

using dyng::test::temp_dir;
namespace io = dyng::io;

TEST(TriadCountBins, KindsAreTriadCountsPatterns) {
  namespace tc = dyng::triad_count;
  for (const tc::pattern p : {tc::pattern::h_motif, tc::pattern::stathyper_type1,
                              tc::pattern::stathyper_type2, tc::pattern::stathyper_type3}) {
    const int want = p == tc::pattern::h_motif ? tc::max_bins : 1;
    EXPECT_EQ(io::triad_bins_size(std::string(tc::to_string(p))), want);
  }
  EXPECT_THROW((void)io::triad_bins_size("open_triads"), dyng::invalid_argument_error);
}

TEST(TriadCountBins, RoundTripOfAResult) {
  // A result's bins: the pattern's name, its version and its counts.
  const dyng::resources res = dyng::resources::sequential();
  using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 5;
  list.add_hyperedge({0, 1, 2});
  list.add_hyperedge({1, 2, 3});
  list.add_hyperedge({2, 3, 4});
  list.add_hyperedge({0, 2, 4});
  const hgraph h = hgraph::from_hyperedges(res, list.view());
  const dyng::triad_count::result r = dyng::triad_count::compute(res, h);
  io::triad_bins b;
  b.kind = std::string(dyng::triad_count::to_string(r.kind()));
  b.graph_version = r.graph_version();
  b.counts.assign(r.counts().begin(), r.counts().end());
  temp_dir dir;
  io::write_triad_bins_json(dir.path("r.json"), b);
  const io::triad_bins back = io::read_triad_bins_json(dir.path("r.json"));
  EXPECT_EQ(back.kind, b.kind);
  EXPECT_EQ(back.graph_version, b.graph_version);
  EXPECT_EQ(back.counts, b.counts);
  EXPECT_TRUE(back.deltas.empty());
  EXPECT_GT(r.total(), 0U);
}

}  // namespace
