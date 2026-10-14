// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_cuda_test.cpp
 * @brief CUDA-only tests of the hypergraph container (M8b, criterion 5): repeated runs give
 *        identical ids, and the device store's ids equal the host backends' on a long stream.
 */
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/hypergraph/hypergraph.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace {

using dyng::test::hyper::rows;
using dyng::test::hyper::stream_mix;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

std::vector<std::vector<std::int64_t>> stream_ids(const dyng::resources& res, std::uint64_t seed,
                                                  dyng::id_reuse reuse) {
  std::mt19937_64 rng(seed);
  const std::int32_t n = 5000;
  const rows initial = dyng::test::hyper::random_rows(rng, n, 6000, 10);
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::none;
  props.reuse = reuse;
  hgraph h = hgraph::from_hyperedges(res, dyng::test::hyper::list_of(n, initial).view(), props);
  std::vector<std::vector<std::int64_t>> out;
  const stream_mix mixes[] = {stream_mix::dense, stream_mix::skewed, stream_mix::larger_than,
                              stream_mix::equal_sizes, stream_mix::delete_all};
  for (const stream_mix mix : mixes) {
    rows current;
    const auto live = dyng::test::hyper::live_ids(h, current);
    const auto tb =
        dyng::test::hyper::next_batch(rng, mix, n, live, h.hyperedge_id_bound(), current);
    out.push_back(h.apply(res, dyng::test::hyper::batch_of(tb).view()).inserted_ids);
  }
  h.check_integrity(res);
  return out;
}

TEST(HypergraphCuda, RepeatedRunsGiveIdenticalIds) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  for (const auto reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    const auto reference = stream_ids(dyng::resources::cuda(0), 11, reuse);
    for (int run = 1; run < 10; ++run) {
      EXPECT_EQ(stream_ids(dyng::resources::cuda(0), 11, reuse), reference) << "run " << run;
    }
    EXPECT_EQ(stream_ids(dyng::resources::sequential(), 11, reuse), reference);
  }
}

}  // namespace
