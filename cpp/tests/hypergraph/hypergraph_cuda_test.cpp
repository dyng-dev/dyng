// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_cuda_test.cpp
 * @brief CUDA-only tests of the hypergraph container (M8b, criteria 5 and 11 C12): repeated runs
 *        give identical ids and line graphs, the device store's equal the host backends' on a
 *        long stream, and two containers on two non-blocking streams, driven from two threads,
 *        equal a sequential run (compute-sanitizer synccheck and racecheck run it in
 *        ci/gpu_local.sh).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/stream.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

using dyng::test::hyper::rows;
using dyng::test::hyper::stream_mix;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

// One stream of batches: per batch the inserted ids and the line graph's layout (ESCHER-GPU's
// payload of the H2H tree).
struct stream_result {
  std::vector<std::vector<std::int64_t>> ids;
  std::vector<std::vector<std::int32_t>> line_graph;
  bool operator==(const stream_result& o) const {
    return ids == o.ids && line_graph == o.line_graph;
  }
};

stream_result stream_ids(const dyng::resources& res, std::uint64_t seed, dyng::id_reuse reuse) {
  std::mt19937_64 rng(seed);
  const std::int32_t n = 5000;
  // Reduced sizes under compute-sanitizer (DYNG_CBST_REDUCED): 1,000 hyperedges.
  const rows initial =
      dyng::test::hyper::random_rows(rng, n, dyng::test::hyper::reduced_sizes() ? 1000 : 6000, 10);
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  props.reuse = reuse;
  hgraph h = hgraph::from_hyperedges(res, dyng::test::hyper::list_of(n, initial).view(), props);
  stream_result out;
  const stream_mix mixes[] = {stream_mix::dense, stream_mix::skewed, stream_mix::larger_than,
                              stream_mix::equal_sizes, stream_mix::delete_all};
  for (const stream_mix mix : mixes) {
    rows current;
    const auto live = dyng::test::hyper::live_ids(h, current);
    const auto tb =
        dyng::test::hyper::next_batch(rng, mix, n, live, h.hyperedge_id_bound(), current);
    out.ids.push_back(h.apply(res, dyng::test::hyper::batch_of(tb).view()).inserted_ids);
    out.line_graph.push_back(dyng::detail::hypergraph_access::store(h).line_graph().payload);
  }
  h.check_integrity(res);
  return out;
}

TEST(HypergraphCuda, RepeatedRunsGiveIdenticalIds) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  for (const auto reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    const auto reference = stream_ids(dyng::resources::cuda(0), 11, reuse);
    // 10 runs (3 under compute-sanitizer, DYNG_CBST_REDUCED).
    const int runs = dyng::test::hyper::reduced_sizes() ? 3 : 10;
    for (int run = 1; run < runs; ++run) {
      EXPECT_EQ(stream_ids(dyng::resources::cuda(0), 11, reuse), reference) << "run " << run;
    }
    EXPECT_EQ(stream_ids(dyng::resources::sequential(), 11, reuse), reference);
  }
}

// C12 (PLAN 8.2): two containers on two non-blocking streams, each driven by its own host thread
// at the same time, equal a run of the same streams one after the other on the default stream
// (ids, h2v rows and the line graph). compute-sanitizer synccheck and racecheck run this test
// (ci/gpu_local.sh).
TEST(HypergraphCuda, TwoContainersOnTwoNonBlockingStreams) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  const dyng::resources ra = dyng::resources::cuda(0, dyng::stream_ref(a));
  const dyng::resources rb = dyng::resources::cuda(0, dyng::stream_ref(b));
  stream_result got[2];
  for (const auto reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    std::thread ta([&] { got[0] = stream_ids(ra, 21, reuse); });
    std::thread tb([&] { got[1] = stream_ids(rb, 22, reuse); });
    ta.join();
    tb.join();
    EXPECT_EQ(got[0], stream_ids(dyng::resources::cuda(0), 21, reuse));
    EXPECT_EQ(got[1], stream_ids(dyng::resources::cuda(0), 22, reuse));
  }
  ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

}  // namespace
