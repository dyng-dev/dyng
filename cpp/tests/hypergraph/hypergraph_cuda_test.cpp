// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_cuda_test.cpp
 * @brief CUDA-only tests of the hypergraph container (M8b, criteria 5 and 11 C12): repeated runs
 *        give identical ids and line graphs, the device store's equal the host backends' on a
 *        long stream, and two containers on two non-blocking streams, driven from two threads,
 *        equal a sequential run (compute-sanitizer synccheck and memcheck run it in
 *        ci/gpu_local.sh); apply() on another stream than the hypergraph's is ordered after the
 *        work queued there (M8b review).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/stream.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace dyng::test::hyper {
cudaError_t queue_delayed_copy(const std::int32_t* from, std::int32_t* to, std::size_t n,
                               long long spin_cycles, cudaStream_t stream);
}  // namespace dyng::test::hyper

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
// (ids, h2v rows and the line graph). compute-sanitizer synccheck and memcheck run this test
// (ci/gpu_local.sh; racecheck leaves it out, M8b retrospective, step 5 deviation 13).
TEST(HypergraphCuda, TwoContainersOnTwoNonBlockingStreams) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  const dyng::resources ra = dyng::resources::cuda(0, dyng::stream_ref(a));
  const dyng::resources rb = dyng::resources::cuda(0, dyng::stream_ref(b));
  stream_result got[2];
  for (const auto reuse : {dyng::id_reuse::erase_first, dyng::id_reuse::insert_first}) {
    // An exception of a thread (a failed check_integrity) is reported, not left to terminate.
    std::string errors[2];
    const auto drive = [&](int i, const dyng::resources& r, std::uint64_t seed) {
      try {
        got[i] = stream_ids(r, seed, reuse);
      } catch (const std::exception& e) {
        errors[i] = e.what();
      }
    };
    std::thread ta(drive, 0, std::cref(ra), 21);
    std::thread tb(drive, 1, std::cref(rb), 22);
    ta.join();
    tb.join();
    ASSERT_EQ(errors[0], "");
    ASSERT_EQ(errors[1], "");
    EXPECT_EQ(got[0], stream_ids(dyng::resources::cuda(0), 21, reuse));
    EXPECT_EQ(got[1], stream_ids(dyng::resources::cuda(0), 22, reuse));
  }
  ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

// M8b review: the store of a hypergraph works on the stream of the resources it was built with;
// apply() with resources on another stream must order the store's work after the work queued so
// far on that stream. A reader of G_t's h2v view queued on stream b before apply(rb, ...) sees
// G_t, although the store (stream a) frees the view and the next view reuses the memory.
TEST(HypergraphCuda, ApplyOnAnotherStreamIsOrderedAfterThatStreamsWork) {
  if (!dyng::backend_available(dyng::backend::cuda)) GTEST_SKIP() << "no CUDA device";
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  const dyng::resources ra = dyng::resources::cuda(0, dyng::stream_ref(a));
  const dyng::resources rb = dyng::resources::cuda(0, dyng::stream_ref(b));
  // Every library kernel loaded now: a lazy module load inside apply() would synchronize the
  // device and hide a missing ordering.
  ra.warm_up();
  const std::int32_t n = 20000;  // apply() takes milliseconds, the reader about half a second
  rows initial(static_cast<std::size_t>(n));
  for (std::int32_t v = 0; v < n; ++v)
    initial[static_cast<std::size_t>(v)] = {v, (v + 1) % n, (v + 7) % n};
  for (auto& r : initial) std::sort(r.begin(), r.end());
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::none;
  hgraph h = hgraph::from_hyperedges(ra, dyng::test::hyper::list_of(n, initial).view(), props);
  const auto view = h.h2v();
  const std::size_t count = view.col_ind.size();
  std::vector<std::int32_t> want(count), got(count, -1);
  ASSERT_EQ(cudaMemcpy(want.data(), view.col_ind.data(), count * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  std::int32_t* copy = nullptr;
  ASSERT_EQ(cudaMalloc(&copy, count * sizeof(std::int32_t)), cudaSuccess);
  // About half a second of spinning on b, then the copy of G_t's view.
  ASSERT_EQ(
      dyng::test::hyper::queue_delayed_copy(view.col_ind.data(), copy, count, 1000000000LL, b),
      cudaSuccess);
  dyng::hyperedge_batch<std::int32_t, dyng::unweighted> batch;
  for (std::int64_t id = 0; id < n; id += 2) batch.delete_hyperedge(id);
  (void)h.apply(rb, batch.view());
  // apply() is synchronous and its store work waited for b's reader: the reader has finished
  // (without the ordering, apply() returns while b still spins).
  EXPECT_EQ(cudaStreamQuery(b), cudaSuccess) << "apply() returned before the earlier work of b";
  const auto next = h.h2v();  // a new view (the pool may reuse G_t's view memory)
  EXPECT_EQ(next.col_ind.size(), count - static_cast<std::size_t>((n + 1) / 2) * 3);
  ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got.data(), copy, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
  std::size_t differ = 0;
  for (std::size_t i = 0; i < count; ++i) differ += got[i] != want[i] ? 1 : 0;
  EXPECT_EQ(differ, 0U) << "entries of G_t's view read differently by the earlier work of b";
  h.check_integrity(ra);
  EXPECT_EQ(cudaFree(copy), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

}  // namespace
