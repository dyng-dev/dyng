// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file stream_order_cuda_test.cpp
 * @brief The commit of a CUDA graph releases the device copy of G_t after the work queued on the
 *        stream of the applying call (ADR 0120). A delayed reader of G_t's resident copy is queued
 *        on stream s; then graph::apply() runs through a handle on s, on a graph whose copy was
 *        built on stream a; then G_{t+1}'s copy is uploaded on a. Released on a (the stream it
 *        was allocated on), the copy's memory returned to the pool while the reader still spun:
 *        compute-sanitizer memcheck reports a use after free on every build (the debug-sync
 *        builds as well: the host apply launches no kernel that would synchronize; ci/gpu_local.sh
 *        runs memcheck on every gpu executable), and the upload on a, which may take the same
 *        memory, changed what the reader copied. Both commits are covered: the host apply (a graph
 *        with weights; the path of cycle_count's C12 failure under memcheck on parity-cuda, found
 *        in M10b) and the device merge of set batches (no weights; it synchronizes the stream of
 *        the call before it replaces the copy). Every test skips without a visible device.
 */
#include "core/cuda_runtime.hpp"
#include "graph/graph_impl.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dyng::test::hyper {
// The delayed reader of the hypergraph's stream test (hypergraph/stream_probe_kernels.cu): spin
// for about `spin_cycles` clock cycles on `stream`, then copy `n` values (one block).
cudaError_t queue_delayed_copy(const std::int32_t* from, std::int32_t* to, std::size_t n,
                               long long spin_cycles, cudaStream_t stream);
}  // namespace dyng::test::hyper

namespace {

bool have_device() {
  return dyng::backend_available(dyng::backend::cuda) && dyng::detail::cuda_device_count() > 0;
}

// G_t: two out-edges per vertex, (v, v + 1) and (v, v + 7) mod n, weight 1 where weighted.
template <typename graph_t>
graph_t ring_graph(const dyng::resources& res, std::int32_t n, dyng::graph_properties props) {
  using weight_t = typename graph_t::weight_type;
  dyng::edge_list<std::int32_t, weight_t> list;
  list.num_vertices = n;
  if constexpr (!dyng::is_unweighted_v<weight_t>) {
    list.num_weights = 1;
    props.num_weights = 1;
  }
  for (std::int32_t v = 0; v < n; ++v) {
    for (const std::int32_t step : {1, 7}) {
      if constexpr (dyng::is_unweighted_v<weight_t>) {
        list.add_edge(v, (v + step) % n);
      } else {
        list.add_edge(v, (v + step) % n, {weight_t{1}});
      }
    }
  }
  return graph_t::from_edges(res, list.view(), props);
}

// The delayed reader of G_t's out-neighbours on s, then g.apply() through a handle on s, then the
// upload of G_{t+1} on a: the reader must copy G_t's neighbours.
template <typename graph_t>
void expect_g_t_read_before_its_release() {
  using weight_t = typename graph_t::weight_type;
  cudaStream_t a = nullptr;
  cudaStream_t s = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  {  // the graph is destroyed (its stream-ordered frees) before the streams
    const dyng::resources ra = dyng::resources::cuda(0, dyng::stream_ref(a));
    const dyng::resources rs = dyng::resources::cuda(0, dyng::stream_ref(s));
    // Every library kernel loaded now: a lazy module load inside apply() (the device merge) would
    // synchronize the device and hide a missing ordering.
    ra.warm_up();
    const std::int32_t n = 20000;  // apply() takes milliseconds, the reader about half a second
    graph_t g = ring_graph<graph_t>(ra, n, dyng::graph_properties::cycle_enum_compatible());
    // G_t's copy, built on a (its upload synchronizes a).
    const auto& before = dyng::detail::graph_access::device_out(ra, g);
    const std::size_t count = before.out_col_ind.size();
    ASSERT_EQ(count, static_cast<std::size_t>(2 * n));
    std::vector<std::int32_t> want(count);
    std::vector<std::int32_t> got(count, -1);
    ASSERT_EQ(cudaMemcpyAsync(want.data(), before.out_col_ind.data(), count * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, a),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    std::int32_t* copy = nullptr;
    ASSERT_EQ(cudaMalloc(&copy, count * sizeof(std::int32_t)), cudaSuccess);
    // About half a second of spinning on s, then the copy of G_t's neighbours.
    ASSERT_EQ(dyng::test::hyper::queue_delayed_copy(before.out_col_ind.data(), copy, count,
                                                    1000000000LL, s),
              cudaSuccess);
    // Delete (v, v + 1) of every even v: G_{t+1} has n / 2 fewer edges, and its neighbour array
    // differs from G_t's from the second row on.
    dyng::edge_batch<std::int32_t, weight_t> batch;
    for (std::int32_t v = 0; v < n; v += 2) {
      batch.delete_edge(v, (v + 1) % n);
    }
    const dyng::apply_summary summary = g.apply(rs, batch.view());
    EXPECT_EQ(summary.deleted_edges, n / 2);
    // G_{t+1}'s copy on a (uploaded after the host apply; resident after the device merge): an
    // upload may take memory the pool got back from G_t's copy.
    const auto& after = dyng::detail::graph_access::device_out(ra, g);
    EXPECT_EQ(after.out_col_ind.size(), count - static_cast<std::size_t>(n / 2));
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(got.data(), copy, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < count; ++i) {
      differ += got[i] != want[i] ? 1 : 0;
    }
    EXPECT_EQ(differ, 0U) << "entries of G_t's device copy read differently by the earlier work "
                             "of the applying call's stream";
    EXPECT_EQ(cudaFree(copy), cudaSuccess);
  }
  EXPECT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

// The host apply (a graph with weights): the copy of G_t is dropped and released.
TEST(GraphStreamOrder, HostApplyReleasesGtsCopyAfterTheCallsStream) {
  if (!have_device()) {
    GTEST_SKIP() << "no CUDA device is visible";
  }
  expect_g_t_read_before_its_release<dyng::graph<std::int32_t, std::int32_t, std::int32_t>>();
}

// The device merge of a set batch (no weights, 32-bit ids, resident): the copy of G_t is replaced
// by G_{t+1}'s and released.
TEST(GraphStreamOrder, DeviceMergeReleasesGtsCopyAfterTheCallsStream) {
  if (!have_device()) {
    GTEST_SKIP() << "no CUDA device is visible";
  }
  expect_g_t_read_before_its_release<dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>>();
}

}  // namespace
