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
 *        the call before it replaces the copy). So are slack rows on the device (ADR 0103): a
 *        vertex batch that grows the graph and moves a row to the tail of a full arena replaces
 *        their per-vertex arrays and their arena inside the call, after queueing the copies of the
 *        old ones. Every test skips without a visible device.
 */
#include "core/cuda_runtime.hpp"
#include "graph/graph_impl.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
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

// row_layout::slack on the device (ADR 0103): a vertex batch that grows the graph and moves a row
// to the tail of a full arena replaces the per-vertex arrays (grow_device_slack) and the arena
// (ensure_arena) after queueing the copies of the old ones on the call's stream. Released on a,
// where the rows were uploaded, they went back to the pool while those copies and a delayed reader
// of G_t's row starts and slots, queued on s before the call, still waited: memcheck reports a
// use after free for each of the six arrays. The apply runs as dyng::update() runs it
// (graph_access::apply, its report settled at the commit); allocations on a follow it, but the
// moves synchronize the call's stream (their pageable sources) before it returns, so outside the
// sanitizer the window is inside the call and the reader's copy is only a further check.
TEST(GraphStreamOrder, SlackDeviceApplyReleasesReplacedRowsAfterTheCallsStream) {
  if (!have_device()) {
    GTEST_SKIP() << "no CUDA device is visible";
  }
  using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
  cudaStream_t a = nullptr;
  cudaStream_t s = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  {  // the graph is destroyed (its stream-ordered frees) before the streams
    const dyng::resources ra = dyng::resources::cuda(0, dyng::stream_ref(a));
    const dyng::resources rs = dyng::resources::cuda(0, dyng::stream_ref(s));
    ra.warm_up();
    dyng::graph_properties props;
    props.directed = false;
    props.layout = dyng::row_layout::slack;
    props.order = dyng::row_order::append;
    props.parallel_edges = dyng::multi_edges::allow;
    props.semantics = dyng::batch_semantics::net_effect();
    props.semantics.allow_vertex_growth = true;
    const std::int32_t n = 20000;
    const std::vector<std::int32_t> hints(static_cast<std::size_t>(n), 2);  // 3 slots per row
    lp_graph g = lp_graph::with_capacity(ra, n, dyng::host_view(hints), props);
    // G_t: every vertex, and the path (v, v + 1); the rows uploaded and applied on a.
    dyng::edge_batch<std::int32_t, float> first;
    for (std::int32_t v = 0; v < n; ++v) {
      first.insert_vertex(v, static_cast<std::int8_t>(v % 7 == 0 ? v % 2 : -1));
    }
    for (std::int32_t v = 0; v + 1 < n; ++v) {
      first.insert_edge(v, v + 1, {1.0F});
    }
    (void)g.apply(ra, first.view());
    const auto& dev = dyng::detail::graph_access::slotted_device(ra, g);
    const std::size_t rows = dev.row_ptr.size();
    const std::size_t slots = dev.col_ind.size();
    ASSERT_EQ(rows, static_cast<std::size_t>(n) + 1);
    std::vector<std::int32_t> want(rows + slots);
    std::vector<std::int32_t> got(rows + slots, -2);
    ASSERT_EQ(cudaMemcpyAsync(want.data(), dev.row_ptr.data(), rows * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, a),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(want.data() + rows, dev.col_ind.data(), slots * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost, a),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    std::int32_t* copy = nullptr;
    ASSERT_EQ(cudaMalloc(&copy, (rows + slots) * sizeof(std::int32_t)), cudaSuccess);
    // About half a second of spinning on s, then the copies of G_t's row starts and slots.
    ASSERT_EQ(
        dyng::test::hyper::queue_delayed_copy(dev.row_ptr.data(), copy, rows, 1000000000LL, s),
        cudaSuccess);
    ASSERT_EQ(dyng::test::hyper::queue_delayed_copy(dev.col_ind.data(), copy + rows, slots, 0, s),
              cudaSuccess);
    // 100 new vertices (the growth), and 200 entries more for vertex 0, whose row of 3 slots
    // moves to the tail of the arena, which has no room left.
    dyng::edge_batch<std::int32_t, float> second;
    for (std::int32_t v = n; v < n + 100; ++v) {
      second.insert_vertex(v);
    }
    for (std::int32_t v = 2; v < 202; ++v) {
      second.insert_edge(0, v, {2.0F});
    }
    dyng::apply_summary summary = dyng::detail::graph_access::apply(
        rs, g, second.view(), static_cast<dyng::detail::apply_delta<std::int32_t>*>(nullptr));
    // Allocations on a, as large as the replaced arrays: they may take memory the pool got back
    // from them.
    std::vector<dyng::buffer<std::int32_t>> takers;
    takers.reserve(6);
    for (const std::size_t size : {rows, rows, rows, rows, slots, slots}) {
      takers.emplace_back(ra, size);
      ASSERT_EQ(cudaMemsetAsync(takers.back().data(), 0x5A, size * sizeof(std::int32_t), a),
                cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    dyng::detail::graph_access::finish_device_apply(rs, g, summary);
    EXPECT_EQ(summary.num_vertices_after, n + 100);
    EXPECT_EQ(summary.relocated_rows, 1);
    EXPECT_EQ(summary.compactions, 0);
    EXPECT_EQ(summary.dropped_insertions, 0);
    ASSERT_EQ(
        cudaMemcpy(got.data(), copy, (rows + slots) * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      differ += got[i] != want[i] ? 1 : 0;
    }
    EXPECT_EQ(differ, 0U) << "entries of G_t's rows read differently by the earlier work of the "
                             "applying call's stream";
    EXPECT_EQ(cudaFree(copy), cudaSuccess);
  }
  EXPECT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(s), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(s), cudaSuccess);
}

}  // namespace
