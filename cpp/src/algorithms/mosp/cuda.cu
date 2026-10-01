// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/combinedGraphGpu.cu (combinedEdge, countEdgesKernel,
// fillEdgesKernel, CombineWorkspace::reserve, combinedGraphSospGpu Step 2)
/**
 * @file cuda.cu
 * @brief The CUDA backend of mosp's combine step: MOSP-CUDA's combinedGraphSospGpu, Step 2.
 *
 * The in-edges of v in the combined graph are the distinct values among Parent_0[v] ..
 * Parent_{K-1}[v], so one thread per vertex compares the K parents of its vertex. Two passes
 * build a CSR of the out-edges (the children lists the push-based search needs): count the
 * out-degree of every parent (the edge count and the weight sum reduced per warp first: one atomic
 * per thread on the same two counters serializes), exclusive scan (CUB), fill. Nothing depends on
 * the order of the atomic fills, because the solve breaks distance ties by the lowest parent id.
 * The host reads the edge count and the weight sum back (one synchronization; the default delta of
 * the combined graph needs them).
 *
 * Changes from the original: templates on the index types (edge offsets of the graph's edge_t;
 * 64-bit offsets use 64-bit atomics); the K parent arrays (one per sssp result) and the preference
 * terms are passed by value in the kernels' parameter block instead of an objective-major array
 * and a device copy of the terms; the stream of the resources handle; the pooled workspace; and
 * mosp_finish_cuda (not in the original): the `affected` count against the previous MOSP tree and
 * the download of the new tree for the host's path costs, behind one synchronization.
 */
#include "algorithms/mosp/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/instantiate.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::detail {

namespace mosp_kernels {

/// Threads per block (MOSP-CUDA's BLOCK_SIZE).
constexpr int block_size = 256;

/// atomicAdd on an edge offset of 32 or 64 bits; returns the old value.
template <typename edge_t>
__device__ __forceinline__ edge_t atomic_add_offset(edge_t* address, edge_t value) {
  if constexpr (sizeof(edge_t) == sizeof(int)) {
    return static_cast<edge_t>(atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(value)));
  } else {
    return static_cast<edge_t>(atomicAdd(reinterpret_cast<unsigned long long*>(address),
                                         static_cast<unsigned long long>(value)));
  }
}

/**
 * MOSP's combinedEdge(): if parent k of v is the first occurrence of its value among the K
 * parents, return true and its combined-graph weight base - sum_{j : Parent_j[v] == p} terms[j].
 */
template <typename vertex_t, typename weight_t>
__device__ __forceinline__ bool combined_edge(const mosp_combine_input<vertex_t>& in, vertex_t v,
                                              int k, vertex_t& p, weight_t& weight) {
  p = in.parents[k][v];
  if (p < 0) {
    return false;
  }
  for (int j = 0; j < k; ++j) {
    if (in.parents[j][v] == p) {
      return false;  // counted with its first occurrence
    }
  }
  int w = in.base - in.terms[k];
  for (int j = k + 1; j < in.num_objectives; ++j) {
    if (in.parents[j][v] == p) {
      w -= in.terms[j];
    }
  }
  weight = static_cast<weight_t>(w);
  return true;
}

/// The global index of the calling thread (one thread per vertex).
__device__ __forceinline__ std::int64_t vertex_index() {
  return static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

/// Pass 1 (countEdgesKernel): out-degree of every parent, number of edges and weight sum, the two
/// totals reduced per warp first.
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    count_edges_kernel(mosp_combine_input<vertex_t> in, edge_t* degree, unsigned long long* sums) {
  const std::int64_t v = vertex_index();
  unsigned long long edges = 0;
  unsigned long long weight_sum = 0;
  if (v < static_cast<std::int64_t>(in.num_vertices) && v != static_cast<std::int64_t>(in.source)) {
    for (int k = 0; k < in.num_objectives; ++k) {
      vertex_t p;
      weight_t weight;
      if (combined_edge(in, static_cast<vertex_t>(v), k, p, weight)) {
        atomic_add_offset(&degree[p], edge_t{1});
        ++edges;
        weight_sum += static_cast<unsigned long long>(weight);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    edges += __shfl_down_sync(0xffffffffu, edges, offset);
    weight_sum += __shfl_down_sync(0xffffffffu, weight_sum, offset);
  }
  if ((threadIdx.x & 31) == 0 && edges > 0) {
    atomicAdd(&sums[0], edges);
    atomicAdd(&sums[1], weight_sum);
  }
}

/// Pass 2 (fillEdgesKernel): write every edge (p, v) into row p of the children CSR.
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    fill_edges_kernel(mosp_combine_input<vertex_t> in, edge_t* cursor, vertex_t* col_ind,
                      weight_t* weights) {
  const std::int64_t v = vertex_index();
  if (v >= static_cast<std::int64_t>(in.num_vertices) ||
      v == static_cast<std::int64_t>(in.source)) {
    return;
  }
  for (int k = 0; k < in.num_objectives; ++k) {
    vertex_t p;
    weight_t weight;
    if (combined_edge(in, static_cast<vertex_t>(v), k, p, weight)) {
      const edge_t position = atomic_add_offset(&cursor[p], edge_t{1});
      col_ind[position] = static_cast<vertex_t>(v);
      weights[position] = weight;
    }
  }
}

/// The vertices whose combined distance or parent differ between two trees (warp-reduced).
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    count_changed_kernel(std::int64_t n, const std::int64_t* old_distance,
                         const vertex_t* old_parent, const std::int64_t* new_distance,
                         const vertex_t* new_parent, unsigned long long* changed) {
  const std::int64_t v = vertex_index();
  unsigned long long count = 0;
  if (v < n) {
    count = (old_distance[v] != new_distance[v] || old_parent[v] != new_parent[v]) ? 1 : 0;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    count += __shfl_down_sync(0xffffffffu, count, offset);
  }
  if ((threadIdx.x & 31) == 0 && count > 0) {
    atomicAdd(changed, count);
  }
}

}  // namespace mosp_kernels

namespace {

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// Blocks of a one-thread-per-vertex launch (MOSP-CUDA's blocks()).
unsigned int blocks(std::int64_t work) {
  const std::int64_t b =
      (std::max<std::int64_t>(work, 1) + mosp_kernels::block_size - 1) / mosp_kernels::block_size;
  return static_cast<unsigned int>(b);
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Workspace (MOSP-CUDA@e220ee2:src/combinedGraphGpu.cu CombineWorkspace::reserve)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void mosp_cuda_workspace<vertex_t, edge_t, weight_t>::reserve(const resources& res, std::int64_t n,
                                                              int k) {
  if (n <= capacity && k <= trees) {
    return;
  }
  const scoped_device guard(res.device());
  const auto nn = static_cast<std::size_t>(std::max<std::int64_t>(n, 1));
  const auto kk = static_cast<std::size_t>(std::max(k, 1));
  row_ptr.reserve(res, nn + 1);
  cursor.reserve(res, nn + 1);
  col_ind.reserve(res, nn * kk);
  weights.reserve(res, nn * kk);
  sums.reserve(res, 3);
  std::size_t scan_bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, cursor.data(), row_ptr.data(),
                                              nn + 1, native(res)));
  scan_temp.reserve(res, std::max<std::size_t>(scan_bytes, 1));
  if (host_sums.size() < 3) {
    note_reservation();
    host_sums = buffer<unsigned long long>(3, res.stream(), resources_access::staging_memory(res),
                                           res.device());
  }
  if (host_parents.size() < nn) {
    note_reservation();
    host_parents = buffer<vertex_t>();  // release first: the peak holds one array
    host_parents =
        buffer<vertex_t>(nn, res.stream(), resources_access::staging_memory(res), res.device());
  }
  capacity = std::max(capacity, n);
  trees = std::max(trees, k);
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::size_t mosp_cuda_workspace<vertex_t, edge_t, weight_t>::bytes() const noexcept {
  return row_ptr.bytes() + cursor.bytes() + col_ind.bytes() + weights.bytes() + sums.bytes() +
         scan_temp.bytes() + host_sums.size() * sizeof(unsigned long long) +
         host_parents.size() * sizeof(vertex_t);
}

// ------------------------------------------------------------------------------------------------
// Step 2 (MOSP-CUDA@e220ee2:src/combinedGraphGpu.cu combinedGraphSospGpu)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_cuda(
    const resources& res, const mosp_combine_input<vertex_t>& in,
    mosp_cuda_workspace<vertex_t, edge_t, weight_t>& ws) {
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto n = static_cast<std::int64_t>(in.num_vertices);
  const auto rows = static_cast<std::size_t>(n) + 1;
  edge_t* cursor = ws.cursor.data();
  edge_t* row_ptr = ws.row_ptr.data();
  unsigned long long* sums = ws.sums.data();
  // Step 2: count, scan, fill.
  DYNG_CUDA_TRY(cudaMemsetAsync(cursor, 0, rows * sizeof(edge_t), stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(sums, 0, 2 * sizeof(unsigned long long), stream));
  mosp_kernels::count_edges_kernel<vertex_t, edge_t, weight_t>
      <<<blocks(n), mosp_kernels::block_size, 0, stream>>>(in, cursor, sums);
  DYNG_CHECK_KERNEL(stream);
  std::size_t scan_bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, cursor, row_ptr, rows, stream));
  unsigned char* temp = ws.scan_temp.reserve(res, std::max<std::size_t>(scan_bytes, 1));
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp, scan_bytes, cursor, row_ptr, rows, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(cursor, row_ptr, static_cast<std::size_t>(n) * sizeof(edge_t),
                                cudaMemcpyDeviceToDevice, stream));
  mosp_kernels::fill_edges_kernel<vertex_t, edge_t, weight_t>
      <<<blocks(n), mosp_kernels::block_size, 0, stream>>>(in, cursor, ws.col_ind.data(),
                                                           ws.weights.data());
  DYNG_CHECK_KERNEL(stream);
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_sums.data(), sums, 2 * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // the update's budget (I9) counts it
  mosp_combined<vertex_t, edge_t, weight_t> out;
  out.view.num_vertices = in.num_vertices;
  out.view.out_row_ptr = row_ptr;
  out.view.out_col_ind = ws.col_ind.data();
  out.view.out_weights = ws.weights.data();
  out.edges = static_cast<std::int64_t>(ws.host_sums.data()[0]);
  out.weight_sum = static_cast<std::int64_t>(ws.host_sums.data()[1]);
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::int64_t mosp_finish_cuda(const resources& res, std::int64_t n,
                              const std::int64_t* old_distance, const vertex_t* old_parent,
                              const std::int64_t* new_distance, const vertex_t* new_parent,
                              bool count, bool download,
                              mosp_cuda_workspace<vertex_t, edge_t, weight_t>& ws) {
  if (!count && !download) {
    return 0;
  }
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  unsigned long long* changed = ws.sums.data() + 2;
  if (count) {
    DYNG_CUDA_TRY(cudaMemsetAsync(changed, 0, sizeof(unsigned long long), stream));
    mosp_kernels::count_changed_kernel<vertex_t>
        <<<blocks(n), mosp_kernels::block_size, 0, stream>>>(n, old_distance, old_parent,
                                                             new_distance, new_parent, changed);
    DYNG_CHECK_KERNEL(stream);
    DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_sums.data() + 2, changed, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost, stream));
  }
  if (download && n > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_parents.data(), new_parent,
                                  static_cast<std::size_t>(n) * sizeof(vertex_t),
                                  cudaMemcpyDeviceToHost, stream));
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // the update's budget (I9) counts it
  return count ? static_cast<std::int64_t>(ws.host_sums.data()[2]) : 0;
}

#define DYNG_INSTANTIATE_MOSP_CUDA(V, E, W)                                               \
  template struct mosp_cuda_workspace<V, E, W>;                                           \
  template mosp_combined<V, E, W> mosp_combine_cuda<V, E, W>(                             \
      const resources&, const mosp_combine_input<V>&, mosp_cuda_workspace<V, E, W>&);     \
  template std::int64_t mosp_finish_cuda<V, E, W>(                                        \
      const resources&, std::int64_t, const std::int64_t*, const V*, const std::int64_t*, \
      const V*, bool, bool, mosp_cuda_workspace<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_CUDA)
#undef DYNG_INSTANTIATE_MOSP_CUDA

// The kernels of resources::warm_up() (eager loading outside the timed regions).
DYNG_REGISTER_KERNEL(mosp_kernels::count_edges_kernel<std::int32_t, std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::count_edges_kernel<std::int32_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::count_edges_kernel<std::int64_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::fill_edges_kernel<std::int32_t, std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::fill_edges_kernel<std::int32_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::fill_edges_kernel<std::int64_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::count_changed_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(mosp_kernels::count_changed_kernel<std::int64_t>);

}  // namespace dyng::detail
