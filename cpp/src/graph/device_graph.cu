// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/deviceGraph.cu (inDegreeKernel, fillReverseKernel,
// uploadDeviceGraph)
/**
 * @file device_graph.cu
 * @brief Upload a graph state once and derive its in-edges on the device (the resident device
 *        graph of the CUDA backend).
 *
 * Straight port of MOSP-CUDA's uploadDeviceGraph(), with mechanical changes only: names, templates
 * on the index types, namespace dyng::detail, dyng buffers from the memory resource of `res`, the
 * stream of `res` instead of the legacy default stream, exceptions instead of `bool` + `cerr`, no
 * cudaDeviceSynchronize() at the end (the caller's next work is ordered on the same stream). The
 * host CSR already stores its weights objective-major, so they are uploaded as they are and the
 * split kernel (edge-major to objective-major) is gone.
 */
#include "core/cuda_runtime.hpp"
#include "graph/device_graph.hpp"
#include "graph/instantiate.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace {

constexpr int block_size = 256;

unsigned int blocks(long long work) {
  return static_cast<unsigned int>((std::max(work, 1LL) + block_size - 1) / block_size);
}

__device__ __forceinline__ std::int32_t atomic_add(std::int32_t* p, std::int32_t v) {
  return atomicAdd(p, v);
}

__device__ __forceinline__ std::int64_t atomic_add(std::int64_t* p, std::int64_t v) {
  return static_cast<std::int64_t>(
      atomicAdd(reinterpret_cast<unsigned long long*>(p), static_cast<unsigned long long>(v)));
}

template <typename vertex_t, typename edge_t>
__global__ void in_degree_kernel(const vertex_t* col_ind, edge_t m, edge_t* degree) {
  const edge_t e = static_cast<edge_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e < m) {
    atomic_add(&degree[col_ind[e]], edge_t{1});
  }
}

/// One thread per source vertex u writes its out-edges into the rows of their heads (order inside
/// a row is irrelevant to the algorithms).
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void fill_reverse_kernel(vertex_t n, edge_t m, int num_weights, const edge_t* row_ptr,
                                    const vertex_t* col_ind, const weight_t* out_columns,
                                    edge_t* cursor, vertex_t* in_col_ind, weight_t* in_columns) {
  const vertex_t u = static_cast<vertex_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (u >= n) {
    return;
  }
  for (edge_t e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
    const edge_t position = atomic_add(&cursor[col_ind[e]], edge_t{1});
    in_col_ind[position] = u;
    for (int k = 0; k < num_weights; ++k) {
      in_columns[static_cast<std::size_t>(k) * static_cast<std::size_t>(m) +
                 static_cast<std::size_t>(position)] =
          out_columns[static_cast<std::size_t>(k) * static_cast<std::size_t>(m) +
                      static_cast<std::size_t>(e)];
    }
  }
}

template <typename value_t>
void upload(const resources& res, buffer<value_t>& dst, const std::vector<value_t>& src) {
  dst = buffer<value_t>(res, src.size());
  if (!src.empty()) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(dst.data(), src.data(), src.size() * sizeof(value_t),
                                  cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(res.stream().get())));
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_graph(const resources& res, const csr<vertex_t, edge_t, weight_t>& host,
                        device_graph<vertex_t, edge_t, weight_t>& out) {
  const scoped_device guard(res.device());
  const auto stream = static_cast<cudaStream_t>(res.stream().get());
  out = device_graph<vertex_t, edge_t, weight_t>();  // release first: one copy at a time
  const vertex_t n = host.num_vertices();
  const edge_t m = host.num_edges();
  const int num_weights = host.num_weights;
  out.device = res.device();
  out.num_vertices = n;
  out.num_edges = m;
  out.num_weights = num_weights;

  upload(res, out.out_row_ptr, host.row_ptr);
  upload(res, out.out_col_ind, host.col_ind);
  upload(res, out.out_weights, host.weights);
  const auto rows = static_cast<std::size_t>(n) + 1;
  out.in_row_ptr = buffer<edge_t>(res, rows);
  out.in_col_ind = buffer<vertex_t>(res, static_cast<std::size_t>(m));
  out.in_weights = buffer<weight_t>(res, host.weights.size());

  // Reverse CSR: in-degrees, exclusive scan, fill.
  buffer<edge_t> cursor(res, rows);
  DYNG_CUDA_TRY(cudaMemsetAsync(cursor.data(), 0, rows * sizeof(edge_t), stream));
  if (m > 0) {
    in_degree_kernel<vertex_t, edge_t>
        <<<blocks(static_cast<long long>(m)), block_size, 0, stream>>>(out.out_col_ind.data(), m,
                                                                       cursor.data());
    DYNG_CHECK_KERNEL(stream);
  }
  std::size_t bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, bytes, cursor.data(), out.in_row_ptr.data(),
                                              rows, stream));
  buffer<unsigned char> scratch(res, std::max<std::size_t>(bytes, 1));
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scratch.data(), bytes, cursor.data(),
                                              out.in_row_ptr.data(), rows, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(cursor.data(), out.in_row_ptr.data(),
                                static_cast<std::size_t>(n) * sizeof(edge_t),
                                cudaMemcpyDeviceToDevice, stream));
  if (n > 0) {
    fill_reverse_kernel<vertex_t, edge_t, weight_t>
        <<<blocks(static_cast<long long>(n)), block_size, 0, stream>>>(
            n, m, num_weights, out.out_row_ptr.data(), out.out_col_ind.data(),
            out.out_weights.data(), cursor.data(), out.in_col_ind.data(), out.in_weights.data());
    DYNG_CHECK_KERNEL(stream);
  }
}

#define DYNG_INSTANTIATE_DEVICE_GRAPH(V, E, W)                                     \
  template void build_device_graph<V, E, W>(const resources&, const csr<V, E, W>&, \
                                            device_graph<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_DEVICE_GRAPH)
#undef DYNG_INSTANTIATE_DEVICE_GRAPH

// One registration per line (the registrar's name is made unique by __LINE__).
DYNG_REGISTER_KERNEL(in_degree_kernel<std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(in_degree_kernel<std::int32_t, std::int64_t>);
DYNG_REGISTER_KERNEL(in_degree_kernel<std::int64_t, std::int64_t>);
DYNG_REGISTER_KERNEL(fill_reverse_kernel<std::int32_t, std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(fill_reverse_kernel<std::int32_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(fill_reverse_kernel<std::int64_t, std::int64_t, std::int32_t>);

}  // namespace dyng::detail
