// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (accumulateNeighbors,
// processVertexGroup, blockSum, processVertexBlock, propagateGroupKernel, propagateBlockKernel)
/**
 * @file neighbor_reduce.cuh
 * @brief neighbor_reduce's CUDA half (neighbor_reduce.hpp): one launch on the stream of the
 *        resources, with the original's thread mappings: `lanes` threads per vertex inside a warp
 *        (8 or 32, a shuffle tree) or a block per vertex (256 or 1024 threads, the shuffle tree per
 *        warp and the warps' sums in shared memory, added in warp order). CUDA translation units
 *        only.
 */
#pragma once

#include "algorithms/label_propagation/operators/neighbor_reduce.hpp"
#include "operators/cuda_execution.cuh"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace dyng::detail::lp_ops {

/// Threads per block of the lane-group reduction kernels.
inline constexpr int reduce_block_threads = 256;

/**
 * @brief The partial sums of one thread: the entries t, t + stride, ... of u's row (the original's
 *        accumulateNeighbors, the damped read of the values).
 */
template <typename edge_t>
__device__ __forceinline__ void partial_sums(const slot_rows<edge_t>& rows, const float* x,
                                             edge_t row_start, edge_t row_end, int t, int stride,
                                             float& w_sum, float& wx_sum) {
  w_sum = 0.0F;
  wx_sum = 0.0F;
  for (edge_t e = row_start + t; e < row_end; e += stride) {
    const std::int32_t v = rows.col_ind[e];
    if (v < 0) continue;                                          // soft-deleted entry
    if (rows.status != nullptr && rows.status[v] == 1) continue;  // deleted vertex
    const float w = rows.weights[e];
    w_sum += w;
    wx_sum = fmaf(w, x[v], wx_sum);
  }
}

/// The sum over the whole block (the original's blockSum).
__device__ __forceinline__ float block_sum(float value, float* s_partial) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  __syncthreads();  // s_partial may still be read by a previous call
  if (lane == 0) s_partial[warp] = value;
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < num_warps; ++w) total += s_partial[w];
  return total;
}

/// neighbor_reduce with `lanes` threads per position (lanes 8 or 32).
template <int lanes, typename edge_t, typename apply_t>
__global__ void __launch_bounds__(reduce_block_threads)
    neighbor_reduce_group_kernel(slot_rows<edge_t> rows, const float* x, const std::int32_t* list,
                                 int count, apply_t apply) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(t / lanes);
  const int lane = static_cast<int>(t % lanes);
  if (i >= count) return;
  const unsigned mask =
      (lanes == 32) ? 0xffffffffU : (((1U << lanes) - 1U) << ((threadIdx.x & 31) & ~(lanes - 1)));
  const std::int32_t u = list[i];
  float w_sum;
  float wx_sum;
  partial_sums(rows, x, rows.row_start[u], rows.row_end[u], lane, lanes, w_sum, wx_sum);
#pragma unroll
  for (int offset = lanes / 2; offset > 0; offset >>= 1) {
    w_sum += __shfl_down_sync(mask, w_sum, offset, lanes);
    wx_sum += __shfl_down_sync(mask, wx_sum, offset, lanes);
  }
  if (lane == 0) apply(i, u, guarded_average(w_sum, wx_sum, x[u]));
}

/// neighbor_reduce with a block of `block` threads per position.
template <int block, typename edge_t, typename apply_t>
__global__ void __launch_bounds__(block)
    neighbor_reduce_block_kernel(slot_rows<edge_t> rows, const float* x, const std::int32_t* list,
                                 int count, apply_t apply) {
  __shared__ float s_partial[block / 32];
  const int i = static_cast<int>(blockIdx.x);
  if (i >= count) return;
  const std::int32_t u = list[i];
  float w_sum;
  float wx_sum;
  partial_sums(rows, x, rows.row_start[u], rows.row_end[u], static_cast<int>(threadIdx.x), block,
               w_sum, wx_sum);
  w_sum = block_sum(w_sum, s_partial);
  wx_sum = block_sum(wx_sum, s_partial);
  if (threadIdx.x == 0) apply(i, u, guarded_average(w_sum, wx_sum, x[u]));
}

/// Registers neighbor_reduce_group_kernel's instantiation for resources::warm_up().
template <int lanes, typename edge_t, typename apply_t>
inline const kernel_registrar neighbor_reduce_group_registrar(
    reinterpret_cast<const void*>(&neighbor_reduce_group_kernel<lanes, edge_t, apply_t>),
    "lp_ops::neighbor_reduce_group_kernel");

/// Registers neighbor_reduce_block_kernel's instantiation for resources::warm_up().
template <int block, typename edge_t, typename apply_t>
inline const kernel_registrar neighbor_reduce_block_registrar(
    reinterpret_cast<const void*>(&neighbor_reduce_block_kernel<block, edge_t, apply_t>),
    "lp_ops::neighbor_reduce_block_kernel");

/// neighbor_reduce on cuda_exec: one launch on the stream of the resources.
template <>
struct neighbor_reduce_impl<operators::cuda_exec> {
  /// See neighbor_reduce().
  template <typename edge_t, typename apply_t>
  static void run(const resources& res, reduce_mapping m, const slot_rows<edge_t>& rows,
                  const float* x, const std::int32_t* list, int count, apply_t apply) {
    (void)&neighbor_reduce_group_registrar<8, edge_t, apply_t>;
    (void)&neighbor_reduce_group_registrar<32, edge_t, apply_t>;
    (void)&neighbor_reduce_block_registrar<256, edge_t, apply_t>;
    (void)&neighbor_reduce_block_registrar<1024, edge_t, apply_t>;
    if (count <= 0) {
      return;
    }
    [[maybe_unused]] const operators::cuda_exec exec(res);  // makes the device current
    const auto stream = static_cast<cudaStream_t>(res.stream().get());
    const auto groups = [&](int threads) {
      return static_cast<unsigned>(
          (static_cast<long long>(count) * threads + reduce_block_threads - 1) /
          reduce_block_threads);
    };
    switch (m) {
      case reduce_mapping::lanes_8:
        neighbor_reduce_group_kernel<8>
            <<<groups(8), reduce_block_threads, 0, stream>>>(rows, x, list, count, apply);
        break;
      case reduce_mapping::warp:
        neighbor_reduce_group_kernel<32>
            <<<groups(32), reduce_block_threads, 0, stream>>>(rows, x, list, count, apply);
        break;
      case reduce_mapping::block_256:
        neighbor_reduce_block_kernel<256>
            <<<static_cast<unsigned>(count), 256, 0, stream>>>(rows, x, list, count, apply);
        break;
      case reduce_mapping::block_1024:
        neighbor_reduce_block_kernel<1024>
            <<<static_cast<unsigned>(count), 1024, 0, stream>>>(rows, x, list, count, apply);
        break;
    }
    DYNG_CHECK_KERNEL(stream);
  }
};

}  // namespace dyng::detail::lp_ops
