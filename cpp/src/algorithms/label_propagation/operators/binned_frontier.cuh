// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (appendNextFrontier and
// its thread mappings in processVertexGroup / processVertexBlock)
/**
 * @file binned_frontier.cuh
 * @brief binned_frontier's CUDA half (binned_frontier.hpp): expand on cuda_exec, one launch on
 *        the stream of the resources with `lanes` threads per vertex (8 or 32 inside a warp; a
 *        block of 256 or 1024 threads above). CUDA translation units only.
 */
#pragma once

#include "algorithms/label_propagation/operators/binned_frontier.hpp"
#include "operators/cuda_execution.cuh"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace dyng::detail::lp_ops {

/// Threads per block of the lane-group kernels.
inline constexpr int group_block_threads = 256;

/**
 * @brief expand with `lanes` threads per position (lanes 8 or 32; the original's
 *        appendNextFrontier in processVertexGroup).
 */
template <int lanes, typename edge_t, typename id_t, int num_bins, typename eligible_t>
__global__ void __launch_bounds__(group_block_threads)
    expand_group_kernel(slot_rows<edge_t> rows, const id_t* list, const std::uint8_t* changed,
                        int count, eligible_t eligible, binned_push<id_t, num_bins> out) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(t / lanes);
  const int lane = static_cast<int>(t % lanes);
  if (i >= count || changed[i] == 0) return;
  const id_t u = list[i];
  for (edge_t e = rows.row_start[u] + lane; e < rows.row_end[u]; e += lanes) {
    const std::int32_t v = rows.col_ind[e];
    if (v >= 0 && eligible(v)) out.push(v, rows.degree(v));
  }
  if (lane == 0) out.push(u, rows.degree(static_cast<std::int32_t>(u)));
}

/// expand with a block of `block` threads per position (the original's large-bin mapping).
template <int block, typename edge_t, typename id_t, int num_bins, typename eligible_t>
__global__ void __launch_bounds__(block)
    expand_block_kernel(slot_rows<edge_t> rows, const id_t* list, const std::uint8_t* changed,
                        int count, eligible_t eligible, binned_push<id_t, num_bins> out) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= count || changed[i] == 0) return;
  const id_t u = list[i];
  for (edge_t e = rows.row_start[u] + threadIdx.x; e < rows.row_end[u]; e += block) {
    const std::int32_t v = rows.col_ind[e];
    if (v >= 0 && eligible(v)) out.push(v, rows.degree(v));
  }
  if (threadIdx.x == 0) out.push(u, rows.degree(static_cast<std::int32_t>(u)));
}

/// Registers expand_group_kernel's instantiation for resources::warm_up() (one per instantiation).
template <int lanes, typename edge_t, typename id_t, int num_bins, typename eligible_t>
inline const kernel_registrar expand_group_registrar(
    reinterpret_cast<const void*>(&expand_group_kernel<lanes, edge_t, id_t, num_bins, eligible_t>),
    "lp_ops::expand_group_kernel");

/// Registers expand_block_kernel's instantiation for resources::warm_up().
template <int block, typename edge_t, typename id_t, int num_bins, typename eligible_t>
inline const kernel_registrar expand_block_registrar(
    reinterpret_cast<const void*>(&expand_block_kernel<block, edge_t, id_t, num_bins, eligible_t>),
    "lp_ops::expand_block_kernel");

/// expand on cuda_exec: one launch on the stream of the resources (see expand()).
template <>
struct expand_impl<operators::cuda_exec> {
  /// See expand().
  template <typename edge_t, typename id_t, int num_bins, typename eligible_t>
  static void run(const resources& res, int lanes, const slot_rows<edge_t>& rows, const id_t* list,
                  const std::uint8_t* changed, int count, eligible_t eligible,
                  const binned_push<id_t, num_bins>& out) {
    (void)&expand_group_registrar<8, edge_t, id_t, num_bins, eligible_t>;
    (void)&expand_group_registrar<32, edge_t, id_t, num_bins, eligible_t>;
    (void)&expand_block_registrar<256, edge_t, id_t, num_bins, eligible_t>;
    (void)&expand_block_registrar<1024, edge_t, id_t, num_bins, eligible_t>;
    if (count <= 0) {
      return;
    }
    [[maybe_unused]] const operators::cuda_exec exec(res);  // makes the device current
    const auto stream = static_cast<cudaStream_t>(res.stream().get());
    const auto groups = [&](int threads) {
      return static_cast<unsigned>(
          (static_cast<long long>(count) * threads + group_block_threads - 1) /
          group_block_threads);
    };
    switch (lanes) {
      case 8:
        expand_group_kernel<8><<<groups(8), group_block_threads, 0, stream>>>(rows, list, changed,
                                                                              count, eligible, out);
        break;
      case 32:
        expand_group_kernel<32><<<groups(32), group_block_threads, 0, stream>>>(
            rows, list, changed, count, eligible, out);
        break;
      case 256:
        expand_block_kernel<256><<<static_cast<unsigned>(count), 256, 0, stream>>>(
            rows, list, changed, count, eligible, out);
        break;
      default:
        expand_block_kernel<1024><<<static_cast<unsigned>(count), 1024, 0, stream>>>(
            rows, list, changed, count, eligible, out);
        break;
    }
    DYNG_CHECK_KERNEL(stream);
  }
};

}  // namespace dyng::detail::lp_ops
