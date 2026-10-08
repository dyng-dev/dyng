// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:escher/kernel/payload.cu (compactSizesKernel, compactCopyKernel)
/**
 * @file compact.cu
 * @brief The payload compaction of the CBST store (MOSP_ESCHER's compactCBST kernels; M8a step 4,
 *        ADR 0039): two passes, one warp per node.
 *
 * Fill only appends into a row's tail segment and overflow segments are bump-allocated, so the
 * space unfill frees in earlier segments, and the segments an overflowing tail leaves behind, are
 * not used again. Compaction rewrites every row as one segment in key order: a live row of L
 * values gets the smallest multiple of 4 above L slots (the construct padding of flatten_rows), a
 * deleted row keeps its first segment of `length` slots (its reusable best-fit slot), emptied, and
 * drops its chain. Keys, node positions and avail do not change. The rows are walked with the
 * bounded warp walk of chain.cuh; a broken chain raises the store's device error word.
 */
#include "hypergraph/escher/chain.cuh"
#include "hypergraph/escher/device_utils.cuh"
#include "hypergraph/escher/kernels.cuh"
#include "util/device_error_flags.cuh"
#include "util/kernel_registry.hpp"

#include <climits>
#include <cstdint>

namespace dyng::detail::escher {

// Pass 1, one warp per node: the live values of the row and the slots of its compacted segment,
// stored by in-order rank (key order).
__global__ void compact_sizes_kernel(const cbst_node* nodes, const int* avail, int n,
                                     const int* flat, long long used, int* live_count,
                                     long long* slots_by_rank, std::uint32_t* control) {
  const long long warp = (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) >> 5;
  if (warp >= n) return;
  const cbst_node& node = nodes[warp];
  const bool live = avail[warp] == 0;
  int count = live ? warp_walk_chain(flat, used, node.value, nullptr) : 0;
  if ((threadIdx.x & 31) == 0) {
    if (count < 0) {
      raise_device_error(control + control_errors, device_error::internal);
      atomicAdd(reinterpret_cast<unsigned int*>(control + control_overruns), 1U);
      count = 0;
    }
    live_count[warp] = count;
    const int slots = live ? (count + 4) / 4 * 4 : node.length;
    slots_by_rank[cbst_rank_of_position(static_cast<int>(warp), n)] = slots;
  }
}

// Pass 2, one warp per node: copies the row to its compacted offset in `dst` (zero-filled) and
// points the node at it: one segment, which is the tail.
__global__ void compact_copy_kernel(cbst_node* nodes, const int* avail, int n, const int* flat,
                                    long long used, const int* live_count,
                                    const long long* offset_by_rank, int* dst, int* start_offsets) {
  const long long warp = (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) >> 5;
  if (warp >= n) return;
  cbst_node& node = nodes[warp];
  const int rank = cbst_rank_of_position(static_cast<int>(warp), n);
  const int base = static_cast<int>(offset_by_rank[rank]);
  const int slots = static_cast<int>(offset_by_rank[rank + 1] - base);
  const int count = live_count[warp];
  if (avail[warp] == 0 && count > 0) (void)warp_walk_chain(flat, used, node.value, dst + base);
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    dst[base + count] = INT_MIN;
    node.value = base;
    node.length = slots;
    node.tail_base = base;
    node.tail_capacity = slots - 1;
    node.occupancy = count;
    start_offsets[rank] = base;
  }
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(compact_sizes_kernel);
DYNG_REGISTER_KERNEL(compact_copy_kernel);

}  // namespace dyng::detail::escher
