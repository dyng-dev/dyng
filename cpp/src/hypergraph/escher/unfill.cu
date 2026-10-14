// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/unfill.cu
// Derived from MOSP_ESCHER@4b86159:escher/kernel/find.cu (the bounded chain walk)
/**
 * @file unfill.cu
 * @brief Unfill kernels: degree-binned removal of values from rows (serial over chained rows, warp
 *        and block compaction of one-segment rows).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/chain.cuh"
#include "hypergraph/escher/kernels.cuh"
#include "hypergraph/escher/lookup.cuh"
#include "util/device_error_flags.cuh"
#include "util/kernel_registry.hpp"

#include <climits>
#include <cstdint>

namespace dyng::detail::escher {

// ═══════════════════════════════════════════════════════════════════════════
// Degree-binned unfill kernels (using occupancy/tail_base/tail_capacity)
// ═══════════════════════════════════════════════════════════════════════════

// ── Serial unfill of one row (all segments) ─────────────────────────────
// Removes values_to_remove[start, end) from the row of `node`, compacting each
// segment in place and keeping the chain pointers. Returns the number of
// live entries left in the tail segment, which is what `occupancy` counts:
// fill appends at tail_base + occupancy. (The original subtracted removals
// from every segment from the tail occupancy, so a later fill overwrote
// live entries of the tail segment.)
// The row is walked with the bounded walk of chain.cuh (M8a step 4, ADR 0039): a chain that leaves
// the `used` payload slots or reads more slots than it has raises the store's device error word
// (control) and leaves the occupancy as it was; it never hangs.
static __device__ int unfill_row_serial(const cbst_node* node, int* flat_values,
                                        const int* values_to_remove, int start, int end,
                                        long long used, std::uint32_t* control) {
  long long seg_base = node->value;  // the segment being compacted
  int w = 0;                         // its values kept so far
  int tail_live = 0;
  const chain_end e = walk_chain(
      flat_values, used, node->value,
      [&](long long slot, int val) {
        (void)slot;  // seg_base + w <= slot: the kept value moves down or stays
        bool remove_it = false;
        for (int r = start; r < end; ++r) {
          if (values_to_remove[r] == val) {
            remove_it = true;
            break;
          }
        }
        if (!remove_it) {
          flat_values[seg_base + w] = val;
          ++w;
        }
      },
      [&](long long begin, long long end_slot, int end_val) {
        (void)begin;  // == seg_base
        bool chained = end_val < 0 && end_val != INT_MIN;
        flat_values[seg_base + w] = chained ? end_val : INT_MIN;
        for (long long z = seg_base + w + 1; z <= end_slot; ++z) flat_values[z] = 0;
        tail_live = w;
        w = 0;
        if (chained) seg_base = -static_cast<long long>(end_val);
      });
  if (e == chain_end::overrun) {
    raise_device_error(control + control_errors, device_error::internal);
    atomicAdd(reinterpret_cast<unsigned int*>(control + control_overruns), 1U);
    return node->occupancy;
  }
  return tail_live;
}

// The end of an item's removal list, [start, end). The mutation hook of PLAN 8.4
// DYNG_MUTATION_CBST_UNFILL_DROPS_LAST (M8a acceptance criterion 17; MOSP_ESCHER's mutation #4:
// "unfill ignores the last value of each removal list") drops the last value; the CTest
// cbst.mutation.unfill_drops_last and parity/mutate.py must see the suite fail.
__device__ inline int removal_end(int start, int end) {
#if defined(DYNG_MUTATION_CBST_UNFILL_DROPS_LAST)
  return end > start ? end - 1 : end;  // MUTATION
#else
  (void)start;
  return end;
#endif
}

// ── Thread-level unfill with metadata (small bins) ──────────────────────
__global__ void unfill_thread_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                     int* values_to_remove, int* remove_prefix_sizes,
                                     int* bin_indices, int bin_count, int used,
                                     std::uint32_t* control, const int* avail) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= bin_count) return;

  int orig_idx = bin_indices[tid];
  int key = keys[orig_idx];
  int start = (orig_idx == 0) ? 0 : remove_prefix_sizes[orig_idx - 1];
  int end = removal_end(start, remove_prefix_sizes[orig_idx]);

  // An absent or deleted key is skipped and reported (M8a step 4, lookup.cuh).
  cbst_node* cur = find_item_node(nodes, avail, key, control, true);
  if (cur == nullptr) return;
  cur->occupancy = unfill_row_serial(cur, flat_values, values_to_remove, start, end, used, control);
}

// ── Warp-level unfill (medium bins, 32 <= occupancy < 1024) ─────────────
// One warp cooperatively processes a single-segment node.
// Uses ballot + popc for warp-level stream compaction.
// Falls back to thread-level logic for multi-segment (overflow chain) nodes.
__global__ void unfill_warp_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                   int* values_to_remove, int* remove_prefix_sizes,
                                   int* bin_indices, int bin_count, int used,
                                   std::uint32_t* control, const int* avail) {
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_idx = global_tid >> 5;
  int lane = threadIdx.x & 31;
  if (warp_idx >= bin_count) return;

  int orig_idx = bin_indices[warp_idx];
  int key = keys[orig_idx];
  int rem_start = (orig_idx == 0) ? 0 : remove_prefix_sizes[orig_idx - 1];
  int rem_end = removal_end(rem_start, remove_prefix_sizes[orig_idx]);

  // All lanes do BST traversal (same key, same result); an absent or deleted key is skipped and
  // reported by lane 0 (M8a step 4, lookup.cuh).
  cbst_node* cur = find_item_node(nodes, avail, key, control, lane == 0);
  if (cur == nullptr) return;

  int seg_base = cur->tail_base;
  // Clamped to the tail capacity, as the fill kernels clamp it (M8a step 4, ADR 0039): a row with
  // inconsistent metadata is not read or compacted past its segment.
  int occ = min(max(cur->occupancy, 0), cur->tail_capacity);

  // For single-segment nodes (tail_base == value), do cooperative compaction.
  // For multi-segment (overflow chain), fall back to lane 0 doing serial work.
  if (cur->tail_base != cur->value) {
    if (lane == 0) {
      cur->occupancy =
          unfill_row_serial(cur, flat_values, values_to_remove, rem_start, rem_end, used, control);
      // A path counter (M8a step 4): warp items that took the serial walk of a chained row.
      atomicAdd(reinterpret_cast<unsigned int*>(control + control_serial_unfills), 1U);
    }
    return;
  }

  // Single-segment cooperative compaction using ballot
  // Process elements in warp-sized chunks
  int write_offset = 0;  // Accumulated across chunks (identical in all lanes)

  for (int chunk = 0; chunk < occ; chunk += 32) {
    int elem_idx = chunk + lane;
    int val = 0;
    bool keep = false;

    if (elem_idx < occ) {
      val = flat_values[seg_base + elem_idx];
      if (val != 0 && val != INT_MIN && val > 0) {
        keep = true;
        // Check if this value should be removed
        for (int r = rem_start; r < rem_end; ++r) {
          if (values_to_remove[r] == val) {
            keep = false;
            break;
          }
        }
      }
    }

    // Ballot: which lanes have a "keep" element?
    unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
    int kept_in_chunk = __popc(mask);
    // My position within the kept elements in this chunk
    unsigned lower_mask = (1u << lane) - 1u;
    int my_pos = __popc(mask & lower_mask);

    // Write kept elements compactly
    if (keep) {
      flat_values[seg_base + write_offset + my_pos] = val;
    }

    write_offset += kept_in_chunk;
  }

  // Lane 0 writes sentinel and zeros, updates metadata
  if (lane == 0) {
    flat_values[seg_base + write_offset] = INT_MIN;
    // Zero remaining positions (between write_offset+1 and occ)
    for (int z = write_offset + 1; z <= occ; ++z) {
      flat_values[seg_base + z] = 0;
    }
    cur->occupancy = write_offset;
  }
}

// ── Block-level unfill (large bins, occupancy >= 1024) ──────────────────
// One CTA cooperatively processes a single-segment node using shared memory
// for the removal set and block-wide scan for compaction.
__global__ void unfill_block_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                    int* values_to_remove, int* remove_prefix_sizes,
                                    int* bin_indices, int bin_count, int used,
                                    std::uint32_t* control, const int* avail) {
  int idx = blockIdx.x;
  if (idx >= bin_count) return;

  // Shared metadata
  __shared__ int sh_seg_base;
  __shared__ int sh_occ;
  __shared__ int sh_rem_start;
  __shared__ int sh_rem_end;
  __shared__ cbst_node* sh_node;
  __shared__ int sh_write_offset;

  if (threadIdx.x == 0) {
    int orig_idx = bin_indices[idx];
    int key = keys[orig_idx];
    sh_rem_start = (orig_idx == 0) ? 0 : remove_prefix_sizes[orig_idx - 1];
    sh_rem_end = removal_end(sh_rem_start, remove_prefix_sizes[orig_idx]);

    // An absent or deleted key is skipped and reported (M8a step 4, lookup.cuh).
    sh_node = find_item_node(nodes, avail, key, control, true);
    if (sh_node != nullptr) {
      sh_seg_base = sh_node->tail_base;
      // Clamped to the tail capacity, as the fill kernels clamp it (M8a step 4, ADR 0039).
      sh_occ = min(max(sh_node->occupancy, 0), sh_node->tail_capacity);
    }
    sh_write_offset = 0;
  }
  __syncthreads();

  if (sh_node == nullptr) return;

  // Multi-segment fallback: thread 0 does serial
  if (sh_node->tail_base != sh_node->value) {
    if (threadIdx.x == 0) {
      sh_node->occupancy = unfill_row_serial(sh_node, flat_values, values_to_remove, sh_rem_start,
                                             sh_rem_end, used, control);
      // A path counter (M8a step 4): block items that took the serial walk of a chained row.
      atomicAdd(reinterpret_cast<unsigned int*>(control + control_serial_unfills), 1U);
    }
    return;
  }

  // Single-segment: block-cooperative compaction
  // Process in block-sized chunks using warp ballots + shared prefix sums
  extern __shared__ int sh_compact[];  // blockDim.x ints for compaction
  int warp_lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int num_warps = blockDim.x >> 5;

  for (int chunk = 0; chunk < sh_occ; chunk += blockDim.x) {
    int elem_idx = chunk + threadIdx.x;
    int val = 0;
    bool keep = false;

    if (elem_idx < sh_occ) {
      val = flat_values[sh_seg_base + elem_idx];
      if (val != 0 && val != INT_MIN && val > 0) {
        keep = true;
        for (int r = sh_rem_start; r < sh_rem_end; ++r) {
          if (values_to_remove[r] == val) {
            keep = false;
            break;
          }
        }
      }
    }

    // Warp-level ballot
    unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
    int warp_kept = __popc(mask);
    unsigned lower_mask = (1u << warp_lane) - 1u;
    int lane_offset = __popc(mask & lower_mask);

    // Per-warp kept counts into shared memory for block prefix sum
    if (warp_lane == 0) {
      sh_compact[warp_id] = warp_kept;
    }
    __syncthreads();

    // Simple serial prefix sum (num_warps is small, max ~8 for 256 threads)
    if (threadIdx.x == 0) {
      int sum = 0;
      for (int w = 0; w < num_warps; ++w) {
        int tmp = sh_compact[w];
        sh_compact[w] = sum;
        sum += tmp;
      }
      // Store total kept in this chunk at sh_compact[num_warps]
      sh_compact[num_warps] = sum;
    }
    __syncthreads();

    int warp_base = sh_compact[warp_id];
    int chunk_kept = sh_compact[num_warps];

    // Write kept elements
    if (keep) {
      flat_values[sh_seg_base + sh_write_offset + warp_base + lane_offset] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      sh_write_offset += chunk_kept;
    }
    __syncthreads();
  }

  // Thread 0 writes sentinel, zeros remaining, updates metadata
  if (threadIdx.x == 0) {
    flat_values[sh_seg_base + sh_write_offset] = INT_MIN;
    for (int z = sh_write_offset + 1; z <= sh_occ; ++z) {
      flat_values[sh_seg_base + z] = 0;
    }
    sh_node->occupancy = sh_write_offset;
  }
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(unfill_thread_kernel);
DYNG_REGISTER_KERNEL(unfill_warp_kernel);
DYNG_REGISTER_KERNEL(unfill_block_kernel);

}  // namespace dyng::detail::escher
