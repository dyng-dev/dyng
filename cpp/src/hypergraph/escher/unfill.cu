// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/unfill.cu
/**
 * @file unfill.cu
 * @brief Unfill kernels: degree-binned removal of values from rows (serial over chained rows, warp
 *        and block compaction of one-segment rows).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/kernels.cuh"
#include "util/kernel_registry.hpp"

#include <climits>

namespace dyng::detail::escher {

// ═══════════════════════════════════════════════════════════════════════════
// Degree-binned unfill kernels (using occupancy/tail_base/tail_capacity)
// ═══════════════════════════════════════════════════════════════════════════

// ── Helper: BST traversal ───────────────────────────────────────────────
static __device__ cbst_node* unfill_bst_find(cbst_node* nodes, int key) {
  cbst_node* current = nodes;
  while (current != nullptr && current->index != key) {
    current = (current->index > key) ? current->left : current->right;
  }
  return current;
}

// ── Serial unfill of one row (all segments) ─────────────────────────────
// Removes values_to_remove[start, end) from the row of `node`, compacting each
// segment in place and keeping the chain pointers. Returns the number of
// live entries left in the tail segment, which is what `occupancy` counts:
// fill appends at tail_base + occupancy. (The original subtracted removals
// from every segment from the tail occupancy, so a later fill overwrote
// live entries of the tail segment.)
static __device__ int unfill_row_serial(const cbst_node* node, int* flat_values,
                                        const int* values_to_remove, int start, int end) {
  int seg_base = node->value;
  for (;;) {
    int w = 0;
    int i = 0;
    for (;;) {
      int val = flat_values[seg_base + i];
      if (val == INT_MIN || val == 0 || val < 0) break;
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
      ++i;
    }
    int end_val = flat_values[seg_base + i];
    bool chained = end_val < 0 && end_val != INT_MIN;
    flat_values[seg_base + w] = chained ? end_val : INT_MIN;
    for (int z = w + 1; z <= i; ++z) flat_values[seg_base + z] = 0;
    if (!chained) return w;
    seg_base = -end_val;
  }
}

// ── Thread-level unfill with metadata (small bins) ──────────────────────
__global__ void unfill_thread_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                     int* values_to_remove, int* remove_prefix_sizes,
                                     int* bin_indices, int bin_count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= bin_count) return;

  int orig_idx = bin_indices[tid];
  int key = keys[orig_idx];
  int start = (orig_idx == 0) ? 0 : remove_prefix_sizes[orig_idx - 1];
  int end = remove_prefix_sizes[orig_idx];

  cbst_node* cur = unfill_bst_find(nodes, key);
  if (cur == nullptr) return;
  cur->occupancy = unfill_row_serial(cur, flat_values, values_to_remove, start, end);
}

// ── Warp-level unfill (medium bins, 32 <= occupancy < 1024) ─────────────
// One warp cooperatively processes a single-segment node.
// Uses ballot + popc for warp-level stream compaction.
// Falls back to thread-level logic for multi-segment (overflow chain) nodes.
__global__ void unfill_warp_kernel(cbst_node* nodes, int* flat_values, int* keys,
                                   int* values_to_remove, int* remove_prefix_sizes,
                                   int* bin_indices, int bin_count) {
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_idx = global_tid >> 5;
  int lane = threadIdx.x & 31;
  if (warp_idx >= bin_count) return;

  int orig_idx = bin_indices[warp_idx];
  int key = keys[orig_idx];
  int rem_start = (orig_idx == 0) ? 0 : remove_prefix_sizes[orig_idx - 1];
  int rem_end = remove_prefix_sizes[orig_idx];

  // All lanes do BST traversal (same key, same result)
  cbst_node* cur = unfill_bst_find(nodes, key);
  if (cur == nullptr) return;

  int seg_base = cur->tail_base;
  int occ = cur->occupancy;

  // For single-segment nodes (tail_base == value), do cooperative compaction.
  // For multi-segment (overflow chain), fall back to lane 0 doing serial work.
  if (cur->tail_base != cur->value) {
    if (lane == 0)
      cur->occupancy = unfill_row_serial(cur, flat_values, values_to_remove, rem_start, rem_end);
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
                                    int* bin_indices, int bin_count) {
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
    sh_rem_end = remove_prefix_sizes[orig_idx];

    sh_node = unfill_bst_find(nodes, key);
    if (sh_node != nullptr) {
      sh_seg_base = sh_node->tail_base;
      sh_occ = sh_node->occupancy;
    }
    sh_write_offset = 0;
  }
  __syncthreads();

  if (sh_node == nullptr) return;

  // Multi-segment fallback: thread 0 does serial
  if (sh_node->tail_base != sh_node->value) {
    if (threadIdx.x == 0)
      sh_node->occupancy =
          unfill_row_serial(sh_node, flat_values, values_to_remove, sh_rem_start, sh_rem_end);
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
