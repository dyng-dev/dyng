// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/insert_reuse.cu
/**
 * @file insert_reuse.cu
 * @brief Best-fit reuse kernels of insert: the order-statistic walk over deleted slots, the
 *        matching pipeline and the degree-binned reuse.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/device_utils.cuh"
#include "hypergraph/escher/kernels.cuh"
#include "util/kernel_registry.hpp"

#include <climits>

namespace dyng::detail::escher {

// Phase 1: Read-only order-statistic tree walk.
// Each thread finds the k-th deleted node and writes its array position.
// No modification to avail[] or nodes[], so concurrent reads are safe.
__global__ void locate_reusable_slots_kernel(int* subtree_avail, int* avail, int num_records,
                                             int* out_positions, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= K) return;
  int k = tid + 1;  // 1-based order statistic
  int idx = 0;
  while (idx < num_records) {
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    int left_count = (left < num_records) ? subtree_avail[left] : 0;
    int self = avail[idx];
    if (k <= left_count) {
      idx = left;
      continue;
    }
    if (self == 1 && k == left_count + 1) {
      break;
    }
    k -= left_count + self;
    idx = right;
  }
  out_positions[tid] = (idx < num_records) ? idx : -1;
}

// ── Best-fit metadata extraction ────────────────────────────────────────

// Extract usable capacity for each located deleted slot.
// capacity = node->length - 1 (last position reserved for INT_MIN sentinel).
__global__ void extract_slot_capacities_kernel(cbst_node* nodes, int* positions,
                                               int* out_capacities, int D) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= D) return;
  int pos = positions[tid];
  if (pos >= 0) {
    out_capacities[tid] = nodes[pos].length - 1;
  } else {
    out_capacities[tid] = 0;
  }
}

// Compute per-item payload size from prefix-sum array.
__global__ void compute_item_sizes_kernel(int* prefix_sizes, int* out_sizes, int K) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= K) return;
  out_sizes[tid] = (tid == 0) ? prefix_sizes[0] : prefix_sizes[tid] - prefix_sizes[tid - 1];
}

// ── GPU-parallel best-fit matching kernels ──────────────────────────────

// Binary search: for each sorted item, find first slot (in sorted capacity
// order) with capacity >= item size.  Equivalent to std::lower_bound.
__global__ void lower_bound_kernel(int* sorted_caps, int D, int* sorted_sizes, int M, int* out_lo) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= M) return;
  int target = sorted_sizes[tid];
  int lo = 0, hi = D;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (sorted_caps[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  out_lo[tid] = lo;
}

// In-place transform: b[i] = lo[i] - i
__global__ void compute_b_in_place_kernel(int* lo, int M) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < M) lo[tid] -= tid;
}

// assigned[i] = i + prefix_max[i]
__global__ void compute_assigned_kernel(int* prefix_max, int* assigned, int M) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < M) assigned[tid] = tid + prefix_max[tid];
}

// Recover the original key for each located deleted-slot position.
// Uses the same CBST layout formula as store_items_kernel (inverse mapping:
// array position → in-order rank → d_keys[rank]).
__global__ void extract_keys_from_positions_kernel(int* d_keys, int* positions, int* out_keys,
                                                   int num_records, int D) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= D) return;
  int pos = positions[tid];
  if (pos < 0) {
    out_keys[tid] = 0;
    return;
  }
  out_keys[tid] = d_keys[cbst_rank_of_position(pos, num_records)];
}

// Pairs each matched item with its slot. Items are sorted by size and slots
// by capacity; assigned[i] (the slot rank of sorted item i) is strictly
// increasing and >= the first slot that fits, so the matched items are
// exactly sorted items 0..match_count-1 and every one fits its slot.
__global__ void pair_matches_kernel(const int* item_order, const int* assigned,
                                    const int* slot_order, int match_count,
                                    int* matched_item_indices, int* matched_slot_indices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= match_count) return;
  matched_item_indices[tid] = item_order[tid];
  matched_slot_indices[tid] = slot_order[assigned[tid]];
}

// ── Phase 2: Apply reuse ────────────────────────────────────────────────
// Best-fit matching guarantees len <= capacity, so a matched item always
// fits its slot (the original assigned items to slots in BST order and
// truncated rows that did not fit).

// Value of slot position i (0 <= i <= capacity) after reusing it for a row
// of len values: the row, its INT_MIN terminator, then zeros (the rest of the
// deleted row's data is cleared so the free part of the segment is empty).
static __device__ int reused_slot_value(const int* new_payload, int start, int len, int i) {
  return i < len ? new_payload[start + i] : (i == len ? INT_MIN : 0);
}

// Metadata of a reused slot.
static __device__ void finish_reuse(cbst_node* node, int* avail, int pos, int base, int len,
                                    int capacity, int key) {
  node->occupancy = len;
  // Restore the slot's own key so the BST order is preserved
  node->index = key;
  node->tail_base = base;
  node->tail_capacity = capacity;
  avail[pos] = 0;
}

// Thread-level apply_reuse_thread_kernel (small payloads, len < 32): one thread per matched
// item. bin_indices[t] -> index into matched_item_indices/matched_slot_indices.
__global__ void apply_reuse_thread_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                          int* positions, int* new_payload, int* new_prefix_sizes,
                                          int* matched_item_indices, int* matched_slot_indices,
                                          int* deleted_keys, int* bin_indices, int bin_count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= bin_count) return;
  int match_idx = bin_indices[tid];
  int item_idx = matched_item_indices[match_idx];
  int slot_idx = matched_slot_indices[match_idx];
  int pos = positions[slot_idx];
  if (pos < 0) return;
  cbst_node* node = &nodes[pos];
  int start = (item_idx == 0) ? 0 : new_prefix_sizes[item_idx - 1];
  int len = new_prefix_sizes[item_idx] - start;
  int base = node->value;
  int capacity = node->length - 1;
  for (int i = 0; i <= capacity; ++i) {
    flat_values[base + i] = reused_slot_value(new_payload, start, len, i);
  }
  finish_reuse(node, avail, pos, base, len, capacity, deleted_keys[slot_idx]);
}

// ── Warp-level apply_reuse_thread_kernel (medium payloads, 32 <= len < 1024) ───────────
// One warp (32 lanes) cooperatively copies payload for one matched item.
// bin_indices[warp_idx] → index into matched_item_indices/matched_slot_indices.
__global__ void apply_reuse_warp_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                        int* positions, int* new_payload, int* new_prefix_sizes,
                                        int* matched_item_indices, int* matched_slot_indices,
                                        int* deleted_keys, int* bin_indices, int bin_count) {
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_idx = global_tid >> 5;
  int lane = threadIdx.x & 31;
  if (warp_idx >= bin_count) return;

  int match_idx = bin_indices[warp_idx];
  int item_idx = matched_item_indices[match_idx];
  int slot_idx = matched_slot_indices[match_idx];
  int pos = positions[slot_idx];
  if (pos < 0) return;

  cbst_node* node = &nodes[pos];
  int start = (item_idx == 0) ? 0 : new_prefix_sizes[item_idx - 1];
  int len = new_prefix_sizes[item_idx] - start;
  int base = node->value;
  int capacity = node->length - 1;

  // Warp-strided cooperative copy
  for (int i = lane; i <= capacity; i += 32) {
    flat_values[base + i] = reused_slot_value(new_payload, start, len, i);
  }

  // Lane 0 handles the metadata
  if (lane == 0) finish_reuse(node, avail, pos, base, len, capacity, deleted_keys[slot_idx]);
}

// ── Block-level apply_reuse_thread_kernel (large payloads, len >= 1024) ────────────────
// One CTA cooperatively copies payload for one matched item.
// bin_indices[blockIdx.x] → index into matched_item_indices/matched_slot_indices.
__global__ void apply_reuse_block_kernel(cbst_node* nodes, int* flat_values, int* avail,
                                         int* positions, int* new_payload, int* new_prefix_sizes,
                                         int* matched_item_indices, int* matched_slot_indices,
                                         int* deleted_keys, int* bin_indices, int bin_count) {
  int idx = blockIdx.x;
  if (idx >= bin_count) return;

  int match_idx = bin_indices[idx];
  int item_idx = matched_item_indices[match_idx];
  int slot_idx = matched_slot_indices[match_idx];
  int pos = positions[slot_idx];
  if (pos < 0) return;

  cbst_node* node = &nodes[pos];
  int start = (item_idx == 0) ? 0 : new_prefix_sizes[item_idx - 1];
  int len = new_prefix_sizes[item_idx] - start;
  int base = node->value;
  int capacity = node->length - 1;

  // Block-strided cooperative copy
  for (int i = threadIdx.x; i <= capacity; i += blockDim.x) {
    flat_values[base + i] = reused_slot_value(new_payload, start, len, i);
  }

  // Thread 0 handles the metadata
  if (threadIdx.x == 0) finish_reuse(node, avail, pos, base, len, capacity, deleted_keys[slot_idx]);
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(locate_reusable_slots_kernel);
DYNG_REGISTER_KERNEL(extract_slot_capacities_kernel);
DYNG_REGISTER_KERNEL(compute_item_sizes_kernel);
DYNG_REGISTER_KERNEL(lower_bound_kernel);
DYNG_REGISTER_KERNEL(compute_b_in_place_kernel);
DYNG_REGISTER_KERNEL(compute_assigned_kernel);
DYNG_REGISTER_KERNEL(extract_keys_from_positions_kernel);
DYNG_REGISTER_KERNEL(pair_matches_kernel);
DYNG_REGISTER_KERNEL(apply_reuse_thread_kernel);
DYNG_REGISTER_KERNEL(apply_reuse_warp_kernel);
DYNG_REGISTER_KERNEL(apply_reuse_block_kernel);

}  // namespace dyng::detail::escher
