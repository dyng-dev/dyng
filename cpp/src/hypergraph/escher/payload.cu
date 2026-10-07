// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/payload.cu
/**
 * @file payload.cu
 * @brief Fill kernels: the degree-binned appends into tail segments, the overflow plan, the
 *        overflow segments and their metadata.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/device_utils.cuh"
#include "hypergraph/escher/kernels.cuh"

#include <climits>

namespace dyng::detail::escher {

__global__ void compute_next_multiple_of_4_kernel(int* partial_solution, int* tmp, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K) {
    int val = partial_solution[3 * idx + 2];
    tmp[idx] = next_multiple_of_4(val);
  }
}

__global__ void update_partial_solution_kernel(int* partial_solution, int* tmp, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K) {
    partial_solution[3 * idx + 2] = tmp[idx];
  }
}

__global__ void allocate_space_kernel(int* partial_solution, int* flat_values,
                                      int space_available_from, int* insert_indices,
                                      int* insert_values, int* insert_sizes, int insert_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < insert_size) {
    int* values;
    int num_values;
    if (tid == 0) {
      values = insert_values;
      num_values = insert_sizes[tid];
    } else {
      values = insert_values + insert_sizes[tid - 1];
      num_values = insert_sizes[tid] - insert_sizes[tid - 1];
    }

    int idx_partial_solution = tid * 3;
    int start_partial_solution = idx_partial_solution + 1;
    int len_partial_solution = idx_partial_solution + 2;

    if (tid == 0) {
      if (partial_solution[len_partial_solution] == 0) return;
    } else {
      if (partial_solution[len_partial_solution] == partial_solution[len_partial_solution - 3])
        return;
    }

    int start_idx;
    int store_start_idx;
    if (tid == 0) {
      start_idx = space_available_from;

    } else {
      start_idx = space_available_from + partial_solution[idx_partial_solution - 1];
    }

    store_start_idx = start_idx;

    for (int i = partial_solution[start_partial_solution]; i < num_values; i++, start_idx++) {
      flat_values[start_idx] = values[i];
    }

    // Compute per-thread padded size (cumulative minus previous cumulative)
    int per_thread_padded = (tid == 0) ? partial_solution[len_partial_solution]
                                       : partial_solution[len_partial_solution] -
                                             partial_solution[len_partial_solution - 3];

    // Zero gap between written data and sentinel
    int num_written = num_values - partial_solution[start_partial_solution];
    for (int z = num_written; z < per_thread_padded - 1; ++z) flat_values[store_start_idx + z] = 0;

    // Sentinel at last position within padded space
    flat_values[store_start_idx + per_thread_padded - 1] = INT_MIN;

    // Back-pointer from original chunk to this overflow location
    flat_values[partial_solution[idx_partial_solution]] = store_start_idx * (-1);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Degree-binned insert_node kernels (using occupancy/tail_base/tail_capacity)
// ═══════════════════════════════════════════════════════════════════════════

// ── Helper: BST traversal to find node by key ───────────────────────────
// Returns pointer to the found node, or nullptr.
static __device__ cbst_node* bst_find(cbst_node* nodes, int key) {
  cbst_node* current = nodes;
  while (current != nullptr && current->index != key) {
    current = (current->index > key) ? current->left : current->right;
  }
  return current;
}

// ── Thread-level insert_node (small payloads, num_values < 32) ────────────
// Uses occupancy to skip serial scan. Direct write at tail_base + occupancy.
__global__ void insert_node_thread_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                          int* insert_values, int* insert_sizes,
                                          int* partial_solution, int* bin_indices, int bin_count) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= bin_count) return;

  int orig_idx = bin_indices[tid];
  int insert_index = insert_indices[orig_idx];
  int* values = (orig_idx == 0) ? insert_values : insert_values + insert_sizes[orig_idx - 1];
  int num_values = (orig_idx == 0) ? insert_sizes[orig_idx]
                                   : insert_sizes[orig_idx] - insert_sizes[orig_idx - 1];

  cbst_node* current = bst_find(nodes, insert_index);
  if (current == nullptr) return;

  int write_pos = current->tail_base + current->occupancy;
  int remaining = current->tail_capacity - current->occupancy;

  if (num_values <= remaining) {
    // All values fit in the current segment
    for (int i = 0; i < num_values; ++i) {
      flat_values[write_pos + i] = values[i];
    }
    flat_values[write_pos + num_values] = INT_MIN;
    current->occupancy += num_values;
  } else {
    // Partial fit: write what we can, record overflow
    for (int i = 0; i < remaining; ++i) {
      flat_values[write_pos + i] = values[i];
    }
    // Record overflow in partial_solution (using orig_idx for correct
    // positioning)
    partial_solution[orig_idx * 3] = write_pos + remaining;
    partial_solution[orig_idx * 3 + 1] = remaining;
    partial_solution[orig_idx * 3 + 2] = num_values - remaining;
    current->occupancy = current->tail_capacity;
  }
}

// ── Warp-level insert_node (medium payloads, 32 <= num_values < 1024) ─────
// One warp (32 lanes) cooperatively writes payload values.
// bin_indices[warp_idx] → original insert index.
__global__ void insert_node_warp_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                        int* insert_values, int* insert_sizes,
                                        int* partial_solution, int* bin_indices, int bin_count) {
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_idx = global_tid >> 5;
  int lane = threadIdx.x & 31;
  if (warp_idx >= bin_count) return;

  int orig_idx = bin_indices[warp_idx];

  // Lane 0 does BST traversal and reads metadata
  int insert_index = insert_indices[orig_idx];
  int values_start = (orig_idx == 0) ? 0 : insert_sizes[orig_idx - 1];
  int num_values = (orig_idx == 0) ? insert_sizes[orig_idx]
                                   : insert_sizes[orig_idx] - insert_sizes[orig_idx - 1];

  // We need shared data across the warp. Use __shfl_sync for broadcast.
  // Lane 0 finds the node and broadcasts metadata.
  int write_pos = 0, remaining = 0;
  int node_found = 0;
  // All lanes participate in the bst_find independently (same key, same result)
  // This avoids the need for shfl broadcast of the pointer.
  cbst_node* current = bst_find(nodes, insert_index);
  if (current != nullptr) {
    node_found = 1;
    write_pos = current->tail_base + current->occupancy;
    remaining = current->tail_capacity - current->occupancy;
  }

  // Broadcast from lane 0 (consistent across lanes since same key)
  node_found = __shfl_sync(0xFFFFFFFF, node_found, 0);
  if (!node_found) return;
  write_pos = __shfl_sync(0xFFFFFFFF, write_pos, 0);
  remaining = __shfl_sync(0xFFFFFFFF, remaining, 0);

  int copy_len = (num_values <= remaining) ? num_values : remaining;

  // Warp-strided cooperative copy
  for (int i = lane; i < copy_len; i += 32) {
    flat_values[write_pos + i] = insert_values[values_start + i];
  }

  // Lane 0 handles sentinel, metadata, overflow plan
  if (lane == 0) {
    if (num_values <= remaining) {
      flat_values[write_pos + num_values] = INT_MIN;
      current->occupancy += num_values;
    } else {
      partial_solution[orig_idx * 3] = write_pos + remaining;
      partial_solution[orig_idx * 3 + 1] = remaining;
      partial_solution[orig_idx * 3 + 2] = num_values - remaining;
      current->occupancy = current->tail_capacity;
    }
  }
}

// ── Block-level insert_node (large payloads, num_values >= 1024) ──────────
// One CTA cooperatively writes payload values.
// bin_indices[blockIdx.x] → original insert index.
__global__ void insert_node_block_kernel(cbst_node* nodes, int* flat_values, int* insert_indices,
                                         int* insert_values, int* insert_sizes,
                                         int* partial_solution, int* bin_indices, int bin_count) {
  int idx = blockIdx.x;
  if (idx >= bin_count) return;

  // Shared metadata for the block
  __shared__ int sh_write_pos;
  __shared__ int sh_remaining;
  __shared__ int sh_num_values;
  __shared__ int sh_values_start;
  __shared__ int sh_orig_idx;
  __shared__ cbst_node* sh_node;

  if (threadIdx.x == 0) {
    sh_orig_idx = bin_indices[idx];
    int insert_index = insert_indices[sh_orig_idx];
    sh_values_start = (sh_orig_idx == 0) ? 0 : insert_sizes[sh_orig_idx - 1];
    sh_num_values = (sh_orig_idx == 0) ? insert_sizes[sh_orig_idx]
                                       : insert_sizes[sh_orig_idx] - insert_sizes[sh_orig_idx - 1];

    sh_node = bst_find(nodes, insert_index);
    if (sh_node != nullptr) {
      sh_write_pos = sh_node->tail_base + sh_node->occupancy;
      sh_remaining = sh_node->tail_capacity - sh_node->occupancy;
    }
  }
  __syncthreads();

  if (sh_node == nullptr) return;

  int copy_len = (sh_num_values <= sh_remaining) ? sh_num_values : sh_remaining;

  // Block-strided cooperative copy
  for (int i = threadIdx.x; i < copy_len; i += blockDim.x) {
    flat_values[sh_write_pos + i] = insert_values[sh_values_start + i];
  }
  __syncthreads();

  // Thread 0 handles sentinel, metadata, overflow plan
  if (threadIdx.x == 0) {
    if (sh_num_values <= sh_remaining) {
      flat_values[sh_write_pos + sh_num_values] = INT_MIN;
      sh_node->occupancy += sh_num_values;
    } else {
      partial_solution[sh_orig_idx * 3] = sh_write_pos + sh_remaining;
      partial_solution[sh_orig_idx * 3 + 1] = sh_remaining;
      partial_solution[sh_orig_idx * 3 + 2] = sh_num_values - sh_remaining;
      sh_node->occupancy = sh_node->tail_capacity;
    }
  }
}

// ── Metadata fixup after allocate_space_kernel ──────────────────────────────────
// Updates tail_base, tail_capacity, occupancy for nodes that overflowed.
// Called after allocate_space_kernel to keep metadata consistent.
__global__ void fixup_overflow_metadata_kernel(cbst_node* nodes, int* insert_indices,
                                               int* insert_sizes, int* partial_solution,
                                               int space_available_from, int insert_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= insert_size) return;

  int idx_ps = tid * 3;
  int len_ps = idx_ps + 2;

  // Check if this item had overflow (same condition as allocate_space_kernel)
  bool had_overflow;
  if (tid == 0) {
    had_overflow = (partial_solution[len_ps] != 0);
  } else {
    had_overflow = (partial_solution[len_ps] != partial_solution[len_ps - 3]);
  }
  if (!had_overflow) return;

  // Compute the overflow segment base (same as allocate_space_kernel)
  int store_start_idx;
  if (tid == 0) {
    store_start_idx = space_available_from;
  } else {
    store_start_idx = space_available_from + partial_solution[idx_ps - 1];
  }

  // Compute overflow segment capacity
  int per_thread_padded = (tid == 0) ? partial_solution[len_ps]
                                     : partial_solution[len_ps] - partial_solution[len_ps - 3];

  // Find the node and update its metadata
  int insert_index = insert_indices[tid];
  cbst_node* current = bst_find(nodes, insert_index);
  if (current != nullptr) {
    int num_values = (tid == 0) ? insert_sizes[tid] : insert_sizes[tid] - insert_sizes[tid - 1];
    int overflow_start = partial_solution[idx_ps + 1];  // how many values fit in original segment
    int overflow_count = num_values - overflow_start;
    current->tail_base = store_start_idx;
    current->tail_capacity = per_thread_padded - 1;  // -1 for sentinel
    current->occupancy = overflow_count;
  }
}

}  // namespace dyng::detail::escher
