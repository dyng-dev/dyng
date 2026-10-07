// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/delete_avail.cu
/**
 * @file delete_avail.cu
 * @brief Erase kernels: locate the nodes, mark them available, and the level-wise subtree_avail
 *        reduction.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/kernels.cuh"

namespace dyng::detail::escher {

// Phase 1: Read-only BST traversal to locate array positions of delete targets.
// Safe for concurrent threads since no node is modified.
__global__ void locate_delete_targets_kernel(cbst_node* nodes, int* delete_indices, int delete_size,
                                             int* out_positions) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < delete_size) {
    int delete_index = delete_indices[tid];
    cbst_node* current = nodes;
    while (current != nullptr && current->index != delete_index) {
      if (current->index > delete_index) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      out_positions[tid] = static_cast<int>(current - nodes);
    } else {
      out_positions[tid] = -1;
    }
  }
}

// Phase 2: Apply deletions using precomputed positions. No traversal needed.
__global__ void apply_deletes_kernel(cbst_node* nodes, int* positions, int delete_size,
                                     int* avail) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < delete_size) {
    int pos = positions[tid];
    if (pos >= 0) {
      // The node keeps its key: overwriting it (the original set -1) broke
      // the BST order, so searches for keys in the node's left subtree
      // failed until the next rebuild. Deletion is recorded in avail[].
      avail[pos] = 1;
      nodes[pos].occupancy = 0;
      nodes[pos].tail_base = nodes[pos].value;
      nodes[pos].tail_capacity = (nodes[pos].length > 0) ? nodes[pos].length - 1 : 0;
    }
  }
}

__global__ void reduce_avail_level_kernel(int level_start, int level_end, int num_records,
                                          int* avail, int* subtree_avail) {
  int g = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = level_start + g;
  if (idx < 0 || idx > level_end || idx >= num_records) return;
  int left = 2 * idx + 1;
  int right = 2 * idx + 2;
  int left_sum = (left < num_records) ? subtree_avail[left] : 0;
  int right_sum = (right < num_records) ? subtree_avail[right] : 0;
  subtree_avail[idx] = avail[idx] + left_sum + right_sum;
}

}  // namespace dyng::detail::escher
