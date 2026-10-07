// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/find.cu
// Derived from ESCHER-GPU@abcf9b2:kernel/motif_utils.cuh
/**
 * @file find.cu
 * @brief Debug lookups of keys and rows (ESCHER-GPU's findAndPrint).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/kernels.cuh"

#include <climits>
#include <cstdio>

namespace dyng::detail::escher {

// Follow back-pointer chains in a flat payload array.
// Negative values (other than INT_MIN) are chain pointers: jump to -val.
// On return, loc points to the resolved position.
static inline __device__ int read_flat(const int* flat, int& loc) {
  while (true) {
    int val = flat[loc];
    if (val < 0 && val != INT_MIN) {
      loc = -val;
      continue;
    }
    return val;
  }
}

__global__ void find_node_kernel(cbst_node* nodes, int* search_indices, int search_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < search_size) {
    int search_index = search_indices[tid];
    cbst_node* current = nodes;
    while (current != nullptr && current->index != search_index) {
      if (current->index > search_index) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      printf("Node %d: Index = %d, Value = %d, Length = %d\n", search_index, current->index,
             current->value, current->length);
    } else {
      printf("Node %d: Not Found\n", search_index);
    }
  }
}

__global__ void find_contents_kernel(cbst_node* nodes, int* search_indices, int search_size,
                                     int* flat_values) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < search_size) {
    int search_index = search_indices[tid];
    cbst_node* current = nodes;
    while (current != nullptr && current->index != search_index) {
      if (current->index > search_index) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    if (current != nullptr) {
      int curr_loc = current->value;
      printf("\n");
      while (true) {
        int val = read_flat(flat_values, curr_loc);
        if (val == 0 || val == INT_MIN) break;
        printf("%d ", val);
        curr_loc++;
      }
      printf("\n");
      printf("Node %d: Index = %d, Value = %d, Length = %d\n", search_index, current->index,
             current->value, current->length);
    } else {
      printf("Node %d: Not Found\n", search_index);
    }
  }
}

}  // namespace dyng::detail::escher
