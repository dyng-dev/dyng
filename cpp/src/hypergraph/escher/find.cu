// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/find.cu
/**
 * @file find.cu
 * @brief Debug lookup of keys (ESCHER-GPU's findNode, which printed what it found, now writes the
 *        heap position of every key's node).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/kernels.cuh"

namespace dyng::detail::escher {

__global__ void find_node_kernel(const cbst_node* nodes, const int* search_indices, int search_size,
                                 int* positions) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < search_size) {
    int search_index = search_indices[tid];
    const cbst_node* current = nodes;
    while (current != nullptr && current->index != search_index) {
      if (current->index > search_index) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
    positions[tid] = current != nullptr ? static_cast<int>(current - nodes) : -1;
  }
}

}  // namespace dyng::detail::escher
