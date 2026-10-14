// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/find.cu
// Derived from MOSP_ESCHER@4b86159:escher/kernel/find.cu (findContents, the bounded readFlat)
/**
 * @file find.cu
 * @brief Lookup of keys (ESCHER-GPU's findNode, which printed what it found, now writes the heap
 *        position of every key's node) and of their rows (its findContents, through chain.cuh).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/chain.cuh"
#include "hypergraph/escher/kernels.cuh"
#include "util/device_error_flags.cuh"
#include "util/kernel_registry.hpp"

#include <cstdint>

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

// The rows of keys (dynG; ESCHER-GPU's and MOSP_ESCHER's findContents printed them with readFlat,
// MOSP_ESCHER's bounded to 1,024 hops): read with the bounded walk of chain.cuh (M8a step 4,
// ADR 0039). A live key's row; an absent or deleted key has none (length 0). Pass 1 counts the
// values (lengths), pass 2 copies them to `out` at `offsets`; a broken chain raises the store's
// device error word.
static __device__ const cbst_node* find_live(const cbst_node* nodes, const int* avail, int key) {
  const cbst_node* current = nodes;
  while (current != nullptr && current->index != key)
    current = current->index > key ? current->left : current->right;
  if (current == nullptr || avail[current - nodes] != 0) return nullptr;
  return current;
}

__global__ void find_row_lengths_kernel(const cbst_node* nodes, const int* avail, const int* flat,
                                        long long used, const int* keys, int num_keys, int* lengths,
                                        std::uint32_t* control) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys) return;
  const cbst_node* node = find_live(nodes, avail, keys[tid]);
  int n = 0;
  if (node != nullptr &&
      walk_chain(flat, used, node->value, [&](long long, int) { ++n; }) == chain_end::overrun) {
    raise_device_error(control + control_errors, device_error::internal);
    atomicAdd(reinterpret_cast<unsigned int*>(control + control_overruns), 1U);
    n = 0;
  }
  lengths[tid] = n;
}

__global__ void find_row_values_kernel(const cbst_node* nodes, const int* avail, const int* flat,
                                       long long used, const int* keys, int num_keys,
                                       const long long* offsets, const int* lengths, int* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_keys || lengths[tid] == 0) return;
  const cbst_node* node = find_live(nodes, avail, keys[tid]);
  long long o = offsets[tid];
  const long long end = o + lengths[tid];
  walk_chain(flat, used, node->value, [&](long long, int v) {
    if (o < end) out[o++] = v;
  });
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(find_node_kernel);
DYNG_REGISTER_KERNEL(find_row_lengths_kernel);
DYNG_REGISTER_KERNEL(find_row_values_kernel);

}  // namespace dyng::detail::escher
