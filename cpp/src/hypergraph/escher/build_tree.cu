// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/build_tree.cu
// Derived from MOSP_ESCHER@4b86159:escher/structure/operations.cu (setInitialOccupancy)
/**
 * @file build_tree.cu
 * @brief Tree construction kernels: the empty heap-ordered tree, node records at their in-order
 *        rank, the tail metadata scan, the rebuild of the surplus insert.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/device_utils.cuh"
#include "hypergraph/escher/kernels.cuh"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <climits>

namespace dyng::detail::escher {

__global__ void build_empty_tree_kernel(cbst_node* nodes, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    nodes[tid].index = tid;
    nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
    nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
    nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
    nodes[tid].occupancy = 0;
    nodes[tid].tail_base = 0;
    nodes[tid].tail_capacity = 0;
  }
}

__global__ void store_items_kernel(cbst_node* nodes, int* indices, int* values, int n,
                                   int total_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    int index2 = cbst_rank_of_position(tid, n);
    nodes[tid].size = total_size;
    if (index2 < n) {
      nodes[tid].index = indices[index2];
      nodes[tid].value = values[index2];
      int seg_len;
      if (index2 < n - 1) {
        seg_len = values[index2 + 1] - values[index2];
      } else {
        seg_len = total_size - values[index2];
      }
      nodes[tid].length = seg_len;
      nodes[tid].tail_base = values[index2];
      nodes[tid].tail_capacity = (seg_len > 0) ? seg_len - 1 : 0;
      // occupancy is set by init_tail_metadata_kernel once the payload is on the device
      nodes[tid].occupancy = 0;
    }
  }
}

// Tail metadata after construction: every row is a single segment, so the
// tail segment starts at the row offset and `occupancy` is the number of
// live entries before the terminator. fill appends at tail_base + occupancy;
// leaving occupancy at 0 made the first fill overwrite existing entries.
__global__ void init_tail_metadata_kernel(cbst_node* nodes, const int* flat_values, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  cbst_node& node = nodes[tid];
  int count = 0;
  while (count < node.tail_capacity) {
    int v = flat_values[node.value + count];
    if (v == 0 || v == INT_MIN) break;
    ++count;
  }
  node.tail_base = node.value;
  node.occupancy = count;
}

// The caller's row counts as the tail occupancies (MOSP_ESCHER's setInitialOccupancy,
// escher/structure/operations.cu@4b86159:123-132): the node at heap position t holds the record of
// in-order rank cbst_rank_of_position(t, n). The tail segment is the row's first segment
// (store_items_kernel set it), as after init_tail_metadata_kernel.
__global__ void set_initial_occupancy_kernel(cbst_node* nodes, const int* row_counts, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    int rank = cbst_rank_of_position(tid, n);
    if (rank < n) {
      nodes[tid].tail_base = nodes[tid].value;
      nodes[tid].occupancy = row_counts[rank];
    }
  }
}

// Debug builds: the number of nodes whose occupancy is not the scan of their row
// (init_tail_metadata_kernel's count), i.e. whose caller count was wrong.
__global__ void count_occupancy_mismatches_kernel(const cbst_node* nodes, const int* flat_values,
                                                  int n, int* mismatches) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const cbst_node& node = nodes[tid];
  int count = 0;
  while (count < node.tail_capacity) {
    int v = flat_values[node.value + count];
    if (v == 0 || v == INT_MIN) break;
    ++count;
  }
  if (count != node.occupancy) atomicAdd(mismatches, 1);
}

// The child and parent pointers of a copied tree (store clone; dynG addition): CBST nodes hold
// absolute device pointers, so the copy of a node array points into its source until they are
// re-wired to the copy's own heap positions (the layout of build_empty_tree_kernel).
__global__ void rewire_tree_kernel(cbst_node* nodes, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    nodes[tid].left = (2 * tid + 1 < n) ? &nodes[2 * tid + 1] : nullptr;
    nodes[tid].right = (2 * tid + 2 < n) ? &nodes[2 * tid + 2] : nullptr;
    nodes[tid].parent = (tid == 0) ? nullptr : &nodes[(tid - 1) / 2];
  }
}

// Rebuild after surplus inserts: places full node records (sorted by key) at
// their heap positions, so surviving rows keep their offset, length and
// tail metadata. Child/parent pointers come from build_empty_tree_kernel.
__global__ void place_node_records_kernel(cbst_node* nodes, const cbst_node* sorted_records,
                                          int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  const cbst_node& r = sorted_records[cbst_rank_of_position(tid, n)];
  cbst_node& node = nodes[tid];
  node.index = r.index;
  node.value = r.value;
  node.length = r.length;
  node.size = r.size;
  node.occupancy = r.occupancy;
  node.tail_base = r.tail_base;
  node.tail_capacity = r.tail_capacity;
}

// Brings the live nodes into key order: the node at heap position t has
// in-order rank cbst_rank_of_position(t, n), and keys are sorted by rank.
__global__ void rank_order_nodes_kernel(const cbst_node* nodes, const int* avail, int n,
                                        cbst_node* ranked, int* ranked_live) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  int rank = cbst_rank_of_position(tid, n);
  ranked[rank] = nodes[tid];
  ranked_live[rank] = avail[tid] == 0;
}

// Sorted key and row offset arrays of a record list.
__global__ void record_keys_and_starts_kernel(const cbst_node* records, int n, int* keys,
                                              int* starts) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  keys[tid] = records[tid].index;
  starts[tid] = records[tid].value;
}

void launch_build_layout(cudaStream_t stream, cbst_node* nodes, int* keys, int* starts, int n,
                         int total_size) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  build_empty_tree_kernel<<<grid, block, 0, stream>>>(nodes, n);
  store_items_kernel<<<grid, block, 0, stream>>>(nodes, keys, starts, n, total_size);
  DYNG_CHECK_KERNEL(stream);
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(build_empty_tree_kernel);
DYNG_REGISTER_KERNEL(rewire_tree_kernel);
DYNG_REGISTER_KERNEL(store_items_kernel);
DYNG_REGISTER_KERNEL(init_tail_metadata_kernel);
DYNG_REGISTER_KERNEL(set_initial_occupancy_kernel);
DYNG_REGISTER_KERNEL(count_occupancy_mismatches_kernel);
DYNG_REGISTER_KERNEL(place_node_records_kernel);
DYNG_REGISTER_KERNEL(rank_order_nodes_kernel);
DYNG_REGISTER_KERNEL(record_keys_and_starts_kernel);

}  // namespace dyng::detail::escher
