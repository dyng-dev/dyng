// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/locator.cu
// Derived from ESCHER-GPU@abcf9b2:kernel/delta_merge.cu
/**
 * @file locator.cu
 * @brief Builds the row view of a CBST: directly over one-segment rows, or over a sorted copy of
 *        chained rows.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/row_view.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace dyng::detail::escher {

namespace {

__device__ __forceinline__ int pair_value(unsigned long long p) {
  return static_cast<int>(p & 0xffffffffULL);
}

__device__ __forceinline__ unsigned long long pack_pair(int key, int value) {
  return (static_cast<unsigned long long>(key) << 32) | static_cast<unsigned int>(value);
}

// Calls f(value) for every entry of a (possibly chained) CBST row.
template <class F>
__device__ __forceinline__ void for_each_chained_entry(const int* flat, int loc, F&& f) {
  if (loc < 0) return;
  while (true) {
    int v = flat[loc];
    if (v < 0 && v != INT_MIN) {
      loc = -v;
      continue;
    }
    if (v == 0 || v == INT_MIN) return;
    f(v);
    ++loc;
  }
}

}  // namespace

__global__ void row_heads_kernel(const cbst_node* nodes, const int* avail, int num_nodes,
                                 int max_key, int* head) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_nodes || (avail != nullptr && avail[t] != 0)) return;
  int key = nodes[t].index;
  if (key >= 1 && key <= max_key) head[key] = nodes[t].value;
}

__global__ void chained_row_lengths_kernel(const int* flat, const int* head, int max_key,
                                           int* len) {
  int key = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (key > max_key) return;
  int n = 0;
  for_each_chained_entry(flat, head[key], [&](int) { ++n; });
  len[key] = n;
}

__global__ void gather_chained_rows_kernel(const int* flat, const int* head, int max_key,
                                           const int* loc, unsigned long long* out) {
  int key = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (key > max_key) return;
  int o = loc[key];
  for_each_chained_entry(flat, head[key], [&](int v) { out[o++] = pack_pair(key, v); });
}

__global__ void pair_values_kernel(const unsigned long long* pairs, long long n, int* values) {
  long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i < n) values[i] = pair_value(pairs[i]);
}

// One thread per tree node: records the row of each live key. `occupancy`
// is the row length of a single-segment row (tail segment == first segment).
__global__ void build_row_locator_kernel(const cbst_node* nodes, const int* avail, int num_nodes,
                                         const int* flat, int max_key, int* loc, int* len, int* out,
                                         int* chained_rows) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_nodes || (avail != nullptr && avail[t] != 0)) return;
  const cbst_node node = nodes[t];
  int key = node.index;
  if (key < 1 || key > max_key) return;
  if (node.tail_base != node.value) {
    atomicAdd(chained_rows, 1);
    return;
  }
  loc[key] = node.value;
  len[key] = node.occupancy;
  if (out != nullptr) {
    // first position whose entry is greater than key (rows are sorted)
    int lo = 0, hi = node.occupancy;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (flat[node.value + mid] > key)
        hi = mid;
      else
        lo = mid + 1;
    }
    out[key] = node.value + lo;
  }
}

static void check_cuda_locator(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    std::exit(-1);
  }
}

row_locator::~row_locator() {
  cudaFree(loc_);
  cudaFree(len_);
  cudaFree(out_);
  cudaFree(copy_);
}

static void reserve_keys(int max_key, int& capacity, int*& loc, int*& len, int*& out) {
  if (max_key + 1 <= capacity) return;
  cudaFree(loc);
  cudaFree(len);
  cudaFree(out);
  capacity = max_key + 1;
  check_cuda_locator(cudaMalloc(&loc, capacity * sizeof(int)));
  check_cuda_locator(cudaMalloc(&len, capacity * sizeof(int)));
  check_cuda_locator(cudaMalloc(&out, capacity * sizeof(int)));
}

void row_locator::build_sorted_copy(const cbst_context& tree, int max_key) {
  reserve_keys(max_key, capacity_, loc_, len_, out_);
  const int block = 256;
  int* d_head;
  check_cuda_locator(cudaMalloc(&d_head, (max_key + 1) * sizeof(int)));
  check_cuda_locator(cudaMemset(d_head, 0xff, (max_key + 1) * sizeof(int)));
  if (tree.num_records > 0)
    row_heads_kernel<<<(tree.num_records + block - 1) / block, block>>>(
        tree.d_nodes, tree.d_avail, tree.num_records, max_key, d_head);
  check_cuda_locator(cudaMemset(len_, 0, (max_key + 1) * sizeof(int)));
  chained_row_lengths_kernel<<<(max_key + block - 1) / block, block>>>(tree.d_flat_payload, d_head,
                                                                       max_key, len_);
  thrust::device_ptr<int> len(len_), loc(loc_);
  thrust::exclusive_scan(len, len + max_key + 1, loc);
  int last_loc = 0, last_len = 0;
  check_cuda_locator(cudaMemcpy(&last_loc, loc_ + max_key, sizeof(int), cudaMemcpyDeviceToHost));
  check_cuda_locator(cudaMemcpy(&last_len, len_ + max_key, sizeof(int), cudaMemcpyDeviceToHost));
  long long total = static_cast<long long>(last_loc) + last_len;

  // (key, value) pairs sorted globally = rows in key order, each sorted
  unsigned long long* d_pairs;
  check_cuda_locator(cudaMalloc(&d_pairs, std::max(1LL, total) * sizeof(unsigned long long)));
  gather_chained_rows_kernel<<<(max_key + block - 1) / block, block>>>(tree.d_flat_payload, d_head,
                                                                       max_key, loc_, d_pairs);
  thrust::sort(thrust::device_ptr<unsigned long long>(d_pairs),
               thrust::device_ptr<unsigned long long>(d_pairs) + total);
  cudaFree(copy_);
  check_cuda_locator(cudaMalloc(&copy_, std::max(1LL, total) * sizeof(int)));
  copy_bytes_ = std::max(1LL, total) * sizeof(int);
  if (total > 0)
    pair_values_kernel<<<static_cast<int>((total + block - 1) / block), block>>>(d_pairs, total,
                                                                                 copy_);
  check_cuda_locator(cudaGetLastError());
  check_cuda_locator(cudaFree(d_pairs));
  check_cuda_locator(cudaFree(d_head));

  view_.flat = copy_;
  view_.loc = loc_;
  view_.len = len_;
  view_.out = nullptr;
  view_.max_key = max_key;
}

void row_locator::build(const cbst_context& tree, int max_key, bool with_out) {
  reserve_keys(max_key, capacity_, loc_, len_, out_);
  size_t bytes = (max_key + 1) * sizeof(int);
  check_cuda_locator(cudaMemset(loc_, 0, bytes));
  check_cuda_locator(cudaMemset(len_, 0, bytes));
  check_cuda_locator(cudaMemset(out_, 0, bytes));
  int* d_chained;
  check_cuda_locator(cudaMalloc(&d_chained, sizeof(int)));
  check_cuda_locator(cudaMemset(d_chained, 0, sizeof(int)));
  if (tree.num_records > 0) {
    int block = 256;
    int grid = (tree.num_records + block - 1) / block;
    build_row_locator_kernel<<<grid, block>>>(tree.d_nodes, tree.d_avail, tree.num_records,
                                              tree.d_flat_payload, max_key, loc_, len_,
                                              with_out ? out_ : nullptr, d_chained);
  }
  int chained = 0;
  check_cuda_locator(cudaMemcpy(&chained, d_chained, sizeof(int), cudaMemcpyDeviceToHost));
  check_cuda_locator(cudaFree(d_chained));
  if (chained) {
    std::cerr << "[" << tree.dataset_name << "] ERROR: " << chained
              << " rows span several payload segments; the counting view "
                 "needs contiguous rows"
              << std::endl;
    std::exit(2);
  }
  view_.flat = tree.d_flat_payload;
  view_.loc = loc_;
  view_.len = len_;
  view_.out = with_out ? out_ : nullptr;
  view_.max_key = max_key;
}

}  // namespace dyng::detail::escher
