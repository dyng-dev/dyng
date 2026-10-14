// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/locator.cu
// Derived from ESCHER-GPU@abcf9b2:kernel/delta_merge.cu
// Derived from MOSP_ESCHER@4b86159:escher/kernel/find.cu (the bounded chain walk)
/**
 * @file locator.cu
 * @brief Builds the row view of a CBST: directly over one-segment rows, or over a sorted copy of
 *        chained rows.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "hypergraph/escher/chain.cuh"
#include "hypergraph/escher/row_view.cuh"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.cuh"
#include "util/kernel_registry.hpp"
#include "util/thrust_allocator.cuh"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace dyng::detail::escher {

namespace {

__device__ __forceinline__ int pair_value(unsigned long long p) {
  return static_cast<int>(p & 0xffffffffULL);
}

__device__ __forceinline__ unsigned long long pack_pair(int key, int value) {
  return (static_cast<unsigned long long>(key) << 32) | static_cast<unsigned int>(value);
}

// Calls f(value) for every entry of a (possibly chained) CBST row, with the bounded walk of
// chain.cuh (M8a step 4, ADR 0039): a broken chain raises the tree's device error word (the
// locator throws at its read point) instead of hanging.
template <class F>
__device__ __forceinline__ void for_each_chained_entry(const int* flat, long long used, int loc,
                                                       std::uint32_t* control, F&& f) {
  if (loc < 0) return;
  if (walk_chain(flat, used, loc, [&](long long, int v) { f(v); }) == chain_end::overrun) {
    raise_device_error(control + control_errors, device_error::internal);
    atomicAdd(reinterpret_cast<unsigned int*>(control + control_overruns), 1U);
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

__global__ void chained_row_lengths_kernel(const int* flat, long long used, const int* head,
                                           int max_key, int* len, std::uint32_t* control) {
  int key = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (key > max_key) return;
  int n = 0;
  for_each_chained_entry(flat, used, head[key], control, [&](int) { ++n; });
  len[key] = n;
}

__global__ void gather_chained_rows_kernel(const int* flat, long long used, const int* head,
                                           int max_key, const int* loc, unsigned long long* out,
                                           std::uint32_t* control) {
  int key = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (key > max_key) return;
  int o = loc[key];
  for_each_chained_entry(flat, used, head[key], control,
                         [&](int v) { out[o++] = pack_pair(key, v); });
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

// The stream of the tree's resources (every kernel, copy and Thrust call of the locator).
static cudaStream_t stream_of(const cbst_context& tree) {
  return static_cast<cudaStream_t>(tree.res.stream().get());
}

// A read point of the original (the total length, the chained-row count), counted.
static void synchronize(const cbst_context& tree) {
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(tree)));
  note_host_sync();
}

// (Re)allocates an array of the locator from the tree's resources.
static int* allocate(const cbst_context& tree, buffer<int>& storage, size_t count) {
  storage = buffer<int>();
  storage = buffer<int>(tree.res, count);
  return storage.data();
}

static void reserve_keys(const cbst_context& tree, int max_key, int& capacity, buffer<int>& loc_s,
                         buffer<int>& len_s, buffer<int>& out_s, int*& loc, int*& len, int*& out) {
  if (max_key + 1 <= capacity) return;
  capacity = max_key + 1;
  loc = allocate(tree, loc_s, capacity);
  len = allocate(tree, len_s, capacity);
  out = allocate(tree, out_s, capacity);
}

void row_locator::build_sorted_copy(const cbst_context& tree, int max_key) {
  DYNG_EXPECTS(max_key >= 0, "CBST ", tree.dataset_name, ": row locator: max_key ", max_key,
               " is negative");
  const scoped_device guard(tree.res.device());
  reserve_keys(tree, max_key, capacity_, loc_storage_, len_storage_, out_storage_, loc_, len_,
               out_);
  if (max_key == 0) {
    // No keys (an empty V2H tree, a hypergraph without vertices): an empty view, nothing to
    // launch (dynG addition, M8a review: the row kernels were launched with 0 blocks).
    DYNG_CUDA_TRY(cudaMemsetAsync(loc_, 0, sizeof(int), stream_of(tree)));
    DYNG_CUDA_TRY(cudaMemsetAsync(len_, 0, sizeof(int), stream_of(tree)));
    copy_ = allocate(tree, copy_storage_, 1);
    copy_bytes_ = sizeof(int);
    view_.flat = copy_;
    view_.loc = loc_;
    view_.len = len_;
    view_.out = nullptr;
    view_.max_key = 0;
    return;
  }
  const int block = 256;
  buffer<int> head_storage(tree.res, static_cast<size_t>(max_key) + 1);
  int* d_head = head_storage.data();
  DYNG_CUDA_TRY(cudaMemsetAsync(d_head, 0xff, (max_key + 1) * sizeof(int), stream_of(tree)));
  if (tree.num_records > 0)
    row_heads_kernel<<<(tree.num_records + block - 1) / block, block, 0, stream_of(tree)>>>(
        tree.d_nodes, tree.d_avail, tree.num_records, max_key, d_head);
  DYNG_CHECK_KERNEL(stream_of(tree));
  DYNG_CUDA_TRY(cudaMemsetAsync(len_, 0, (max_key + 1) * sizeof(int), stream_of(tree)));
  chained_row_lengths_kernel<<<(max_key + block - 1) / block, block, 0, stream_of(tree)>>>(
      tree.d_flat_payload, tree.initial_payload_size, d_head, max_key, len_, tree.d_control);
  DYNG_CHECK_KERNEL(stream_of(tree));
  thrust::device_ptr<int> len(len_), loc(loc_);
  thrust_allocator alloc(tree.res);
  thrust::exclusive_scan(thrust_policy(tree.res, alloc), len, len + max_key + 1, loc);
  int last_loc = 0, last_len = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&last_loc, loc_ + max_key, sizeof(int), cudaMemcpyDeviceToHost,
                                stream_of(tree)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(&last_len, len_ + max_key, sizeof(int), cudaMemcpyDeviceToHost,
                                stream_of(tree)));
  enqueue_control_readback(tree);
  synchronize(tree);
  // A broken chain stops the view here (the lengths are not all known); the store itself is
  // poisoned at its next read point.
  throw_if_control_raised(tree, "row locator");
  long long total = static_cast<long long>(last_loc) + last_len;

  // (key, value) pairs sorted globally = rows in key order, each sorted
  buffer<unsigned long long> pairs_storage(tree.res, static_cast<size_t>(std::max(1LL, total)));
  unsigned long long* d_pairs = pairs_storage.data();
  gather_chained_rows_kernel<<<(max_key + block - 1) / block, block, 0, stream_of(tree)>>>(
      tree.d_flat_payload, tree.initial_payload_size, d_head, max_key, loc_, d_pairs,
      tree.d_control);
  DYNG_CHECK_KERNEL(stream_of(tree));
  thrust::sort(thrust_policy(tree.res, alloc), thrust::device_ptr<unsigned long long>(d_pairs),
               thrust::device_ptr<unsigned long long>(d_pairs) + total);
  copy_ = allocate(tree, copy_storage_, static_cast<size_t>(std::max(1LL, total)));
  copy_bytes_ = std::max(1LL, total) * sizeof(int);
  if (total > 0)
    pair_values_kernel<<<static_cast<int>((total + block - 1) / block), block, 0,
                         stream_of(tree)>>>(d_pairs, total, copy_);
  DYNG_CHECK_KERNEL(stream_of(tree));

  view_.flat = copy_;
  view_.loc = loc_;
  view_.len = len_;
  view_.out = nullptr;
  view_.max_key = max_key;
}

void row_locator::build(const cbst_context& tree, int max_key, bool with_out) {
  DYNG_EXPECTS(max_key >= 0, "CBST ", tree.dataset_name, ": row locator: max_key ", max_key,
               " is negative");
  const scoped_device guard(tree.res.device());
  reserve_keys(tree, max_key, capacity_, loc_storage_, len_storage_, out_storage_, loc_, len_,
               out_);
  size_t bytes = (max_key + 1) * sizeof(int);
  DYNG_CUDA_TRY(cudaMemsetAsync(loc_, 0, bytes, stream_of(tree)));
  DYNG_CUDA_TRY(cudaMemsetAsync(len_, 0, bytes, stream_of(tree)));
  DYNG_CUDA_TRY(cudaMemsetAsync(out_, 0, bytes, stream_of(tree)));
  buffer<int> chained_storage(tree.res, 1);
  int* d_chained = chained_storage.data();
  DYNG_CUDA_TRY(cudaMemsetAsync(d_chained, 0, sizeof(int), stream_of(tree)));
  if (tree.num_records > 0) {
    int block = 256;
    int grid = (tree.num_records + block - 1) / block;
    build_row_locator_kernel<<<grid, block, 0, stream_of(tree)>>>(
        tree.d_nodes, tree.d_avail, tree.num_records, tree.d_flat_payload, max_key, loc_, len_,
        with_out ? out_ : nullptr, d_chained);
    DYNG_CHECK_KERNEL(stream_of(tree));
  }
  int chained = 0;
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(&chained, d_chained, sizeof(int), cudaMemcpyDeviceToHost, stream_of(tree)));
  synchronize(tree);
  if (chained) {
    DYNG_FAIL("CBST ", tree.dataset_name, ": ", chained,
              " rows span several payload segments; the counting view needs contiguous rows");
  }
  view_.flat = tree.d_flat_payload;
  view_.loc = loc_;
  view_.len = len_;
  view_.out = with_out ? out_ : nullptr;
  view_.max_key = max_key;
}

// Every kernel of this file is loaded by resources::warm_up().
DYNG_REGISTER_KERNEL(row_heads_kernel);
DYNG_REGISTER_KERNEL(chained_row_lengths_kernel);
DYNG_REGISTER_KERNEL(gather_chained_rows_kernel);
DYNG_REGISTER_KERNEL(pair_values_kernel);
DYNG_REGISTER_KERNEL(build_row_locator_kernel);

}  // namespace dyng::detail::escher
