// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu
/**
 * @file cbst_store.cu
 * @brief The host side of the CBST store: construction, rebuild, best-fit insert with the surplus
 *        rebuild, fill with overflow segments and payload growth, erase, unfill, and the store
 *        object.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/flatten.hpp"
#include "hypergraph/escher/transplant_shims.hpp"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>

// Functor for thrust predicate: returns true when value < threshold
struct less_than {
  int threshold;
  __host__ __device__ bool operator()(int x) const {
    return x < threshold;
  }
};

// Kernel prototypes moved to kernel/kernels.cuh
#include "hypergraph/escher/kernels.cuh"

namespace dyng::detail::escher {

// Local CUDA error checker for this TU
static inline void check_cuda(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    std::exit(-1);
  }
}

// Stream-ordered temporary buffer (cudaMallocAsync/cudaFreeAsync on the
// default stream): operations queue their kernels without host round trips
// or device-wide synchronization from cudaMalloc/cudaFree.
template <class T>
class temp_buffer {
 public:
  explicit temp_buffer(size_t n) {
    check_cuda(
        cudaMallocAsync(reinterpret_cast<void**>(&ptr_), std::max<size_t>(n, 1) * sizeof(T), 0));
  }
  ~temp_buffer() {
    cudaFreeAsync(ptr_, 0);
  }
  temp_buffer(const temp_buffer&) = delete;
  temp_buffer& operator=(const temp_buffer&) = delete;
  T* get() const {
    return ptr_;
  }

 private:
  T* ptr_ = nullptr;
};

// Degree binning of per-item work (ESCHER+ thread / warp / block levels):
// items with fewer than 32 values run one per thread, fewer than 1024 one
// per warp, larger ones one per block. The three index lists are uploaded
// with a single copy.
struct degree_bins {
  explicit degree_bins(const std::vector<int>& sizes) : index(sizes.size()) {
    std::vector<int> lists[3];
    for (size_t i = 0; i < sizes.size(); ++i)
      lists[sizes[i] < 32 ? 0 : (sizes[i] < 1024 ? 1 : 2)].push_back(static_cast<int>(i));
    std::vector<int> all;
    all.reserve(sizes.size());
    for (int b = 0; b < 3; ++b) {
      count[b] = static_cast<int>(lists[b].size());
      all.insert(all.end(), lists[b].begin(), lists[b].end());
    }
    if (!all.empty())
      check_cuda(cudaMemcpyAsync(index.get(), all.data(), all.size() * sizeof(int),
                                 cudaMemcpyHostToDevice, 0));
  }
  // Counts one launch for every non-empty bin.
  void record(long long (&launches)[3]) const {
    for (int b = 0; b < 3; ++b) launches[b] += count[b] > 0;
  }
  int* bin(int b) const {
    return index.get() + (b == 0 ? 0 : (b == 1 ? count[0] : count[0] + count[1]));
  }
  temp_buffer<int> index;
  int count[3] = {0, 0, 0};
};

static degree_bin_launches g_bin_launches;
const degree_bin_launches& degree_bin_launch_counts() {
  return g_bin_launches;
}

// Sizes of the items of a prefix-sum layout.
static std::vector<int> item_sizes(const std::vector<int>& prefix_sizes) {
  std::vector<int> sizes(prefix_sizes.size());
  for (size_t i = 0; i < prefix_sizes.size(); ++i)
    sizes[i] = prefix_sizes[i] - (i == 0 ? 0 : prefix_sizes[i - 1]);
  return sizes;
}

// Grows the per-operation input buffers so that an operation with num_items
// keys and payload_size values fits (they were sized by num_records only and
// overflowed for batches with more values than 3 x num_records).
static void reserve_scratch(cbst_context& ctx, int num_items, size_t payload_size) {
  if (num_items > ctx.scratch_item_capacity) {
    check_cuda(cudaFree(ctx.d_insert_keys));
    check_cuda(cudaFree(ctx.d_insert_prefix_sizes));
    check_cuda(cudaFree(ctx.d_relocation_plan));
    check_cuda(cudaMalloc(&ctx.d_insert_keys, num_items * sizeof(int)));
    check_cuda(cudaMalloc(&ctx.d_insert_prefix_sizes, num_items * sizeof(int)));
    check_cuda(cudaMalloc(&ctx.d_relocation_plan, 3 * num_items * sizeof(int)));
    ctx.scratch_item_capacity = num_items;
  }
  if (static_cast<long long>(payload_size) > ctx.scratch_payload_capacity) {
    check_cuda(cudaFree(ctx.d_insert_payload));
    check_cuda(cudaMalloc(&ctx.d_insert_payload, payload_size * sizeof(int)));
    ctx.scratch_payload_capacity = static_cast<long long>(payload_size);
  }
}

// Recomputes subtree_avail (number of deleted slots per subtree) bottom-up,
// one launch per tree level (stream-ordered, no host synchronization).
static void recompute_subtree_avail(cbst_context& ctx) {
  const int block_size = 256;
  int last_level_start = 1;
  while (last_level_start * 2 <= ctx.num_records) last_level_start <<= 1;
  int level_start = last_level_start - 1;
  for (int level_end = ctx.num_records - 1; level_start >= 0;) {
    int count = level_end - level_start + 1;
    int blocks = (count + block_size - 1) / block_size;
    reduce_avail_level_kernel<<<blocks, block_size>>>(level_start, level_end, ctx.num_records,
                                                      ctx.d_avail, ctx.d_subtree_avail);
    check_cuda(cudaGetLastError());
    if (level_start == 0) break;
    level_end = level_start - 1;
    level_start = (level_start - 1) / 2;
  }
}

// Grows the payload buffer so that `needed` entries fit (appends by fill
// overflow and by surplus inserts). Row offsets stay valid: the used prefix
// is copied to the new buffer and the rest is zeroed.
static void ensure_payload_capacity(cbst_context& ctx, long long needed) {
  if (needed <= ctx.fixed_size) return;
  long long capacity = std::max(needed + needed / 4, 3LL * ctx.fixed_size / 2);
  capacity = std::min<long long>(capacity, INT_MAX);
  if (needed > capacity) {
    std::cerr << "[" << ctx.dataset_name << "] ERROR: " << needed
              << " payload entries exceed the int32 offsets of the CBST" << std::endl;
    std::exit(2);
  }
  int* grown;
  check_cuda(cudaMalloc(&grown, capacity * sizeof(int)));
  check_cuda(cudaMemcpy(grown, ctx.d_flat_payload,
                        static_cast<size_t>(ctx.initial_payload_size) * sizeof(int),
                        cudaMemcpyDeviceToDevice));
  check_cuda(cudaMemset(grown + ctx.initial_payload_size, 0,
                        (capacity - ctx.initial_payload_size) * sizeof(int)));
  check_cuda(cudaFree(ctx.d_flat_payload));
  ctx.d_flat_payload = grown;
  ctx.fixed_size = static_cast<int>(capacity);
}

void construct_cbst(int* keys, int* start_offsets, int num_records, int* flat_payload,
                    int flat_payload_size, int payload_capacity, const char* dataset_name,
                    cbst_context& ctx) {
  ctx.fixed_size = payload_capacity;
  ctx.num_records = num_records;
  ctx.initial_payload_size = flat_payload_size;
  ctx.dataset_name = dataset_name;
  // ctx.alignment should be set by the owner (cbst_store) before calling

  if (num_records == 0) {
    return;
  }

  if (ctx.fixed_size < flat_payload_size) {
    std::cerr << "Overflow: fixed_size is less than flat_payload_size" << std::endl;
    return;
  }

  check_cuda(cudaMalloc(&ctx.d_nodes, num_records * sizeof(cbst_node)));
  check_cuda(cudaMemset(ctx.d_nodes, 0, num_records * sizeof(cbst_node)));
  check_cuda(cudaMalloc(&ctx.d_keys, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_start_offsets, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_flat_payload, ctx.fixed_size * sizeof(int)));

  check_cuda(cudaMemcpy(ctx.d_keys, keys, num_records * sizeof(int), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(ctx.d_start_offsets, start_offsets, num_records * sizeof(int),
                        cudaMemcpyHostToDevice));

  check_cuda(cudaMemcpy(ctx.d_flat_payload, flat_payload, flat_payload_size * sizeof(int),
                        cudaMemcpyHostToDevice));
  check_cuda(cudaMemset(ctx.d_flat_payload + flat_payload_size, 0,
                        (ctx.fixed_size - flat_payload_size) * sizeof(int)));

  check_cuda(cudaMalloc(&ctx.d_insert_keys, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_insert_payload, num_records * 3 * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_insert_prefix_sizes, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_relocation_plan, 3 * num_records * sizeof(int)));
  ctx.scratch_item_capacity = num_records;
  ctx.scratch_payload_capacity = 3LL * num_records;
  // Availability arrays (0/1 per node) and subtree sums (per node)
  check_cuda(cudaMalloc(&ctx.d_avail, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_subtree_avail, num_records * sizeof(int)));
  check_cuda(cudaMemset(ctx.d_avail, 0, num_records * sizeof(int)));
  check_cuda(cudaMemset(ctx.d_subtree_avail, 0, num_records * sizeof(int)));

  int block_size = 256;
  int num_blocks = (num_records + block_size - 1) / block_size;

  build_empty_tree_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, num_records);
  store_items_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, ctx.d_keys, ctx.d_start_offsets,
                                                 num_records, flat_payload_size);
  init_tail_metadata_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, ctx.d_flat_payload,
                                                        num_records);
  check_cuda(cudaGetLastError());

  if (verbose()) {
    std::cout << "Printing the tree from the device (" << dataset_name << "):" << std::endl;
    print_nodes_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, num_records);
    check_cuda(cudaDeviceSynchronize());
  }
}

void rebuild_cbst_from_device(int* d_start_offsets, int num_records, int* d_flat_payload,
                              int flat_payload_size, int payload_capacity, cbst_context& ctx) {
  check_cuda(cudaFree(ctx.d_nodes));
  check_cuda(cudaFree(ctx.d_keys));
  check_cuda(cudaFree(ctx.d_start_offsets));
  check_cuda(cudaFree(ctx.d_flat_payload));
  check_cuda(cudaFree(ctx.d_avail));
  check_cuda(cudaFree(ctx.d_subtree_avail));
  ctx.d_start_offsets = d_start_offsets;
  ctx.d_flat_payload = d_flat_payload;
  ctx.fixed_size = payload_capacity;
  ctx.initial_payload_size = flat_payload_size;
  ctx.num_records = num_records;
  check_cuda(cudaMalloc(&ctx.d_nodes, num_records * sizeof(cbst_node)));
  check_cuda(cudaMemset(ctx.d_nodes, 0, num_records * sizeof(cbst_node)));
  check_cuda(cudaMalloc(&ctx.d_keys, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_avail, num_records * sizeof(int)));
  check_cuda(cudaMalloc(&ctx.d_subtree_avail, num_records * sizeof(int)));
  check_cuda(cudaMemset(ctx.d_avail, 0, num_records * sizeof(int)));
  check_cuda(cudaMemset(ctx.d_subtree_avail, 0, num_records * sizeof(int)));
  thrust::sequence(thrust::device_pointer_cast(ctx.d_keys),
                   thrust::device_pointer_cast(ctx.d_keys) + num_records, 1);

  int block_size = 256;
  int num_blocks = (num_records + block_size - 1) / block_size;
  build_empty_tree_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, num_records);
  store_items_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, ctx.d_keys, ctx.d_start_offsets,
                                                 num_records, flat_payload_size);
  init_tail_metadata_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, ctx.d_flat_payload,
                                                        num_records);
  check_cuda(cudaGetLastError());
}

void fill_cbst(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
               const std::vector<int>& insert_prefix_sizes, cbst_context& ctx) {
  if (insert_keys.empty()) return;
  int K = static_cast<int>(insert_keys.size());
  reserve_scratch(ctx, K, insert_payload.size());

  check_cuda(cudaMemcpyAsync(ctx.d_insert_keys, insert_keys.data(), K * sizeof(int),
                             cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_payload, insert_payload.data(),
                             insert_payload.size() * sizeof(int), cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, insert_prefix_sizes.data(), K * sizeof(int),
                             cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemsetAsync(ctx.d_relocation_plan, 0, 3 * K * sizeof(int), 0));

  int block_size = 256;

  // ── Degree-binned insert_node dispatch ────────────────────────────────
  degree_bins bins(item_sizes(insert_prefix_sizes));
  bins.record(g_bin_launches.fill);
  if (bins.count[0] > 0)
    insert_node_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(0), bins.count[0]);
  if (bins.count[1] > 0)
    insert_node_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(1), bins.count[1]);
  if (bins.count[2] > 0)
    insert_node_block_kernel<<<bins.count[2], block_size>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(2), bins.count[2]);
  check_cuda(cudaGetLastError());

  // ── Overflow handling ────────────────────────────────────────────────
  // Padded overflow sizes -> cumulative offsets of the appended segments.
  temp_buffer<int> tmp(K);
  int num_blocks = (K + block_size - 1) / block_size;
  compute_next_multiple_of_4_kernel<<<num_blocks, block_size>>>(ctx.d_relocation_plan, tmp.get(),
                                                                K);
  thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(tmp.get());
  thrust::inclusive_scan(tmp_ptr, tmp_ptr + K, tmp_ptr);
  update_partial_solution_kernel<<<num_blocks, block_size>>>(ctx.d_relocation_plan, tmp.get(), K);

  // Only the total appended size is needed on the host.
  int total_appended = 0;
  check_cuda(cudaMemcpy(&total_appended, ctx.d_relocation_plan + 3 * (K - 1) + 2, sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (verbose()) {
    std::vector<int> relocation_plan_host_out(K * 3);
    check_cuda(cudaMemcpy(relocation_plan_host_out.data(), ctx.d_relocation_plan,
                          K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    print_vector(relocation_plan_host_out, "Cumulative Relocation Plan");
  }
  if (total_appended > 0) {
    ensure_payload_capacity(ctx, static_cast<long long>(ctx.initial_payload_size) + total_appended);
    if (verbose())
      printf("[%s] Space available from: %d \n", ctx.dataset_name, ctx.initial_payload_size);
    allocate_space_kernel<<<num_blocks, block_size>>>(
        ctx.d_relocation_plan, ctx.d_flat_payload, ctx.initial_payload_size, ctx.d_insert_keys,
        ctx.d_insert_payload, ctx.d_insert_prefix_sizes, K);

    // Fixup metadata for overflowed nodes
    fixup_overflow_metadata_kernel<<<num_blocks, block_size>>>(
        ctx.d_nodes, ctx.d_insert_keys, ctx.d_insert_prefix_sizes, ctx.d_relocation_plan,
        ctx.initial_payload_size, K);
    check_cuda(cudaGetLastError());
    ctx.initial_payload_size += total_appended;
  }

  if (verbose()) {
    std::vector<int> updated_flat(ctx.fixed_size);
    check_cuda(cudaMemcpy(updated_flat.data(), ctx.d_flat_payload, ctx.fixed_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
    print_vector(updated_flat, "Updated Flattened Values (vec1d)");
  }
}

void delete_cbst(const std::vector<int>& delete_keys, cbst_context& ctx) {
  if (delete_keys.empty()) return;
  int delete_size = static_cast<int>(delete_keys.size());
  temp_buffer<int> d_delete_keys(delete_size);
  check_cuda(cudaMemcpyAsync(d_delete_keys.get(), delete_keys.data(), delete_size * sizeof(int),
                             cudaMemcpyHostToDevice, 0));

  // Temporary buffer for located node positions
  temp_buffer<int> d_delete_positions(delete_size);

  int block_size = 256;
  int num_blocks = (delete_size + block_size - 1) / block_size;

  // Phase 1: Read-only traversal to locate targets (no races)
  locate_delete_targets_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, d_delete_keys.get(),
                                                           delete_size, d_delete_positions.get());

  // Phase 2: Apply deletions + mark avail using precomputed positions (no
  // traversal)
  apply_deletes_kernel<<<num_blocks, block_size>>>(ctx.d_nodes, d_delete_positions.get(),
                                                   delete_size, ctx.d_avail);
  check_cuda(cudaGetLastError());

  // Bottom-up level-wise reduction to recompute subtree_avail
  recompute_subtree_avail(ctx);
}

// Functor for thrust::copy_if on a 0/1 stencil
struct is_set {
  __host__ __device__ bool operator()(int x) const {
    return x != 0;
  }
};

insert_mapping insert_cbst(const std::vector<int>& new_keys, const std::vector<int>& new_payload,
                           const std::vector<int>& new_prefix_sizes, cbst_context& ctx) {
  insert_mapping mapping;
  int K = static_cast<int>(new_keys.size());
  mapping.item_to_key.resize(K, 0);
  if (K == 0) return mapping;

  int block_size = 256;
  reserve_scratch(ctx, K, new_payload.size());

  // Copy inputs to device
  check_cuda(cudaMemcpyAsync(ctx.d_insert_keys, new_keys.data(), K * sizeof(int),
                             cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_payload, new_payload.data(),
                             new_payload.size() * sizeof(int), cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, new_prefix_sizes.data(), K * sizeof(int),
                             cudaMemcpyHostToDevice, 0));

  // Determine number of deleted slots (root's subtree_avail)
  int D = 0;
  check_cuda(cudaMemcpy(&D, ctx.d_subtree_avail, sizeof(int), cudaMemcpyDeviceToHost));

  int reuse_k = std::min(K, D);
  std::vector<char> matched(K, 0);

  // ── GPU Best-Fit Matching Pipeline ──────────────────────────────────
  if (reuse_k > 0) {
    // Locate ALL D deleted slots via order-statistic tree
    temp_buffer<int> d_all_positions(D), d_deleted_keys(D), d_slot_caps(D), d_slot_order(D);
    int num_blocks_d = (D + block_size - 1) / block_size;
    locate_reusable_slots_kernel<<<num_blocks_d, block_size>>>(
        ctx.d_subtree_avail, ctx.d_avail, ctx.num_records, d_all_positions.get(), D);

    // Recover original keys of deleted slots via CBST layout formula
    extract_keys_from_positions_kernel<<<num_blocks_d, block_size>>>(
        ctx.d_keys, d_all_positions.get(), d_deleted_keys.get(), ctx.num_records, D);

    // Step 1: Extract slot capacities (GPU parallel)
    extract_slot_capacities_kernel<<<num_blocks_d, block_size>>>(ctx.d_nodes, d_all_positions.get(),
                                                                 d_slot_caps.get(), D);

    // Step 2: Compute item sizes for eligible items (GPU parallel)
    temp_buffer<int> d_item_sizes(reuse_k), d_item_order(reuse_k), d_lo(reuse_k),
        d_prefix_max(reuse_k), d_assigned(reuse_k);
    int num_blocks_r = (reuse_k + block_size - 1) / block_size;
    compute_item_sizes_kernel<<<num_blocks_r, block_size>>>(ctx.d_insert_prefix_sizes,
                                                            d_item_sizes.get(), reuse_k);
    check_cuda(cudaGetLastError());

    // Step 3: Sort slot capacities, remembering which slot each one is
    // (the original sorted the capacities alone and lost the slot identity)
    thrust::device_ptr<int> caps_ptr = thrust::device_pointer_cast(d_slot_caps.get());
    thrust::device_ptr<int> slot_order_ptr = thrust::device_pointer_cast(d_slot_order.get());
    thrust::sequence(slot_order_ptr, slot_order_ptr + D);
    thrust::sort_by_key(caps_ptr, caps_ptr + D, slot_order_ptr);

    // Step 4: Sort item sizes with original-index tracking
    thrust::device_ptr<int> item_order_ptr = thrust::device_pointer_cast(d_item_order.get());
    thrust::sequence(item_order_ptr, item_order_ptr + reuse_k);
    thrust::device_ptr<int> sizes_ptr = thrust::device_pointer_cast(d_item_sizes.get());
    thrust::sort_by_key(sizes_ptr, sizes_ptr + reuse_k, item_order_ptr);

    // Step 5: Binary search — lo[i] = first slot with capacity >=
    // sorted_size[i]
    lower_bound_kernel<<<num_blocks_r, block_size>>>(d_slot_caps.get(), D, d_item_sizes.get(),
                                                     reuse_k, d_lo.get());

    // Step 6: b[i] = lo[i] - i  (in-place, d_lo becomes d_b)
    compute_b_in_place_kernel<<<num_blocks_r, block_size>>>(d_lo.get(), reuse_k);
    check_cuda(cudaGetLastError());

    // Step 7: prefix_max = inclusive_scan(b, max)
    thrust::device_ptr<int> b_ptr = thrust::device_pointer_cast(d_lo.get());
    thrust::device_ptr<int> pmax_ptr = thrust::device_pointer_cast(d_prefix_max.get());
    thrust::inclusive_scan(b_ptr, b_ptr + reuse_k, pmax_ptr, thrust::maximum<int>());

    // Step 8: assigned[i] = i + prefix_max[i] (rank of the slot for sorted
    // item i; strictly increasing, and >= lo[i] so the slot fits the item)
    compute_assigned_kernel<<<num_blocks_r, block_size>>>(d_prefix_max.get(), d_assigned.get(),
                                                          reuse_k);

    // Step 9: Count matched items (assigned[i] < D)
    thrust::device_ptr<int> assigned_ptr = thrust::device_pointer_cast(d_assigned.get());
    int match_count =
        static_cast<int>(thrust::count_if(assigned_ptr, assigned_ptr + reuse_k, less_than{D}));

    if (verbose())
      printf(
          "[%s] GPU best-fit: %d/%d eligible items matched to deleted slots "
          "(D=%d)\n",
          ctx.dataset_name, match_count, reuse_k, D);

    if (match_count > 0) {
      // Step 10: matched (item, slot) pairs: sorted item i -> slot of rank
      // assigned[i]
      temp_buffer<int> d_matched_item_idx(match_count), d_matched_slot_idx(match_count);
      pair_matches_kernel<<<(match_count + block_size - 1) / block_size, block_size>>>(
          d_item_order.get(), d_assigned.get(), d_slot_order.get(), match_count,
          d_matched_item_idx.get(), d_matched_slot_idx.get());

      // D2H: matched pairs and slot keys for the mapping and the binning
      std::vector<int> h_items(match_count), h_slots(match_count), h_keys(D);
      check_cuda(cudaMemcpy(h_items.data(), d_matched_item_idx.get(), match_count * sizeof(int),
                            cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(h_slots.data(), d_matched_slot_idx.get(), match_count * sizeof(int),
                            cudaMemcpyDeviceToHost));
      check_cuda(
          cudaMemcpy(h_keys.data(), d_deleted_keys.get(), D * sizeof(int), cudaMemcpyDeviceToHost));
      std::vector<int> matched_sizes(match_count);
      for (int k = 0; k < match_count; ++k) {
        int item_idx = h_items[k];
        mapping.item_to_key[item_idx] = h_keys[h_slots[k]];
        matched[item_idx] = 1;
        matched_sizes[k] =
            new_prefix_sizes[item_idx] - (item_idx == 0 ? 0 : new_prefix_sizes[item_idx - 1]);
      }

      // ── Degree-binned apply_reuse_thread_kernel dispatch (thread / warp / block) ──────
      degree_bins bins(matched_sizes);
      bins.record(g_bin_launches.reuse);
      if (bins.count[0] > 0)
        apply_reuse_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(0), bins.count[0]);
      if (bins.count[1] > 0)
        apply_reuse_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(1), bins.count[1]);
      if (bins.count[2] > 0)
        apply_reuse_block_kernel<<<bins.count[2], block_size>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(2), bins.count[2]);
      check_cuda(cudaGetLastError());
    }
  }

  // ── Build surplus list ──────────────────────────────────────────────
  // Unmatched items from the first reuse_k plus all items beyond reuse_k.
  std::vector<int> surplus_indices;
  for (int i = 0; i < K; ++i)
    if (!matched[i]) surplus_indices.push_back(i);
  int surplus = static_cast<int>(surplus_indices.size());

  // ── Surplus inserts: append at tail, then reconstruct ───────────────
  if (surplus > 0) {
    auto next_multiple = [](int num, int a) {
      if (num <= 0) return 0;
      int q = (num + a - 1) / a;
      return q * a;
    };
    // Pack every surplus row (values, zero padding, INT_MIN) on the host and
    // append them with a single copy.
    std::vector<int> packed;
    std::vector<cbst_node> surplus_records(surplus);
    const long long base = ctx.initial_payload_size;
    for (int s = 0; s < surplus; ++s) {
      int global_idx = surplus_indices[s];
      int start = (global_idx == 0) ? 0 : new_prefix_sizes[global_idx - 1];
      int len = new_prefix_sizes[global_idx] - start;
      int aligned = next_multiple(len, ctx.alignment);
      cbst_node& r = surplus_records[s];
      r = cbst_node{};
      r.value = static_cast<int>(base + packed.size());
      r.length = aligned + 1;
      r.occupancy = len;
      r.tail_base = r.value;
      r.tail_capacity = aligned;
      packed.insert(packed.end(), new_payload.begin() + start, new_payload.begin() + start + len);
      packed.insert(packed.end(), aligned - len, 0);
      packed.push_back(INT_MIN);
    }
    ensure_payload_capacity(ctx, base + static_cast<long long>(packed.size()));
    check_cuda(cudaMemcpyAsync(ctx.d_flat_payload + base, packed.data(),
                               packed.size() * sizeof(int), cudaMemcpyHostToDevice, 0));
    ctx.initial_payload_size = static_cast<int>(base + packed.size());

    // Reconstruct the CBST from the surviving (non-deleted) nodes plus the
    // surplus rows, on the device. Surviving nodes keep their full records
    // (offset, length, tail metadata) and are brought into key order through
    // their in-order rank (heap position -> rank), then compacted.
    int old_n = ctx.num_records;
    temp_buffer<cbst_node> d_ranked(old_n), d_records(old_n + surplus);
    temp_buffer<int> d_ranked_live(old_n);
    int blocks_old = (old_n + block_size - 1) / block_size;
    rank_order_nodes_kernel<<<blocks_old, block_size>>>(ctx.d_nodes, ctx.d_avail, old_n,
                                                        d_ranked.get(), d_ranked_live.get());
    check_cuda(cudaGetLastError());
    thrust::device_ptr<cbst_node> ranked_ptr(d_ranked.get()), records_ptr(d_records.get());
    thrust::device_ptr<int> live_ptr(d_ranked_live.get());
    int valid_old_count = static_cast<int>(
        thrust::copy_if(ranked_ptr, ranked_ptr + old_n, live_ptr, records_ptr, is_set()) -
        records_ptr);

    // ── Option A: Preserve original keys (no compaction) ────────────
    // Surviving nodes keep their original keys.  Surplus items get keys
    // beyond the current maximum so no external references are invalidated.
    int next_key = 1;
    if (valid_old_count > 0) {
      int last_key = 0;
      check_cuda(cudaMemcpy(&last_key, &d_records.get()[valid_old_count - 1].index, sizeof(int),
                            cudaMemcpyDeviceToHost));
      next_key = last_key + 1;
    }
    for (int i = 0; i < surplus; ++i) {
      surplus_records[i].index = next_key;
      surplus_records[i].size = ctx.initial_payload_size;
      mapping.item_to_key[surplus_indices[i]] = next_key;
      next_key++;
    }
    check_cuda(cudaMemcpyAsync(d_records.get() + valid_old_count, surplus_records.data(),
                               surplus * sizeof(cbst_node), cudaMemcpyHostToDevice, 0));

    int new_n = valid_old_count + surplus;
    check_cuda(cudaFree(ctx.d_keys));
    check_cuda(cudaFree(ctx.d_start_offsets));
    check_cuda(cudaFree(ctx.d_nodes));
    check_cuda(cudaFree(ctx.d_avail));
    check_cuda(cudaFree(ctx.d_subtree_avail));
    ctx.num_records = new_n;
    check_cuda(cudaMalloc(&ctx.d_nodes, new_n * sizeof(cbst_node)));
    check_cuda(cudaMemset(ctx.d_nodes, 0, new_n * sizeof(cbst_node)));
    check_cuda(cudaMalloc(&ctx.d_keys, new_n * sizeof(int)));
    check_cuda(cudaMalloc(&ctx.d_start_offsets, new_n * sizeof(int)));
    check_cuda(cudaMalloc(&ctx.d_avail, new_n * sizeof(int)));
    check_cuda(cudaMalloc(&ctx.d_subtree_avail, new_n * sizeof(int)));
    check_cuda(cudaMemset(ctx.d_avail, 0, new_n * sizeof(int)));
    check_cuda(cudaMemset(ctx.d_subtree_avail, 0, new_n * sizeof(int)));

    int blocks_build = (new_n + block_size - 1) / block_size;
    record_keys_and_starts_kernel<<<blocks_build, block_size>>>(d_records.get(), new_n, ctx.d_keys,
                                                                ctx.d_start_offsets);
    build_empty_tree_kernel<<<blocks_build, block_size>>>(ctx.d_nodes, new_n);
    place_node_records_kernel<<<blocks_build, block_size>>>(ctx.d_nodes, d_records.get(), new_n);
    check_cuda(cudaGetLastError());
  }

  // Recompute subtree_avail bottom-up (reused slots are no longer available)
  recompute_subtree_avail(ctx);

  return mapping;
}

// cbst_store implementation
cbst_store::cbst_store(const char* dataset_name, int payload_capacity, int alignment) {
  ctx_.dataset_name = dataset_name;
  ctx_.fixed_size = payload_capacity;
  ctx_.alignment = alignment;
}

cbst_store::~cbst_store() {
  if (ctx_.d_insert_keys) check_cuda(cudaFree(ctx_.d_insert_keys));
  if (ctx_.d_insert_payload) check_cuda(cudaFree(ctx_.d_insert_payload));
  if (ctx_.d_insert_prefix_sizes) check_cuda(cudaFree(ctx_.d_insert_prefix_sizes));
  if (ctx_.d_relocation_plan) check_cuda(cudaFree(ctx_.d_relocation_plan));
  if (ctx_.d_keys) check_cuda(cudaFree(ctx_.d_keys));
  if (ctx_.d_start_offsets) check_cuda(cudaFree(ctx_.d_start_offsets));
  if (ctx_.d_nodes) check_cuda(cudaFree(ctx_.d_nodes));
  if (ctx_.d_flat_payload) check_cuda(cudaFree(ctx_.d_flat_payload));
  if (ctx_.d_avail) check_cuda(cudaFree(ctx_.d_avail));
  if (ctx_.d_subtree_avail) check_cuda(cudaFree(ctx_.d_subtree_avail));
}

cbst_store::cbst_store(cbst_store&& other) noexcept {
  ctx_ = other.ctx_;
  constructed_ = other.constructed_;
  // Null out other's pointers to avoid double free
  other.ctx_.d_nodes = nullptr;
  other.ctx_.d_keys = nullptr;
  other.ctx_.d_start_offsets = nullptr;
  other.ctx_.d_flat_payload = nullptr;
  other.ctx_.d_insert_keys = nullptr;
  other.ctx_.d_insert_payload = nullptr;
  other.ctx_.d_insert_prefix_sizes = nullptr;
  other.ctx_.d_relocation_plan = nullptr;
  other.ctx_.d_avail = nullptr;
  other.ctx_.d_subtree_avail = nullptr;
  other.constructed_ = false;
}

cbst_store& cbst_store::operator=(cbst_store&& other) noexcept {
  if (this != &other) {
    // Free current resources
    this->~cbst_store();
    // Steal other's resources
    ctx_ = other.ctx_;
    constructed_ = other.constructed_;
    // Null out other's pointers
    other.ctx_.d_nodes = nullptr;
    other.ctx_.d_keys = nullptr;
    other.ctx_.d_start_offsets = nullptr;
    other.ctx_.d_flat_payload = nullptr;
    other.ctx_.d_insert_keys = nullptr;
    other.ctx_.d_insert_payload = nullptr;
    other.ctx_.d_insert_prefix_sizes = nullptr;
    other.ctx_.d_relocation_plan = nullptr;
    other.ctx_.d_avail = nullptr;
    other.ctx_.d_subtree_avail = nullptr;
    other.constructed_ = false;
  }
  return *this;
}

void cbst_store::construct(int* keys, int* start_offsets, int num_records, int* flat_payload,
                           int flat_payload_size) {
  construct_cbst(keys, start_offsets, num_records, flat_payload, flat_payload_size, ctx_.fixed_size,
                 ctx_.dataset_name, ctx_);
  constructed_ = true;
}

void cbst_store::rebuild_from_device(int* d_start_offsets, int num_records, int* d_flat_payload,
                                     int flat_payload_size, int payload_capacity) {
  rebuild_cbst_from_device(d_start_offsets, num_records, d_flat_payload, flat_payload_size,
                           payload_capacity, ctx_);
  constructed_ = true;
}

insert_mapping cbst_store::insert(const std::vector<int>& insert_keys,
                                  const std::vector<int>& insert_payload,
                                  const std::vector<int>& insert_prefix_sizes) {
  return insert_cbst(insert_keys, insert_payload, insert_prefix_sizes, ctx_);
}

void cbst_store::append(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                        const std::vector<int>& insert_prefix_sizes) {
  fill_cbst(insert_keys, insert_payload, insert_prefix_sizes, ctx_);
}

void cbst_store::erase(const std::vector<int>& delete_keys) {
  delete_cbst(delete_keys, ctx_);
}

void cbst_store::remove_values(const std::vector<int>& keys_to_unfill,
                               const std::vector<int>& values_to_remove,
                               const std::vector<int>& remove_prefix_sizes) {
  unfill_cbst(keys_to_unfill, values_to_remove, remove_prefix_sizes, ctx_);
}

void cbst_store::find_and_print(const std::vector<int>& ids) const {
  if (ids.empty()) return;
  int* d_search;
  check_cuda(cudaMalloc(&d_search, ids.size() * sizeof(int)));
  check_cuda(cudaMemcpy(d_search, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
  find_contents_kernel<<<(ids.size() + 256 - 1) / 256, 256>>>(ctx_.d_nodes, d_search, ids.size(),
                                                              ctx_.d_flat_payload);
  check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaFree(d_search));
}

const cbst_context& cbst_store::context() const {
  return ctx_;
}

size_t cbst_store::device_bytes() const {
  size_t n = static_cast<size_t>(ctx_.num_records);
  return n * (sizeof(cbst_node) + 4 * sizeof(int)) +  // nodes, keys, starts, avail, subtree_avail
         static_cast<size_t>(ctx_.fixed_size) * sizeof(int) +
         static_cast<size_t>(ctx_.scratch_item_capacity) * 5 * sizeof(int) +
         static_cast<size_t>(ctx_.scratch_payload_capacity) * sizeof(int);
}

void unfill_cbst(const std::vector<int>& keys_to_unfill, const std::vector<int>& values_to_remove,
                 const std::vector<int>& remove_prefix_sizes, cbst_context& ctx) {
  if (keys_to_unfill.empty()) return;
  int K = static_cast<int>(keys_to_unfill.size());
  reserve_scratch(ctx, K, values_to_remove.size());
  // Reuse insert buffers for passing inputs
  check_cuda(cudaMemcpyAsync(ctx.d_insert_keys, keys_to_unfill.data(), K * sizeof(int),
                             cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_payload, values_to_remove.data(),
                             values_to_remove.size() * sizeof(int), cudaMemcpyHostToDevice, 0));
  check_cuda(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, remove_prefix_sizes.data(),
                             remove_prefix_sizes.size() * sizeof(int), cudaMemcpyHostToDevice, 0));
  int block_size = 256;

  // ── Degree-binned unfill dispatch ────────────────────────────────────
  // Binned by the number of removals per key, a proxy for the work (the
  // row lengths are only known on the device).
  degree_bins bins(item_sizes(remove_prefix_sizes));
  bins.record(g_bin_launches.unfill);
  if (bins.count[0] > 0)
    unfill_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, bins.bin(0), bins.count[0]);
  if (bins.count[1] > 0)
    unfill_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, bins.bin(1), bins.count[1]);
  if (bins.count[2] > 0) {
    size_t shmem = sizeof(int) * static_cast<size_t>(block_size + 1);
    unfill_block_kernel<<<bins.count[2], block_size, shmem>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, bins.bin(2), bins.count[2]);
  }
  check_cuda(cudaGetLastError());
}

}  // namespace dyng::detail::escher
