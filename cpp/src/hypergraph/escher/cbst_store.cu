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
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "hypergraph/escher/cbst_store.hpp"
#include "util/cuda_check.hpp"
#include "util/thrust_allocator.cuh"

#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>
#include <dyng/core/profiler.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

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

// The stream of the store's resources: every kernel, copy and Thrust or CUB call of the store is
// ordered on it (PLAN 4.6 rule 7).
static cudaStream_t stream_of(const cbst_context& ctx) {
  return static_cast<cudaStream_t>(ctx.res.stream().get());
}

// One host synchronization with the store's stream, at a point where ESCHER-GPU reads a value
// back (ADR 0038 lists them); counted per store and by the budget counters.
static void synchronize(cbst_context& ctx) {
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx)));
  note_host_sync();
  ++ctx.host_syncs;
}

// A Thrust algorithm that returns a value to the host waited for the stream (count_if, the end of
// copy_if): counted as a synchronization.
static void count_implicit_synchronization(cbst_context& ctx) {
  note_host_sync();
  ++ctx.host_syncs;
}

// The trace-level debug output of ESCHER-GPU's --verbose mode (the relocation plan, the payload,
// the best-fit counts) goes to the log sink.
static void log_ints(const cbst_context& ctx, const char* what, const std::vector<int>& values) {
  std::ostringstream text;
  text << "dyng: CBST " << ctx.dataset_name << ": " << what << ": [ ";
  for (int v : values) text << v << " ";
  text << "]";
  log_message(log_level::trace, text.str());
}

// Stream-ordered temporary buffer: a buffer of at least one element from the memory resource of
// the store's resources (by default the stream-ordered pool, cudaMallocAsync), ordered on their
// stream, so operations queue their kernels without host round trips or the device-wide
// synchronization of cudaMalloc/cudaFree.
template <class T>
class temp_buffer {
 public:
  temp_buffer(const resources& res, size_t n) : storage_(res, std::max<size_t>(n, 1)) {}
  T* get() {
    return storage_.data();
  }

 private:
  buffer<T> storage_;
};

// (Re)allocates one array of the context from its resources; `view` is the device pointer the
// kernels use. The previous array is released first, so the peak holds one of them.
template <typename value_t>
static void allocate(cbst_context& ctx, buffer<value_t>& storage, value_t*& view, size_t count) {
  storage = buffer<value_t>();
  storage = buffer<value_t>(ctx.res, count);
  view = storage.data();
}

// Releases one array of the context.
template <typename value_t>
static void release(buffer<value_t>& storage, value_t*& view) {
  storage = buffer<value_t>();
  view = nullptr;
}

// Degree binning of per-item work (ESCHER+ thread / warp / block levels):
// items with fewer than 32 values run one per thread, fewer than 1024 one
// per warp, larger ones one per block. The three index lists are uploaded
// with a single copy.
struct degree_bins {
  degree_bins(const resources& res, const std::vector<int>& sizes) : index(res, sizes.size()) {
    const auto stream = static_cast<cudaStream_t>(res.stream().get());
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
      DYNG_CUDA_TRY(cudaMemcpyAsync(index.get(), all.data(), all.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
  }
  // Counts one launch for every non-empty bin.
  void record(long long (&launches)[3]) const {
    for (int b = 0; b < 3; ++b) launches[b] += count[b] > 0;
  }
  int* bin(int b) {
    return index.get() + (b == 0 ? 0 : (b == 1 ? count[0] : count[0] + count[1]));
  }
  temp_buffer<int> index;
  int count[3] = {0, 0, 0};
};

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
    release(ctx.insert_keys_storage, ctx.d_insert_keys);
    release(ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes);
    release(ctx.relocation_plan_storage, ctx.d_relocation_plan);
    allocate(ctx, ctx.insert_keys_storage, ctx.d_insert_keys, num_items);
    allocate(ctx, ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes, num_items);
    allocate(ctx, ctx.relocation_plan_storage, ctx.d_relocation_plan, 3 * num_items);
    ctx.scratch_item_capacity = num_items;
  }
  if (static_cast<long long>(payload_size) > ctx.scratch_payload_capacity) {
    release(ctx.insert_payload_storage, ctx.d_insert_payload);
    allocate(ctx, ctx.insert_payload_storage, ctx.d_insert_payload, payload_size);
    ctx.scratch_payload_capacity = static_cast<long long>(payload_size);
  }
}

// Recomputes subtree_avail (number of deleted slots per subtree) bottom-up,
// one launch per tree level (stream-ordered, no host synchronization).
static void recompute_subtree_avail(cbst_context& ctx) {
  if (ctx.num_records <= 0) return;  // an empty tree has no levels (dynG addition)
  const int block_size = 256;
  int last_level_start = 1;
  while (last_level_start * 2 <= ctx.num_records) last_level_start <<= 1;
  int level_start = last_level_start - 1;
  for (int level_end = ctx.num_records - 1; level_start >= 0;) {
    int count = level_end - level_start + 1;
    int blocks = (count + block_size - 1) / block_size;
    reduce_avail_level_kernel<<<blocks, block_size, 0, stream_of(ctx)>>>(
        level_start, level_end, ctx.num_records, ctx.d_avail, ctx.d_subtree_avail);
    DYNG_CHECK_KERNEL(stream_of(ctx));
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
    throw capacity_error("dyng: CBST " + ctx.dataset_name + ": " + std::to_string(needed) +
                         " payload entries exceed the int32 offsets of the CBST (at most " +
                         std::to_string(INT_MAX) + ")");
  }
  buffer<int> grown_storage(ctx.res, static_cast<size_t>(capacity));
  int* grown = grown_storage.data();
  DYNG_CUDA_TRY(cudaMemcpyAsync(grown, ctx.d_flat_payload,
                                static_cast<size_t>(ctx.initial_payload_size) * sizeof(int),
                                cudaMemcpyDeviceToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemsetAsync(grown + ctx.initial_payload_size, 0,
                                (capacity - ctx.initial_payload_size) * sizeof(int),
                                stream_of(ctx)));
  ctx.flat_payload_storage = std::move(grown_storage);
  ctx.d_flat_payload = grown;
  ctx.fixed_size = static_cast<int>(capacity);
}

void construct_cbst(int* keys, int* start_offsets, int num_records, int* flat_payload,
                    int flat_payload_size, int payload_capacity, const std::string& dataset_name,
                    cbst_context& ctx) {
  // A capacity below the payload is rejected before anything changes or is allocated (ESCHER-GPU
  // printed "Overflow" and returned a tree without nodes).
  DYNG_EXPECTS(num_records == 0 || payload_capacity >= flat_payload_size, "CBST ", dataset_name,
               ": payload capacity ", payload_capacity, " is below the payload size ",
               flat_payload_size);
  ctx.fixed_size = payload_capacity;
  ctx.num_records = num_records;
  ctx.initial_payload_size = flat_payload_size;
  ctx.dataset_name = dataset_name;
  // ctx.alignment should be set by the owner (cbst_store) before calling

  if (num_records == 0) {
    // An empty tree (dynG addition): its payload is allocated as for any other tree, so that a
    // later insert or append has a buffer to write to (ESCHER-GPU and MOSP_ESCHER kept none, and
    // an insert into the empty tree failed).
    allocate(ctx, ctx.flat_payload_storage, ctx.d_flat_payload, ctx.fixed_size);
    if (ctx.fixed_size > 0)
      DYNG_CUDA_TRY(
          cudaMemsetAsync(ctx.d_flat_payload, 0, ctx.fixed_size * sizeof(int), stream_of(ctx)));
    return;
  }

  allocate(ctx, ctx.nodes_storage, ctx.d_nodes, num_records);
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_nodes, 0, num_records * sizeof(cbst_node), stream_of(ctx)));
  allocate(ctx, ctx.keys_storage, ctx.d_keys, num_records);
  allocate(ctx, ctx.start_offsets_storage, ctx.d_start_offsets, num_records);
  allocate(ctx, ctx.flat_payload_storage, ctx.d_flat_payload, ctx.fixed_size);

  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_keys, keys, num_records * sizeof(int), cudaMemcpyHostToDevice,
                                stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_start_offsets, start_offsets, num_records * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));

  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_flat_payload, flat_payload, flat_payload_size * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_flat_payload + flat_payload_size, 0,
                                (ctx.fixed_size - flat_payload_size) * sizeof(int),
                                stream_of(ctx)));

  allocate(ctx, ctx.insert_keys_storage, ctx.d_insert_keys, num_records);
  allocate(ctx, ctx.insert_payload_storage, ctx.d_insert_payload, num_records * 3);
  allocate(ctx, ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes, num_records);
  allocate(ctx, ctx.relocation_plan_storage, ctx.d_relocation_plan, 3 * num_records);
  ctx.scratch_item_capacity = num_records;
  ctx.scratch_payload_capacity = 3LL * num_records;
  // Availability arrays (0/1 per node) and subtree sums (per node)
  allocate(ctx, ctx.avail_storage, ctx.d_avail, num_records);
  allocate(ctx, ctx.subtree_avail_storage, ctx.d_subtree_avail, num_records);
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_avail, 0, num_records * sizeof(int), stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_subtree_avail, 0, num_records * sizeof(int), stream_of(ctx)));

  int block_size = 256;
  int num_blocks = (num_records + block_size - 1) / block_size;

  build_empty_tree_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(ctx.d_nodes, num_records);
  store_items_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_keys, ctx.d_start_offsets, num_records, flat_payload_size);
  init_tail_metadata_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_flat_payload, num_records);
  DYNG_CHECK_KERNEL(stream_of(ctx));

  if (log_enabled(log_level::debug)) {
    log_message(log_level::debug, "dyng: CBST " + dataset_name + ": constructed " +
                                      std::to_string(num_records) + " records over " +
                                      std::to_string(flat_payload_size) + " payload entries");
  }
}

void rebuild_cbst_from_device(buffer<int> start_offsets, int num_records, buffer<int> flat_payload,
                              int flat_payload_size, int payload_capacity, cbst_context& ctx) {
  DYNG_EXPECTS(start_offsets.size() >= static_cast<size_t>(std::max(num_records, 0)) &&
                   flat_payload.size() >= static_cast<size_t>(std::max(payload_capacity, 0)) &&
                   flat_payload_size <= payload_capacity,
               "CBST ", ctx.dataset_name, ": rebuild_from_device: ", start_offsets.size(),
               " row offsets and a payload of ", flat_payload.size(), " for ", num_records,
               " rows, ", flat_payload_size, " used of a capacity of ", payload_capacity);
  release(ctx.nodes_storage, ctx.d_nodes);
  release(ctx.keys_storage, ctx.d_keys);
  release(ctx.start_offsets_storage, ctx.d_start_offsets);
  release(ctx.flat_payload_storage, ctx.d_flat_payload);
  release(ctx.avail_storage, ctx.d_avail);
  release(ctx.subtree_avail_storage, ctx.d_subtree_avail);
  ctx.start_offsets_storage = std::move(start_offsets);
  ctx.d_start_offsets = ctx.start_offsets_storage.data();
  ctx.flat_payload_storage = std::move(flat_payload);
  ctx.d_flat_payload = ctx.flat_payload_storage.data();
  ctx.fixed_size = payload_capacity;
  ctx.initial_payload_size = flat_payload_size;
  ctx.num_records = num_records;
  allocate(ctx, ctx.nodes_storage, ctx.d_nodes, num_records);
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_nodes, 0, num_records * sizeof(cbst_node), stream_of(ctx)));
  allocate(ctx, ctx.keys_storage, ctx.d_keys, num_records);
  allocate(ctx, ctx.avail_storage, ctx.d_avail, num_records);
  allocate(ctx, ctx.subtree_avail_storage, ctx.d_subtree_avail, num_records);
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_avail, 0, num_records * sizeof(int), stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_subtree_avail, 0, num_records * sizeof(int), stream_of(ctx)));
  thrust_allocator alloc(ctx.res);
  thrust::sequence(thrust_policy(ctx.res, alloc), thrust::device_pointer_cast(ctx.d_keys),
                   thrust::device_pointer_cast(ctx.d_keys) + num_records, 1);

  int block_size = 256;
  int num_blocks = (num_records + block_size - 1) / block_size;
  build_empty_tree_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(ctx.d_nodes, num_records);
  store_items_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_keys, ctx.d_start_offsets, num_records, flat_payload_size);
  init_tail_metadata_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_flat_payload, num_records);
  DYNG_CHECK_KERNEL(stream_of(ctx));
}

void fill_cbst(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
               const std::vector<int>& insert_prefix_sizes, cbst_context& ctx) {
  if (insert_keys.empty()) return;
  int K = static_cast<int>(insert_keys.size());
  reserve_scratch(ctx, K, insert_payload.size());

  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_keys, insert_keys.data(), K * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_payload, insert_payload.data(),
                                insert_payload.size() * sizeof(int), cudaMemcpyHostToDevice,
                                stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, insert_prefix_sizes.data(),
                                K * sizeof(int), cudaMemcpyHostToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_relocation_plan, 0, 3 * K * sizeof(int), stream_of(ctx)));

  int block_size = 256;

  // ── Degree-binned insert_node dispatch ────────────────────────────────
  degree_bins bins(ctx.res, item_sizes(insert_prefix_sizes));
  bins.record(ctx.bin_launches.fill);
  if (bins.count[0] > 0)
    insert_node_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size, 0,
                                stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(0), bins.count[0]);
  if (bins.count[1] > 0)
    insert_node_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size, 0,
                              stream_of(ctx)>>>(ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys,
                                                ctx.d_insert_payload, ctx.d_insert_prefix_sizes,
                                                ctx.d_relocation_plan, bins.bin(1), bins.count[1]);
  if (bins.count[2] > 0)
    insert_node_block_kernel<<<bins.count[2], block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(2), bins.count[2]);
  DYNG_CHECK_KERNEL(stream_of(ctx));

  // ── Overflow handling ────────────────────────────────────────────────
  // Padded overflow sizes -> cumulative offsets of the appended segments.
  temp_buffer<int> tmp(ctx.res, K);
  int num_blocks = (K + block_size - 1) / block_size;
  compute_next_multiple_of_4_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_relocation_plan, tmp.get(), K);
  thrust::device_ptr<int> tmp_ptr = thrust::device_pointer_cast(tmp.get());
  thrust_allocator alloc(ctx.res);
  thrust::inclusive_scan(thrust_policy(ctx.res, alloc), tmp_ptr, tmp_ptr + K, tmp_ptr);
  update_partial_solution_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_relocation_plan, tmp.get(), K);

  // Only the total appended size is needed on the host.
  int total_appended = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&total_appended, ctx.d_relocation_plan + 3 * (K - 1) + 2,
                                sizeof(int), cudaMemcpyDeviceToHost, stream_of(ctx)));
  synchronize(ctx);
  if (log_enabled(log_level::trace)) {
    std::vector<int> relocation_plan_host_out(K * 3);
    DYNG_CUDA_TRY(cudaMemcpyAsync(relocation_plan_host_out.data(), ctx.d_relocation_plan,
                                  K * 3 * sizeof(int), cudaMemcpyDeviceToHost, stream_of(ctx)));
    synchronize(ctx);
    log_ints(ctx, "cumulative relocation plan", relocation_plan_host_out);
  }
  if (total_appended > 0) {
    ensure_payload_capacity(ctx, static_cast<long long>(ctx.initial_payload_size) + total_appended);
    if (log_enabled(log_level::trace))
      log_message(log_level::trace, "dyng: CBST " + ctx.dataset_name + ": space available from " +
                                        std::to_string(ctx.initial_payload_size));
    allocate_space_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
        ctx.d_relocation_plan, ctx.d_flat_payload, ctx.initial_payload_size, ctx.d_insert_keys,
        ctx.d_insert_payload, ctx.d_insert_prefix_sizes, K);

    // Fixup metadata for overflowed nodes
    fixup_overflow_metadata_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_insert_keys, ctx.d_insert_prefix_sizes, ctx.d_relocation_plan,
        ctx.initial_payload_size, K);
    DYNG_CHECK_KERNEL(stream_of(ctx));
    ctx.initial_payload_size += total_appended;
  }

  if (log_enabled(log_level::trace)) {
    std::vector<int> updated_flat(ctx.fixed_size);
    DYNG_CUDA_TRY(cudaMemcpyAsync(updated_flat.data(), ctx.d_flat_payload,
                                  ctx.fixed_size * sizeof(int), cudaMemcpyDeviceToHost,
                                  stream_of(ctx)));
    synchronize(ctx);
    log_ints(ctx, "updated payload", updated_flat);
  }
}

void delete_cbst(const std::vector<int>& delete_keys, cbst_context& ctx) {
  if (delete_keys.empty()) return;
  int delete_size = static_cast<int>(delete_keys.size());
  temp_buffer<int> d_delete_keys(ctx.res, delete_size);
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_delete_keys.get(), delete_keys.data(), delete_size * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));

  // Temporary buffer for located node positions
  temp_buffer<int> d_delete_positions(ctx.res, delete_size);

  int block_size = 256;
  int num_blocks = (delete_size + block_size - 1) / block_size;

  // Phase 1: Read-only traversal to locate targets (no races)
  locate_delete_targets_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, d_delete_keys.get(), delete_size, d_delete_positions.get());

  // Phase 2: Apply deletions + mark avail using precomputed positions (no
  // traversal)
  apply_deletes_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, d_delete_positions.get(), delete_size, ctx.d_avail);
  DYNG_CHECK_KERNEL(stream_of(ctx));

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
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_keys, new_keys.data(), K * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_payload, new_payload.data(),
                                new_payload.size() * sizeof(int), cudaMemcpyHostToDevice,
                                stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, new_prefix_sizes.data(), K * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));

  // Determine number of deleted slots (root's subtree_avail)
  // (An empty tree has no deleted slots and nothing to read; dynG addition.)
  int D = 0;
  if (ctx.num_records > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(&D, ctx.d_subtree_avail, sizeof(int), cudaMemcpyDeviceToHost,
                                  stream_of(ctx)));
    synchronize(ctx);
  }

  int reuse_k = std::min(K, D);
  std::vector<char> matched(K, 0);

  // ── GPU Best-Fit Matching Pipeline ──────────────────────────────────
  if (reuse_k > 0) {
    // Locate ALL D deleted slots via order-statistic tree
    temp_buffer<int> d_all_positions(ctx.res, D), d_deleted_keys(ctx.res, D),
        d_slot_caps(ctx.res, D), d_slot_order(ctx.res, D);
    int num_blocks_d = (D + block_size - 1) / block_size;
    locate_reusable_slots_kernel<<<num_blocks_d, block_size, 0, stream_of(ctx)>>>(
        ctx.d_subtree_avail, ctx.d_avail, ctx.num_records, d_all_positions.get(), D);

    // Recover original keys of deleted slots via CBST layout formula
    extract_keys_from_positions_kernel<<<num_blocks_d, block_size, 0, stream_of(ctx)>>>(
        ctx.d_keys, d_all_positions.get(), d_deleted_keys.get(), ctx.num_records, D);

    // Step 1: Extract slot capacities (GPU parallel)
    extract_slot_capacities_kernel<<<num_blocks_d, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, d_all_positions.get(), d_slot_caps.get(), D);

    // Step 2: Compute item sizes for eligible items (GPU parallel)
    temp_buffer<int> d_item_sizes(ctx.res, reuse_k), d_item_order(ctx.res, reuse_k),
        d_lo(ctx.res, reuse_k), d_prefix_max(ctx.res, reuse_k), d_assigned(ctx.res, reuse_k);
    int num_blocks_r = (reuse_k + block_size - 1) / block_size;
    compute_item_sizes_kernel<<<num_blocks_r, block_size, 0, stream_of(ctx)>>>(
        ctx.d_insert_prefix_sizes, d_item_sizes.get(), reuse_k);
    DYNG_CHECK_KERNEL(stream_of(ctx));

    // Step 3: Sort slot capacities, remembering which slot each one is
    // (the original sorted the capacities alone and lost the slot identity)
    thrust::device_ptr<int> caps_ptr = thrust::device_pointer_cast(d_slot_caps.get());
    thrust::device_ptr<int> slot_order_ptr = thrust::device_pointer_cast(d_slot_order.get());
    thrust_allocator alloc(ctx.res);
    thrust::sequence(thrust_policy(ctx.res, alloc), slot_order_ptr, slot_order_ptr + D);
    // A stable sort (ESCHER-GPU's sort_by_key on int keys is Thrust's radix sort, which is
    // stable): slots of equal capacity stay in the order of the order-statistic walk, which is
    // key order (ADR 0038, the tie rule).
    thrust::stable_sort_by_key(thrust_policy(ctx.res, alloc), caps_ptr, caps_ptr + D,
                               slot_order_ptr);

    // Step 4: Sort item sizes with original-index tracking
    thrust::device_ptr<int> item_order_ptr = thrust::device_pointer_cast(d_item_order.get());
    thrust::sequence(thrust_policy(ctx.res, alloc), item_order_ptr, item_order_ptr + reuse_k);
    thrust::device_ptr<int> sizes_ptr = thrust::device_pointer_cast(d_item_sizes.get());
    // Stable: items of equal size stay in input order (the tie rule).
    thrust::stable_sort_by_key(thrust_policy(ctx.res, alloc), sizes_ptr, sizes_ptr + reuse_k,
                               item_order_ptr);

    // Step 5: Binary search — lo[i] = first slot with capacity >=
    // sorted_size[i]
    lower_bound_kernel<<<num_blocks_r, block_size, 0, stream_of(ctx)>>>(
        d_slot_caps.get(), D, d_item_sizes.get(), reuse_k, d_lo.get());

    // Step 6: b[i] = lo[i] - i  (in-place, d_lo becomes d_b)
    compute_b_in_place_kernel<<<num_blocks_r, block_size, 0, stream_of(ctx)>>>(d_lo.get(), reuse_k);
    DYNG_CHECK_KERNEL(stream_of(ctx));

    // Step 7: prefix_max = inclusive_scan(b, max)
    thrust::device_ptr<int> b_ptr = thrust::device_pointer_cast(d_lo.get());
    thrust::device_ptr<int> pmax_ptr = thrust::device_pointer_cast(d_prefix_max.get());
    thrust::inclusive_scan(thrust_policy(ctx.res, alloc), b_ptr, b_ptr + reuse_k, pmax_ptr,
                           thrust::maximum<int>());

    // Step 8: assigned[i] = i + prefix_max[i] (rank of the slot for sorted
    // item i; strictly increasing, and >= lo[i] so the slot fits the item)
    compute_assigned_kernel<<<num_blocks_r, block_size, 0, stream_of(ctx)>>>(
        d_prefix_max.get(), d_assigned.get(), reuse_k);

    // Step 9: Count matched items (assigned[i] < D)
    thrust::device_ptr<int> assigned_ptr = thrust::device_pointer_cast(d_assigned.get());
    int match_count = static_cast<int>(thrust::count_if(thrust_policy(ctx.res, alloc), assigned_ptr,
                                                        assigned_ptr + reuse_k, less_than{D}));
    count_implicit_synchronization(ctx);

    if (log_enabled(log_level::debug))
      log_message(log_level::debug,
                  "dyng: CBST " + ctx.dataset_name + ": GPU best-fit: " +
                      std::to_string(match_count) + "/" + std::to_string(reuse_k) +
                      " eligible items matched to deleted slots (D=" + std::to_string(D) + ")");

    if (match_count > 0) {
      // Step 10: matched (item, slot) pairs: sorted item i -> slot of rank
      // assigned[i]
      temp_buffer<int> d_matched_item_idx(ctx.res, match_count),
          d_matched_slot_idx(ctx.res, match_count);
      pair_matches_kernel<<<(match_count + block_size - 1) / block_size, block_size, 0,
                            stream_of(ctx)>>>(d_item_order.get(), d_assigned.get(),
                                              d_slot_order.get(), match_count,
                                              d_matched_item_idx.get(), d_matched_slot_idx.get());

      // D2H: matched pairs and slot keys for the mapping and the binning
      std::vector<int> h_items(match_count), h_slots(match_count), h_keys(D);
      DYNG_CUDA_TRY(cudaMemcpyAsync(h_items.data(), d_matched_item_idx.get(),
                                    match_count * sizeof(int), cudaMemcpyDeviceToHost,
                                    stream_of(ctx)));
      DYNG_CUDA_TRY(cudaMemcpyAsync(h_slots.data(), d_matched_slot_idx.get(),
                                    match_count * sizeof(int), cudaMemcpyDeviceToHost,
                                    stream_of(ctx)));
      DYNG_CUDA_TRY(cudaMemcpyAsync(h_keys.data(), d_deleted_keys.get(), D * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream_of(ctx)));
      synchronize(ctx);
      std::vector<int> matched_sizes(match_count);
      for (int k = 0; k < match_count; ++k) {
        int item_idx = h_items[k];
        mapping.item_to_key[item_idx] = h_keys[h_slots[k]];
        matched[item_idx] = 1;
        matched_sizes[k] =
            new_prefix_sizes[item_idx] - (item_idx == 0 ? 0 : new_prefix_sizes[item_idx - 1]);
      }

      // ── Degree-binned apply_reuse_thread_kernel dispatch (thread / warp / block) ──────
      degree_bins bins(ctx.res, matched_sizes);
      bins.record(ctx.bin_launches.reuse);
      if (bins.count[0] > 0)
        apply_reuse_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size, 0,
                                    stream_of(ctx)>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(0), bins.count[0]);
      if (bins.count[1] > 0)
        apply_reuse_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size, 0,
                                  stream_of(ctx)>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(1), bins.count[1]);
      if (bins.count[2] > 0)
        apply_reuse_block_kernel<<<bins.count[2], block_size, 0, stream_of(ctx)>>>(
            ctx.d_nodes, ctx.d_flat_payload, ctx.d_avail, d_all_positions.get(),
            ctx.d_insert_payload, ctx.d_insert_prefix_sizes, d_matched_item_idx.get(),
            d_matched_slot_idx.get(), d_deleted_keys.get(), bins.bin(2), bins.count[2]);
      DYNG_CHECK_KERNEL(stream_of(ctx));
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
    DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_flat_payload + base, packed.data(),
                                  packed.size() * sizeof(int), cudaMemcpyHostToDevice,
                                  stream_of(ctx)));
    ctx.initial_payload_size = static_cast<int>(base + packed.size());

    // Reconstruct the CBST from the surviving (non-deleted) nodes plus the
    // surplus rows, on the device. Surviving nodes keep their full records
    // (offset, length, tail metadata) and are brought into key order through
    // their in-order rank (heap position -> rank), then compacted.
    int old_n = ctx.num_records;
    temp_buffer<cbst_node> d_ranked(ctx.res, old_n), d_records(ctx.res, old_n + surplus);
    temp_buffer<int> d_ranked_live(ctx.res, old_n);
    int blocks_old = (old_n + block_size - 1) / block_size;
    thrust_allocator surplus_alloc(ctx.res);
    thrust::device_ptr<cbst_node> ranked_ptr(d_ranked.get()), records_ptr(d_records.get());
    thrust::device_ptr<int> live_ptr(d_ranked_live.get());
    int valid_old_count = 0;
    if (old_n > 0) {  // an empty tree has no surviving records (dynG addition)
      rank_order_nodes_kernel<<<blocks_old, block_size, 0, stream_of(ctx)>>>(
          ctx.d_nodes, ctx.d_avail, old_n, d_ranked.get(), d_ranked_live.get());
      DYNG_CHECK_KERNEL(stream_of(ctx));
      valid_old_count =
          static_cast<int>(thrust::copy_if(thrust_policy(ctx.res, surplus_alloc), ranked_ptr,
                                           ranked_ptr + old_n, live_ptr, records_ptr, is_set()) -
                           records_ptr);
      count_implicit_synchronization(ctx);
    }

    // ── Option A: Preserve original keys (no compaction) ────────────
    // Surviving nodes keep their original keys.  Surplus items get keys
    // beyond the current maximum so no external references are invalidated.
    int next_key = 1;
    if (valid_old_count > 0) {
      int last_key = 0;
      DYNG_CUDA_TRY(cudaMemcpyAsync(&last_key, &d_records.get()[valid_old_count - 1].index,
                                    sizeof(int), cudaMemcpyDeviceToHost, stream_of(ctx)));
      synchronize(ctx);
      next_key = last_key + 1;
    }
    for (int i = 0; i < surplus; ++i) {
      surplus_records[i].index = next_key;
      surplus_records[i].size = ctx.initial_payload_size;
      mapping.item_to_key[surplus_indices[i]] = next_key;
      next_key++;
    }
    DYNG_CUDA_TRY(cudaMemcpyAsync(d_records.get() + valid_old_count, surplus_records.data(),
                                  surplus * sizeof(cbst_node), cudaMemcpyHostToDevice,
                                  stream_of(ctx)));

    int new_n = valid_old_count + surplus;
    release(ctx.keys_storage, ctx.d_keys);
    release(ctx.start_offsets_storage, ctx.d_start_offsets);
    release(ctx.nodes_storage, ctx.d_nodes);
    release(ctx.avail_storage, ctx.d_avail);
    release(ctx.subtree_avail_storage, ctx.d_subtree_avail);
    ctx.num_records = new_n;
    allocate(ctx, ctx.nodes_storage, ctx.d_nodes, new_n);
    DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_nodes, 0, new_n * sizeof(cbst_node), stream_of(ctx)));
    allocate(ctx, ctx.keys_storage, ctx.d_keys, new_n);
    allocate(ctx, ctx.start_offsets_storage, ctx.d_start_offsets, new_n);
    allocate(ctx, ctx.avail_storage, ctx.d_avail, new_n);
    allocate(ctx, ctx.subtree_avail_storage, ctx.d_subtree_avail, new_n);
    DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_avail, 0, new_n * sizeof(int), stream_of(ctx)));
    DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_subtree_avail, 0, new_n * sizeof(int), stream_of(ctx)));

    int blocks_build = (new_n + block_size - 1) / block_size;
    record_keys_and_starts_kernel<<<blocks_build, block_size, 0, stream_of(ctx)>>>(
        d_records.get(), new_n, ctx.d_keys, ctx.d_start_offsets);
    build_empty_tree_kernel<<<blocks_build, block_size, 0, stream_of(ctx)>>>(ctx.d_nodes, new_n);
    place_node_records_kernel<<<blocks_build, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, d_records.get(), new_n);
    DYNG_CHECK_KERNEL(stream_of(ctx));
  }

  // Recompute subtree_avail bottom-up (reused slots are no longer available)
  recompute_subtree_avail(ctx);

  return mapping;
}

// cbst_store implementation
cbst_store::cbst_store(const resources& res, std::string dataset_name, int payload_capacity,
                       int alignment) {
  DYNG_EXPECTS(res.get_backend() == backend::cuda, "CBST ", dataset_name,
               ": the store needs resources of the CUDA backend");
  ctx_.res = res;
  ctx_.dataset_name = std::move(dataset_name);
  ctx_.fixed_size = payload_capacity;
  ctx_.alignment = alignment;
}

cbst_store::~cbst_store() = default;

// The device pointers of the context are views of its buffers, which a move keeps; the source's
// views are cleared with them.
static void clear_views(cbst_context& ctx) {
  ctx.d_nodes = nullptr;
  ctx.d_keys = nullptr;
  ctx.d_start_offsets = nullptr;
  ctx.d_flat_payload = nullptr;
  ctx.d_insert_keys = nullptr;
  ctx.d_insert_payload = nullptr;
  ctx.d_insert_prefix_sizes = nullptr;
  ctx.d_relocation_plan = nullptr;
  ctx.d_avail = nullptr;
  ctx.d_subtree_avail = nullptr;
}

cbst_store::cbst_store(cbst_store&& other) noexcept
    : ctx_(std::move(other.ctx_)), constructed_(other.constructed_), poisoned_(other.poisoned_) {
  clear_views(other.ctx_);
  other.constructed_ = false;
}

cbst_store& cbst_store::operator=(cbst_store&& other) noexcept {
  if (this != &other) {
    ctx_ = std::move(other.ctx_);
    constructed_ = other.constructed_;
    poisoned_ = other.poisoned_;
    clear_views(other.ctx_);
    other.constructed_ = false;
  }
  return *this;
}

// Every operation that changes the tree marks the store poisoned when it fails after it started
// changing it (PLAN 4.7.3, basic guarantee): its state is then undefined, and every later
// operation but construct and rebuild_from_device throws.
void cbst_store::check_not_poisoned(const char* op) const {
  if (poisoned_) {
    throw invalid_argument_error("dyng: CBST " + ctx_.dataset_name + ": " + op +
                                 " on a store that an earlier failed operation left poisoned; "
                                 "construct or rebuild it first");
  }
}

template <typename op_t>
auto cbst_store::guarded(const char* op, op_t&& body) {
  check_not_poisoned(op);
  if (!constructed_) {
    throw invalid_argument_error("dyng: CBST " + ctx_.dataset_name + ": " + op +
                                 " before construct or rebuild_from_device");
  }
  try {
    return body();
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

void cbst_store::construct(int* keys, int* start_offsets, int num_records, int* flat_payload,
                           int flat_payload_size) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.construct");
  try {
    construct_cbst(keys, start_offsets, num_records, flat_payload, flat_payload_size,
                   ctx_.fixed_size, ctx_.dataset_name, ctx_);
  } catch (const invalid_argument_error&) {
    throw;  // the validation, before anything changed
  } catch (...) {
    poisoned_ = true;
    throw;
  }
  poisoned_ = false;
  constructed_ = true;
}

void cbst_store::rebuild_from_device(buffer<int> start_offsets, int num_records,
                                     buffer<int> flat_payload, int flat_payload_size,
                                     int payload_capacity) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.rebuild");
  try {
    rebuild_cbst_from_device(std::move(start_offsets), num_records, std::move(flat_payload),
                             flat_payload_size, payload_capacity, ctx_);
  } catch (...) {
    poisoned_ = true;
    throw;
  }
  poisoned_ = false;
  constructed_ = true;
}

insert_mapping cbst_store::insert(const std::vector<int>& insert_keys,
                                  const std::vector<int>& insert_payload,
                                  const std::vector<int>& insert_prefix_sizes) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.insert");
  return guarded("insert", [&] {
    return insert_cbst(insert_keys, insert_payload, insert_prefix_sizes, ctx_);
  });
}

void cbst_store::append(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                        const std::vector<int>& insert_prefix_sizes) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.append");
  guarded("append", [&] { fill_cbst(insert_keys, insert_payload, insert_prefix_sizes, ctx_); });
}

void cbst_store::erase(const std::vector<int>& delete_keys) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.erase");
  guarded("erase", [&] { delete_cbst(delete_keys, ctx_); });
}

void cbst_store::remove_values(const std::vector<int>& keys_to_unfill,
                               const std::vector<int>& values_to_remove,
                               const std::vector<int>& remove_prefix_sizes) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.remove_values");
  guarded("remove_values",
          [&] { unfill_cbst(keys_to_unfill, values_to_remove, remove_prefix_sizes, ctx_); });
}

// A device copy of one array of a context into the matching array of another (same size).
template <typename value_t>
static void copy_array(cbst_context& to, buffer<value_t>& to_storage, value_t*& to_view,
                       const buffer<value_t>& from) {
  allocate(to, to_storage, to_view, from.size());
  if (!from.empty())
    DYNG_CUDA_TRY(cudaMemcpyAsync(to_view, from.data(), from.size() * sizeof(value_t),
                                  cudaMemcpyDeviceToDevice, stream_of(to)));
}

cbst_store cbst_store::clone(const resources& res) const {
  check_not_poisoned("clone");
  const scoped_device guard(ctx_.res.device());
  cbst_store copy(res, ctx_.dataset_name, ctx_.fixed_size, ctx_.alignment);
  cbst_context& c = copy.ctx_;
  // The copy is ordered after the work queued on the source's stream.
  if (stream_of(c) != stream_of(ctx_)) {
    cudaEvent_t ready;
    DYNG_CUDA_TRY(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    DYNG_CUDA_TRY(cudaEventRecord(ready, stream_of(ctx_)));
    DYNG_CUDA_TRY(cudaStreamWaitEvent(stream_of(c), ready, 0));
    DYNG_CUDA_TRY(cudaEventDestroy(ready));
  }
  c.num_records = ctx_.num_records;
  c.initial_payload_size = ctx_.initial_payload_size;
  c.scratch_item_capacity = ctx_.scratch_item_capacity;
  c.scratch_payload_capacity = ctx_.scratch_payload_capacity;
  copy_array(c, c.nodes_storage, c.d_nodes, ctx_.nodes_storage);
  copy_array(c, c.keys_storage, c.d_keys, ctx_.keys_storage);
  copy_array(c, c.start_offsets_storage, c.d_start_offsets, ctx_.start_offsets_storage);
  copy_array(c, c.flat_payload_storage, c.d_flat_payload, ctx_.flat_payload_storage);
  copy_array(c, c.avail_storage, c.d_avail, ctx_.avail_storage);
  copy_array(c, c.subtree_avail_storage, c.d_subtree_avail, ctx_.subtree_avail_storage);
  // The operation scratch: the same capacities (its contents are inputs of the next operation).
  allocate(c, c.insert_keys_storage, c.d_insert_keys, ctx_.insert_keys_storage.size());
  allocate(c, c.insert_payload_storage, c.d_insert_payload, ctx_.insert_payload_storage.size());
  allocate(c, c.insert_prefix_sizes_storage, c.d_insert_prefix_sizes,
           ctx_.insert_prefix_sizes_storage.size());
  allocate(c, c.relocation_plan_storage, c.d_relocation_plan, ctx_.relocation_plan_storage.size());
  if (c.nodes_storage.size() > 0) {
    const int n = static_cast<int>(c.nodes_storage.size());
    rewire_tree_kernel<<<(n + 255) / 256, 256, 0, stream_of(c)>>>(c.d_nodes, n);
    DYNG_CHECK_KERNEL(stream_of(c));
  }
  // Later work on the source's stream waits for the copy's reads.
  if (stream_of(c) != stream_of(ctx_)) {
    cudaEvent_t copied;
    DYNG_CUDA_TRY(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    DYNG_CUDA_TRY(cudaEventRecord(copied, stream_of(c)));
    DYNG_CUDA_TRY(cudaStreamWaitEvent(stream_of(ctx_), copied, 0));
    DYNG_CUDA_TRY(cudaEventDestroy(copied));
  }
  copy.constructed_ = constructed_;
  return copy;
}

// A device array on the host (part of download()).
template <typename value_t>
static std::vector<value_t> download_array(const cbst_context& ctx, const value_t* device,
                                           size_t count) {
  std::vector<value_t> host(count);
  if (count > 0 && device != nullptr)
    DYNG_CUDA_TRY(cudaMemcpyAsync(host.data(), device, count * sizeof(value_t),
                                  cudaMemcpyDeviceToHost, stream_of(ctx)));
  return host;
}

cbst_host_state cbst_store::download() const {
  const scoped_device guard(ctx_.res.device());
  cbst_host_state h;
  h.num_records = ctx_.num_records;
  h.fixed_size = ctx_.fixed_size;
  h.initial_payload_size = ctx_.initial_payload_size;
  h.alignment = ctx_.alignment;
  h.scratch_item_capacity = ctx_.scratch_item_capacity;
  h.scratch_payload_capacity = ctx_.scratch_payload_capacity;
  const size_t n = ctx_.d_nodes != nullptr ? static_cast<size_t>(std::max(ctx_.num_records, 0)) : 0;
  const std::vector<cbst_node> nodes = download_array(ctx_, ctx_.d_nodes, n);
  h.keys = download_array(ctx_, ctx_.d_keys, ctx_.d_keys != nullptr ? n : 0);
  h.start_offsets =
      download_array(ctx_, ctx_.d_start_offsets, ctx_.d_start_offsets != nullptr ? n : 0);
  h.avail = download_array(ctx_, ctx_.d_avail, ctx_.d_avail != nullptr ? n : 0);
  h.subtree_avail =
      download_array(ctx_, ctx_.d_subtree_avail, ctx_.d_subtree_avail != nullptr ? n : 0);
  const long long used =
      std::max(0LL, std::min<long long>(ctx_.initial_payload_size, ctx_.fixed_size));
  h.payload = download_array(ctx_, ctx_.d_flat_payload,
                             ctx_.d_flat_payload != nullptr ? static_cast<size_t>(used) : 0);
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx_)));
  note_host_sync();
  const auto base = reinterpret_cast<std::uintptr_t>(ctx_.d_nodes);
  auto position = [&](const cbst_node* p) -> long long {
    if (p == nullptr) return -1;
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    if (a < base || (a - base) % sizeof(cbst_node) != 0) return -2;
    const auto t = static_cast<long long>((a - base) / sizeof(cbst_node));
    return t < static_cast<long long>(n) ? t : -2;
  };
  h.nodes.resize(n);
  for (size_t t = 0; t < n; ++t) {
    const cbst_node& d = nodes[t];
    cbst_host_node& x = h.nodes[t];
    x.index = d.index;
    x.value = d.value;
    x.length = d.length;
    x.size = d.size;
    x.occupancy = d.occupancy;
    x.tail_base = d.tail_base;
    x.tail_capacity = d.tail_capacity;
    x.left = position(d.left);
    x.right = position(d.right);
    x.parent = position(d.parent);
  }
  return h;
}

// ESCHER-GPU's findAndPrint printed the rows from the device; the lookup now returns the heap
// position of every key's node (-1: not in the tree) and prints nothing.
std::vector<int> cbst_store::find(const std::vector<int>& ids) const {
  const scoped_device guard(ctx_.res.device());
  std::vector<int> positions(ids.size(), -1);
  if (ids.empty() || ctx_.d_nodes == nullptr) return positions;
  buffer<int> search(ctx_.res, ids.size());
  buffer<int> found(ctx_.res, ids.size());
  int* d_search = search.data();
  int* d_positions = found.data();
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_search, ids.data(), ids.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx_)));
  find_node_kernel<<<(ids.size() + 256 - 1) / 256, 256, 0, stream_of(ctx_)>>>(
      ctx_.d_nodes, d_search, static_cast<int>(ids.size()), d_positions);
  DYNG_CHECK_KERNEL(stream_of(ctx_));
  DYNG_CUDA_TRY(cudaMemcpyAsync(positions.data(), d_positions, ids.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_of(ctx_)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx_)));
  note_host_sync();
  return positions;
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
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_keys, keys_to_unfill.data(), K * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_payload, values_to_remove.data(),
                                values_to_remove.size() * sizeof(int), cudaMemcpyHostToDevice,
                                stream_of(ctx)));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_insert_prefix_sizes, remove_prefix_sizes.data(),
                                remove_prefix_sizes.size() * sizeof(int), cudaMemcpyHostToDevice,
                                stream_of(ctx)));
  int block_size = 256;

  // ── Degree-binned unfill dispatch ────────────────────────────────────
  // Binned by the number of removals per key, a proxy for the work (the
  // row lengths are only known on the device).
  degree_bins bins(ctx.res, item_sizes(remove_prefix_sizes));
  bins.record(ctx.bin_launches.unfill);
  if (bins.count[0] > 0)
    unfill_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size, 0,
                           stream_of(ctx)>>>(ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys,
                                             ctx.d_insert_payload, ctx.d_insert_prefix_sizes,
                                             bins.bin(0), bins.count[0]);
  if (bins.count[1] > 0)
    unfill_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size, 0,
                         stream_of(ctx)>>>(ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys,
                                           ctx.d_insert_payload, ctx.d_insert_prefix_sizes,
                                           bins.bin(1), bins.count[1]);
  if (bins.count[2] > 0) {
    size_t shmem = sizeof(int) * static_cast<size_t>(block_size + 1);
    unfill_block_kernel<<<bins.count[2], block_size, shmem, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, bins.bin(2), bins.count[2]);
  }
  DYNG_CHECK_KERNEL(stream_of(ctx));
}

}  // namespace dyng::detail::escher
