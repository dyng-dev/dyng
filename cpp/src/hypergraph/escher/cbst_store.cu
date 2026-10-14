// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu
// Derived from MOSP_ESCHER@4b86159:escher/structure/operations.cu (constructCBST validation, the
// caller's row counts, ensureScratchCapacity, compactCBST)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (escherDeviceBytes)
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
#include "core/resources_access.hpp"
#include "hypergraph/escher/cbst_store.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"
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
#include <thrust/reverse.h>
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

void enqueue_control_readback(const cbst_context& ctx) {
  if (ctx.d_control == nullptr) return;
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.h_control, ctx.d_control, control_words * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream_of(ctx)));
}

void throw_if_control_raised(const cbst_context& ctx, const char* what) {
  if (ctx.h_control == nullptr) return;
  const volatile std::uint32_t* mirror = ctx.h_control;
  const std::uint32_t bits = mirror[control_errors];
  if (bits == 0) return;
  std::ostringstream detail;
  detail << mirror[control_overruns]
         << " row walks overran the used payload (a cyclic or broken chain), "
         << mirror[control_rejected] << " append / remove_values items named a dead or absent key";
  throw_device_errors(bits, "CBST " + ctx.dataset_name + ": " + what, detail.str());
}

// One host synchronization with the store's stream, at a point where ESCHER-GPU reads a value
// back (ADR 0038 lists them); counted per store and by the budget counters. The control block
// comes back with it (M8a step 4): a device error raised since the last read point throws here.
static void synchronize(cbst_context& ctx) {
  enqueue_control_readback(ctx);
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx)));
  note_host_sync();
  ++ctx.host_syncs;
  throw_if_control_raised(ctx, "a device error");
}

// Clears the control block (a new tree).
static void reset_control(cbst_context& ctx) {
  if (ctx.d_control == nullptr) return;
  DYNG_CUDA_TRY(
      cudaMemsetAsync(ctx.d_control, 0, control_words * sizeof(std::uint32_t), stream_of(ctx)));
  for (int w = 0; w < control_words; ++w) ctx.h_control[w] = 0;
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

// Releases the per-record arrays of a tree (nodes, keys, row offsets, avail, subtree_avail).
static void release_tree_arrays(cbst_context& ctx) {
  release(ctx.nodes_storage, ctx.d_nodes);
  release(ctx.keys_storage, ctx.d_keys);
  release(ctx.start_offsets_storage, ctx.d_start_offsets);
  release(ctx.avail_storage, ctx.d_avail);
  release(ctx.subtree_avail_storage, ctx.d_subtree_avail);
}

// Releases the operation scratch (the per-operation input buffers).
static void release_scratch(cbst_context& ctx) {
  release(ctx.insert_keys_storage, ctx.d_insert_keys);
  release(ctx.insert_payload_storage, ctx.d_insert_payload);
  release(ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes);
  release(ctx.relocation_plan_storage, ctx.d_relocation_plan);
  ctx.scratch_item_capacity = 0;
  ctx.scratch_payload_capacity = 0;
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
// The capacities follow the policy's scratch rule (M8a step 4, behaviour (g)): exact under
// escher_gpu() (ESCHER-GPU's reserveScratch, which the memory gate measures), geometric under the
// MOSP_ESCHER presets (its ensureScratchCapacity, escher/structure/operations.cu@4b86159:139-165,
// which reallocated all four buffers when one grew; here only the arrays that grow are).
static void reserve_scratch(cbst_context& ctx, int num_items, size_t payload_size) {
  const long long items =
      ctx.policy.scratch_capacity(num_items, ctx.scratch_item_capacity, 1, INT_MAX / 3);
  const long long values = ctx.policy.scratch_capacity(static_cast<long long>(payload_size),
                                                       ctx.scratch_payload_capacity, 4, LLONG_MAX);
  if (items > ctx.scratch_item_capacity) {
    release(ctx.insert_keys_storage, ctx.d_insert_keys);
    release(ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes);
    release(ctx.relocation_plan_storage, ctx.d_relocation_plan);
    allocate(ctx, ctx.insert_keys_storage, ctx.d_insert_keys, static_cast<size_t>(items));
    allocate(ctx, ctx.insert_prefix_sizes_storage, ctx.d_insert_prefix_sizes,
             static_cast<size_t>(items));
    allocate(ctx, ctx.relocation_plan_storage, ctx.d_relocation_plan,
             3 * static_cast<size_t>(items));
    ctx.scratch_item_capacity = static_cast<int>(items);
  }
  if (values > ctx.scratch_payload_capacity) {
    release(ctx.insert_payload_storage, ctx.d_insert_payload);
    allocate(ctx, ctx.insert_payload_storage, ctx.d_insert_payload, static_cast<size_t>(values));
    ctx.scratch_payload_capacity = values;
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
  // The policy's growth (escher_gpu(): ESCHER-GPU's rule).
  const long long capacity = ctx.policy.grown_capacity(needed, ctx.fixed_size, ctx.dataset_name);
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
  ++ctx.growths;
}

// The hysteresis of an automatic compaction (store_policy::compaction_headroom; M8a review, ADR
// 0040): after an operation that compacted at a trigger point, the payload grows by the policy's
// rule when fewer than capacity / headroom slots are free, so that the next compaction comes only
// after that many slots were used again (a compaction is O(payload)).
static void keep_headroom_after_compaction(cbst_context& ctx, long long compactions_before) {
  const long long headroom = ctx.policy.compaction_headroom;
  if (headroom <= 0 || ctx.compactions == compactions_before ||
      ctx.policy.growth == store_policy::growth_rule::none)
    return;
  const long long slack = ctx.fixed_size / headroom;
  if (static_cast<long long>(ctx.fixed_size) - ctx.initial_payload_size >= slack) return;
  const long long needed = std::min(static_cast<long long>(ctx.initial_payload_size) + slack,
                                    std::min<long long>(ctx.policy.max_payload, INT_MAX));
  ensure_payload_capacity(ctx, needed);
}

// MOSP_ESCHER's construct validation (escher/structure/operations.cu@4b86159:196-211; M8a step
// 4): negative sizes, a capacity below the payload (also for a tree of 0 records) and missing
// arrays are rejected with invalid_argument_error before anything changes or is allocated
// (ESCHER-GPU printed "Overflow" and returned a tree without nodes, or read the null arrays);
// a record count whose operation scratch (3 ints per record, ESCHER-GPU's sizing) exceeds the
// int32 indexing of the store is a capacity_error. MOSP_ESCHER threw escher::EscherError for all
// of them; its device synchronizations after the construct kernels (operations.cu:259, 263, 274)
// are not taken over (the store orders the kernels on its stream).
void validate_construct(const int* keys, const int* start_offsets, int num_records,
                        const int* flat_payload, int flat_payload_size, int payload_capacity,
                        const std::string& dataset_name, const int* row_counts) {
  DYNG_EXPECTS(num_records >= 0, "CBST ", dataset_name, ": construct: the record count ",
               num_records, " is negative");
  DYNG_EXPECTS(flat_payload_size >= 0 && payload_capacity >= 0, "CBST ", dataset_name,
               ": construct: the payload size ", flat_payload_size, " and capacity ",
               payload_capacity, " must not be negative");
  DYNG_EXPECTS(payload_capacity >= flat_payload_size, "CBST ", dataset_name, ": payload capacity ",
               payload_capacity, " is below the payload size ", flat_payload_size);
  DYNG_EXPECTS(num_records == 0 || (keys != nullptr && start_offsets != nullptr), "CBST ",
               dataset_name, ": construct: the keys and row offsets of ", num_records,
               " records must not be null");
  DYNG_EXPECTS(flat_payload_size == 0 || flat_payload != nullptr, "CBST ", dataset_name,
               ": construct: a payload of ", flat_payload_size, " entries must not be null");
  if (3LL * num_records > INT_MAX) {
    throw capacity_error("dyng: CBST " + dataset_name + ": " + std::to_string(num_records) +
                         " records exceed the int32 indexing of the store's operation scratch "
                         "(3 entries per record; at most " +
                         std::to_string(INT_MAX / 3) + " records)");
  }
  // The layout (dynG, M8a review; neither original checked it): keys strictly increasing (the
  // tree is a BST over them), every row at least one slot (its terminator) and inside the
  // payload, the rows in key order without overlap. A zero-length row would put its chain pointer
  // or terminator into the next row's first slot.
  for (int r = 0; r < num_records; ++r) {
    const long long end = r + 1 < num_records ? start_offsets[r + 1] : flat_payload_size;
    DYNG_EXPECTS(start_offsets[r] >= 0 && start_offsets[r] < end, "CBST ", dataset_name,
                 ": construct: row ", r, " (key ", keys[r], ") starts at ", start_offsets[r],
                 " and ends at ", end,
                 "; every row needs at least one slot (its terminator) inside the payload of ",
                 flat_payload_size, ", and the row offsets must increase");
    DYNG_EXPECTS(r == 0 || keys[r - 1] < keys[r], "CBST ", dataset_name, ": construct: key ",
                 keys[r], " of row ", r, " does not follow key ", r == 0 ? 0 : keys[r - 1],
                 "; the keys must be strictly increasing");
  }
  if (row_counts == nullptr) return;
  // The caller's row counts (the fast path of construct): each within the tail capacity of its
  // row (the slots up to the next row's offset, or the payload's end, but the terminator), as
  // store_items_kernel derives it.
  for (int r = 0; r < num_records; ++r) {
    const long long end = r + 1 < num_records ? start_offsets[r + 1] : flat_payload_size;
    const long long tail_capacity = std::max(0LL, end - start_offsets[r] - 1);
    DYNG_EXPECTS(row_counts[r] >= 0 && row_counts[r] <= tail_capacity, "CBST ", dataset_name,
                 ": construct: the row count ", row_counts[r], " of key ", keys[r],
                 " does not fit its row of ", tail_capacity, " value slots");
  }
}

void validate_batch(const std::vector<int>& keys, const std::vector<int>& payload,
                    const std::vector<int>& prefix_sizes, const std::string& dataset_name,
                    const char* op) {
  DYNG_EXPECTS(prefix_sizes.size() == keys.size(), "CBST ", dataset_name, ": ", op, ": ",
               keys.size(), " keys but ", prefix_sizes.size(), " prefix sizes");
  if (keys.empty()) return;
  // The operation's int32 indexing: 3 relocation-plan entries per item (the fill), the payload
  // offsets of the prefix sums.
  if (3LL * static_cast<long long>(keys.size()) > INT_MAX) {
    throw capacity_error("dyng: CBST " + dataset_name + ": " + op + ": " +
                         std::to_string(keys.size()) +
                         " items exceed the int32 indexing of the store's operation scratch "
                         "(3 entries per item; at most " +
                         std::to_string(INT_MAX / 3) + " items)");
  }
  int previous = 0;
  for (size_t i = 0; i < prefix_sizes.size(); ++i) {
    DYNG_EXPECTS(prefix_sizes[i] >= previous, "CBST ", dataset_name, ": ", op,
                 ": the prefix sizes must be non-decreasing and non-negative (item ", i, ": ",
                 prefix_sizes[i], " after ", previous, ")");
    previous = prefix_sizes[i];
  }
  DYNG_EXPECTS(static_cast<size_t>(prefix_sizes.back()) == payload.size(), "CBST ", dataset_name,
               ": ", op, ": the last prefix size ", prefix_sizes.back(),
               " is not the payload size ", payload.size());
}

void construct_cbst(int* keys, int* start_offsets, int num_records, int* flat_payload,
                    int flat_payload_size, int payload_capacity, const std::string& dataset_name,
                    cbst_context& ctx, const int* row_counts) {
  // Rejected before anything changes or is allocated.
  validate_construct(keys, start_offsets, num_records, flat_payload, flat_payload_size,
                     payload_capacity, dataset_name, row_counts);
  reset_control(ctx);
  ctx.fixed_size = payload_capacity;
  ctx.num_records = num_records;
  ctx.initial_payload_size = flat_payload_size;
  ctx.dataset_name = dataset_name;
  // ctx.alignment should be set by the owner (cbst_store) before calling

  if (num_records == 0) {
    // An empty tree (dynG addition): its payload is allocated as for any other tree, so that a
    // later insert or append has a buffer to write to (ESCHER-GPU and MOSP_ESCHER kept none, and
    // an insert into the empty tree failed). The arrays of an earlier tree go (M8a review: a
    // lookup walked the old nodes and an append wrote into stale rows), and so does its
    // operation scratch, which a construct sizes by its record count (0 here).
    release_tree_arrays(ctx);
    release_scratch(ctx);
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
  if (row_counts == nullptr) {
    // The default: the occupancies scanned from the payload (ESCHER-GPU's initTailMetadata).
    init_tail_metadata_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, num_records);
    DYNG_CHECK_KERNEL(stream_of(ctx));
  } else {
    // The fast path: the caller's row counts (MOSP_ESCHER's rowOccupancy,
    // escher/structure/operations.cu@4b86159:268-275, without its device synchronization).
    temp_buffer<int> d_row_counts(ctx.res, num_records);
    DYNG_CUDA_TRY(cudaMemcpyAsync(d_row_counts.get(), row_counts, num_records * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_of(ctx)));
    set_initial_occupancy_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, d_row_counts.get(), num_records);
    DYNG_CHECK_KERNEL(stream_of(ctx));
#ifndef NDEBUG
    // Debug builds: the counts must be the scan (an extra synchronization, Debug only; counted
    // like every other, ADR 0038 section 5).
    temp_buffer<int> d_mismatches(ctx.res, 1);
    DYNG_CUDA_TRY(cudaMemsetAsync(d_mismatches.get(), 0, sizeof(int), stream_of(ctx)));
    count_occupancy_mismatches_kernel<<<num_blocks, block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, num_records, d_mismatches.get());
    DYNG_CHECK_KERNEL(stream_of(ctx));
    int mismatches = 0;
    DYNG_CUDA_TRY(cudaMemcpyAsync(&mismatches, d_mismatches.get(), sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_of(ctx)));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx)));
    count_implicit_synchronization(ctx);
    if (mismatches != 0) {
      DYNG_FAIL("CBST ", dataset_name, ": ", mismatches,
                " of the caller's row counts differ from the rows of the payload");
    }
#endif
  }

  if (log_enabled(log_level::debug)) {
    log_message(log_level::debug, "dyng: CBST " + dataset_name + ": constructed " +
                                      std::to_string(num_records) + " records over " +
                                      std::to_string(flat_payload_size) + " payload entries");
  }
}

void rebuild_cbst_from_device(buffer<int> start_offsets, int num_records, buffer<int> flat_payload,
                              int flat_payload_size, int payload_capacity, cbst_context& ctx) {
  DYNG_EXPECTS(num_records >= 0 && flat_payload_size >= 0 &&
                   start_offsets.size() >= static_cast<size_t>(std::max(num_records, 0)) &&
                   flat_payload.size() >= static_cast<size_t>(std::max(payload_capacity, 0)) &&
                   flat_payload_size <= payload_capacity,
               "CBST ", ctx.dataset_name, ": rebuild_from_device: ", start_offsets.size(),
               " row offsets and a payload of ", flat_payload.size(), " for ", num_records,
               " rows, ", flat_payload_size, " used of a capacity of ", payload_capacity);
  reset_control(ctx);
  release_tree_arrays(ctx);
  release(ctx.flat_payload_storage, ctx.d_flat_payload);
  ctx.start_offsets_storage = std::move(start_offsets);
  ctx.d_start_offsets = ctx.start_offsets_storage.data();
  ctx.flat_payload_storage = std::move(flat_payload);
  ctx.d_flat_payload = ctx.flat_payload_storage.data();
  ctx.fixed_size = payload_capacity;
  ctx.initial_payload_size = flat_payload_size;
  ctx.num_records = num_records;
  // A merged tree of 0 records (every row deleted) is valid: the payload is kept for later
  // inserts and appends, and there is nothing to build (dynG addition, M8a review: the build
  // kernels were launched with 0 blocks, a CUDA error that poisoned the store).
  if (num_records == 0) {
    release(ctx.start_offsets_storage, ctx.d_start_offsets);
    return;
  }
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

static void fill_impl(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                      const std::vector<int>& insert_prefix_sizes, cbst_context& ctx,
                      bool allow_compact);

// The compaction of a fill whose overflow segments do not fit (MOSP_ESCHER's fillImpl): the
// payload is compacted and the values of the overflowing items that did not fit their tails are
// appended once more, without compaction. `plan` is the relocation plan on the host: plan[3i + 1]
// is how many of item i's values the tail took; item i overflowed iff its padded overflow size is
// not 0 (`cumulative`: plan[3i + 2] holds the inclusive prefix sums of the padded sizes; else the
// raw overflow sizes).
static void compact_and_refill(const std::vector<int>& insert_keys,
                               const std::vector<int>& insert_payload,
                               const std::vector<int>& insert_prefix_sizes,
                               const std::vector<int>& plan, bool cumulative, cbst_context& ctx) {
  const int K = static_cast<int>(insert_keys.size());
  std::vector<int> rest_keys, rest_payload, rest_prefix;
  for (int i = 0; i < K; ++i) {
    const int here = plan[3 * static_cast<size_t>(i) + 2];
    const int before = !cumulative || i == 0 ? 0 : plan[3 * static_cast<size_t>(i - 1) + 2];
    if (here == before) continue;
    const int start = i == 0 ? 0 : insert_prefix_sizes[static_cast<size_t>(i - 1)];
    rest_keys.push_back(insert_keys[static_cast<size_t>(i)]);
    rest_payload.insert(rest_payload.end(),
                        insert_payload.begin() + start + plan[3 * static_cast<size_t>(i) + 1],
                        insert_payload.begin() + insert_prefix_sizes[static_cast<size_t>(i)]);
    rest_prefix.push_back(static_cast<int>(rest_payload.size()));
  }
  compact_cbst(ctx);
  fill_impl(rest_keys, rest_payload, rest_prefix, ctx, false);
}

// The fill. When its overflow segments do not fit behind the bump pointer and the policy compacts
// (MOSP_ESCHER's fillImpl, escher/structure/operations.cu@4b86159:409-444; M8a step 5), the
// payload is compacted and the values that did not fit their rows' tails are appended once more,
// without compaction (`allow_compact` false); what still does not fit grows by the policy or is a
// capacity_error.
static void fill_impl(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                      const std::vector<int>& insert_prefix_sizes, cbst_context& ctx,
                      bool allow_compact) {
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
  DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_relocation_plan, 0, 3 * static_cast<size_t>(K) * sizeof(int),
                                stream_of(ctx)));

  int block_size = 256;

  // ── Degree-binned insert_node dispatch ────────────────────────────────
  const std::vector<int> sizes = item_sizes(insert_prefix_sizes);
  degree_bins bins(ctx.res, sizes);
  bins.record(ctx.bin_launches.fill);
  if (bins.count[0] > 0)
    insert_node_thread_kernel<<<(bins.count[0] + block_size - 1) / block_size, block_size, 0,
                                stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(0), bins.count[0], ctx.d_avail,
        ctx.d_control);
  if (bins.count[1] > 0)
    insert_node_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size, 0,
                              stream_of(ctx)>>>(ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys,
                                                ctx.d_insert_payload, ctx.d_insert_prefix_sizes,
                                                ctx.d_relocation_plan, bins.bin(1), bins.count[1],
                                                ctx.d_avail, ctx.d_control);
  if (bins.count[2] > 0)
    insert_node_block_kernel<<<bins.count[2], block_size, 0, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, ctx.d_relocation_plan, bins.bin(2), bins.count[2], ctx.d_avail,
        ctx.d_control);
  DYNG_CHECK_KERNEL(stream_of(ctx));

  // ── Overflow handling ────────────────────────────────────────────────
  // The int32 range of the overflow plan (dynG, M8a review): an item overflows by at most its
  // size, padded to a multiple of 4, so the bump pointer plus the sizes plus 3 per item bounds
  // the end of the appended segments. Beyond INT_MAX (batches of several GB) the plan's padded
  // overflow is summed exactly on the host first (one more read point): when the segments
  // would end past the int32 offsets, the store compacts first where the policy compacts, or
  // throws capacity_error (the tails were written in place: the store is poisoned).
  long long bound = ctx.initial_payload_size;
  for (int v : sizes) bound += static_cast<long long>(v) + 3;
  if (bound > INT_MAX) {
    std::vector<int> raw(3 * static_cast<size_t>(K));
    DYNG_CUDA_TRY(cudaMemcpyAsync(raw.data(), ctx.d_relocation_plan, raw.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_of(ctx)));
    synchronize(ctx);
    long long padded_total = 0;
    for (int i = 0; i < K; ++i) {
      const long long over = raw[3 * static_cast<size_t>(i) + 2];
      padded_total += over > 0 ? (over + 3) / 4 * 4 : 0;
    }
    if (static_cast<long long>(ctx.initial_payload_size) + padded_total > INT_MAX) {
      if (allow_compact && ctx.policy.compaction != store_policy::compaction_rule::never) {
        compact_and_refill(insert_keys, insert_payload, insert_prefix_sizes, raw, false, ctx);
        return;
      }
      throw capacity_error("dyng: CBST " + ctx.dataset_name +
                           ": fill: the overflow segments end at " +
                           std::to_string(ctx.initial_payload_size + padded_total) +
                           " payload slots, beyond the int32 offsets of the CBST");
    }
  }
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
  const long long needed = static_cast<long long>(ctx.initial_payload_size) + total_appended;
  if (total_appended > 0 && needed > ctx.fixed_size && allow_compact &&
      ctx.policy.compaction != store_policy::compaction_rule::never) {
    // The insert_node kernels have written the values that fit into each row's tail (such a
    // row's tail is now full and still ends in its terminator slot); plan[3i + 1] is how many of
    // item i's values that was, and item i overflowed iff its padded overflow size (the cumulative
    // plan[3i + 2]) is not 0. One more read point (MOSP_ESCHER's copy of the plan).
    std::vector<int> plan(3 * static_cast<size_t>(K));
    DYNG_CUDA_TRY(cudaMemcpyAsync(plan.data(), ctx.d_relocation_plan, plan.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_of(ctx)));
    synchronize(ctx);
    compact_and_refill(insert_keys, insert_payload, insert_prefix_sizes, plan, true, ctx);
    return;
  }
  if (total_appended > 0) {
    ensure_payload_capacity(ctx, needed);
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

void fill_cbst(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
               const std::vector<int>& insert_prefix_sizes, cbst_context& ctx) {
  validate_batch(insert_keys, insert_payload, insert_prefix_sizes, ctx.dataset_name, "fill");
  const long long compactions = ctx.compactions;
  fill_impl(insert_keys, insert_payload, insert_prefix_sizes, ctx, true);
  keep_headroom_after_compaction(ctx, compactions);
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
  validate_batch(new_keys, new_payload, new_prefix_sizes, ctx.dataset_name, "insert");
  insert_mapping mapping;
  int K = static_cast<int>(new_keys.size());
  mapping.item_to_key.resize(K, 0);
  if (K == 0) return mapping;
  const long long compactions = ctx.compactions;

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
#if defined(DYNG_MUTATION_CBST_REVERSED_TIES)
    // MUTATION (PLAN 8.4, M8a acceptance criterion 17): slots of equal capacity in reverse key
    // order (the best-fit tie order reversed).
    thrust::reverse(thrust_policy(ctx.res, alloc), caps_ptr, caps_ptr + D);
    thrust::reverse(thrust_policy(ctx.res, alloc), slot_order_ptr, slot_order_ptr + D);
#endif
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
    // The slots of the surplus rows. When they do not fit behind the bump pointer and the policy
    // compacts, the payload is compacted first (MOSP_ESCHER's insertCBST,
    // escher/structure/operations.cu@4b86159:694-701; the best-fit reuse above is complete);
    // what still does not fit grows by the policy, or is a capacity_error naming the first row
    // that does not fit (MOSP_ESCHER threw per row, :713-721, before writing any of them).
    long long surplus_slots = 0;
    for (int s = 0; s < surplus; ++s) {
      const int global_idx = surplus_indices[s];
      const int len =
          new_prefix_sizes[global_idx] - (global_idx == 0 ? 0 : new_prefix_sizes[global_idx - 1]);
      surplus_slots += next_multiple(len, ctx.alignment) + 1;
    }
    if (ctx.initial_payload_size + surplus_slots > ctx.fixed_size &&
        ctx.policy.compaction != store_policy::compaction_rule::never) {
      compact_cbst(ctx);
    }
    if (ctx.initial_payload_size + surplus_slots > ctx.fixed_size &&
        ctx.policy.growth == store_policy::growth_rule::none) {
      long long end = ctx.initial_payload_size;
      int row = 0;
      for (; row < surplus; ++row) {
        const int global_idx = surplus_indices[row];
        const int len =
            new_prefix_sizes[global_idx] - (global_idx == 0 ? 0 : new_prefix_sizes[global_idx - 1]);
        end += next_multiple(len, ctx.alignment) + 1;
        if (end > ctx.fixed_size) break;
      }
      throw capacity_error("dyng: CBST " + ctx.dataset_name + ": insert: surplus row " +
                           std::to_string(row) + " of " + std::to_string(surplus) +
                           " exceeds the payload capacity (" + std::to_string(end) + " > " +
                           std::to_string(ctx.fixed_size) + "; policy " + ctx.policy.name +
                           " does not grow the payload)");
    }
    // Pack every surplus row (values, zero padding, INT_MIN) on the host and
    // append them with a single copy.
    std::vector<int> packed;
    std::vector<cbst_node> surplus_records(surplus);
    const long long base = ctx.initial_payload_size;
    for (int s = 0; s < surplus; ++s) {
      int global_idx = surplus_indices[s];
      int start = (global_idx == 0) ? 0 : new_prefix_sizes[global_idx - 1];
      int len = new_prefix_sizes[global_idx] - start;
#if defined(DYNG_MUTATION_CBST_SURPLUS_RULE1)
      // MUTATION (PLAN 8.4, M8a acceptance criterion 17): a surplus row padded by padding rule 1
      // (the construct's (n + 4) / 4 * 4 slots) instead of rule 2.
      int aligned = (len + 4) / 4 * 4 - 1;
#else
      int aligned = next_multiple(len, ctx.alignment);
#endif
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
#if defined(DYNG_MUTATION_CBST_REQUESTED_KEYS)
      // MUTATION (PLAN 8.4, M8a acceptance criterion 17): a surplus row takes its requested key
      // when that key is above the next free one (requested keys honoured).
      next_key = std::max(next_key, new_keys[surplus_indices[i]]);
#endif
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
  keep_headroom_after_compaction(ctx, compactions);

  return mapping;
}

// cbst_store implementation
cbst_store::cbst_store(const resources& res, std::string dataset_name, int payload_capacity,
                       int alignment, store_policy policy) {
  DYNG_EXPECTS(res.get_backend() == backend::cuda, "CBST ", dataset_name,
               ": the store needs resources of the CUDA backend");
  ctx_.res = res;
  ctx_.policy = std::move(policy);
  ctx_.dataset_name = std::move(dataset_name);
  ctx_.fixed_size = payload_capacity;
  ctx_.alignment = alignment;
  // The control block (M8a step 4): a few words on the device and their pinned mirror.
  const scoped_device guard(res.device());
  ctx_.control_storage = buffer<std::uint32_t>(res, control_words);
  ctx_.control_mirror_storage =
      buffer<std::uint32_t>(control_words, res.stream(), resources_access::staging_memory(res), -1);
  ctx_.d_control = ctx_.control_storage.data();
  ctx_.h_control = ctx_.control_mirror_storage.data();
  reset_control(ctx_);
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
  ctx.d_control = nullptr;
  ctx.h_control = nullptr;
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
  construct_impl(keys, start_offsets, num_records, flat_payload, flat_payload_size, nullptr);
}

void cbst_store::construct(int* keys, int* start_offsets, int num_records, int* flat_payload,
                           int flat_payload_size, const int* row_counts) {
  // The caller's counts are required here: MOSP_ESCHER's nullptr (occupancy 0, its legacy
  // behaviour) is not taken over.
  DYNG_EXPECTS(num_records == 0 || row_counts != nullptr, "CBST ", ctx_.dataset_name,
               ": construct: the row counts of ", num_records, " records must not be null");
  construct_impl(keys, start_offsets, num_records, flat_payload, flat_payload_size, row_counts);
}

void cbst_store::construct_impl(int* keys, int* start_offsets, int num_records, int* flat_payload,
                                int flat_payload_size, const int* row_counts) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.construct");
  // The validation, before anything changes (the store is not poisoned by it).
  validate_construct(keys, start_offsets, num_records, flat_payload, flat_payload_size,
                     ctx_.fixed_size, ctx_.dataset_name, row_counts);
  try {
    construct_cbst(keys, start_offsets, num_records, flat_payload, flat_payload_size,
                   ctx_.fixed_size, ctx_.dataset_name, ctx_, row_counts);
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
  check_not_poisoned("insert");
  // The batch's shape, before anything changes (the store is not poisoned by it).
  validate_batch(insert_keys, insert_payload, insert_prefix_sizes, ctx_.dataset_name, "insert");
  return guarded("insert", [&] {
    return insert_cbst(insert_keys, insert_payload, insert_prefix_sizes, ctx_);
  });
}

void cbst_store::append(const std::vector<int>& insert_keys, const std::vector<int>& insert_payload,
                        const std::vector<int>& insert_prefix_sizes) {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.append");
  check_not_poisoned("append");
  validate_batch(insert_keys, insert_payload, insert_prefix_sizes, ctx_.dataset_name, "append");
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
  check_not_poisoned("remove_values");
  validate_batch(keys_to_unfill, values_to_remove, remove_prefix_sizes, ctx_.dataset_name,
                 "remove_values");
  guarded("remove_values",
          [&] { unfill_cbst(keys_to_unfill, values_to_remove, remove_prefix_sizes, ctx_); });
}

int cbst_store::compact() {
  const scoped_device guard(ctx_.res.device());
  scoped_stage stage(ctx_.res, "hypergraph.cbst.compact");
  check_not_poisoned("compact");
  if (!constructed_) {
    throw invalid_argument_error("dyng: CBST " + ctx_.dataset_name +
                                 ": compact before construct or rebuild_from_device");
  }
  const int bump = ctx_.initial_payload_size;
  const int capacity = ctx_.fixed_size;
  try {
    return compact_cbst(ctx_);
  } catch (const capacity_error&) {
    // The compacted rows do not fit a payload that cannot grow: thrown before anything changed.
    if (ctx_.initial_payload_size == bump && ctx_.fixed_size == capacity) throw;
    poisoned_ = true;
    throw;
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

// A device copy of `bytes` from a context's device to another's, on the destination's stream
// with its device current: a peer copy between two devices (the stream-ordered pool's memory is
// not mapped on its peers, so a plain device-to-device copy is rejected), a device-to-device copy
// within one.
static void copy_bytes(const cbst_context& to, void* dst, const cbst_context& from, const void* src,
                       size_t bytes) {
  if (bytes == 0) return;
  if (to.res.device() == from.res.device()) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_of(to)));
  } else {
    DYNG_CUDA_TRY(
        cudaMemcpyPeerAsync(dst, to.res.device(), src, from.res.device(), bytes, stream_of(to)));
  }
}

// A device copy of one array of a context into the matching array of another (same size).
template <typename value_t>
static void copy_array(cbst_context& to, buffer<value_t>& to_storage, value_t*& to_view,
                       const cbst_context& from, const buffer<value_t>& from_storage) {
  allocate(to, to_storage, to_view, from_storage.size());
  copy_bytes(to, to_view, from, from_storage.data(), from_storage.size() * sizeof(value_t));
}

// Makes `waiter` wait for the work queued so far on `signaller` (an event recorded on the
// signaller's device, waited for on the waiter's).
static void order_after(const cbst_context& signaller, const cbst_context& waiter) {
  // (One stream handle names one stream only on one device: the per-thread default stream's
  // handle is the same on every device.)
  if (stream_of(signaller) == stream_of(waiter) && signaller.res.device() == waiter.res.device())
    return;
  cudaEvent_t event = nullptr;
  {
    const scoped_device guard(signaller.res.device());
    DYNG_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    DYNG_CUDA_TRY(cudaEventRecord(event, stream_of(signaller)));
  }
  const scoped_device guard(waiter.res.device());
  DYNG_CUDA_TRY(cudaStreamWaitEvent(stream_of(waiter), event, 0));
  DYNG_CUDA_TRY(cudaEventDestroy(event));
}

cbst_store cbst_store::clone(const resources& res) const {
  check_not_poisoned("clone");
  DYNG_EXPECTS(res.get_backend() == backend::cuda, "CBST ", ctx_.dataset_name,
               ": clone: the store needs resources of the CUDA backend");
  cbst_store copy(res, ctx_.dataset_name, ctx_.fixed_size, ctx_.alignment, ctx_.policy);
  cbst_context& c = copy.ctx_;
  // The copy is ordered after the work queued on the source's stream; it is made on the copy's
  // device (the source's or another: M8a review).
  order_after(ctx_, c);
  {
    const scoped_device guard(res.device());
    c.num_records = ctx_.num_records;
    c.initial_payload_size = ctx_.initial_payload_size;
    c.scratch_item_capacity = ctx_.scratch_item_capacity;
    c.scratch_payload_capacity = ctx_.scratch_payload_capacity;
    copy_array(c, c.nodes_storage, c.d_nodes, ctx_, ctx_.nodes_storage);
    copy_array(c, c.keys_storage, c.d_keys, ctx_, ctx_.keys_storage);
    copy_array(c, c.start_offsets_storage, c.d_start_offsets, ctx_, ctx_.start_offsets_storage);
    copy_array(c, c.flat_payload_storage, c.d_flat_payload, ctx_, ctx_.flat_payload_storage);
    copy_array(c, c.avail_storage, c.d_avail, ctx_, ctx_.avail_storage);
    copy_array(c, c.subtree_avail_storage, c.d_subtree_avail, ctx_, ctx_.subtree_avail_storage);
    // The device error word and its counters: an error the source raised but has not read yet
    // (erase and remove_values have no read point) is the copy's too, and its next read point
    // throws it (M8a review: the copy started clean).
    if (ctx_.d_control != nullptr && c.d_control != nullptr)
      copy_bytes(c, c.d_control, ctx_, ctx_.d_control, control_words * sizeof(std::uint32_t));
    // The operation scratch: the same capacities (its contents are inputs of the next operation).
    allocate(c, c.insert_keys_storage, c.d_insert_keys, ctx_.insert_keys_storage.size());
    allocate(c, c.insert_payload_storage, c.d_insert_payload, ctx_.insert_payload_storage.size());
    allocate(c, c.insert_prefix_sizes_storage, c.d_insert_prefix_sizes,
             ctx_.insert_prefix_sizes_storage.size());
    allocate(c, c.relocation_plan_storage, c.d_relocation_plan,
             ctx_.relocation_plan_storage.size());
    if (c.nodes_storage.size() > 0) {
      const int n = static_cast<int>(c.nodes_storage.size());
      rewire_tree_kernel<<<(n + 255) / 256, 256, 0, stream_of(c)>>>(c.d_nodes, n);
      DYNG_CHECK_KERNEL(stream_of(c));
    }
  }
  // Later work on the source's stream waits for the copy's reads.
  order_after(c, ctx_);
  copy.constructed_ = constructed_;
  return copy;
}

void cbst_store::check_errors() {
  // A poisoned store's state is undefined: its end-of-update read point throws (M8a review).
  check_not_poisoned("check_errors");
  if (!constructed_) return;
  const scoped_device guard(ctx_.res.device());
  try {
    synchronize(ctx_);
  } catch (...) {
    poisoned_ = true;
    throw;
  }
}

void cbst_store::check_errors(std::initializer_list<cbst_store*> stores) {
  cbst_store* first = nullptr;
  bool one_stream = true;
  for (cbst_store* s : stores) {
    if (s == nullptr) continue;
    s->check_not_poisoned("check_errors");
    if (!s->constructed_) continue;
    if (first == nullptr) {
      first = s;
    } else if (stream_of(s->ctx_) != stream_of(first->ctx_) ||
               s->ctx_.res.device() != first->ctx_.res.device()) {
      one_stream = false;
    }
  }
  if (first == nullptr) return;
  if (!one_stream) {
    for (cbst_store* s : stores)
      if (s != nullptr) s->check_errors();
    return;
  }
  const scoped_device guard(first->ctx_.res.device());
  try {
    for (cbst_store* s : stores)
      if (s != nullptr && s->constructed_) enqueue_control_readback(s->ctx_);
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(first->ctx_)));
  } catch (...) {
    for (cbst_store* s : stores)
      if (s != nullptr && s->constructed_) s->poisoned_ = true;
    throw;
  }
  note_host_sync();
  ++first->ctx_.host_syncs;
  for (cbst_store* s : stores) {
    if (s == nullptr || !s->constructed_) continue;
    try {
      throw_if_control_raised(s->ctx_, "a device error");
    } catch (...) {
      s->poisoned_ = true;
      throw;
    }
  }
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
  check_not_poisoned("find");
  const scoped_device guard(ctx_.res.device());
  std::vector<int> positions(ids.size(), -1);
  if (ids.empty() || ctx_.d_nodes == nullptr || ctx_.num_records <= 0) return positions;
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

std::vector<std::vector<int>> cbst_store::rows(const std::vector<int>& keys) const {
  check_not_poisoned("rows");
  const scoped_device guard(ctx_.res.device());
  std::vector<std::vector<int>> out(keys.size());
  if (keys.empty() || ctx_.d_nodes == nullptr || ctx_.num_records <= 0) return out;
  const int k = static_cast<int>(keys.size());
  const int block = 256, grid = (k + block - 1) / block;
  temp_buffer<int> d_keys(ctx_.res, keys.size()), d_lengths(ctx_.res, keys.size());
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_keys.get(), keys.data(), keys.size() * sizeof(int),
                                cudaMemcpyHostToDevice, stream_of(ctx_)));
  find_row_lengths_kernel<<<grid, block, 0, stream_of(ctx_)>>>(
      ctx_.d_nodes, ctx_.d_avail, ctx_.d_flat_payload, ctx_.initial_payload_size, d_keys.get(), k,
      d_lengths.get(), ctx_.d_control);
  DYNG_CHECK_KERNEL(stream_of(ctx_));
  std::vector<int> lengths(keys.size());
  DYNG_CUDA_TRY(cudaMemcpyAsync(lengths.data(), d_lengths.get(), keys.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_of(ctx_)));
  enqueue_control_readback(ctx_);
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx_)));
  note_host_sync();
  throw_if_control_raised(ctx_, "rows");
  std::vector<long long> offsets(keys.size() + 1, 0);
  for (size_t i = 0; i < keys.size(); ++i) offsets[i + 1] = offsets[i] + lengths[i];
  const long long total = offsets.back();
  if (total == 0) return out;
  temp_buffer<long long> d_offsets(ctx_.res, offsets.size());
  temp_buffer<int> d_values(ctx_.res, static_cast<size_t>(total));
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_offsets.get(), offsets.data(), offsets.size() * sizeof(long long),
                                cudaMemcpyHostToDevice, stream_of(ctx_)));
  find_row_values_kernel<<<grid, block, 0, stream_of(ctx_)>>>(
      ctx_.d_nodes, ctx_.d_avail, ctx_.d_flat_payload, ctx_.initial_payload_size, d_keys.get(), k,
      d_offsets.get(), d_lengths.get(), d_values.get());
  DYNG_CHECK_KERNEL(stream_of(ctx_));
  std::vector<int> values(static_cast<size_t>(total));
  DYNG_CUDA_TRY(cudaMemcpyAsync(values.data(), d_values.get(), values.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_of(ctx_)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(ctx_)));
  note_host_sync();
  for (size_t i = 0; i < keys.size(); ++i)
    out[i].assign(values.begin() + offsets[i], values.begin() + offsets[i + 1]);
  return out;
}

cbst_store_stats cbst_store::stats() const {
  const cbst_context& c = ctx_;
  cbst_store_stats st;
  st.num_records = c.num_records;
  st.payload_capacity = c.fixed_size;
  st.payload_used = c.initial_payload_size;
  st.growths = c.growths;
  st.compactions = c.compactions;
  st.compaction_temp_bytes = c.compaction_temp_bytes;
  st.device_bytes = static_cast<long long>(device_bytes());
  st.scratch_bytes = (5LL * c.scratch_item_capacity + c.scratch_payload_capacity) *
                     static_cast<long long>(sizeof(int));
  st.mosp_escher_bytes = st.device_bytes - st.scratch_bytes;
  st.host_syncs = c.host_syncs;
  st.bin_launches = c.bin_launches;
  if (c.h_control != nullptr) {
    const volatile std::uint32_t* mirror = c.h_control;
    st.serial_unfills = mirror[control_serial_unfills];
    st.chain_overruns = mirror[control_overruns];
    st.rejected_items = mirror[control_rejected];
  }
  return st;
}

size_t cbst_store::allocated_bytes() const {
  const cbst_context& c = ctx_;
  return c.nodes_storage.size() * sizeof(cbst_node) +
         (c.keys_storage.size() + c.start_offsets_storage.size() + c.flat_payload_storage.size() +
          c.insert_keys_storage.size() + c.insert_payload_storage.size() +
          c.insert_prefix_sizes_storage.size() + c.relocation_plan_storage.size() +
          c.avail_storage.size() + c.subtree_avail_storage.size()) *
             sizeof(int);
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

void cbst_store::set_used_payload(long long used) {
  check_not_poisoned("set_used_payload");
  DYNG_EXPECTS(constructed_, "CBST ", ctx_.dataset_name,
               ": set_used_payload on a tree that is not constructed");
  DYNG_EXPECTS(used >= ctx_.initial_payload_size && used <= ctx_.fixed_size, "CBST ",
               ctx_.dataset_name, ": set_used_payload(", used, ") is outside [",
               ctx_.initial_payload_size, ", ", ctx_.fixed_size, "]");
  ctx_.initial_payload_size = static_cast<int>(used);
}

int compact_cbst(cbst_context& ctx) {
  const int n = ctx.num_records;
  if (n <= 0 || ctx.d_nodes == nullptr) return ctx.initial_payload_size;
  const int block_size = 256;
  const int warp_blocks = static_cast<int>((32LL * n + block_size - 1) / block_size);
  const long long limit = ctx.initial_payload_size;

  // Pass 1: live values and compacted slots per row; offsets by key order.
  temp_buffer<int> live_count(ctx.res, n);
  temp_buffer<long long> offsets(ctx.res, static_cast<size_t>(n) + 1);
  DYNG_CUDA_TRY(cudaMemsetAsync(offsets.get() + n, 0, sizeof(long long), stream_of(ctx)));
  compact_sizes_kernel<<<warp_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_avail, n, ctx.d_flat_payload, limit, live_count.get(), offsets.get(),
      ctx.d_control);
  DYNG_CHECK_KERNEL(stream_of(ctx));
  thrust::device_ptr<long long> off_ptr(offsets.get());
  thrust_allocator alloc(ctx.res);
  thrust::exclusive_scan(thrust_policy(ctx.res, alloc), off_ptr, off_ptr + n + 1, off_ptr);
  long long total = 0;
  DYNG_CUDA_TRY(cudaMemcpyAsync(&total, offsets.get() + n, sizeof(long long),
                                cudaMemcpyDeviceToHost, stream_of(ctx)));
  synchronize(ctx);  // the read point (and a broken chain of pass 1 throws here)
  if (total > INT_MAX) {
    throw capacity_error("dyng: CBST " + ctx.dataset_name + ": the compacted rows need " +
                         std::to_string(total) +
                         " payload slots, more than the int32 offsets of the CBST hold");
  }
  // MOSP_ESCHER threw when the compacted rows exceed the capacity; dynG grows the payload by the
  // policy first (capacity_error without growth), before anything changed.
  ensure_payload_capacity(ctx, total);

  // Pass 2: copy every row into a zeroed buffer at its new offset, then back to the front of the
  // payload; the rest of the used prefix is cleared so the tail beyond the new bump pointer is
  // zero again.
  temp_buffer<int> compacted(ctx.res, static_cast<size_t>(total));
  ctx.compaction_temp_bytes =
      std::max(ctx.compaction_temp_bytes,
               static_cast<long long>(std::max<long long>(total, 1) * sizeof(int) +
                                      (static_cast<long long>(n) + 1) * sizeof(long long) +
                                      static_cast<long long>(n) * sizeof(int)));
  DYNG_CUDA_TRY(cudaMemsetAsync(compacted.get(), 0, static_cast<size_t>(total) * sizeof(int),
                                stream_of(ctx)));
  compact_copy_kernel<<<warp_blocks, block_size, 0, stream_of(ctx)>>>(
      ctx.d_nodes, ctx.d_avail, n, ctx.d_flat_payload, limit, live_count.get(), offsets.get(),
      compacted.get(), ctx.d_start_offsets);
  DYNG_CHECK_KERNEL(stream_of(ctx));
  DYNG_CUDA_TRY(cudaMemcpyAsync(ctx.d_flat_payload, compacted.get(),
                                static_cast<size_t>(total) * sizeof(int), cudaMemcpyDeviceToDevice,
                                stream_of(ctx)));
  if (total < limit)
    DYNG_CUDA_TRY(cudaMemsetAsync(ctx.d_flat_payload + total, 0,
                                  static_cast<size_t>(limit - total) * sizeof(int),
                                  stream_of(ctx)));
  ctx.initial_payload_size = static_cast<int>(total);
  ++ctx.compactions;
  if (log_enabled(log_level::debug))
    log_message(log_level::debug, "dyng: CBST " + ctx.dataset_name + ": compacted " +
                                      std::to_string(limit) + " payload slots to " +
                                      std::to_string(total));
  return ctx.initial_payload_size;
}

void unfill_cbst(const std::vector<int>& keys_to_unfill, const std::vector<int>& values_to_remove,
                 const std::vector<int>& remove_prefix_sizes, cbst_context& ctx) {
  validate_batch(keys_to_unfill, values_to_remove, remove_prefix_sizes, ctx.dataset_name, "unfill");
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
                                             bins.bin(0), bins.count[0], ctx.initial_payload_size,
                                             ctx.d_control, ctx.d_avail);
  if (bins.count[1] > 0)
    unfill_warp_kernel<<<(bins.count[1] * 32 + block_size - 1) / block_size, block_size, 0,
                         stream_of(ctx)>>>(ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys,
                                           ctx.d_insert_payload, ctx.d_insert_prefix_sizes,
                                           bins.bin(1), bins.count[1], ctx.initial_payload_size,
                                           ctx.d_control, ctx.d_avail);
  if (bins.count[2] > 0) {
    size_t shmem = sizeof(int) * static_cast<size_t>(block_size + 1);
    unfill_block_kernel<<<bins.count[2], block_size, shmem, stream_of(ctx)>>>(
        ctx.d_nodes, ctx.d_flat_payload, ctx.d_insert_keys, ctx.d_insert_payload,
        ctx.d_insert_prefix_sizes, bins.bin(2), bins.count[2], ctx.initial_payload_size,
        ctx.d_control, ctx.d_avail);
  }
  DYNG_CHECK_KERNEL(stream_of(ctx));
  // No read point: a broken chain the walks found surfaces at the next one (or check_errors()).
}

}  // namespace dyng::detail::escher
