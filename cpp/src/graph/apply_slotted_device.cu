// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/dynamicGraph.cu (setStatusKernel, insertKernel,
// halfEdgesKernel, appendKernel, advanceRowEndKernel, createDeviceGraph, launchApplyVertexChanges,
// launchAppendEdges, launchReadErrorFlags) and src/dynLP.cu (the transfer of a batch)
/**
 * @file apply_slotted_device.cu
 * @brief row_layout::slotted on the CUDA backend (device_slotted.hpp; ADR 0074): the resident
 *        device copy and the device apply of a vertex batch.
 *
 * Straight port of LabelPropagation-CUDA's dynamicGraph.cu (its file comment: half-edge keys, a
 * stable radix sort, an exclusive sum by key, the append, the advance of the row ends). Mechanical
 * changes: names, templates on the index types, namespace dyng::detail, dyng buffers from the
 * memory resource of `res` on its stream instead of cudaMalloc, exceptions instead of exit(), the
 * statuses with vertex_status's values (active 0, deleted 1, inactive 2; the original's ground
 * truth is a seed label of the algorithm's result, ADR 0069), and the sticky error flags widened
 * into the report of device_slotted.hpp. Three additions, none of them floating-point:
 *   - the removed live entries of the deletions are counted before the states change
 *     (graph::num_edges(); the original does not keep the count): a vertex of Del_eff is first
 *     marked `deleting`, then each one sums its row (2 per entry to an active vertex, 1 per entry
 *     to another vertex being deleted or to itself), as the host apply counts them;
 *   - halfEdgesKernel checks the endpoints' range (an out-of-range endpoint is recorded and its
 *     edge dropped instead of read out of bounds: a documented fix of the original; Step 0 refuses
 *     such a batch first, so this is defense in depth);
 *   - the dropped half-edges are counted from the sorted keys (the apply summary).
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/device_slotted.hpp"
#include "graph/instantiate.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dyng::detail {

namespace {

constexpr int threads_per_block = 256;  // the original's THREADS

constexpr std::int32_t status_active = 0;    // vertex_status::active
constexpr std::int32_t status_deleted = 1;   // vertex_status::deleted
constexpr std::int32_t status_inactive = 2;  // vertex_status::inactive
constexpr std::int32_t status_deleting = 3;  // transient: in Del_eff, its entries being counted

inline int blocks_for(long long n) {
  return static_cast<int>((std::max(n, 1LL) + threads_per_block - 1) / threads_per_block);
}

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

__device__ __forceinline__ void report_or(std::int64_t* report, int entry, std::uint64_t bits) {
  atomicOr(reinterpret_cast<unsigned long long*>(report + entry),
           static_cast<unsigned long long>(bits));
}

__device__ __forceinline__ void report_max(std::int64_t* report, int entry, std::int64_t value) {
  atomicMax(reinterpret_cast<unsigned long long*>(report + entry),
            static_cast<unsigned long long>(value));
}

__device__ __forceinline__ void report_add(std::int64_t* report, int entry, std::int64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(report + entry),
            static_cast<unsigned long long>(value));
}

// ---- kernels (LabelPropagation-CUDA@a276a3a:src/dynamicGraph.cu) --------------------------------

/// Del_eff (active before the batch) is marked `deleting` before its entries are counted.
template <typename vertex_t>
__global__ void mark_deleting_kernel(const vertex_t* ids, int count, std::int32_t* status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    status[ids[i]] = status_deleting;
  }
}

/// The live entries a vertex of Del_eff takes with it (the host apply's count): 2 per entry to an
/// active vertex (its entry and the reverse one), 1 per entry to another vertex being deleted (the
/// other side counts the reverse one) and 1 per entry to itself.
template <typename vertex_t, typename edge_t>
__global__ void count_removed_kernel(const vertex_t* ids, int count, const edge_t* row_ptr,
                                     const edge_t* row_end, const vertex_t* col_ind,
                                     const std::int32_t* status, std::int64_t* report) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const vertex_t v = ids[i];
  std::int64_t removed = 0;
  for (edge_t e = row_ptr[v]; e < row_end[v]; ++e) {
    const vertex_t c = col_ind[e];
    if (c == v) {
      removed += 1;
    } else if (status[c] == status_active) {
      removed += 2;
    } else if (status[c] == status_deleting) {
      removed += 1;
    }
  }
  if (removed != 0) {
    report_add(report, report_removed, removed);
  }
}

/// setStatusKernel.
template <typename vertex_t>
__global__ void set_status_kernel(const vertex_t* ids, int count, std::int32_t value,
                                  std::int32_t* status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    status[ids[i]] = value;
  }
}

/// insertKernel: inserted vertices become active; one that was not inactive (inserted before) is
/// recorded as 1 + its id.
template <typename vertex_t>
__global__ void insert_kernel(const vertex_t* ids, int count, std::int32_t* status,
                              std::int64_t* report) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const vertex_t v = ids[i];
  const std::int32_t old = atomicExch(&status[v], status_active);
  if (old != status_inactive) {
    report_max(report, report_reinserted, static_cast<std::int64_t>(v) + 1);
    report_or(report, report_error_bits, bits_of(device_error::invalid_input));
  }
}

/// halfEdgesKernel: two half-edges per new edge, keyed by their source vertex (n: dropped). An
/// endpoint out of range is recorded and the edge dropped (a documented fix of the original).
template <typename vertex_t>
__global__ void half_edges_kernel(const vertex_t* src, const vertex_t* dst, int num_edges,
                                  const std::int32_t* status, int num_nodes, std::int32_t* keys,
                                  std::int32_t* items, std::int64_t* report) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_edges) {
    return;
  }
  const vertex_t u = src[e];
  const vertex_t v = dst[e];
  bool keep = false;
  if (u < 0 || u >= num_nodes || v < 0 || v >= num_nodes) {
    report_max(report, report_bad_endpoint, static_cast<std::int64_t>(e) + 1);
    report_or(report, report_error_bits, bits_of(device_error::invalid_input));
  } else {
    keep = status[u] == status_active && status[v] == status_active;
  }
  keys[2 * e] = keep ? static_cast<std::int32_t>(u) : num_nodes;
  keys[2 * e + 1] = keep ? static_cast<std::int32_t>(v) : num_nodes;
  items[2 * e] = 2 * e;
  items[2 * e + 1] = 2 * e + 1;
}

/// appendKernel: write each sorted half-edge to its slot in its source's row (dropped and
/// recorded past the end of the row).
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void append_kernel(const std::int32_t* keys, const std::int32_t* items,
                              const std::int32_t* rank, int num_half_edges, int num_nodes,
                              const vertex_t* src, const vertex_t* dst, const weight_t* weight,
                              int num_weights, const edge_t* row_ptr, const edge_t* row_end,
                              vertex_t* col_ind, weight_t* values, std::int64_t* report) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_half_edges) {
    return;
  }
  const std::int32_t source = keys[i];
  if (source >= num_nodes) {
    return;  // dropped edge
  }
  // In 64 bits: with 32-bit offsets a row near the end of a slot array of almost 2^31 slots
  // would wrap past the row's end check (the original's arithmetic is edge_t; a documented fix).
  const std::int64_t slot =
      static_cast<std::int64_t>(row_end[source]) + static_cast<std::int64_t>(rank[i]);
  if (slot >= static_cast<std::int64_t>(row_ptr[source + 1])) {  // row full: contract violated
    report_max(report, report_overflow_row, static_cast<std::int64_t>(source) + 1);
    report_or(report, report_error_bits, bits_of(device_error::capacity));
    return;
  }
  const std::int32_t item = items[i];
  const std::int32_t e = item >> 1;
  const vertex_t neighbor = (item & 1) ? src[e] : dst[e];
  col_ind[slot] = neighbor;
  if (num_weights != 0) {
    values[slot] = weight[e];
  }
}

/// advanceRowEndKernel: the last new half-edge of each row advances the row's end (at most to the
/// end of the row).
template <typename edge_t>
__global__ void advance_row_end_kernel(const std::int32_t* keys, const std::int32_t* rank,
                                       int num_half_edges, int num_nodes, const edge_t* row_ptr,
                                       edge_t* row_end) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_half_edges) {
    return;
  }
  const std::int32_t source = keys[i];
  if (source >= num_nodes) {
    return;
  }
  if (i + 1 < num_half_edges && keys[i + 1] == source) {
    return;
  }
  // In 64 bits, as append_kernel's slot (the end of the row bounds the result).
  const std::int64_t end =
      static_cast<std::int64_t>(row_end[source]) + static_cast<std::int64_t>(rank[i]) + 1;
  const auto limit = static_cast<std::int64_t>(row_ptr[source + 1]);
  row_end[source] = static_cast<edge_t>(end < limit ? end : limit);
}

/// The dropped half-edges: the keys n sort last, so the last key's rank says how many there are.
__global__ void count_dropped_kernel(const std::int32_t* keys, const std::int32_t* rank,
                                     int num_half_edges, int num_nodes, std::int64_t* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && num_half_edges > 0 &&
      keys[num_half_edges - 1] >= num_nodes) {
    report[report_dropped] = static_cast<std::int64_t>(rank[num_half_edges - 1]) + 1;
  }
}

/// The states of a host graph moved to CUDA (vertex_status's values widened to int32).
__global__ void widen_status_kernel(const std::uint8_t* in, int count, std::int32_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = static_cast<std::int32_t>(in[i]);
  }
}

/// Every vertex inactive (createDeviceGraph's initial states).
__global__ void fill_status_kernel(std::int32_t* status, int count, std::int32_t value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    status[i] = value;
  }
}

/// The CUB temporary storage of an append of up to `half_edges` half-edges (createDeviceGraph).
std::size_t append_temp_bytes(int half_edges, int key_bits) {
  std::size_t sort_bytes = 0;
  std::size_t scan_bytes = 0;
  cub::DoubleBuffer<std::int32_t> keys(nullptr, nullptr);
  cub::DoubleBuffer<std::int32_t> ids(nullptr, nullptr);
  DYNG_CUDA_TRY(
      cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, ids, half_edges, 0, key_bits));
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSumByKey(
      nullptr, scan_bytes, static_cast<const std::int32_t*>(nullptr),
      thrust::constant_iterator<std::int32_t>(1), static_cast<std::int32_t*>(nullptr), half_edges));
  return std::max(sort_bytes, scan_bytes);
}

/// The size a batch buffer of `size` elements grows to for `count` (> size): the first batch's
/// size exactly (the original sizes its batch buffers once, for the largest batch), a later
/// larger batch with headroom (x 1.5 of the larger of the two), so that a stream whose batch
/// sizes vary reserves a bounded number of times, not on every new maximum (M10a review).
std::size_t grown(std::size_t size, std::size_t count) {
  if (size == 0) {
    return count;
  }
  const std::size_t larger = std::max(size, count);
  return larger + larger / 2;
}

/// A device buffer of at least `count` elements (contents not kept; see grown()).
template <typename value_t>
void ensure(const resources& res, buffer<value_t>& b, std::size_t count) {
  if (b.size() < count) {
    note_reservation();
    const std::size_t size = std::max<std::size_t>(grown(b.size(), count), 1);
    b = buffer<value_t>();
    b = buffer<value_t>(res, size);
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_slotted(const resources& res, const std::vector<edge_t>& row_ptr, int num_weights,
                          device_slotted<vertex_t, edge_t, weight_t>& out) {
  static_assert(device_slotted_supported_v<vertex_t>);
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto n = static_cast<std::size_t>(row_ptr.size() - 1);
  out.device = res.device();
  out.num_vertices = static_cast<vertex_t>(n);
  out.num_slots = row_ptr.back();
  out.num_weights = num_weights;
  const auto slots = std::max<std::size_t>(static_cast<std::size_t>(out.num_slots), 1);
  out.row_ptr = buffer<edge_t>(res, n + 1);
  out.row_end = buffer<edge_t>(res, std::max<std::size_t>(n, 1));
  out.col_ind = buffer<vertex_t>(res, slots);
  out.weights = num_weights > 0 ? buffer<weight_t>(res, slots) : buffer<weight_t>();
  out.status = buffer<std::int32_t>(res, std::max<std::size_t>(n, 1));
  // Slots past the row ends are never read by the kernels; clearing them once keeps the download
  // (and compute-sanitizer initcheck) free of uninitialized memory, as createDeviceGraph does.
  DYNG_CUDA_TRY(cudaMemsetAsync(out.col_ind.data(), 0xFF, slots * sizeof(vertex_t), stream));
  if (num_weights > 0) {
    DYNG_CUDA_TRY(cudaMemsetAsync(out.weights.data(), 0, slots * sizeof(weight_t), stream));
  }
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.row_ptr.data(), row_ptr.data(), (n + 1) * sizeof(edge_t),
                                cudaMemcpyHostToDevice, stream));
  // Every row starts empty, every vertex inactive.
  DYNG_CUDA_TRY(cudaMemcpyAsync(out.row_end.data(), out.row_ptr.data(), n * sizeof(edge_t),
                                cudaMemcpyDeviceToDevice, stream));
  if (n > 0) {
    fill_status_kernel<<<blocks_for(static_cast<long long>(n)), threads_per_block, 0, stream>>>(
        out.status.data(), static_cast<int>(n), status_inactive);
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  out.key_bits = 1;
  while ((1LL << out.key_bits) <= static_cast<long long>(n)) {
    ++out.key_bits;
  }
  out.report = buffer<std::int64_t>(res, report_entries);
  out.report_host = buffer<std::int64_t>(report_entries, res.stream(),
                                         resources_access::staging_memory(res), res.device());
  std::memset(out.report_host.data(), 0, report_entries * sizeof(std::int64_t));
  DYNG_CUDA_TRY(
      cudaMemsetAsync(out.report.data(), 0, report_entries * sizeof(std::int64_t), stream));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void upload_device_slotted(const resources& res, const csr<vertex_t, edge_t, weight_t>& slots,
                           const std::vector<edge_t>& row_end,
                           const std::vector<vertex_status>& status,
                           device_slotted<vertex_t, edge_t, weight_t>& out) {
  build_device_slotted(res, slots.row_ptr, slots.num_weights, out);
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto n = row_end.size();
  const auto m = slots.col_ind.size();
  if (m > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.col_ind.data(), slots.col_ind.data(), m * sizeof(vertex_t),
                                  cudaMemcpyHostToDevice, stream));
    if (slots.num_weights > 0) {
      DYNG_CUDA_TRY(cudaMemcpyAsync(out.weights.data(), slots.weights.data(), m * sizeof(weight_t),
                                    cudaMemcpyHostToDevice, stream));
    }
  }
  if (n > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.row_end.data(), row_end.data(), n * sizeof(edge_t),
                                  cudaMemcpyHostToDevice, stream));
    buffer<std::uint8_t> narrow(res, n);
    DYNG_CUDA_TRY(cudaMemcpyAsync(narrow.data(), status.data(), n, cudaMemcpyHostToDevice, stream));
    widen_status_kernel<<<blocks_for(static_cast<long long>(n)), threads_per_block, 0, stream>>>(
        narrow.data(), static_cast<int>(n), out.status.data());
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void reserve_device_slotted(const resources& res, device_slotted<vertex_t, edge_t, weight_t>& g,
                            std::int64_t max_edges) {
  if (max_edges >= (std::int64_t{1} << 30)) {
    throw capacity_error(concat_message(
        "dyng: a vertex batch of ", max_edges,
        " edges cannot be appended on the device (its half-edges are numbered with 32-bit "
        "integers: at most 2^30 - 1 edges per batch)"));
  }
  if (max_edges <= g.max_edges) {
    return;
  }
  const scoped_device guard(res.device());
  note_reservation();
  const auto edges = static_cast<std::size_t>(std::max<std::int64_t>(max_edges, 1));
  const std::size_t half_edges = 2 * edges;
  for (int k = 0; k < 2; ++k) {
    g.keys[k] = buffer<std::int32_t>();
    g.items[k] = buffer<std::int32_t>();
    g.keys[k] = buffer<std::int32_t>(res, half_edges);
    g.items[k] = buffer<std::int32_t>(res, half_edges);
  }
  g.temp = buffer<unsigned char>();
  g.temp = buffer<unsigned char>(
      res, std::max<std::size_t>(append_temp_bytes(static_cast<int>(half_edges), g.key_bits), 1));
  g.edge_src = buffer<vertex_t>();
  g.edge_dst = buffer<vertex_t>();
  g.edge_src = buffer<vertex_t>(res, edges);
  g.edge_dst = buffer<vertex_t>(res, edges);
  if (g.num_weights > 0) {
    g.edge_weight = buffer<weight_t>();
    g.edge_weight = buffer<weight_t>(res, edges);
  }
  g.max_edges = static_cast<std::int64_t>(edges);
}

namespace {

/// One array of the batch to the device: directly from pinned host memory, through the staging
/// otherwise (the bytes are placed at `offset` of the staging, which the caller sized).
template <typename value_t>
void put(const resources& res, unsigned char* staging, std::size_t& offset, value_t* dst,
         const value_t* src, memory_space space, std::size_t count) {
  if (count == 0) {
    return;
  }
  const std::size_t bytes = count * sizeof(value_t);
  const void* from = src;
  if (space != memory_space::pinned_host) {
    std::memcpy(staging + offset, src, bytes);
    from = staging + offset;
    offset += (bytes + 255) / 256 * 256;
  }
  DYNG_CUDA_TRY(cudaMemcpyAsync(dst, from, bytes, cudaMemcpyHostToDevice, native(res)));
}

constexpr std::size_t aligned(std::size_t bytes) {
  return (bytes + 255) / 256 * 256;
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void upload_vertex_batch(const resources& res, device_slotted<vertex_t, edge_t, weight_t>& g,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         const normalized_batch<vertex_t>& normalized) {
  const scoped_device guard(res.device());
  const std::size_t num_del = batch.delete_vertices.size();
  const std::size_t num_del_eff = normalized.deleted_vertices.size();
  const std::size_t num_ins = normalized.inserted_vertices.size();
  const std::size_t num_edges = batch.insert_src.size();
  // Ins \ Del and Del_eff are the batch's own arrays when nothing was removed from them (Step 0
  // keeps the list order), so the common case copies only the caller's (pinned) arrays.
  const bool ins_is_batch = num_ins == batch.insert_vertices.size();
  const bool del_eff_is_batch = num_del_eff == num_del;
  const bool labels = !batch.insert_vertex_labels.empty();
  // The edge scratch: what graph::reserve() sized, else grown like the vertex arrays (at most
  // the 2^30 - 1 edges a batch may have).
  if (static_cast<std::int64_t>(num_edges) > g.max_edges) {
    const auto want = static_cast<std::int64_t>(
        grown(static_cast<std::size_t>(std::max<std::int64_t>(g.max_edges, 0)), num_edges));
    reserve_device_slotted(res, g,
                           std::max(static_cast<std::int64_t>(num_edges),
                                    std::min(want, (std::int64_t{1} << 30) - 1)));
  }
  ensure(res, g.deleted, num_del);
  // Del_eff is a subset of the batch's deletions: sized with them, so that a later batch whose
  // Del_eff first differs from its deletions does not reserve again (M10a review).
  if (num_del > 0) {
    ensure(res, g.deleted_effective, num_del);
  }
  ensure(res, g.inserted, num_ins);
  if (labels) {
    ensure(res, g.inserted_labels, num_ins);
  }
  // The staging of the pageable arrays.
  const auto pageable = [](memory_space s) { return s != memory_space::pinned_host; };
  std::size_t staged = 0;
  if (pageable(batch.delete_vertices.space())) {
    staged += aligned(num_del * sizeof(vertex_t));
  }
  staged += !del_eff_is_batch ? aligned(num_del_eff * sizeof(vertex_t)) : 0;
  if (!ins_is_batch || pageable(batch.insert_vertices.space())) {
    staged += aligned(num_ins * sizeof(vertex_t));
  }
  if (labels && (!ins_is_batch || pageable(batch.insert_vertex_labels.space()))) {
    staged += aligned(num_ins);
  }
  if (pageable(batch.insert_src.space())) {
    staged += aligned(num_edges * sizeof(vertex_t));
  }
  if (pageable(batch.insert_dst.space())) {
    staged += aligned(num_edges * sizeof(vertex_t));
  }
  if (g.num_weights > 0 && pageable(batch.insert_weights.space())) {
    staged += aligned(num_edges * sizeof(weight_t));
  }
  // The previous batch's copies out of the staging must be complete before it is overwritten.
  g.staging_fence.synchronize();
  if (g.staging.size() < staged) {
    note_reservation();
    g.staging = buffer<unsigned char>();
    g.staging = buffer<unsigned char>(staged + staged / 2, res.stream(),
                                      resources_access::staging_memory(res), res.device());
  }
  std::size_t offset = 0;
  unsigned char* st = g.staging.data();
  // The ids first (the original's idsReady), then the edges (edgesReady).
  put(res, st, offset, g.deleted.data(), batch.delete_vertices.data(),
      batch.delete_vertices.space(), num_del);
  if (!del_eff_is_batch) {
    put(res, st, offset, g.deleted_effective.data(), normalized.deleted_vertices.data(),
        memory_space::host, num_del_eff);
  }
  if (ins_is_batch) {
    put(res, st, offset, g.inserted.data(), batch.insert_vertices.data(),
        batch.insert_vertices.space(), num_ins);
  } else {
    put(res, st, offset, g.inserted.data(), normalized.inserted_vertices.data(), memory_space::host,
        num_ins);
  }
  if (labels) {
    if (ins_is_batch) {
      put(res, st, offset, g.inserted_labels.data(), batch.insert_vertex_labels.data(),
          batch.insert_vertex_labels.space(), num_ins);
    } else {
      put(res, st, offset, g.inserted_labels.data(), normalized.inserted_labels.data(),
          memory_space::host, num_ins);
    }
  }
  put(res, st, offset, g.edge_src.data(), batch.insert_src.data(), batch.insert_src.space(),
      num_edges);
  put(res, st, offset, g.edge_dst.data(), batch.insert_dst.data(), batch.insert_dst.space(),
      num_edges);
  if (g.num_weights > 0) {
    put(res, st, offset, g.edge_weight.data(), batch.insert_weights.data(),
        batch.insert_weights.space(), num_edges);
  }
  if (offset > 0) {
    g.staging_fence.record(res.device(), res.stream());
  }
  g.num_deleted = static_cast<std::int64_t>(num_del);
  g.num_deleted_effective = static_cast<std::int64_t>(num_del_eff);
  g.num_inserted = static_cast<std::int64_t>(num_ins);
  g.num_edges = static_cast<std::int64_t>(num_edges);
  g.has_labels = labels;
  g.deleted_effective_ptr = del_eff_is_batch ? g.deleted.data() : g.deleted_effective.data();
  g.batch_current = true;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void apply_vertex_batch_device(const resources& res,
                               device_slotted<vertex_t, edge_t, weight_t>& g) {
  DYNG_EXPECTS(g.batch_current, "apply (slotted, device): the batch was not uploaded");
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const int n = static_cast<int>(g.num_vertices);
  std::int64_t* report = g.report.data();
  g.last_settled = device_slotted_settled{};
  DYNG_CUDA_TRY(cudaMemsetAsync(report, 0, report_entries * sizeof(std::int64_t), stream));
  // The live entries Del_eff takes with it (before the states change).
  const auto num_del_eff = static_cast<int>(g.num_deleted_effective);
  if (num_del_eff > 0) {
    mark_deleting_kernel<<<blocks_for(num_del_eff), threads_per_block, 0, stream>>>(
        g.deleted_effective_ptr, num_del_eff, g.status.data());
    DYNG_CUDA_TRY(cudaGetLastError());
    count_removed_kernel<<<blocks_for(num_del_eff), threads_per_block, 0, stream>>>(
        g.deleted_effective_ptr, num_del_eff, g.row_ptr.data(), g.row_end.data(), g.col_ind.data(),
        g.status.data(), report);
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  // launchApplyVertexChanges: the deletions, then Ins \ Del (the seeds are the result's).
  const auto num_del = static_cast<int>(g.num_deleted);
  if (num_del > 0) {
    set_status_kernel<<<blocks_for(num_del), threads_per_block, 0, stream>>>(
        g.deleted.data(), num_del, status_deleted, g.status.data());
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  const auto num_ins = static_cast<int>(g.num_inserted);
  if (num_ins > 0) {
    insert_kernel<<<blocks_for(num_ins), threads_per_block, 0, stream>>>(g.inserted.data(), num_ins,
                                                                         g.status.data(), report);
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  // launchAppendEdges.
  const auto num_edges = static_cast<int>(g.num_edges);
  if (num_edges > 0) {
    const int half_edges = 2 * num_edges;
    half_edges_kernel<<<blocks_for(num_edges), threads_per_block, 0, stream>>>(
        g.edge_src.data(), g.edge_dst.data(), num_edges, g.status.data(), n, g.keys[0].data(),
        g.items[0].data(), report);
    DYNG_CUDA_TRY(cudaGetLastError());
    cub::DoubleBuffer<std::int32_t> keys(g.keys[0].data(), g.keys[1].data());
    cub::DoubleBuffer<std::int32_t> ids(g.items[0].data(), g.items[1].data());
    std::size_t bytes = g.temp.size();
    DYNG_CUDA_TRY(cub::DeviceRadixSort::SortPairs(g.temp.data(), bytes, keys, ids, half_edges, 0,
                                                  g.key_bits, stream));
    const std::int32_t* sorted_keys = keys.Current();
    const std::int32_t* sorted_ids = ids.Current();
    std::int32_t* rank = keys.Alternate();  // free after the sort
    bytes = g.temp.size();
    DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSumByKey(
        g.temp.data(), bytes, sorted_keys, thrust::constant_iterator<std::int32_t>(1), rank,
        half_edges, ::cuda::std::equal_to<>(), stream));
    append_kernel<<<blocks_for(half_edges), threads_per_block, 0, stream>>>(
        sorted_keys, sorted_ids, rank, half_edges, n, g.edge_src.data(), g.edge_dst.data(),
        g.edge_weight.data(), g.num_weights, g.row_ptr.data(), g.row_end.data(), g.col_ind.data(),
        g.weights.data(), report);
    DYNG_CUDA_TRY(cudaGetLastError());
    advance_row_end_kernel<<<blocks_for(half_edges), threads_per_block, 0, stream>>>(
        sorted_keys, rank, half_edges, n, g.row_ptr.data(), g.row_end.data());
    DYNG_CUDA_TRY(cudaGetLastError());
    count_dropped_kernel<<<1, 32, 0, stream>>>(sorted_keys, rank, half_edges, n, report);
    DYNG_CUDA_TRY(cudaGetLastError());
  }
  // launchReadErrorFlags (read after the caller's next synchronization: settle()).
  DYNG_CUDA_TRY(cudaMemcpyAsync(g.report_host.data(), report, report_entries * sizeof(std::int64_t),
                                cudaMemcpyDeviceToHost, stream));
  g.batch_current = false;
  g.pending = true;
}

template <typename vertex_t, typename edge_t, typename weight_t>
device_slotted_settled settle_device_slotted(const device_slotted<vertex_t, edge_t, weight_t>& g) {
  const std::int64_t* r = g.report_host.data();
  device_slotted_settled out;
  out.error_bits = static_cast<std::uint32_t>(r[report_error_bits]);
  out.reinserted = r[report_reinserted] - 1;
  out.bad_endpoint = r[report_bad_endpoint] - 1;
  out.overflow_row = r[report_overflow_row] - 1;
  out.dropped = r[report_dropped];
  out.appended = 2 * g.num_edges - out.dropped;
  out.removed = r[report_removed];
  g.pending = false;
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void download_device_slotted(const device_slotted<vertex_t, edge_t, weight_t>& g,
                             csr<vertex_t, edge_t, weight_t>& slots, std::vector<edge_t>& row_end) {
  const scoped_device guard(g.device);
  const auto stream = static_cast<cudaStream_t>(g.col_ind.stream().get());
  const auto n = static_cast<std::size_t>(g.num_vertices);
  const auto m = static_cast<std::size_t>(g.num_slots);
  slots.col_ind.resize(m);
  slots.weights.resize(m * static_cast<std::size_t>(g.num_weights));
  row_end.resize(n);
  if (m > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(slots.col_ind.data(), g.col_ind.data(), m * sizeof(vertex_t),
                                  cudaMemcpyDeviceToHost, stream));
    if (g.num_weights > 0) {
      DYNG_CUDA_TRY(cudaMemcpyAsync(slots.weights.data(), g.weights.data(), m * sizeof(weight_t),
                                    cudaMemcpyDeviceToHost, stream));
    }
  }
  if (n > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(row_end.data(), g.row_end.data(), n * sizeof(edge_t),
                                  cudaMemcpyDeviceToHost, stream));
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
}

#define DYNG_INSTANTIATE_DEVICE_SLOTTED(V, E, W)                                                  \
  template void build_device_slotted<V, E, W>(const resources&, const std::vector<E>&, int,       \
                                              device_slotted<V, E, W>&);                          \
  template void upload_device_slotted<V, E, W>(                                                   \
      const resources&, const csr<V, E, W>&, const std::vector<E>&,                               \
      const std::vector<vertex_status>&, device_slotted<V, E, W>&);                               \
  template void reserve_device_slotted<V, E, W>(const resources&, device_slotted<V, E, W>&,       \
                                                std::int64_t);                                    \
  template void upload_vertex_batch<V, E, W>(const resources&, device_slotted<V, E, W>&,          \
                                             const edge_batch_view<V, W>&,                        \
                                             const normalized_batch<V>&);                         \
  template void apply_vertex_batch_device<V, E, W>(const resources&, device_slotted<V, E, W>&);   \
  template device_slotted_settled settle_device_slotted<V, E, W>(const device_slotted<V, E, W>&); \
  template void download_device_slotted<V, E, W>(const device_slotted<V, E, W>&, csr<V, E, W>&,   \
                                                 std::vector<E>&);

DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int32_t, std::int32_t)
DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int64_t, std::int32_t)
DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int32_t, ::dyng::unweighted)
DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int64_t, ::dyng::unweighted)
DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int32_t, float)
DYNG_INSTANTIATE_DEVICE_SLOTTED(std::int32_t, std::int64_t, float)
#undef DYNG_INSTANTIATE_DEVICE_SLOTTED

DYNG_REGISTER_KERNEL(count_removed_kernel<std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(count_removed_kernel<std::int32_t, std::int64_t>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int32_t, std::int32_t>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int64_t, std::int32_t>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int32_t, ::dyng::unweighted>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int64_t, ::dyng::unweighted>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int32_t, float>);
DYNG_REGISTER_KERNEL(append_kernel<std::int32_t, std::int64_t, float>);
DYNG_REGISTER_KERNEL(advance_row_end_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(advance_row_end_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(mark_deleting_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(set_status_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(insert_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(half_edges_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(count_dropped_kernel);
DYNG_REGISTER_KERNEL(widen_status_kernel);
DYNG_REGISTER_KERNEL(fill_status_kernel);

}  // namespace dyng::detail
