// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_cuda_kernel.cu (mark_owners_kernel,
// change_rows_kernel, next_degree_kernel, build_next_rows_kernel, grid_for, exclusive_scan and the
// G_{t+1} part of count_update_cycles_device)
/**
 * @file apply_set_device.cu
 * @brief The device set apply: a normalized batch (batch_semantics::as_sets) merged into the
 *        sorted rows of a resident graph on the device, so the graph stays on the device across
 *        batches (PLAN Sections 6.4.1 and 6.4.3).
 *
 * Straight port of the part of CycleEnumeration-GPU's count_update_cycles_device() that builds
 * G_{t+1} on the device: the deleted positions of G_t are marked (mark_owners_kernel on an owner
 * array cleared to kNoOwner), the first change of every row is found in both sorted lists
 * (change_rows_kernel), the degrees of G_{t+1} are counted (next_degree_kernel) and scanned, and
 * one warp per vertex writes its row (build_next_rows_kernel: a ballot compaction of the kept
 * entries of a row without insertions, a sorted merge otherwise), writing the insertion id of every
 * inserted edge (next_owner, here device_graph::insertion_ids).
 *
 * Mechanical changes: names, the offsets templated on the graph's edge type (uint32_t for int32_t
 * offsets, the original's layout), dyng buffers from the memory resource of `res` on its stream
 * instead of cudaMalloc and cudaMemcpy, the change lists uploaded once per update from pinned
 * staging (upload_normalized_batch; the cycle_count phases read the same copy), G_{t+1} built
 * whether or not the batch has insertions (the original builds it only for its insert phase; here
 * it is the graph's next state), and a stream synchronization at the end instead of the original's
 * cudaDeviceSynchronize before the histograms are copied.
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/device_graph.hpp"
#include "graph/instantiate.hpp"
#include "graph/normalized_batch.hpp"
#include "util/cuda_check.hpp"
#include "util/device_csr.cuh"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace dyng::detail {

namespace {

constexpr unsigned int set_apply_block_size = 128;  // the original's kBlockSize

// ---- kernels (CycleEnumeration-GPU@0a976ad:src/dynamic/update_cuda_kernel.cu) -------------------

/// owner[position of change w in the CSR] = w (mark_owners_kernel). Deletions are validated, so
/// each one is present.
template <typename offset_t>
__global__ void mark_owners_kernel(const device_csr<offset_t> graph, const device_edge* changes,
                                   const std::uint32_t change_count, int* owner) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < change_count; w += stride) {
    const device_vertex source = changes[w].source;
    const device_vertex target = changes[w].target;
    const offset_t begin = graph.offsets[source];
    const offset_t end = graph.offsets[source + 1];
    const offset_t position = lower_bound_u32(graph.neighbors, begin, end, target);
    if (position < end && graph.neighbors[position] == target) {
      owner[position] = static_cast<int>(w);
    }
  }
}

/// First index of each vertex's changes in a change list sorted by source (change_rows_kernel).
__global__ void change_rows_kernel(const device_edge* changes, const std::uint32_t change_count,
                                   const std::uint32_t vertex_count, std::uint32_t* row_begin) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v <= vertex_count; v += stride) {
    std::uint32_t low = 0;
    std::uint32_t high = change_count;
    while (low < high) {
      const std::uint32_t mid = low + ((high - low) >> 1);
      if (changes[mid].source < v) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    row_begin[v] = low;
  }
}

/// Out-degree of every vertex of G_{t+1} (next_degree_kernel).
template <typename offset_t>
__global__ void next_degree_kernel(const device_csr<offset_t> base,
                                   const std::uint32_t next_vertex_count,
                                   const std::uint32_t* deletion_rows,
                                   const std::uint32_t* insertion_rows, offset_t* degree) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < next_vertex_count;
       v += stride) {
    const offset_t old_degree =
        v < base.vertex_count ? base.offsets[v + 1] - base.offsets[v] : offset_t{0};
    degree[v] = old_degree - (deletion_rows[v + 1] - deletion_rows[v]) +
                (insertion_rows[v + 1] - insertion_rows[v]);
  }
}

/// One warp per vertex writes its G_{t+1} row: rows without insertions are the old row minus
/// deleted positions (owner of the deletion phase set), rows with insertions merge the two sorted
/// lists. The owner array of the insertion phase holds the insertion id of each inserted edge
/// (build_next_rows_kernel).
template <typename offset_t>
__global__ void build_next_rows_kernel(const device_csr<offset_t> base,
                                       const int* __restrict__ deletion_owner,
                                       const device_edge* insertions,
                                       const std::uint32_t* insertion_rows,
                                       const std::uint32_t next_vertex_count,
                                       const offset_t* next_offsets, device_vertex* next_neighbors,
                                       int* next_owner) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warps = (gridDim.x * blockDim.x) >> 5;
  for (std::uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; v < next_vertex_count;
       v += warps) {
    const offset_t old_begin = v < base.vertex_count ? base.offsets[v] : offset_t{0};
    const offset_t old_end = v < base.vertex_count ? base.offsets[v + 1] : offset_t{0};
    const std::uint32_t insert_begin = insertion_rows[v];
    const std::uint32_t insert_end = insertion_rows[v + 1];
    offset_t out = next_offsets[v];
    if (insert_begin == insert_end) {
      // Kept old entries in order: a warp-wide compaction by ballot.
      for (offset_t base_index = old_begin; base_index < old_end; base_index += 32U) {
        const offset_t index = base_index + lane;
        const bool keep = index < old_end && deletion_owner[index] == no_change_id;
        const unsigned int ballot = __ballot_sync(0xffffffffU, keep);
        if (keep) {
          const std::uint32_t rank = __popc(ballot & ((1U << lane) - 1U));
          next_neighbors[out + rank] = base.neighbors[index];
          next_owner[out + rank] = no_change_id;
        }
        out += static_cast<offset_t>(__popc(ballot));
      }
    } else if (lane == 0) {
      offset_t a = old_begin;
      std::uint32_t b = insert_begin;
      while (a < old_end || b < insert_end) {
        if (a < old_end && deletion_owner[a] != no_change_id) {
          ++a;
          continue;
        }
        const bool take_old =
            b >= insert_end || (a < old_end && base.neighbors[a] < insertions[b].target);
        if (take_old) {
          next_neighbors[out] = base.neighbors[a++];
          next_owner[out] = no_change_id;
        } else {
          next_neighbors[out] = insertions[b].target;
          next_owner[out] = static_cast<int>(b);
          ++b;
        }
        ++out;
      }
    }
  }
}

/// present[w] = 1 if the requested change w is an edge of the graph (has_edge on the device: a
/// binary search in the sorted row of its source; 0 for a source >= n).
template <typename offset_t>
__global__ void edge_membership_kernel(const device_csr<offset_t> graph, const device_edge* changes,
                                       const std::uint32_t change_count, std::uint8_t* present) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < change_count; w += stride) {
    const device_vertex source = changes[w].source;
    const device_vertex target = changes[w].target;
    std::uint8_t found = 0;
    if (source < graph.vertex_count) {
      const offset_t begin = graph.offsets[source];
      const offset_t end = graph.offsets[source + 1];
      const offset_t position = lower_bound_u32(graph.neighbors, begin, end, target);
      found = position < end && graph.neighbors[position] == target ? 1 : 0;
    }
    present[w] = found;
  }
}

// ---- host helpers (grid_for, exclusive_scan) ----------------------------------------------------

/// The grid is capped at 2^20 blocks (2^27 threads at the block size), so every kernel launched
/// with it loops with a grid stride (grid_for).
unsigned int grid_for(std::uint64_t work) {
  const std::uint64_t blocks = (work + set_apply_block_size - 1) / set_apply_block_size;
  return static_cast<unsigned int>(
      std::min<std::uint64_t>(std::max<std::uint64_t>(blocks, 1), 1U << 20));
}

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// A scratch buffer of at least `size` elements from the memory resource of `res` (kept, and
/// reallocated only to grow or to follow another memory resource).
template <typename value_t>
value_t* scratch(const resources& res, buffer<value_t>& b, std::size_t size) {
  size = std::max<std::size_t>(size, 1);
  if (b.size() < size || b.memory_resource() != res.memory()) {
    b = buffer<value_t>();  // release first: the peak holds one array
    b = buffer<value_t>(res, size);
  }
  return b.data();
}

/// Exclusive prefix sum of `counts` (n values) into `offsets` (n + 1 values; counts[n] is 0), with
/// the scan's scratch in `temp`.
template <typename value_t>
void exclusive_scan(const resources& res, const value_t* counts, value_t* offsets, std::size_t n,
                    buffer<unsigned char>& temp) {
  std::size_t temp_bytes = 0;
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, offsets, n + 1, native(res)));
  unsigned char* bytes = scratch(res, temp, temp_bytes);
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(bytes, temp_bytes, counts, offsets, n + 1, native(res)));
}

}  // namespace

template <typename vertex_t>
const std::uint32_t* upload_normalized_batch(const resources& res,
                                             const normalized_batch<vertex_t>& nb) {
  static_assert(sizeof(vertex_t) == 4, "the device change lists hold 32-bit vertex ids");
  const std::size_t pairs = nb.deletions.size() + nb.insertions.size();
  const std::size_t words = std::max<std::size_t>(2 * pairs, 2);
  if (nb.device_current && nb.device_lists.size() >= words) {
    return nb.device_lists.data();
  }
  const scoped_device guard(res.device());
  if (nb.device_lists.size() < words || nb.device_lists.memory_resource() != res.memory()) {
    note_reservation();  // a deliberate growth of a reusable array (I9)
    nb.device_lists = buffer<std::uint32_t>();
    nb.device_lists = buffer<std::uint32_t>(res, words);
  }
  if (nb.staging.size() < words) {
    note_reservation();  // a deliberate growth of a reusable array (I9)
    nb.staging = buffer<std::uint32_t>();
    nb.staging = buffer<std::uint32_t>(words, res.stream(), resources_access::staging_memory(res),
                                       res.device());
  } else {
    // The staging buffer may still be read by the copy of an earlier update on this stream.
    DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
    note_host_sync();  // counted by the budgets (I9)
  }
  std::uint32_t* host = nb.staging.data();
  std::size_t k = 0;
  for (const auto& c : nb.deletions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  for (const auto& c : nb.insertions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  if (k > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(nb.device_lists.data(), host, k * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice, native(res)));
  }
  nb.device_current = true;
  return nb.device_lists.data();
}

template <typename vertex_t, typename edge_t, typename weight_t>
const int* mark_normalized_deletions(const resources& res,
                                     const device_graph<vertex_t, edge_t, weight_t>& base,
                                     const normalized_batch<vertex_t>& normalized) {
  static_assert(device_set_apply_supported_v<vertex_t>,
                "the device deletion marks read 32-bit vertex ids");
  using offset_t = device_offset_t<edge_t>;
  DYNG_EXPECTS(normalized.vertices_before == static_cast<std::int64_t>(base.num_vertices) &&
                   normalized.edges_before == static_cast<std::int64_t>(base.num_edges),
               "mark_normalized_deletions: the normalized batch belongs to another graph state");
  const auto m = std::max<std::size_t>(static_cast<std::size_t>(base.num_edges), 1);
  if (normalized.deletions_marked && normalized.deletion_owner.size() >= m) {
    return normalized.deletion_owner.data();
  }
  DYNG_EXPECTS(static_cast<std::int64_t>(normalized.deletions.size()) < no_change_id,
               "mark_normalized_deletions: the batch exceeds the 32-bit change ids");
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const std::uint32_t* lists = upload_normalized_batch(res, normalized);
  if (normalized.deletion_owner.size() < m ||
      normalized.deletion_owner.memory_resource() != res.memory()) {
    note_reservation();                         // a deliberate growth of a reusable array (I9)
    normalized.deletion_owner = buffer<int>();  // release first: the peak holds one array
    normalized.deletion_owner = buffer<int>(res, m);
  }
  int* owner = normalized.deletion_owner.data();
  DYNG_CUDA_TRY(cudaMemsetAsync(owner, 0x7f, sizeof(int) * m, stream));  // no_change_id
  const auto deletion_count = static_cast<std::uint32_t>(normalized.deletions.size());
  if (deletion_count > 0) {
    const device_csr<offset_t> graph{
        static_cast<std::uint32_t>(base.num_vertices),
        reinterpret_cast<const offset_t*>(base.out_row_ptr.data()),
        reinterpret_cast<const device_vertex*>(base.out_col_ind.data())};
    mark_owners_kernel<offset_t><<<grid_for(deletion_count), set_apply_block_size, 0, stream>>>(
        graph, reinterpret_cast<const device_edge*>(lists), deletion_count, owner);
    DYNG_CHECK_KERNEL(stream);
  }
  normalized.deletions_marked = true;
  return owner;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void apply_set_batch_device(const resources& res,
                            const device_graph<vertex_t, edge_t, weight_t>& base,
                            const normalized_batch<vertex_t>& normalized,
                            device_graph<vertex_t, edge_t, weight_t>& next) {
  static_assert(device_set_apply_supported_v<vertex_t>,
                "the device set apply reads 32-bit vertex ids");
  using offset_t = device_offset_t<edge_t>;
  DYNG_EXPECTS(base.num_weights == 0,
               "apply (device): the device set apply merges graphs without "
               "weight columns only");
  DYNG_EXPECTS(normalized.vertices_before == static_cast<std::int64_t>(base.num_vertices) &&
                   normalized.edges_before == static_cast<std::int64_t>(base.num_edges),
               "apply (device): the normalized batch belongs to another graph state");
  // The original's 32-bit limits of the change ids and rows (owner values are int, below
  // kNoOwner).
  constexpr std::int64_t id_limit = no_change_id;
  if (static_cast<std::int64_t>(normalized.deletions.size()) >= id_limit ||
      static_cast<std::int64_t>(normalized.insertions.size()) >= id_limit ||
      normalized.vertices_after >=
          static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw capacity_error(
        "dyng: apply (device): the batch or the graph exceeds the 32-bit change "
        "ids of the device set apply");
  }
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const auto next_count = static_cast<std::uint32_t>(normalized.vertices_after);
  const auto next_edges = static_cast<std::size_t>(normalized.edges_after);
  const auto deletion_count = static_cast<std::uint32_t>(normalized.deletions.size());
  const auto insertion_count = static_cast<std::uint32_t>(normalized.insertions.size());
  const std::uint32_t* lists = upload_normalized_batch(res, normalized);
  const auto* deletions = reinterpret_cast<const device_edge*>(lists);
  const auto* insertions = reinterpret_cast<const device_edge*>(lists) + deletion_count;
  const device_csr<offset_t> old_graph{
      static_cast<std::uint32_t>(base.num_vertices),
      reinterpret_cast<const offset_t*>(base.out_row_ptr.data()),
      reinterpret_cast<const device_vertex*>(base.out_col_ind.data())};

  // Deleted positions of G_t: the owner array of the deletion phase (shared with the cycle_count
  // delete phase, which may have marked them already).
  const int* deletion_owner = mark_normalized_deletions(res, base, normalized);

  // G_{t+1}: change rows, degrees, offsets, rows.
  next = device_graph<vertex_t, edge_t, weight_t>();
  next.device = res.device();
  next.num_vertices = static_cast<vertex_t>(next_count);
  next.num_edges = static_cast<edge_t>(next_edges);
  next.num_weights = 0;
  // The scratch of the merge is kept in `normalized` (a pooled workspace): a steady workload
  // allocates only the arrays of G_{t+1} below.
  const std::size_t rows = std::size_t{next_count} + 1;
  std::uint32_t* deletion_rows = scratch(res, normalized.apply_rows, 2 * rows);
  std::uint32_t* insertion_rows = deletion_rows + rows;
  auto* degree = reinterpret_cast<offset_t*>(
      scratch(res, normalized.apply_degree, rows * sizeof(offset_t)));  // aligned by the resource
  change_rows_kernel<<<grid_for(std::uint64_t{next_count} + 1), set_apply_block_size, 0, stream>>>(
      deletions, deletion_count, next_count, deletion_rows);
  DYNG_CHECK_KERNEL(stream);
  change_rows_kernel<<<grid_for(std::uint64_t{next_count} + 1), set_apply_block_size, 0, stream>>>(
      insertions, insertion_count, next_count, insertion_rows);
  DYNG_CHECK_KERNEL(stream);
  DYNG_CUDA_TRY(cudaMemsetAsync(degree, 0, sizeof(offset_t) * rows, stream));
  if (next_count > 0) {
    next_degree_kernel<offset_t><<<grid_for(next_count), set_apply_block_size, 0, stream>>>(
        old_graph, next_count, deletion_rows, insertion_rows, degree);
    DYNG_CHECK_KERNEL(stream);
  }
  next.out_row_ptr = buffer<edge_t>(res, rows);
  exclusive_scan(res, degree, reinterpret_cast<offset_t*>(next.out_row_ptr.data()), next_count,
                 normalized.apply_scan_temp);
  next.out_col_ind = buffer<vertex_t>(res, next_edges);
  next.insertion_ids = buffer<std::int32_t>(res, std::max<std::size_t>(next_edges, 1));
  if (next_count > 0) {
    build_next_rows_kernel<offset_t><<<grid_for(static_cast<std::uint64_t>(next_count) * 32U),
                                       set_apply_block_size, 0, stream>>>(
        old_graph, deletion_owner, insertions, insertion_rows, next_count,
        reinterpret_cast<const offset_t*>(next.out_row_ptr.data()),
        reinterpret_cast<device_vertex*>(next.out_col_ind.data()), next.insertion_ids.data());
    DYNG_CHECK_KERNEL(stream);
  }
  // The state is complete before it becomes the graph's (its host copy may be downloaded on
  // another stream; the scratch above is released on this stream).
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // counted by the budgets (I9)
}

template <typename vertex_t, typename edge_t, typename weight_t>
void device_edge_membership(const resources& res,
                            const device_graph<vertex_t, edge_t, weight_t>& base,
                            const std::vector<set_change<vertex_t>>& deletions,
                            const std::vector<set_change<vertex_t>>& insertions,
                            const normalized_batch<vertex_t>& nb,
                            std::vector<std::uint8_t>& present) {
  static_assert(device_set_apply_supported_v<vertex_t>,
                "the device membership test reads 32-bit vertex ids");
  using offset_t = device_offset_t<edge_t>;
  const std::size_t pairs = deletions.size() + insertions.size();
  present.resize(pairs);
  if (pairs == 0) {
    return;
  }
  if (static_cast<std::int64_t>(pairs) >= static_cast<std::int64_t>(no_change_id)) {
    throw capacity_error(
        "dyng: apply (device): the batch exceeds the 32-bit change ids of the device set apply");
  }
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const std::size_t words = 2 * pairs;
  // The lists' device and staging buffers of `nb` serve as scratch: the normalized lists are
  // uploaded into them afterwards (upload_normalized_batch; device_current is false until then).
  if (nb.device_lists.size() < words || nb.device_lists.memory_resource() != res.memory()) {
    nb.device_lists = buffer<std::uint32_t>();
    nb.device_lists = buffer<std::uint32_t>(res, words);
  }
  if (nb.staging.size() < words) {
    nb.staging = buffer<std::uint32_t>();
    nb.staging = buffer<std::uint32_t>(words, res.stream(), resources_access::staging_memory(res),
                                       res.device());
  } else {
    // The staging buffer may still be read by the copy of an earlier update on this stream.
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    note_host_sync();  // counted by the budgets (I9)
  }
  if (nb.device_present.size() < pairs || nb.device_present.memory_resource() != res.memory()) {
    nb.device_present = buffer<std::uint8_t>();
    nb.device_present = buffer<std::uint8_t>(res, pairs);
  }
  if (nb.present_staging.size() < pairs) {
    nb.present_staging = buffer<std::uint8_t>();
    nb.present_staging = buffer<std::uint8_t>(pairs, res.stream(),
                                              resources_access::staging_memory(res), res.device());
  }
  std::uint32_t* host = nb.staging.data();
  std::size_t k = 0;
  for (const auto& c : deletions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  for (const auto& c : insertions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  DYNG_CUDA_TRY(cudaMemcpyAsync(nb.device_lists.data(), host, words * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream));
  const device_csr<offset_t> graph{static_cast<std::uint32_t>(base.num_vertices),
                                   reinterpret_cast<const offset_t*>(base.out_row_ptr.data()),
                                   reinterpret_cast<const device_vertex*>(base.out_col_ind.data())};
  const auto count = static_cast<std::uint32_t>(pairs);
  edge_membership_kernel<offset_t><<<grid_for(count), set_apply_block_size, 0, stream>>>(
      graph, reinterpret_cast<const device_edge*>(nb.device_lists.data()), count,
      nb.device_present.data());
  DYNG_CHECK_KERNEL(stream);
  DYNG_CUDA_TRY(cudaMemcpyAsync(nb.present_staging.data(), nb.device_present.data(), pairs,
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // counted by the budgets (I9)
  std::copy(nb.present_staging.data(), nb.present_staging.data() + pairs, present.begin());
}

#define DYNG_INSTANTIATE_APPLY_SET_DEVICE(V, E, W)                                              \
  template void apply_set_batch_device<V, E, W>(const resources&, const device_graph<V, E, W>&, \
                                                const normalized_batch<V>&,                     \
                                                device_graph<V, E, W>&);                        \
  template const int* mark_normalized_deletions<V, E, W>(                                       \
      const resources&, const device_graph<V, E, W>&, const normalized_batch<V>&);              \
  template void device_edge_membership<V, E, W>(                                                \
      const resources&, const device_graph<V, E, W>&, const std::vector<set_change<V>>&,        \
      const std::vector<set_change<V>>&, const normalized_batch<V>&, std::vector<std::uint8_t>&);
DYNG_INSTANTIATE_APPLY_SET_DEVICE(std::int32_t, std::int32_t, std::int32_t)
DYNG_INSTANTIATE_APPLY_SET_DEVICE(std::int32_t, std::int64_t, std::int32_t)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_APPLY_SET_DEVICE)
#undef DYNG_INSTANTIATE_APPLY_SET_DEVICE

template const std::uint32_t* upload_normalized_batch<std::int32_t>(
    const resources&, const normalized_batch<std::int32_t>&);

// One registration per line (the registrar's name is made unique by __LINE__).
DYNG_REGISTER_KERNEL(mark_owners_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(mark_owners_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(change_rows_kernel);
DYNG_REGISTER_KERNEL(edge_membership_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(edge_membership_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(next_degree_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(next_degree_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(build_next_rows_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(build_next_rows_kernel<std::uint64_t>);

}  // namespace dyng::detail
