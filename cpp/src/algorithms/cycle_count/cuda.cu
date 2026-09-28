// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_cuda_kernel.cu
// (count_update_cycles_device, run_phase, closes_owned, extend_owned_path,
// count_owned_cycles_kernel, mark_owners_kernel, item_counts_kernel) and src/dynamic/update_cuda.cpp
// (update_static_histogram_cuda)
/**
 * @file cuda.cu
 * @brief The CUDA backend of cycle_count::update(): the delete phase on G_t and the insert phase on
 *        G_{t+1}, with the graph resident on the device (Tier B fused kernels).
 *
 * A phase enumerates, for every changed edge (s -> t) with id w, the simple paths
 * t -> ... -> s of at most k - 1 edges that use no changed edge with an id smaller than w; each
 * closes one cycle through (s -> t) that the change owns, so every affected cycle is counted by
 * exactly one change. The path is kept in thread-local memory and membership is a scan of at most
 * k vertices, so memory is O(|changes| + E) instead of O(|changes| x V). Work items are (change,
 * first hop) pairs, so one change with a heavy target does not run on a single thread. An owner
 * array holds, for every CSR position, the ownership id of the change stored there or "none"; it
 * replaces a binary search of the change list on every adjacency read.
 *
 * Straight port; mechanical changes: names, namespace dyng::detail, the offsets templated on the
 * graph's edge type (uint32_t for int32_t offsets: the original's 32-bit CSR), dyng buffers from
 * the workspace pool on the stream of `resources` instead of cudaMalloc on the legacy default
 * stream, the scalar read-back through pinned memory. The data flow of count_update_cycles_device()
 * is split along the hooks: G_t is the graph's resident device copy (the original uploads it per
 * call), the change lists are uploaded once per update (shared with the device apply when the
 * framework normalized the batch), the delete phase runs in before_apply, G_{t+1} is built by the
 * graph's device apply in the commit (graph/apply_set_device.cu, build_next_rows_kernel, whose
 * next_owner is device_graph::insertion_ids), and the insert phase and the copy of both histograms
 * run in after_apply. When the commit applied the batch on the host (other batch semantics, weight
 * columns), G_{t+1} is uploaded and its owner array is built with mark_owners_kernel, as for G_t.
 */
#include "algorithms/cycle_count/dfs.cuh"
#include "algorithms/cycle_count/problem.hpp"
#include "algorithms/cycle_count/work_queue.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/device_graph.hpp"
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

template <typename edge_t>
void cuda_workspace_prepare(const resources& res, cycle_count_cuda_workspace<edge_t>& ws);

namespace {

constexpr unsigned int update_block_size = 128;  // kBlockSize

// ---- kernels (CycleEnumeration-GPU@0a976ad:src/dynamic/update_cuda_kernel.cu) -----------------------

/// Whether from -> to is an edge whose owner id is not below `work` (closes_owned).
template <typename offset_t>
__device__ __forceinline__ bool closes_owned(const device_csr<offset_t> graph,
                                             const int* __restrict__ owner,
                                             const device_vertex from, const device_vertex to,
                                             const int work) {
  const offset_t begin = __ldg(graph.offsets + from);
  const offset_t end = __ldg(graph.offsets + from + 1);
  const offset_t position = lower_bound_u32(graph.neighbors, begin, end, to);
  return position < end && __ldg(graph.neighbors + position) == to &&
         __ldg(owner + position) >= work;
}

/// Search below path[0..prefix-1] (path[0] = t), whose own closing edge was already counted; a
/// path of d vertices closes a cycle of length d + 1 (extend_owned_path).
template <int cap, typename offset_t>
__device__ __forceinline__ void extend_owned_path(const device_csr<offset_t> graph,
                                                  const int* __restrict__ owner,
                                                  const device_vertex source, const int work,
                                                  device_vertex (&path)[cap], const int prefix,
                                                  const int max_length,
                                                  thread_histogram<cap>& histogram) {
  if (prefix + 1 >= max_length) {
    return;
  }
  offset_t cursor[cap];
  offset_t end[cap];
  int depth = prefix;
  cursor[depth - 1] = __ldg(graph.offsets + path[depth - 1]);
  end[depth - 1] = __ldg(graph.offsets + path[depth - 1] + 1);

  while (depth >= prefix) {
    if (cursor[depth - 1] >= end[depth - 1]) {
      --depth;
      continue;
    }
    const offset_t position = cursor[depth - 1]++;
    const device_vertex next = __ldg(graph.neighbors + position);
    if (__ldg(owner + position) < work || next == source) {
      continue;  // a smaller change owns it, or it is the closing edge
    }
    bool on_path = false;
#pragma unroll
    for (int index = 0; index < cap; ++index) {
      if (index < depth && path[index] == next) {
        on_path = true;
      }
    }
    if (on_path) {
      continue;
    }
    if (closes_owned(graph, owner, next, source, work)) {
      ++histogram.count[depth + 2];
    }
    if (depth + 2 < max_length) {
      path[depth] = next;
      cursor[depth] = __ldg(graph.offsets + next);
      end[depth] = __ldg(graph.offsets + next + 1);
      ++depth;
    }
  }
}

/// Items (w, j), j < outdeg(t_w), numbered from item_offset[w]: the path t_w -> (j-th neighbour
/// of t_w). Item j = 0 also counts the 2-cycle (count_owned_cycles_kernel).
template <int cap, typename offset_t>
__global__ void count_owned_cycles_kernel(
    const device_csr<offset_t> graph, const int* __restrict__ owner,
    const device_edge* __restrict__ changes, const unsigned long long* item_offset,
    const std::uint32_t change_count, const unsigned long long item_count, const int max_length,
    unsigned long long* work_counter, unsigned long long* histogram) {
  thread_histogram<cap> counts;
  counts.clear();
  device_vertex path[cap];
  while (true) {
    const unsigned long long item = atomicAdd(work_counter, 1ULL);
    if (item >= item_count) {
      break;
    }
    std::uint32_t low = 0;
    std::uint32_t high = change_count;
    while (high - low > 1) {  // last w with item_offset[w] <= item
      const std::uint32_t mid = low + ((high - low) >> 1);
      if (__ldg(item_offset + mid) <= item) {
        low = mid;
      } else {
        high = mid;
      }
    }
    const int work = static_cast<int>(low);
    const device_vertex source = changes[low].source;
    const device_vertex target = changes[low].target;
    const auto j = static_cast<offset_t>(item - __ldg(item_offset + low));
    if (j == 0 && closes_owned(graph, owner, target, source, work)) {
      ++counts.count[2];
    }
    if (max_length < 3) {
      continue;
    }
    const offset_t position = __ldg(graph.offsets + target) + j;
    const device_vertex next = __ldg(graph.neighbors + position);
    if (__ldg(owner + position) < work || next == source || next == target) {
      continue;
    }
    if (closes_owned(graph, owner, next, source, work)) {
      ++counts.count[3];
    }
    path[0] = target;
    path[1] = next;
    extend_owned_path<cap>(graph, owner, source, work, path, 2, max_length, counts);
  }
  counts.flush(histogram);
}

/// owner[position of change w in the CSR] = w. Deletions are validated, so each one is present
/// (mark_owners_kernel).
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

/// Number of items of each change: the out-degree of its target (0 for a target with no edges,
/// which closes no cycle) (item_counts_kernel).
template <typename offset_t>
__global__ void item_counts_kernel(const device_csr<offset_t> graph, const device_edge* changes,
                                   const std::uint32_t change_count, unsigned long long* counts) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < change_count; w += stride) {
    const device_vertex target = changes[w].target;
    counts[w] =
        target < graph.vertex_count ? graph.offsets[target + 1] - graph.offsets[target] : 0ULL;
  }
}

// ---- host helpers ----------------------------------------------------------------------------

/// The grid is capped at 2^20 blocks (2^27 threads at the block size), so every kernel launched
/// with it loops with a grid stride (grid_for).
unsigned int grid_for(std::uint64_t work) {
  const std::uint64_t blocks = (work + update_block_size - 1) / update_block_size;
  return static_cast<unsigned int>(
      std::min<std::uint64_t>(std::max<std::uint64_t>(blocks, 1), 1U << 20));
}

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// Count the cycles each change owns in `graph` into `histogram` (run_phase).
template <int cap, typename offset_t, typename edge_t>
void run_phase(const resources& res, const device_csr<offset_t> graph, const int* owner,
               const device_edge* changes, const std::uint32_t change_count, const int max_length,
               cycle_count_cuda_workspace<edge_t>& ws, unsigned long long* histogram,
               unsigned long long* work_counter) {
  if (change_count == 0) {
    return;
  }
  const cudaStream_t stream = native(res);
  int resident = 0;
  DYNG_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, count_owned_cycles_kernel<cap, offset_t>, update_block_size, 0));
  const auto grid = static_cast<unsigned int>(std::max(resident, 1) * std::max(ws.sm_count, 1));
  unsigned long long* counts = ws.item_counts.reserve(res, std::size_t{change_count} + 1);
  unsigned long long* offsets = ws.item_offsets.reserve(res, std::size_t{change_count} + 1);
  DYNG_CUDA_TRY(
      cudaMemsetAsync(counts, 0, sizeof(unsigned long long) * (change_count + 1), stream));
  item_counts_kernel<offset_t><<<grid_for(change_count), update_block_size, 0, stream>>>(
      graph, changes, change_count, counts);
  DYNG_CHECK_KERNEL(stream);
  std::size_t temp_bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, offsets,
                                              std::size_t{change_count} + 1, stream));
  unsigned char* temp = ws.scan_temp.reserve(res, std::max<std::size_t>(temp_bytes, 1));
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, counts, offsets,
                                              std::size_t{change_count} + 1, stream));
  unsigned long long* host = ws.host_scalars.data() + 2 * (cycle_count_max_device_length + 1);
  DYNG_CUDA_TRY(cudaMemcpyAsync(host, offsets + change_count, sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  const unsigned long long item_count = *host;
  if (item_count == 0) {
    return;
  }
  DYNG_CUDA_TRY(cudaMemsetAsync(work_counter, 0, sizeof(unsigned long long), stream));
  count_owned_cycles_kernel<cap, offset_t>
      <<<grid, update_block_size, 0, stream>>>(graph, owner, changes, offsets, change_count,
                                               item_count, max_length, work_counter, histogram);
  DYNG_CHECK_KERNEL(stream);
}

}  // namespace

template <typename edge_t>
void cycle_count_cuda_begin_update(const resources& res, cycle_count_cuda_workspace<edge_t>& ws) {
  const scoped_device guard(res.device());
  // The work items of a static count are not read by an update: return them before its arrays
  // are sized (the original frees them after every count; kept, the prior's two-hop items of
  // COLLAB k = 4 alone were 300 MB next to G_t and G_{t+1}).
  ws.forward_begin.release();
  ws.forward_count.release();
  ws.forward_offset.release();
  ws.item_source.release();
  ws.item_target.release();
  ws.hop_count.release();
  ws.hop_offset.release();
  cuda_workspace_prepare(res, ws);
  DYNG_CUDA_TRY(cudaMemsetAsync(
      ws.histograms.data(), 0, sizeof(unsigned long long) * 2 * (cycle_count_max_device_length + 1),
      native(res)));
}

template <typename edge_t, typename vertex_t>
const std::uint32_t* cycle_count_cuda_upload_changes(
    const resources& res, const std::vector<edge_change<vertex_t>>& deletions,
    const std::vector<edge_change<vertex_t>>& insertions, cycle_count_cuda_workspace<edge_t>& ws) {
  const scoped_device guard(res.device());
  const std::size_t words = std::max<std::size_t>(2 * (deletions.size() + insertions.size()), 2);
  std::uint32_t* device = ws.changes.reserve(res, words);
  if (ws.host_changes.size() < words) {
    ws.host_changes = buffer<std::uint32_t>();
    ws.host_changes = buffer<std::uint32_t>(words, res.stream(),
                                            resources_access::staging_memory(res), res.device());
  } else {
    // The staging buffer may still be read by the copy of an earlier update on this stream.
    DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
  }
  std::uint32_t* host = ws.host_changes.data();
  std::size_t k = 0;
  for (const auto& c : deletions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  for (const auto& c : insertions) {
    host[k++] = static_cast<std::uint32_t>(c.source);
    host[k++] = static_cast<std::uint32_t>(c.target);
  }
  if (k > 0) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(device, host, k * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                  native(res)));
  }
  return device;
}

template <typename edge_t>
void cycle_count_cuda_phase(const resources& res, const cycle_device_graph<edge_t>& graph,
                            const std::int32_t* owner, const std::uint32_t* changes,
                            std::uint32_t change_count, std::int64_t length, int slot,
                            cycle_count_cuda_workspace<edge_t>& ws) {
  using offset_t = device_offset_t<edge_t>;
  if (change_count == 0) {
    return;
  }
  DYNG_EXPECTS(length >= 2 && length <= cycle_count_max_device_length,
               "cycle_count: the CUDA backend supports max_cycle_length up to ",
               cycle_count_max_device_length);
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  const device_csr<offset_t> view{static_cast<std::uint32_t>(graph.vertex_count),
                                  reinterpret_cast<const offset_t*>(graph.offsets),
                                  reinterpret_cast<const device_vertex*>(graph.neighbors)};
  const auto* list = reinterpret_cast<const device_edge*>(changes);
  if (owner == nullptr) {
    // owner[position of change w] = w, every other entry kNoOwner (cudaMemset 0x7f).
    const auto m = static_cast<std::size_t>(std::max<std::int64_t>(graph.edge_count, 1));
    std::int32_t* marks = ws.owner.reserve(res, m);
    DYNG_CUDA_TRY(cudaMemsetAsync(marks, 0x7f, sizeof(std::int32_t) * m, stream));
    mark_owners_kernel<offset_t>
        <<<grid_for(change_count), update_block_size, 0, stream>>>(view, list, change_count, marks);
    DYNG_CHECK_KERNEL(stream);
    owner = marks;
  }
  unsigned long long* histogram =
      ws.histograms.data() + static_cast<std::size_t>(slot) * (cycle_count_max_device_length + 1);
  unsigned long long* work_counter = ws.work_counters.data() + slot;
  dispatch_capacity(static_cast<std::size_t>(length), [&](auto capacity) {
    constexpr int cap = decltype(capacity)::value;
    run_phase<cap, offset_t>(res, view, owner, list, change_count, static_cast<int>(length), ws,
                             histogram, work_counter);
  });
}

template <typename edge_t>
void cycle_count_cuda_end_update(const resources& res, std::int64_t length,
                                 cycle_count_cuda_workspace<edge_t>& ws,
                                 std::vector<std::uint64_t>& removed,
                                 std::vector<std::uint64_t>& added) {
  const scoped_device guard(res.device());
  constexpr std::size_t slot = cycle_count_max_device_length + 1;
  unsigned long long* host = ws.host_scalars.data();
  DYNG_CUDA_TRY(cudaMemcpyAsync(host, ws.histograms.data(), sizeof(unsigned long long) * 2 * slot,
                                cudaMemcpyDeviceToHost, native(res)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
  const auto size = static_cast<std::size_t>(length) + 1;
  removed.assign(size, 0);
  added.assign(size, 0);
  for (std::size_t len = 2; len < size && len < slot; ++len) {
    removed[len] = host[len];
    added[len] = host[slot + len];
  }
}

#define DYNG_INSTANTIATE_CYCLE_COUNT_CUDA(E)                                                       \
  template void cycle_count_cuda_begin_update<E>(const resources&,                                 \
                                                 cycle_count_cuda_workspace<E>&);                  \
  template const std::uint32_t* cycle_count_cuda_upload_changes<E, std::int32_t>(                  \
      const resources&, const std::vector<edge_change<std::int32_t>>&,                             \
      const std::vector<edge_change<std::int32_t>>&, cycle_count_cuda_workspace<E>&);              \
  template void cycle_count_cuda_phase<E>(                                                         \
      const resources&, const cycle_device_graph<E>&, const std::int32_t*, const std::uint32_t*,   \
      std::uint32_t, std::int64_t, int, cycle_count_cuda_workspace<E>&);                           \
  template void cycle_count_cuda_end_update<E>(                                                    \
      const resources&, std::int64_t, cycle_count_cuda_workspace<E>&, std::vector<std::uint64_t>&, \
      std::vector<std::uint64_t>&);
DYNG_INSTANTIATE_CYCLE_COUNT_CUDA(std::int32_t)
DYNG_INSTANTIATE_CYCLE_COUNT_CUDA(std::int64_t)
#undef DYNG_INSTANTIATE_CYCLE_COUNT_CUDA

// Kernel registry (resources::warm_up(); one registration per line).
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<4, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<8, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<16, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<32, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<64, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<4, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<8, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<16, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<32, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_owned_cycles_kernel<64, std::uint64_t>);
DYNG_REGISTER_KERNEL(mark_owners_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(mark_owners_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(item_counts_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(item_counts_kernel<std::uint64_t>);

}  // namespace dyng::detail
