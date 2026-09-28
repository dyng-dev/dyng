// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/cuda/cuda_static_kernels.cu
// (count_simple_cycles_johnson_device, count_simple_cycles_johnson_queue_device and their kernels)
// and src/cuda/cuda_work_queue.cpp (count_simple_cycles_johnson_work_queue)
/**
 * @file static_cuda.cu
 * @brief The CUDA backend of cycle_count::compute(): the naive and the work-queue static counters
 *        (Tier B fused kernels, PLAN Section 4.5.4).
 *
 * Both schedulers run the same exact pruned depth-first search (extend_prefix, dfs.cuh): each
 * directed cycle is counted once, from its minimum vertex. The naive scheduler maps one root to
 * one thread. The work queue keeps a device-filling grid resident and lets each thread claim work
 * items from a global counter; the items are roots, forward edges (r -> v1, v1 > r) or two-hop
 * paths (r -> v1 -> v2), built on the device, so the search tree of one heavy root is spread over
 * many threads. Every cycle has exactly one prefix of each kind, so the choice never changes the
 * counts.
 *
 * Straight port; mechanical changes: names, namespace dyng::detail, the offsets templated on the
 * graph's edge type (uint32_t for int32_t offsets: the original's 32-bit CSR), the graph read from
 * the resident device copy (graph_access::device_out; the original uploads it per call, outside its
 * kernel_ms region), the scratch arrays leased from the workspace pool of `resources` (the original
 * allocates them with cudaMalloc inside its kernel region) on the stream of `resources` instead of
 * the legacy default stream, the scalar read-backs through pinned memory with a stream
 * synchronization, the launch grids planned before the timed region (as the original's occupancy
 * queries), no environment tuning (the original's CYCLE_ENUM_CUDA_BLOCK_SIZE and
 * CYCLE_ENUM_CUDA_BLOCKS_PER_SM keep their defaults: 128 threads, the occupancy limit), and the
 * NVTX ranges replaced by the profiler stages cycle_count.count (kernel_ms) and
 * cycle_count.finalize (the histogram copy).
 */
#include "algorithms/cycle_count/dfs.cuh"
#include "algorithms/cycle_count/problem.hpp"
#include "algorithms/cycle_count/work_queue.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "util/cuda_check.hpp"
#include "util/device_csr.cuh"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dyng::detail {

namespace {

// ---- kernels (CycleEnumeration-GPU@0a976ad:src/cuda/cuda_static_kernels.cu) -----------------------

/// One root per thread (count_roots_kernel).
template <int cap, typename offset_t>
__global__ void count_roots_kernel(const device_csr<offset_t> graph, const int max_length,
                                   unsigned long long* histogram) {
  thread_histogram<cap> counts;
  counts.clear();
  const std::uint64_t root = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (root < graph.vertex_count) {
    device_vertex path[cap];
    path[0] = static_cast<device_vertex>(root);
    extend_prefix<cap>(graph, path, 1, max_length, counts);
  }
  counts.flush(histogram);
}

/// Roots claimed from a global counter (count_roots_queue_kernel).
template <int cap, typename offset_t>
__global__ void count_roots_queue_kernel(const device_csr<offset_t> graph, const int max_length,
                                         unsigned long long* work_counter,
                                         unsigned long long* histogram) {
  thread_histogram<cap> counts;
  counts.clear();
  device_vertex path[cap];
  while (true) {
    const unsigned long long root = atomicAdd(work_counter, 1ULL);
    if (root >= graph.vertex_count) {
      break;
    }
    path[0] = static_cast<device_vertex>(root);
    extend_prefix<cap>(graph, path, 1, max_length, counts);
  }
  counts.flush(histogram);
}

// ---- prefix work items ------------------------------------------------------------------------

/// Forward row split: for every vertex v, the neighbours greater than v start at forward_begin[v]
/// and there are forward_count[v] of them (forward_rows_kernel).
template <typename offset_t>
__global__ void forward_rows_kernel(const device_csr<offset_t> graph, offset_t* forward_begin,
                                    offset_t* forward_count) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < graph.vertex_count;
       v += stride) {
    const offset_t begin = graph.offsets[v];
    const offset_t end = graph.offsets[v + 1];
    const offset_t split = lower_bound_u32(graph.neighbors, begin, end, v + 1);
    forward_begin[v] = split;
    forward_count[v] = end - split;
  }
}

/// One warp per vertex copies its forward edges into the item arrays (fill_edge_items_kernel).
template <typename offset_t>
__global__ void fill_edge_items_kernel(const device_csr<offset_t> graph,
                                       const offset_t* forward_begin,
                                       const offset_t* forward_offset, device_vertex* item_source,
                                       device_vertex* item_target) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warps = (gridDim.x * blockDim.x) >> 5;
  for (std::uint32_t v = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; v < graph.vertex_count;
       v += warps) {
    const offset_t out = forward_offset[v];
    const offset_t count = forward_offset[v + 1] - out;
    const offset_t begin = forward_begin[v];
    for (offset_t j = lane; j < count; j += 32U) {
      item_source[out + j] = v;
      item_target[out + j] = graph.neighbors[begin + j];
    }
  }
}

/// Out-degree of each edge item's target: the number of two-hop items it spawns
/// (target_degree_kernel).
template <typename offset_t>
__global__ void target_degree_kernel(const device_csr<offset_t> graph,
                                     const device_vertex* item_target, const offset_t item_count,
                                     unsigned long long* degree) {
  const offset_t stride = gridDim.x * blockDim.x;
  for (offset_t q = blockIdx.x * blockDim.x + threadIdx.x; q < item_count; q += stride) {
    const device_vertex target = item_target[q];
    degree[q] = graph.offsets[target + 1] - graph.offsets[target];
  }
}

/// Edge items: item q is the forward edge (r -> v1), v1 > r. It counts the 2-cycle r -> v1 -> r
/// and searches below the prefix (r, v1) (count_edge_items_kernel).
template <int cap, typename offset_t>
__global__ void count_edge_items_kernel(const device_csr<offset_t> graph, const int max_length,
                                        const device_vertex* item_source,
                                        const device_vertex* item_target,
                                        const unsigned long long item_count,
                                        unsigned long long* work_counter,
                                        unsigned long long* histogram) {
  thread_histogram<cap> counts;
  counts.clear();
  device_vertex path[cap];
  while (true) {
    const unsigned long long item = atomicAdd(work_counter, 1ULL);
    if (item >= item_count) {
      break;
    }
    path[0] = __ldg(item_source + item);
    path[1] = __ldg(item_target + item);
    offset_t position = 0;
    offset_t row_end = 0;
    if (find_edge(graph, path[1], path[0], position, row_end)) {
      ++counts.count[2];
    }
    extend_prefix<cap>(graph, path, 2, max_length, counts);
  }
  counts.flush(histogram);
}

/// Two-hop items: the items of edge item q are the pairs (q, j) for j < outdeg(v1), numbered
/// consecutively from hop_offset[q]. Item (q, j) is the path r -> v1 -> v2 with v2 the j-th
/// neighbour of v1; it is skipped unless v2 > r and v2 != v1. The 2-cycle of q is counted by its
/// item j = 0 (count_two_hop_items_kernel).
template <int cap, typename offset_t>
__global__ void count_two_hop_items_kernel(
    const device_csr<offset_t> graph, const int max_length, const device_vertex* item_source,
    const device_vertex* item_target, const unsigned long long* hop_offset,
    const offset_t edge_item_count, const unsigned long long item_count,
    unsigned long long* work_counter, unsigned long long* histogram) {
  thread_histogram<cap> counts;
  counts.clear();
  device_vertex path[cap];
  while (true) {
    const unsigned long long item = atomicAdd(work_counter, 1ULL);
    if (item >= item_count) {
      break;
    }
    // The edge item q is the last one with hop_offset[q] <= item.
    offset_t low = 0;
    offset_t high = edge_item_count;
    while (high - low > 1) {
      const offset_t mid = low + ((high - low) >> 1);
      if (__ldg(hop_offset + mid) <= item) {
        low = mid;
      } else {
        high = mid;
      }
    }
    const offset_t q = low;
    const device_vertex root = __ldg(item_source + q);
    const device_vertex first_hop = __ldg(item_target + q);
    const auto j = static_cast<offset_t>(item - __ldg(hop_offset + q));
    offset_t position = 0;
    offset_t row_end = 0;
    if (j == 0 && find_edge(graph, first_hop, root, position, row_end)) {
      ++counts.count[2];
    }
    const device_vertex second_hop = __ldg(graph.neighbors + __ldg(graph.offsets + first_hop) + j);
    if (second_hop <= root || second_hop == first_hop) {
      continue;
    }
    if (find_edge(graph, second_hop, root, position, row_end)) {
      ++counts.count[3];
    }
    path[0] = root;
    path[1] = first_hop;
    path[2] = second_hop;
    extend_prefix<cap>(graph, path, 3, max_length, counts);
  }
  counts.flush(histogram);
}

// ---- host helpers ----------------------------------------------------------------------------

constexpr unsigned int build_block_size = 256;  // kBuildBlockSize

unsigned int grid_blocks(std::uint64_t work) {
  const std::uint64_t blocks = (work + build_block_size - 1) / build_block_size;
  return static_cast<unsigned int>(
      std::min<std::uint64_t>(std::max<std::uint64_t>(blocks, 1), 1U << 20));
}

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// Exclusive prefix sum of counts[0..n) into offsets[0..n]; counts[n] is 0 (exclusive_scan).
template <typename value_t, typename edge_t>
void exclusive_scan(const resources& res, cycle_count_cuda_workspace<edge_t>& ws,
                    const value_t* counts, value_t* offsets, std::size_t n) {
  std::size_t temp_bytes = 0;
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, offsets, n + 1, native(res)));
  unsigned char* temp = ws.scan_temp.reserve(res, std::max<std::size_t>(temp_bytes, 1));
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(temp, temp_bytes, counts, offsets, n + 1, native(res)));
}

/// One value copied back through pinned memory (read_back; synchronizes the stream).
template <typename value_t, typename edge_t>
value_t read_back(const resources& res, cycle_count_cuda_workspace<edge_t>& ws,
                  const value_t* device_value) {
  static_assert(sizeof(value_t) <= sizeof(unsigned long long), "a scalar read-back");
  unsigned long long* host = ws.host_scalars.data() + 2 * (cycle_count_max_device_length + 1);
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(host, device_value, sizeof(value_t), cudaMemcpyDeviceToHost, native(res)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
  value_t value{};
  std::memcpy(&value, host, sizeof(value_t));
  return value;
}

/// Resident grid of a counting kernel (the occupancy query also loads the kernel, so module
/// loading stays out of the timed region).
template <typename kernel_t>
work_queue_launch grid_for(const kernel_t kernel, std::size_t work_items, int sm_count) {
  int resident = 0;
  DYNG_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, kernel, static_cast<int>(cycle_count_queue_block_size), 0));
  return plan_work_queue_launch(std::max<std::size_t>(work_items, 1), cycle_count_queue_block_size,
                                static_cast<unsigned int>(std::max(resident, 1)),
                                static_cast<unsigned int>(std::max(sm_count, 1)));
}

}  // namespace

template <typename edge_t>
void cuda_workspace_prepare(const resources& res, cycle_count_cuda_workspace<edge_t>& ws) {
  if (ws.sm_count == 0) {
    ws.sm_count = resources_access::device_properties(res).multiprocessor_count;
  }
  constexpr std::size_t scalars = 2 * (cycle_count_max_device_length + 1) + 1;
  if (ws.host_scalars.size() < scalars) {
    ws.host_scalars = buffer<unsigned long long>(
        scalars, res.stream(), resources_access::staging_memory(res), res.device());
  }
  (void)ws.histograms.reserve(res, 2 * (cycle_count_max_device_length + 1));
  (void)ws.work_counters.reserve(res, 2);
}

template <typename edge_t>
void cycle_count_cuda_compute(const resources& res, const cycle_device_graph<edge_t>& graph,
                              std::int64_t length, const cycle_count::options& opt,
                              cycle_count_cuda_workspace<edge_t>& ws,
                              std::vector<std::uint64_t>& counts) {
  using offset_t = device_offset_t<edge_t>;
  const scoped_device guard(res.device());
  const cudaStream_t stream = native(res);
  if (graph.vertex_count == 0 || graph.edge_count == 0) {
    return;  // no cycle
  }
  DYNG_EXPECTS(length >= 2 && length <= cycle_count_max_device_length,
               "cycle_count: the CUDA backend supports max_cycle_length up to ",
               cycle_count_max_device_length);
  const device_csr<offset_t> view{static_cast<std::uint32_t>(graph.vertex_count),
                                  reinterpret_cast<const offset_t*>(graph.offsets),
                                  reinterpret_cast<const device_vertex*>(graph.neighbors)};
  const auto n = static_cast<std::uint64_t>(graph.vertex_count);
  const auto m = static_cast<std::uint64_t>(graph.edge_count);
  const bool naive = opt.scheduler == cycle_count::cuda_scheduler::naive;
  const cycle_count::cuda_work_items items =
      resolve_work_items(opt.work_items, static_cast<std::size_t>(length), n, m);
  const int max_length = static_cast<int>(length);

  dispatch_capacity(static_cast<std::size_t>(length), [&](auto capacity) {
    constexpr int cap = decltype(capacity)::value;
    unsigned long long* histogram = nullptr;
    unsigned long long* work_counter = nullptr;
    work_queue_launch root_launch;
    work_queue_launch edge_launch;
    work_queue_launch hop_launch;
    {
      scoped_stage stage(res, "cycle_count.reset");
      cuda_workspace_prepare(res, ws);
      histogram = ws.histograms.data();
      work_counter = ws.work_counters.data();
      DYNG_CUDA_TRY(cudaMemsetAsync(histogram, 0, sizeof(unsigned long long) * (cap + 1), stream));
      DYNG_CUDA_TRY(cudaMemsetAsync(work_counter, 0, sizeof(unsigned long long), stream));
      if (naive) {
        cudaFuncAttributes attributes{};  // also loads the kernel before timing
        DYNG_CUDA_TRY(cudaFuncGetAttributes(&attributes, count_roots_kernel<cap, offset_t>));
      } else {
        root_launch = grid_for(count_roots_queue_kernel<cap, offset_t>, n, ws.sm_count);
        edge_launch = grid_for(count_edge_items_kernel<cap, offset_t>, m, ws.sm_count);
        hop_launch = grid_for(count_two_hop_items_kernel<cap, offset_t>, m, ws.sm_count);
        cudaFuncAttributes attributes{};
        DYNG_CUDA_TRY(cudaFuncGetAttributes(&attributes, forward_rows_kernel<offset_t>));
        DYNG_CUDA_TRY(cudaFuncGetAttributes(&attributes, fill_edge_items_kernel<offset_t>));
        DYNG_CUDA_TRY(cudaFuncGetAttributes(&attributes, target_degree_kernel<offset_t>));
      }
    }

    // The kernel time covers building the work items as well as counting (kernel_ms).
    scoped_stage stage(res, "cycle_count.count");
    if (naive) {
      const std::uint64_t blocks =
          (n + cycle_count_naive_block_size - 1) / cycle_count_naive_block_size;
      DYNG_EXPECTS(blocks <= std::numeric_limits<unsigned int>::max(),
                   "cycle_count: the vertex count exceeds the naive CUDA grid capacity");
      count_roots_kernel<cap, offset_t>
          <<<static_cast<unsigned int>(blocks), cycle_count_naive_block_size, 0, stream>>>(
              view, max_length, histogram);
      DYNG_CHECK_KERNEL(stream);
    } else if (items == cycle_count::cuda_work_items::roots) {
      count_roots_queue_kernel<cap, offset_t>
          <<<root_launch.grid_blocks, root_launch.block_size, 0, stream>>>(view, max_length,
                                                                           work_counter, histogram);
      DYNG_CHECK_KERNEL(stream);
    } else {
      // Forward edges r -> v1 (v1 > r) are the edge items.
      offset_t* forward_begin = ws.forward_begin.reserve(res, n);
      offset_t* forward_count = ws.forward_count.reserve(res, n + 1);
      offset_t* forward_offset = ws.forward_offset.reserve(res, n + 1);
      DYNG_CUDA_TRY(cudaMemsetAsync(forward_count, 0, sizeof(offset_t) * (n + 1), stream));
      forward_rows_kernel<offset_t>
          <<<grid_blocks(n), build_block_size, 0, stream>>>(view, forward_begin, forward_count);
      DYNG_CHECK_KERNEL(stream);
      exclusive_scan(res, ws, forward_count, forward_offset, n);
      const offset_t edge_items = read_back(res, ws, forward_offset + n);

      device_vertex* item_source = ws.item_source.reserve(res, edge_items);
      device_vertex* item_target = ws.item_target.reserve(res, edge_items);
      if (edge_items > 0) {
        fill_edge_items_kernel<offset_t><<<grid_blocks(n * 32U), build_block_size, 0, stream>>>(
            view, forward_begin, forward_offset, item_source, item_target);
        DYNG_CHECK_KERNEL(stream);
      }

      if (edge_items == 0) {
        // No forward edge: no cycle.
      } else if (items == cycle_count::cuda_work_items::edges) {
        count_edge_items_kernel<cap, offset_t>
            <<<edge_launch.grid_blocks, edge_launch.block_size, 0, stream>>>(
                view, max_length, item_source, item_target, edge_items, work_counter, histogram);
        DYNG_CHECK_KERNEL(stream);
      } else {
        // Two-hop items: one per (edge item, neighbour of its target).
        unsigned long long* hop_count = ws.hop_count.reserve(res, std::size_t{edge_items} + 1);
        unsigned long long* hop_offset = ws.hop_offset.reserve(res, std::size_t{edge_items} + 1);
        DYNG_CUDA_TRY(cudaMemsetAsync(
            hop_count, 0, sizeof(unsigned long long) * (std::size_t{edge_items} + 1), stream));
        target_degree_kernel<offset_t><<<grid_blocks(edge_items), build_block_size, 0, stream>>>(
            view, item_target, edge_items, hop_count);
        DYNG_CHECK_KERNEL(stream);
        exclusive_scan(res, ws, hop_count, hop_offset, edge_items);
        const unsigned long long hop_items = read_back(res, ws, hop_offset + edge_items);
        if (hop_items > 0) {
          count_two_hop_items_kernel<cap, offset_t>
              <<<hop_launch.grid_blocks, hop_launch.block_size, 0, stream>>>(
                  view, max_length, item_source, item_target, hop_offset, edge_items, hop_items,
                  work_counter, histogram);
          DYNG_CHECK_KERNEL(stream);
        }
      }
    }
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    stage.stop();

    // The histogram copy (the original's download region).
    scoped_stage finalize(res, "cycle_count.finalize");
    unsigned long long* host = ws.host_scalars.data();
    DYNG_CUDA_TRY(cudaMemcpyAsync(host, histogram, sizeof(unsigned long long) * (cap + 1),
                                  cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    for (std::int64_t len = 2; len <= length && len < static_cast<std::int64_t>(counts.size());
         ++len) {
      counts[static_cast<std::size_t>(len)] = host[len];
    }
  });
}

#define DYNG_INSTANTIATE_CYCLE_COUNT_STATIC_CUDA(E)                                              \
  template void cuda_workspace_prepare<E>(const resources&, cycle_count_cuda_workspace<E>&);     \
  template void cycle_count_cuda_compute<E>(                                                     \
      const resources&, const cycle_device_graph<E>&, std::int64_t, const cycle_count::options&, \
      cycle_count_cuda_workspace<E>&, std::vector<std::uint64_t>&);
DYNG_INSTANTIATE_CYCLE_COUNT_STATIC_CUDA(std::int32_t)
DYNG_INSTANTIATE_CYCLE_COUNT_STATIC_CUDA(std::int64_t)
#undef DYNG_INSTANTIATE_CYCLE_COUNT_STATIC_CUDA

// Kernel registry (resources::warm_up() loads every kernel; one registration per line: the
// registrar's name is made unique by __LINE__).
DYNG_REGISTER_KERNEL(count_roots_kernel<4, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<4, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<4, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<4, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<8, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<8, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<8, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<8, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<16, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<16, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<16, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<16, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<32, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<32, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<32, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<32, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<64, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<64, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<64, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<64, std::uint32_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<4, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<4, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<4, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<4, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<8, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<8, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<8, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<8, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<16, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<16, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<16, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<16, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<32, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<32, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<32, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<32, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_kernel<64, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_roots_queue_kernel<64, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_edge_items_kernel<64, std::uint64_t>);
DYNG_REGISTER_KERNEL(count_two_hop_items_kernel<64, std::uint64_t>);
DYNG_REGISTER_KERNEL(forward_rows_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(forward_rows_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(fill_edge_items_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(fill_edge_items_kernel<std::uint64_t>);
DYNG_REGISTER_KERNEL(target_degree_kernel<std::uint32_t>);
DYNG_REGISTER_KERNEL(target_degree_kernel<std::uint64_t>);

}  // namespace dyng::detail
