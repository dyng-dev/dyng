// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh
/**
 * @file wedge.cuh
 * @brief The driver's half of the wedge engine on the device (ESCHER-GPU's kernel/wedge.cuh,
 *        countWedgeShape and countKeys): the wedge count of the sources, its scan and the one read
 *        of its total, then the Op's counting kernel launched through the Op's table.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp). Since M9b
 * (ADR 0058) the per-Op kernels (triangle_kernel, pair_kernel, key_kernel), their bins handle,
 * the warp-cooperative wedge loop and the frontier filters are the installed extension header
 * <dyng/experimental/detail/local_patterns_kernels.cuh>, moved there unchanged: triad_count's
 * four built-in operations instantiate them in counters.cu, the Ops of users in their own
 * translation units, and both run through count_shape() below with the same scratch.
 */
#pragma once

// Wedge-parallel counting (ESCHER-GPU's countWedgeShape).
//
// A work item is one oriented wedge (a, b): a source key a and one entry b
// of its adjacency row. Full counts use the out-wedges of every key (b > a).
// Frontier deltas use every neighbour b of each frontier key f and skip the
// frontier members smaller than f, so that each triangle (or pair) is
// counted once, at its smallest frontier member (see include/frontier.hpp).
// The wedges of the sources are counted and scanned first (one thread per
// source), then the Op's kernel runs one thread per wedge.
//
// An op_t provides (the Op concept of experimental::count_local_patterns):
//   static constexpr int num_bins; static constexpr experimental::shape shape;
//   triangles: void begin_wedge(rows, int a, int b); void triangle(rows, int k, bins);
//   pairs:     void pair(rows, int a, int b, bins);
//   keys:      void key(rows, int a, bins);
//   void flush(bins)  -- adds register-held partial sums to the block bins.

#include "algorithms/triad_count/engine/ownership.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/engine/sets.hpp"
#include "core/budget_counters.hpp"
#include "hypergraph/row_views.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/detail/local_patterns_kernels.cuh>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>

namespace dyng::detail::triad {

namespace wedge {

using local_patterns::block_size;

/// The bins handle of the kernels (local_patterns_kernels.cuh), by the names the triad tests use.
template <int count, bool checked>
using block_bins = local_patterns::block_bins<count, checked>;
using local_patterns::flush_bins;

// dynG (M9b): the library's device error bits are the ones the installed kernels raise.
static_assert(local_patterns::error_bit_internal ==
                  static_cast<unsigned long long>(bits_of(device_error::internal)),
              "the engine's invalid-bin bit must be device_error::internal");
static_assert(local_patterns::error_bit_capacity ==
                  static_cast<unsigned long long>(bits_of(device_error::capacity)),
              "the engine's capacity bit must be device_error::capacity");

/// The line graph's rows of the kernels (the internal row view's layout).
inline local_patterns::line_rows line_rows_of(const escher::row_view& v) noexcept {
  return local_patterns::line_rows{v.flat, v.loc, v.len, v.out, v.max_key};
}

// Number of wedges of each source (all neighbours, or out-neighbours only).
static __global__ void count_wedges_kernel(local_patterns::line_rows adj, const int* sources,
                                           int num_sources, bool all_neighbours,
                                           long long* counts) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_sources) return;
  int a = local_patterns::source_key(sources, s);
  counts[s] = adj.loc[a] + adj.len[a] - local_patterns::wedge_begin(adj, a, all_neighbours);
}

// dynG: the scratch of count_shape from the caller's workspace (ESCHER-GPU allocated the
// offsets with cudaMalloc and Thrust's scan storage per call): the wedge offsets of the sources
// (num_sources + 1 entries) and CUB's temporary storage of their scan; the stream of the call's
// resources (ESCHER-GPU ran on the legacy default stream) and a pinned slot for the one read of
// the wedge count.
struct scratch {
  long long* offsets = nullptr;
  void* scan_temp = nullptr;
  std::size_t scan_temp_bytes = 0;
  cudaStream_t stream = nullptr;
  long long* host_wedges = nullptr;
};

// dynG: the bytes of CUB's temporary storage for the scan of n wedge counts (the workspace
// reserves the largest scan of a call).
inline std::size_t scan_temp_bytes(int n) {
  std::size_t bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<long long*>(nullptr),
                                              static_cast<long long*>(nullptr), n));
  return bytes;
}

/// Launch the Op's kernel through its table and check the launch (DYNG_CHECK_KERNEL).
inline void launch_op(const local_patterns::kernels& k, const void* op,
                      const local_patterns::device_pass& pass, cudaStream_t stream) {
  if (k.launch == nullptr) {
    DYNG_FAIL("local patterns: the Op has no CUDA kernel");
  }
  DYNG_CUDA_TRY(static_cast<cudaError_t>(k.launch(op, pass)));
  DYNG_CHECK_KERNEL(stream);
}

// Counts over `num_sources` sources (ESCHER-GPU's countWedgeShape and countKeys, the per-Op kernel
// launched through the Op's table, dynG M9b): for the shapes triangles and pairs over the wedges of
// the keys in d_sources (frontier mode: all neighbours, frontier filter d_in_frontier) or, when
// d_sources is null, of the keys 1..num_sources (full count: out-neighbours only); for keys over
// the keys themselves. Adds the Op's bins to d_out[0..num_bins) (dynG, ADR 0057: unsigned
// magnitudes; the phase's sign is applied on the host).
inline launch count_shape(const local_patterns::kernels& k, const void* op,
                          const escher::row_view& adj, const hyperedge_rows& rows,
                          const int* d_sources, int num_sources, const unsigned char* d_in_frontier,
                          unsigned long long* d_out, unsigned long long* d_error,
                          const scratch& s) {
  if (num_sources <= 0) return launch::no_sources;
  local_patterns::device_pass pass;
  pass.rows = rows;
  pass.sources = d_sources;
  pass.num_sources = num_sources;
  pass.out = d_out;
  pass.error = d_error;
  pass.stream = s.stream;
  if (k.pattern_shape == experimental::shape::keys) {
    launch_op(k, op, pass, s.stream);
    return launch::kernel;
  }
  const bool all_neighbours = d_sources != nullptr;
  long long* d_offsets = s.offsets;  // dynG: the workspace's
  DYNG_CUDA_TRY(cudaMemsetAsync(d_offsets + num_sources, 0, sizeof(long long), s.stream));
  count_wedges_kernel<<<(num_sources + block_size - 1) / block_size, block_size, 0, s.stream>>>(
      line_rows_of(adj), d_sources, num_sources, all_neighbours, d_offsets);
  DYNG_CHECK_KERNEL(s.stream);
  // dynG: CUB's scan on the workspace's storage (Thrust's exclusive_scan allocated it per call).
  std::size_t temp_bytes = s.scan_temp_bytes;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(s.scan_temp, temp_bytes, d_offsets, d_offsets,
                                              num_sources + 1, s.stream));
  // dynG: ESCHER-GPU's read point, on the stream: the wedge count, the one host synchronization
  // of a call (counted, invariant I9).
  DYNG_CUDA_TRY(cudaMemcpyAsync(s.host_wedges, d_offsets + num_sources, sizeof(long long),
                                cudaMemcpyDeviceToHost, s.stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(s.stream));
  note_host_sync();
  const long long num_wedges = *s.host_wedges;
  if (num_wedges <= 0) return launch::no_wedges;
  pass.adj = line_rows_of(adj);
  pass.offsets = d_offsets;
  pass.num_wedges = num_wedges;
  pass.all_neighbours = all_neighbours;
  pass.in_frontier = d_in_frontier;
  launch_op(k, op, pass, s.stream);
  return launch::kernel;
}

}  // namespace wedge

}  // namespace dyng::detail::triad
