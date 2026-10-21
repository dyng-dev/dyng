// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh
/**
 * @file wedge.cuh
 * @brief The wedge-parallel counting kernels: one thread per oriented wedge, the frontier filters
 *        of the exactly-once rule, the block bins and their flush, and the set primitives on
 *        sorted rows (ESCHER-GPU's kernel/wedge.cuh).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

// Wedge-parallel counting kernels.
//
// A work item is one oriented wedge (a, b): a source key a and one entry b
// of its adjacency row. Full counts use the out-wedges of every key (b > a).
// Frontier deltas use every neighbour b of each frontier key f and skip the
// frontier members smaller than f, so that each triangle (or pair) is
// counted once, at its smallest frontier member (see include/frontier.hpp).
// A wedge is closed into triangles (a, b, k) by intersecting the suffixes of
// N(a) after b and of N(b) after b, i.e. k > b.
//
// Mapping one thread per wedge instead of one thread per hyperedge balances
// the work of hub rows across many threads; rows are read through the
// key-indexed escher::row_view (no BST walks in the inner loops). Each block
// accumulates into private 64-bit bins that are flushed once (dynG, ADR 0057: as unsigned
// magnitudes with a carry test; ESCHER-GPU flushed them with a sign, +1 add, -1 subtract, in
// two's complement).
//
// An op_t provides:
//   static constexpr int num_bins;
//   triangles: void begin_wedge(rows, int a, int b); void triangle(rows, int k, bins);
//   pairs:     void pair(rows, int a, int b, bins);
//   keys:      void key(rows, int a, bins);
//   void flush(bins)  -- adds register-held partial sums to the block bins.
//
// dynG (the Op form, c6 of the port): the operations are host-and-device structs; they see
// 0-based hyperedge ids, a rows handle (hyperedge_rows, sets.hpp: the members by id) and a bins
// handle (block_bins: add(bin, value)); the keys (id + 1) of the line graph, its rows and the
// frontier filters (ownership.hpp) stay inside the engine.

#include "algorithms/triad_count/engine/ownership.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/engine/sets.hpp"
#include "core/budget_counters.hpp"
#include "hypergraph/row_views.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>

namespace dyng::detail::triad {

namespace wedge {

constexpr int block_size = 256;

// Largest s in [lo, hi] with offsets[s] <= t (offsets is non-decreasing).
__device__ __forceinline__ int find_source(const long long* offsets, int lo, int hi, long long t) {
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ int source_key(const int* sources, int s) {
  return sources != nullptr ? sources[s] : s + 1;
}

__device__ __forceinline__ int wedge_begin(const escher::row_view& adj, int a,
                                           bool all_neighbours) {
  return all_neighbours ? adj.loc[a] : adj.out[a];
}

// Number of wedges of each source (all neighbours, or out-neighbours only).
static __global__ void count_wedges_kernel(escher::row_view adj, const int* sources,
                                           int num_sources, bool all_neighbours,
                                           long long* counts) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_sources) return;
  int a = source_key(sources, s);
  counts[s] = adj.loc[a] + adj.len[a] - wedge_begin(adj, a, all_neighbours);
}

// Warp-cooperative grid-stride loop over the wedges: lanes 0 and 1 locate
// the first and last source of the warp's 32 wedges, then every lane binary
// searches its own source within that range. Calls body(a, b, p) for each
// wedge that passes the frontier filter (p = position of b in N(a)).
template <class body_t>
__device__ __forceinline__ void for_each_wedge(const escher::row_view& adj, const int* sources,
                                               int num_sources, const long long* offsets,
                                               long long num_wedges, bool all_neighbours,
                                               const unsigned char* in_frontier, body_t& body) {
  const int lane = threadIdx.x & 31;
  const long long warp = (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) >> 5;
  const long long num_warps = (gridDim.x * static_cast<long long>(blockDim.x)) >> 5;
  for (long long base = warp * 32; base < num_wedges; base += num_warps * 32) {
    int lo = 0, hi = 0;
    if (lane == 0) lo = find_source(offsets, 0, num_sources - 1, base);
    if (lane == 1) hi = find_source(offsets, 0, num_sources - 1, min(base + 31, num_wedges - 1));
    lo = __shfl_sync(0xffffffffu, lo, 0);
    hi = __shfl_sync(0xffffffffu, hi, 1);
    long long t = base + lane;
    if (t >= num_wedges) continue;
    int s = find_source(offsets, lo, hi, t);
    int a = source_key(sources, s);
    int p = wedge_begin(adj, a, all_neighbours) + static_cast<int>(t - offsets[s]);
    int b = adj.flat[p];
    if (min_member_skips_wedge(in_frontier, b, a)) continue;
    body(a, b, p);
  }
}

// dynG (the Op form): the bins handle of an operation, the block's bins in shared memory. A
// negative bin (MotifOp's pattern of no class) raises the call's device error word instead of being
// written (ESCHER-GPU's MotifOp added such a triangle to bins[-1]); the engine's operations never
// name a bin at or above their num_bins (their classes and bin 0), so that bound is not tested on
// the hot path (the test costs an instruction per triangle; M9b's operations of users may take a
// handle that tests it).
// dynG (ADR 0057): every bin holds an unsigned magnitude (the phase's sign is applied on the host)
// and a count past 2^64 - 1 sets the capacity bit of the error word (capacity_error) instead of
// wrapping: overflow() for an operation's own sum, a carry test in add() for the operations that
// add more than 1 (`checked`; motif_op adds 1 per triangle, which cannot carry a block's bin: at
// most 2^63 triangles per block, a thread's wedges times fewer than 2^31 closings each).
template <int count, bool checked>
struct block_bins {
  static constexpr unsigned long long invalid_bin =
      static_cast<unsigned long long>(bits_of(device_error::internal));
  static constexpr unsigned long long too_large =
      static_cast<unsigned long long>(bits_of(device_error::capacity));
  unsigned long long* shared;
  unsigned long long* error;
  __device__ __forceinline__ void add(int bin, unsigned long long value) const {
    if (bin < 0) {
      atomicOr(error, invalid_bin);
      return;
    }
    if constexpr (checked) {
      const unsigned long long old = atomicAdd(&shared[bin], value);
      if (old + value < old) atomicOr(error, too_large);
    } else {
      atomicAdd(&shared[bin], value);
    }
  }
  /// An operation's own sum passed 2^64 - 1.
  __device__ __forceinline__ void overflow() const {
    atomicOr(error, too_large);
  }
};

/// The bins handle of operation op_t's kernels.
template <class op_t>
using bins_for = block_bins<op_t::num_bins, !op_t::unit_counts>;

// dynG (ADR 0057): the block's magnitudes into the call's bins with a carry test (once per block
// and bin); ESCHER-GPU added sign * bin in two's complement.
template <class op_t, class bins_t>
__device__ __forceinline__ void flush_bins(op_t& op, bins_t& bins, unsigned long long* out) {
  op.flush(bins);
  __syncthreads();
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) {
    const unsigned long long value = bins.shared[i];
    if (value != 0) {
      const unsigned long long old = atomicAdd(&out[i], value);
      if (old + value < old) atomicOr(bins.error, bins_t::too_large);
    }
  }
}

template <class op_t, class bins_t>
struct triangle_body {
  const escher::row_view& adj;
  const hyperedge_rows& rows;
  const unsigned char* in_frontier;
  op_t& op;
  bins_t& bins;
  __device__ __forceinline__ void operator()(int a, int b, int p) {
    op.begin_wedge(rows, a - 1, b - 1);
    int x = p + 1, end_a = adj.loc[a] + adj.len[a];
    int y = adj.out[b], end_b = adj.loc[b] + adj.len[b];
    while (x < end_a && y < end_b) {
      int ka = adj.flat[x], kb = adj.flat[y];
      if (ka == kb) {
        if (!min_member_skips_closing(in_frontier, ka, a)) op.triangle(rows, ka - 1, bins);
        ++x;
        ++y;
      } else if (ka < kb) {
        ++x;
      } else {
        ++y;
      }
    }
  }
};

template <class op_t, class bins_t>
struct pair_body {
  const hyperedge_rows& rows;
  op_t& op;
  bins_t& bins;
  __device__ __forceinline__ void operator()(int a, int b, int) {
    op.pair(rows, a - 1, b - 1, bins);
  }
};

template <class op_t>
__global__ void triangle_kernel(escher::row_view adj, hyperedge_rows rows, const int* sources,
                                int num_sources, const long long* offsets, long long num_wedges,
                                bool all_neighbours, const unsigned char* in_frontier, op_t op,
                                unsigned long long* out, unsigned long long* error) {
  __shared__ unsigned long long shared[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) shared[i] = 0;
  __syncthreads();
  op_t local = op;
  bins_for<op_t> bins{shared, error};
  triangle_body<op_t, bins_for<op_t>> body{adj, rows, in_frontier, local, bins};
  for_each_wedge(adj, sources, num_sources, offsets, num_wedges, all_neighbours, in_frontier, body);
  flush_bins(local, bins, out);
}

template <class op_t>
__global__ void pair_kernel(escher::row_view adj, hyperedge_rows rows, const int* sources,
                            int num_sources, const long long* offsets, long long num_wedges,
                            bool all_neighbours, const unsigned char* in_frontier, op_t op,
                            unsigned long long* out, unsigned long long* error) {
  __shared__ unsigned long long shared[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) shared[i] = 0;
  __syncthreads();
  op_t local = op;
  bins_for<op_t> bins{shared, error};
  pair_body<op_t, bins_for<op_t>> body{rows, local, bins};
  for_each_wedge(adj, sources, num_sources, offsets, num_wedges, all_neighbours, in_frontier, body);
  flush_bins(local, bins, out);
}

template <class op_t>
__global__ void key_kernel(hyperedge_rows rows, const int* sources, int num_sources, op_t op,
                           unsigned long long* out, unsigned long long* error) {
  __shared__ unsigned long long shared[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) shared[i] = 0;
  __syncthreads();
  op_t local = op;
  bins_for<op_t> bins{shared, error};
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_sources; s += gridDim.x * blockDim.x)
    local.key(rows, source_key(sources, s) - 1, bins);
  flush_bins(local, bins, out);
}

enum class wedge_shape { triangles, pairs };

// dynG: what one count call did (wedge::launch, phases.hpp: the launch counters per mode and
// shape, M9a criterion 9): no sources (the first early out), no wedges (the second), or a
// counting kernel launched.

// dynG: the scratch of count_wedge_shape from the caller's workspace (ESCHER-GPU allocated the
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

// Counts over the wedges of `num_sources` sources: the keys in d_sources
// (frontier mode: all neighbours, frontier filter d_in_frontier) or, when
// d_sources is null, keys 1..num_sources (full count: out-neighbours only).
// Adds the bins of op_t to d_out[0..num_bins) (dynG, ADR 0057: unsigned magnitudes; the phase's
// sign is applied on the host).
template <wedge_shape shape, class op_t>
launch count_wedge_shape(const escher::row_view& adj, const hyperedge_rows& rows,
                         const int* d_sources, int num_sources, const unsigned char* d_in_frontier,
                         const op_t& op, unsigned long long* d_out, unsigned long long* d_error,
                         const scratch& s) {
  if (num_sources <= 0) return launch::no_sources;
  const bool all_neighbours = d_sources != nullptr;
  long long* d_offsets = s.offsets;  // dynG: the workspace's
  DYNG_CUDA_TRY(cudaMemsetAsync(d_offsets + num_sources, 0, sizeof(long long), s.stream));
  count_wedges_kernel<<<(num_sources + block_size - 1) / block_size, block_size, 0, s.stream>>>(
      adj, d_sources, num_sources, all_neighbours, d_offsets);
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
  {
    // One thread per wedge: with fewer threads looping over several wedges,
    // threads that draw heavy wedges set the tail of the kernel.
    int grid =
        static_cast<int>(std::min<long long>((num_wedges + block_size - 1) / block_size, INT_MAX));
    if constexpr (shape == wedge_shape::triangles)
      triangle_kernel<op_t><<<grid, block_size, 0, s.stream>>>(
          adj, rows, d_sources, num_sources, d_offsets, num_wedges, all_neighbours, d_in_frontier,
          op, d_out, d_error);
    else
      pair_kernel<op_t><<<grid, block_size, 0, s.stream>>>(adj, rows, d_sources, num_sources,
                                                           d_offsets, num_wedges, all_neighbours,
                                                           d_in_frontier, op, d_out, d_error);
    DYNG_CHECK_KERNEL(s.stream);
  }
  return launch::kernel;
}

template <class op_t>
launch count_triangles(const escher::row_view& adj, const hyperedge_rows& rows,
                       const int* d_sources, int num_sources, const unsigned char* d_in_frontier,
                       const op_t& op, unsigned long long* d_out, unsigned long long* d_error,
                       const scratch& s) {
  return count_wedge_shape<wedge_shape::triangles>(adj, rows, d_sources, num_sources, d_in_frontier,
                                                   op, d_out, d_error, s);
}

template <class op_t>
launch count_pairs(const escher::row_view& adj, const hyperedge_rows& rows, const int* d_sources,
                   int num_sources, const unsigned char* d_in_frontier, const op_t& op,
                   unsigned long long* d_out, unsigned long long* d_error, const scratch& s) {
  return count_wedge_shape<wedge_shape::pairs>(adj, rows, d_sources, num_sources, d_in_frontier, op,
                                               d_out, d_error, s);
}

// Per-key counts over the keys in d_sources (or keys 1..num_sources).
template <class op_t>
launch count_keys(const hyperedge_rows& rows, const int* d_sources, int num_sources, const op_t& op,
                  unsigned long long* d_out, unsigned long long* d_error, const scratch& s) {
  if (num_sources <= 0) return launch::no_sources;
  int grid = (num_sources + block_size - 1) / block_size;
  key_kernel<op_t>
      <<<grid, block_size, 0, s.stream>>>(rows, d_sources, num_sources, op, d_out, d_error);
  DYNG_CHECK_KERNEL(s.stream);
  return launch::kernel;
}

}  // namespace wedge

}  // namespace dyng::detail::triad
