// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh
/**
 * @file local_patterns_kernels.cuh
 * @brief EXPERIMENTAL PUBLIC (detail); may change in any minor release. The wedge-parallel
 *        counting kernels of the local-pattern engine (ESCHER-GPU's kernel/wedge.cuh): one thread
 *        per oriented wedge, the frontier filters of the exactly-once rule, the block bins and
 *        their flush; and the launch of an Op's kernel that the driver calls. Needs nvcc.
 *
 * The kernels of triad_count's four built-in patterns are these templates instantiated in libdyng
 * (cpp/src/algorithms/triad_count/engine/counters.cu); an Op of a user instantiates them in the
 * user's translation unit (<dyng/experimental/local_patterns.cuh>). The wedge count and its scan,
 * the frontier and the bins' memory belong to the driver in libdyng (ADR 0058).
 *
 * A work item is one oriented wedge (a, b): a source key a and one entry b of its adjacency row
 * (the line graph's keys are hyperedge ids + 1). Full counts use the out-wedges of every key
 * (b > a). Frontier deltas use every neighbour b of each frontier key f and skip the frontier
 * members smaller than f, so that each triangle (or pair) is counted once, at its smallest
 * frontier member. A wedge is closed into triangles (a, b, k) by intersecting the suffixes of
 * N(a) after b and of N(b) after b, i.e. k > b. One thread per wedge balances the work of hub
 * rows across many threads; rows are read through the key-indexed row views. Each block
 * accumulates into private 64-bit bins in shared memory, flushed once per block as unsigned
 * magnitudes with a carry test (dynG, ADR 0057; the driver applies the phase's sign).
 * @ingroup experimental
 */
#pragma once

#if !defined(__CUDACC__)
#error \
    "<dyng/experimental/detail/local_patterns_kernels.cuh> needs nvcc (a CUDA translation unit, .cu)"
#endif

#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/local_patterns.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>

namespace dyng::detail::local_patterns {

/// The threads of a counting block.
constexpr int block_size = 256;

/// The error word's bit of an invalid bin (device_error::internal of the library).
inline constexpr unsigned long long error_bit_internal = 1ULL << 31U;
/// The error word's bit of a count past 2^64 - 1 (device_error::capacity of the library).
inline constexpr unsigned long long error_bit_capacity = 1ULL;

/// Largest s in [lo, hi] with offsets[s] <= t (offsets is non-decreasing).
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

/// The key of source s: the frontier's sources[s], or s + 1 in a full count.
__device__ __forceinline__ int source_key(const int* sources, int s) {
  return sources != nullptr ? sources[s] : s + 1;
}

/// The first wedge of source a: every neighbour (a frontier phase) or the out-neighbours.
__device__ __forceinline__ int wedge_begin(const line_rows& adj, int a, bool all_neighbours) {
  return all_neighbours ? adj.loc[a] : adj.out[a];
}

/// Warp-cooperative grid-stride loop over the wedges: lanes 0 and 1 locate the first and last
/// source of the warp's 32 wedges, then every lane binary searches its own source within that
/// range. Calls body(a, b, p) for each wedge that passes the frontier filter (p = position of b in
/// N(a)).
template <class body_t>
__device__ __forceinline__ void for_each_wedge(const line_rows& adj, const int* sources,
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

/**
 * @brief The bins handle of an Op's kernels: the block's bins in shared memory and the call's
 *        error word.
 *
 * An invalid bin raises the error word instead of being written: a negative one always (the
 * built-in MotifOp's pattern of no class), one at or above `count` when `bounded` (the Ops of
 * users; the built-in Ops never name one, and the test costs an instruction per triangle). Every
 * bin holds an unsigned magnitude; a count past 2^64 - 1 sets the capacity bit instead of
 * wrapping: overflow() for an Op's own sum, a carry test in add() when `checked` (every Op of a
 * user; the built-ins that add more than 1).
 * @tparam count   The bins.
 * @tparam checked Whether add() tests the carry.
 * @tparam bounded Whether add() tests the upper bound.
 */
template <int count, bool checked, bool bounded = false>
struct block_bins {
  /// The error bit of an invalid bin.
  static constexpr unsigned long long invalid_bin = error_bit_internal;
  /// The error bit of a count past 2^64 - 1.
  static constexpr unsigned long long too_large = error_bit_capacity;
  unsigned long long* shared;  ///< the block's bins
  unsigned long long* error;   ///< the call's error word

  /**
   * @brief Add `value` to bin `bin`.
   * @param[in] bin   The bin.
   * @param[in] value The count.
   */
  __device__ __forceinline__ void add(int bin, unsigned long long value) const {
    if constexpr (bounded) {
      if (bin < 0 || bin >= count) {
        atomicOr(error, invalid_bin);
        return;
      }
    } else {
      if (bin < 0) {
        atomicOr(error, invalid_bin);
        return;
      }
    }
    if constexpr (checked) {
      const unsigned long long old = atomicAdd(&shared[bin], value);
      if (old + value < old) atomicOr(error, too_large);
    } else {
      atomicAdd(&shared[bin], value);
    }
  }

  /// An Op's own sum passed 2^64 - 1.
  __device__ __forceinline__ void overflow() const {
    atomicOr(error, too_large);
  }
};

/// Whether the device handle of op_t tests the carry: every Op of a user; a built-in Op unless
/// it adds 1 per call (its `unit_counts`: a block's bin cannot pass 2^64 - 1 then).
template <class op_t>
constexpr bool checked_bins() noexcept {
  if constexpr (builtin_op<op_t>::value) {
    return !op_t::unit_counts;
  } else {
    return true;
  }
}

/// The bins handle of operation op_t's kernels.
template <class op_t>
using bins_for = block_bins<op_t::num_bins, checked_bins<op_t>(), !builtin_op<op_t>::value>;

/// The block's magnitudes into the call's bins with a carry test (once per block and bin);
/// ESCHER-GPU added sign * bin in two's complement.
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

/// The triangles of one wedge (ESCHER-GPU's TriangleBody).
template <class op_t, class bins_t>
struct triangle_body {
  const line_rows& adj;                      ///< the line graph
  const experimental::hyperedge_rows& rows;  ///< the Op's rows
  const unsigned char* in_frontier;          ///< the frontier's flags
  op_t& op;                                  ///< the thread's Op
  bins_t& bins;                              ///< the block's bins
  /// Wedge (a, b) at position p of N(a).
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

/// The pair of one wedge (ESCHER-GPU's PairBody).
template <class op_t, class bins_t>
struct pair_body {
  const experimental::hyperedge_rows& rows;  ///< the Op's rows
  op_t& op;                                  ///< the thread's Op
  bins_t& bins;                              ///< the block's bins
  /// Wedge (a, b).
  __device__ __forceinline__ void operator()(int a, int b, int) {
    op.pair(rows, a - 1, b - 1, bins);
  }
};

/// The triangles shape: one thread per wedge.
template <class op_t>
__global__ void triangle_kernel(line_rows adj, experimental::hyperedge_rows rows,
                                const int* sources, int num_sources, const long long* offsets,
                                long long num_wedges, bool all_neighbours,
                                const unsigned char* in_frontier, op_t op, unsigned long long* out,
                                unsigned long long* error) {
  __shared__ unsigned long long shared[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) shared[i] = 0;
  __syncthreads();
  op_t local = op;
  bins_for<op_t> bins{shared, error};
  triangle_body<op_t, bins_for<op_t>> body{adj, rows, in_frontier, local, bins};
  for_each_wedge(adj, sources, num_sources, offsets, num_wedges, all_neighbours, in_frontier, body);
  flush_bins(local, bins, out);
}

/// The pairs shape: one thread per wedge.
template <class op_t>
__global__ void pair_kernel(line_rows adj, experimental::hyperedge_rows rows, const int* sources,
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

/// The keys shape: one thread per source key (an Op of a user skips the empty row of a dead id).
template <class op_t>
__global__ void key_kernel(experimental::hyperedge_rows rows, const int* sources, int num_sources,
                           op_t op, unsigned long long* out, unsigned long long* error) {
  __shared__ unsigned long long shared[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) shared[i] = 0;
  __syncthreads();
  op_t local = op;
  bins_for<op_t> bins{shared, error};
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_sources;
       s += gridDim.x * blockDim.x) {
    if constexpr (builtin_op<op_t>::value) {
      local.key(rows, source_key(sources, s) - 1, bins);
    } else {
      const int id = source_key(sources, s) - 1;
      if (rows.size(id) != 0) local.key(rows, id, bins);
    }
  }
  flush_bins(local, bins, out);
}

/**
 * @brief Launch op_t's counting kernel for a pass the driver prepared (the wedges counted and
 *        scanned for triangles and pairs); the launch only, on the pass's stream.
 * @tparam op_t The Op.
 * @param[in] op_bytes The Op's value.
 * @param[in] p        The pass.
 * @return cudaGetLastError() after the launch (0: launched).
 */
template <class op_t>
int launch(const void* op_bytes, const device_pass& p) {
  op_t op;
  std::memcpy(static_cast<void*>(&op), op_bytes, sizeof(op_t));
  const auto stream = static_cast<cudaStream_t>(p.stream);
  if constexpr (op_t::shape == experimental::shape::keys) {
    const int grid = (p.num_sources + block_size - 1) / block_size;
    key_kernel<op_t>
        <<<grid, block_size, 0, stream>>>(p.rows, p.sources, p.num_sources, op, p.out, p.error);
  } else {
    // One thread per wedge: with fewer threads looping over several wedges, threads that draw
    // heavy wedges set the tail of the kernel.
    const int grid = static_cast<int>(
        std::min<long long>((p.num_wedges + block_size - 1) / block_size, INT_MAX));
    if constexpr (op_t::shape == experimental::shape::triangles) {
      triangle_kernel<op_t><<<grid, block_size, 0, stream>>>(
          p.adj, p.rows, p.sources, p.num_sources, p.offsets, p.num_wedges, p.all_neighbours,
          p.in_frontier, op, p.out, p.error);
    } else {
      pair_kernel<op_t><<<grid, block_size, 0, stream>>>(p.adj, p.rows, p.sources, p.num_sources,
                                                         p.offsets, p.num_wedges, p.all_neighbours,
                                                         p.in_frontier, op, p.out, p.error);
    }
  }
  return static_cast<int>(cudaGetLastError());
}

/**
 * @brief The table of op_t that the driver runs: its properties, its kernel's launch and its
 *        host work.
 * @tparam op_t The Op (check_op<op_t>() holds).
 * @return The table.
 */
template <class op_t>
kernels make_kernels() noexcept {
  kernels k;
  k.num_bins = op_t::num_bins;
  k.pattern_shape = op_t::shape;
  k.builtin = builtin_op<op_t>::value;
  k.merge_limit = true;
  k.op_size = sizeof(op_t);
  k.launch = &launch<op_t>;
  k.source = &host_source<op_t>;
  return k;
}

}  // namespace dyng::detail::local_patterns
