// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/cuda/cuda_dfs.cuh
/**
 * @file dfs.cuh
 * @brief Device building blocks of the CUDA cycle counters: the capacity dispatch, the per-thread
 *        histogram and the exact pruned depth-first search below a path prefix.
 *
 * Only CUDA translation units include this header. Straight port of CycleEnumeration-GPU's
 * cuda_dfs.cuh; mechanical changes: names, namespace dyng::detail, the row offsets templated on
 * the graph's offset type (uint32_t for int32_t edge offsets: the original's 32-bit CsrView;
 * util/device_csr.cuh), the capacity overflow of dispatch_capacity an internal_error (the entry points
 * reject a longer bound first, as the original's check_length does).
 *
 * - Rows of the CSR are sorted by neighbour id and hold distinct neighbours, so one lower_bound
 *   both tests an edge and splits a row at a vertex id.
 * - Every search keeps its path and cursors in thread-local arrays sized by a compile-time
 *   capacity, and its per-length counts in a thread-local histogram that is reduced across the
 *   warp and added to global memory once per thread, instead of one shared-memory atomic per
 *   cycle.
 */
#pragma once

#include "algorithms/cycle_count/work_queue.hpp"
#include "util/device_csr.cuh"

#include <dyng/core/error.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dyng::detail {

/**
 * @brief Call `body(std::integral_constant<int, cap>{})` with the smallest supported capacity that
 *        holds cycles of length `max_cycle_length` (dispatch_capacity).
 * @tparam body_t Callable.
 * @param[in] max_cycle_length The effective length bound (>= 2).
 * @param[in] body             The callable.
 * @throws internal_error if `max_cycle_length` exceeds cycle_count_max_device_length (the callers
 *         reject such a bound first with invalid_argument_error, as the original's check_length).
 */
template <typename body_t>
void dispatch_capacity(const std::size_t max_cycle_length, body_t&& body) {
  if (max_cycle_length <= 4) {
    body(std::integral_constant<int, 4>{});
  } else if (max_cycle_length <= 8) {
    body(std::integral_constant<int, 8>{});
  } else if (max_cycle_length <= 16) {
    body(std::integral_constant<int, 16>{});
  } else if (max_cycle_length <= 32) {
    body(std::integral_constant<int, 32>{});
  } else if (max_cycle_length <= static_cast<std::size_t>(cycle_count_max_device_length)) {
    body(std::integral_constant<int, cycle_count_max_device_length>{});
  } else {
    DYNG_FAIL("cycle_count: the CUDA backend supports max_cycle_length up to ",
              cycle_count_max_device_length);
  }
}

/**
 * @brief Per-thread cycle counts for lengths 0..cap (ThreadHistogram).
 * @tparam cap The capacity (longest length).
 */
template <int cap>
struct thread_histogram {
  unsigned long long count[cap + 1];  ///< cycles by length

  /// Zero every length.
  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int length = 0; length <= cap; ++length) {
      count[length] = 0ULL;
    }
  }

  /**
   * @brief Add the counts of the whole warp to `histogram` (one atomic per non-empty length per
   *        warp). Every lane of the warp must call this.
   * @param[in,out] histogram The global histogram (cap + 1 entries).
   */
  __device__ __forceinline__ void flush(unsigned long long* histogram) const {
#pragma unroll
    for (int length = 2; length <= cap; ++length) {
      unsigned long long value = count[length];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
      }
      if ((threadIdx.x & 31U) == 0U && value != 0ULL) {
        atomicAdd(histogram + length, value);
      }
    }
  }
};

/**
 * @brief Exact pruned depth-first search below a path prefix (extend_prefix).
 *
 * `path[0]` is the root and `path[1..prefix-1]` are distinct vertices greater than the root; the
 * caller has already counted the cycle that closes the prefix itself. The search counts, in
 * `histogram`, every simple cycle of length at most `max_length` whose minimum vertex is the root
 * and that begins with the prefix, each exactly once.
 *
 * Because rows are sorted, one lower_bound of the root in the row of each pushed vertex gives both
 * the closing test (the root is present) and the first neighbour greater than the root, so
 * neighbours at or below the root are never read and the last level never scans a row.
 * @tparam cap      Capacity of the path.
 * @tparam offset_t Row offset type.
 * @param[in]     graph      The CSR.
 * @param[in,out] path       The path (prefix given).
 * @param[in]     prefix     Length of the prefix.
 * @param[in]     max_length Longest counted cycle.
 * @param[in,out] histogram  The thread's counts.
 */
template <int cap, typename offset_t>
__device__ __forceinline__ void extend_prefix(const device_csr<offset_t> graph,
                                              device_vertex (&path)[cap], const int prefix,
                                              const int max_length,
                                              thread_histogram<cap>& histogram) {
  if (prefix >= max_length) {
    return;
  }
  const device_vertex root = path[0];
  offset_t cursor[cap];
  offset_t end[cap];

  int depth = prefix;
  {
    offset_t position = 0;
    offset_t row_end = 0;
    const bool closes = find_edge(graph, path[depth - 1], root, position, row_end);
    cursor[depth - 1] = position + (closes ? 1U : 0U);
    end[depth - 1] = row_end;
  }

  while (depth >= prefix) {
    if (cursor[depth - 1] >= end[depth - 1]) {
      --depth;
      continue;
    }
    const device_vertex next = __ldg(graph.neighbors + cursor[depth - 1]);
    ++cursor[depth - 1];

    bool on_path = false;
#pragma unroll
    for (int index = 1; index < cap; ++index) {
      if (index < depth && path[index] == next) {
        on_path = true;
      }
    }
    if (on_path) {
      continue;
    }

    offset_t position = 0;
    offset_t row_end = 0;
    const bool closes = find_edge(graph, next, root, position, row_end);
    if (closes) {
#if defined(DYNG_MUTATION_CUDA_DOUBLE_COUNT_5)
      // Recorded mutation (PLAN Section 8.4, "double counting 5-cycles" in the static work-queue
      // kernel, where CycleEnumeration-GPU recorded it): the gpu suite must fail.
      histogram.count[depth + 1] += depth + 1 == 5 ? 2ULL : 1ULL;
#else
      ++histogram.count[depth + 1];  // cycle root, path[1..depth-1], next
#endif
    }
    if (depth + 1 < max_length) {
      path[depth] = next;
      cursor[depth] = position + (closes ? 1U : 0U);
      end[depth] = row_end;
      ++depth;
    }
  }
}

}  // namespace dyng::detail
