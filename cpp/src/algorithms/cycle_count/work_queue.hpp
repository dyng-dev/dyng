// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/cuda/cuda_work_queue.hpp
// (CudaWorkItems, WorkQueueLaunch, WorkQueueTuning), src/cuda/cuda_work_queue.cpp
// (plan_work_queue_launch), src/cuda/cuda_static_kernels.cu (resolve_work_items,
// kDenseAverageDegree, effective_length) and include/cycle_enum/cuda/cuda_dfs.cuh
// (kMaxDeviceCycleLength)
/**
 * @file work_queue.hpp
 * @brief Host-side planning of the CUDA cycle counters: the longest device cycle, the launch of a
 *        persistent work-queue kernel and the automatic choice of work items (host-only, so it is
 *        unit tested without a device).
 */
#pragma once

#include <dyng/core/error.hpp>
#include <dyng/cycle_count.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::detail {

/// Largest cycle length the device counters support (kMaxDeviceCycleLength): the path and the
/// cursors of a search live in thread-local arrays of a compile-time capacity.
inline constexpr int cycle_count_max_device_length = 64;

/**
 * @brief A persistent-kernel launch configuration (WorkQueueLaunch).
 */
struct work_queue_launch {
  unsigned int block_size = 0;   ///< threads per block
  unsigned int grid_blocks = 0;  ///< resident blocks to launch
};

/// Threads per block of the persistent kernels (WorkQueueTuning's default; the original reads
/// CYCLE_ENUM_CUDA_BLOCK_SIZE, dynG has no environment tuning).
inline constexpr unsigned int cycle_count_queue_block_size = 128;

/// Threads per block of the naive one-thread-per-root kernel (kNaiveBlockSize).
inline constexpr unsigned int cycle_count_naive_block_size = 128;

/**
 * @brief Plan a persistent-kernel launch (plan_work_queue_launch): `blocks_per_sm * sm_count`
 *        resident blocks, never more blocks than work items; an empty grid without work.
 * @param[in] work_item_count Number of independent work items.
 * @param[in] block_size      Threads per block (> 0).
 * @param[in] blocks_per_sm   Resident blocks per multiprocessor (> 0).
 * @param[in] sm_count        Multiprocessors (> 0).
 * @return The launch.
 * @throws invalid_argument_error if a configuration value is zero.
 */
inline work_queue_launch plan_work_queue_launch(std::size_t work_item_count,
                                                unsigned int block_size, unsigned int blocks_per_sm,
                                                unsigned int sm_count) {
  DYNG_EXPECTS(block_size != 0 && blocks_per_sm != 0 && sm_count != 0,
               "work-queue launch configuration values must be positive");
  if (work_item_count == 0) {
    return work_queue_launch{block_size, 0};
  }
  const std::size_t resident =
      static_cast<std::size_t>(blocks_per_sm) * static_cast<std::size_t>(sm_count);
  const std::size_t grid = std::min(resident, work_item_count);
  return work_queue_launch{block_size, static_cast<unsigned int>(grid)};
}

/// k = 4 uses two-hop items from this many edges per vertex (kDenseAverageDegree).
inline constexpr std::uint64_t cycle_count_dense_average_degree = 16;

/**
 * @brief The work items a static count distributes (resolve_work_items): the requested kind, or
 *        for cuda_work_items::automatic edge items up to k = 3 and at k = 4 on sparse graphs (fewer
 *        than 16 edges per vertex), two-hop items otherwise.
 *
 * Measured by the original on DD, COLLAB, twitch_egos and github_stargazers: at k = 3 a two-hop
 * item is one or two binary searches and its claim dominates; from k = 5 single edge prefixes of a
 * hub bound the kernel (two-hop items 2.2-2.7x faster); at k = 4 two-hop items win only on dense
 * graphs. Every cycle has exactly one prefix of each kind, so the choice never changes the counts.
 * @param[in] requested        The requested items.
 * @param[in] max_cycle_length The effective length bound.
 * @param[in] vertex_count     n.
 * @param[in] edge_count       m.
 * @return roots, edges or two_hop.
 */
inline cycle_count::cuda_work_items resolve_work_items(cycle_count::cuda_work_items requested,
                                                       std::size_t max_cycle_length,
                                                       std::uint64_t vertex_count,
                                                       std::uint64_t edge_count) {
  using items = cycle_count::cuda_work_items;
  if (requested != items::automatic) {
    if (requested == items::two_hop && max_cycle_length < 3) {
      return items::edges;
    }
    return requested;
  }
  if (max_cycle_length <= 3) {
    return items::edges;
  }
  if (max_cycle_length == 4) {
    const bool dense = edge_count >= cycle_count_dense_average_degree * vertex_count;
    return dense ? items::two_hop : items::edges;
  }
  return items::two_hop;
}

/**
 * @brief The effective length bound of a device count (effective_length): a simple cycle has at
 *        most n vertices, so a larger bound is equivalent to n (and lets tiny graphs use any bound).
 * @param[in] max_length   options::max_length (-1: no bound).
 * @param[in] vertex_count n.
 * @return max(min(k, n), 2).
 */
inline std::int64_t cycle_count_effective_length(std::int64_t max_length,
                                                 std::int64_t vertex_count) {
  const std::int64_t bound =
      max_length < 0 ? vertex_count : std::min<std::int64_t>(max_length, vertex_count);
  return std::max<std::int64_t>(bound, 2);
}

}  // namespace dyng::detail
