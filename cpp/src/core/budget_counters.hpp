// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budget_counters.hpp
 * @brief The allocation and host-synchronization counters behind invariant I9 (PLAN Section
 *        4.5.5): counted only in builds with DYNG_DEBUG_BUDGETS=ON.
 *
 * The library's own memory resources (host, pinned host, CUDA stream-ordered pool) call
 * note_allocation() for every allocation, and the library's stream synchronization
 * (detail::cuda_synchronize(), which resources::synchronize() uses) calls note_host_sync(). The
 * counters are process-wide atomics: a reader takes a snapshot before and after a phase and
 * subtracts (framework/budgets.hpp does that for the algorithm phase of an update). The functions
 * are always called; they count only in a build with DYNG_DEBUG_BUDGETS=ON (budget_counters.cpp is
 * then compiled with the define) and are empty calls otherwise.
 *
 * What is not counted: allocations of a memory resource the user installed
 * (resources::set_memory_resource), std::vector growth of the host engines, and synchronizations
 * an engine makes with the CUDA runtime directly. A test binary can count host allocations too by
 * calling note_allocation() from a replacement of the global operator new (the conformance kit's
 * check C8 does that); engines that synchronize directly call note_host_sync() next to the call.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

/**
 * @brief A snapshot of the process-wide counters (differences of two snapshots are the counts of
 *        a phase).
 */
struct budget_counters {
  std::int64_t allocations = 0;      ///< allocations through the library's memory resources
  std::int64_t allocated_bytes = 0;  ///< bytes of those allocations
  std::int64_t host_syncs = 0;       ///< host synchronizations with a stream

  /**
   * @brief The counts between an earlier snapshot and this one.
   * @param[in] earlier The snapshot taken first.
   * @return This minus `earlier`, field by field.
   */
  [[nodiscard]] budget_counters since(const budget_counters& earlier) const noexcept {
    budget_counters d;
    d.allocations = allocations - earlier.allocations;
    d.allocated_bytes = allocated_bytes - earlier.allocated_bytes;
    d.host_syncs = host_syncs - earlier.host_syncs;
    return d;
  }
};

/**
 * @brief Whether libdyng was built with DYNG_DEBUG_BUDGETS=ON (the counters count).
 * @return true in a budgets build.
 */
[[nodiscard]] bool budgets_enabled() noexcept;

/**
 * @brief The current values of the counters.
 * @return A snapshot (all zero when budgets_enabled() is false).
 */
[[nodiscard]] budget_counters budget_snapshot() noexcept;

/**
 * @brief Count one allocation (a no-op unless budgets_enabled()).
 * @param[in] bytes Its size.
 */
void note_allocation(std::size_t bytes) noexcept;

/**
 * @brief Count one host synchronization with a stream (a no-op unless budgets_enabled()).
 */
void note_host_sync() noexcept;

}  // namespace dyng::detail
