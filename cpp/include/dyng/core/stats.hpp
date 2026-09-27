// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file stats.hpp
 * @brief update_stats: the counters every algorithm's update() reports.
 * @ingroup core
 */
#pragma once

#include <dyng/core/types.hpp>

#include <cstdint>

namespace dyng {

/**
 * @brief Counters common to every update(); each algorithm's `stats` derives from it and appends
 *        its own fields.
 *
 * Each field is documented as *deterministic* (identical across runs and backends, so tests may
 * assert it) or *schedule-dependent* (it may vary with the thread schedule; log it only). Timing
 * is never a statistic: it goes through the profiler.
 *
 * @ingroup core
 */
struct update_stats {
  /// Deterministic: the number of elements whose value changed (for sssp: vertices whose
  /// distance or parent differs after the update). 0 for an update that changes nothing.
  std::int64_t affected = 0;
  /// Schedule-dependent: rounds of the Step 2 loop.
  std::int64_t iterations = 0;
  /// Schedule-dependent: elements expanded by the Step 2 loop (frontier visits).
  std::int64_t frontier_visits = 0;
  /// Deterministic: true if the update recomputed from scratch instead of updating
  /// incrementally (a documented policy such as a budget; never silently).
  bool fallback_used = false;
  /// Deterministic: false if an iteration cap was reached (reported under `on_limit::report`).
  bool converged = true;
  /// Deterministic: the engine that ran (engine::fused for a ported paper engine,
  /// engine::operators for a hook-by-hook composition).
  engine engine_used = engine::automatic;
};

}  // namespace dyng
