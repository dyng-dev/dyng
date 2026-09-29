// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budgets.hpp
 * @brief Budgets of the algorithm phase of an update (invariant I9, PLAN Section 4.5.5; checked by
 *        conformance check C8 in builds with DYNG_DEBUG_BUDGETS=ON).
 *
 * The algorithm phase is everything the update enactor runs after the commit: resume,
 * identify_affected, seed, loop or count(+), finalize, or enact_fused. Once the result and the
 * handle's workspaces have been reserved (by compute(), clone() or an earlier update of the same
 * size), that phase allocates nothing and synchronizes with the host a bounded number of times.
 * A problem states its bound with the hook `budget algorithm_budget(context&)`; the default is
 * budget::unchecked(). The enactor measures the phase with a budget_scope (the process-wide
 * counters of core/budget_counters.hpp) and, in a budgets build, throws internal_error if the
 * phase exceeded the problem's budget. The commit is measured separately (container growth is
 * reported, not failed: PLAN 4.5.5).
 *
 * The measurement of the last update enactor run on the calling thread stays readable through
 * last_budget_report() (for the tests of C8). Counting is process-wide, so run budget checks
 * without concurrent library calls on other threads, and without profiler_options::sync_stages
 * (whose synchronizations count as host synchronizations).
 *
 * PLAN Section 4.2 names this header budget.hpp; the task that extracted the framework named it
 * budgets.hpp (docs/developer/framework.md).
 */
#pragma once

#include "core/budget_counters.hpp"

#include <dyng/core/error.hpp>

#include <cstdint>
#include <string_view>

namespace dyng::detail::framework {

/**
 * @brief The allowance of one algorithm phase (after the commit), once reserved.
 */
struct budget {
  /// No bound on a quantity.
  static constexpr std::int64_t unlimited = -1;

  std::int64_t allocations = unlimited;  ///< allocations through the library's memory resources
  std::int64_t host_syncs = unlimited;   ///< host synchronizations with a stream

  /**
   * @brief The default budget: nothing is checked.
   * @return A budget with both quantities unlimited.
   */
  [[nodiscard]] static constexpr budget unchecked() noexcept {
    return budget{};
  }

  /**
   * @brief A budget that allows no allocation and at most `syncs` host synchronizations.
   * @param[in] syncs Host synchronizations allowed (e.g. 1 for a fused CUDA engine that reads its
   *                  control block back once; 0 for a host backend).
   * @return The budget.
   */
  [[nodiscard]] static constexpr budget steady_state(std::int64_t syncs) noexcept {
    budget b;
    b.allocations = 0;
    b.host_syncs = syncs;
    return b;
  }

  /**
   * @brief Whether the counts of a phase are within the budget.
   * @param[in] used The counts of the phase.
   * @return true if every bounded quantity is at most its bound.
   */
  [[nodiscard]] constexpr bool allows(const budget_counters& used) const noexcept {
    return (allocations == unlimited || used.allocations <= allocations) &&
           (host_syncs == unlimited || used.host_syncs <= host_syncs);
  }
};

/**
 * @brief The measurement of one algorithm phase.
 */
struct budget_report {
  bool measured = false;  ///< false: no phase ran, or counting is off (not a budgets build)
  budget limit;           ///< the problem's budget
  budget_counters used;   ///< what the phase did
};

/**
 * @brief Counts what happens between its construction and used() (both on one thread).
 */
class budget_scope {
 public:
  /// @brief Start counting now.
  budget_scope() noexcept : start_(budget_snapshot()) {}

  /**
   * @brief The counts since construction.
   * @return The differences of the process-wide counters (all zero if counting is off).
   */
  [[nodiscard]] budget_counters used() const noexcept {
    return budget_snapshot().since(start_);
  }

 private:
  budget_counters start_;
};

/**
 * @brief The report of the last algorithm phase measured on the calling thread.
 * @return The report (measured == false before the first).
 */
[[nodiscard]] budget_report last_budget_report() noexcept;

/**
 * @brief Store the report of an algorithm phase and, in a budgets build, check it.
 * @param[in] algorithm The algorithm's name (for the message).
 * @param[in] limit     The problem's budget.
 * @param[in] used      The counts of the phase.
 * @throws internal_error if budgets_enabled() and the counts exceed the budget.
 */
void check_budget(std::string_view algorithm, const budget& limit, const budget_counters& used);

}  // namespace dyng::detail::framework
