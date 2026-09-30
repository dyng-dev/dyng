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
 * budget::unchecked(). The enactor measures the phase with a budget_scope (the calling thread's
 * counters of core/budget_counters.hpp) and records a budget_report. The commit is measured
 * separately (container growth is reported, not failed: PLAN 4.5.5).
 *
 * An over-budget phase. In a budgets build the enactor compares the report with the problem's
 * budget. By default an excess is logged (log_level::warn) and the update succeeds: the check runs
 * after the commit, so throwing would fail, and poison, a result that is correct. With strict
 * budgets (set_strict_budgets(true), strict_budgets_scope, or the environment variable
 * DYNG_STRICT_BUDGETS=1) an excess throws internal_error instead. The conformance kit (check C8)
 * and the framework's tests arm strict budgets; a strict failure poisons the result like any
 * failure after the commit.
 *
 * "Once reserved": a phase that grew a reusable array on purpose (a new or enlarged workspace, a
 * grown scratch buffer, a result grown for new vertices; note_reservation() in
 * core/budget_counters.hpp) is a reserving run. Its allocations are reported, not failed; its
 * host synchronizations are still checked. A phase that reserved nothing must stay within the
 * whole budget, so the steady state (the same shapes again) allocates nothing. The graph's own
 * materializations inside the phase (its device copy uploaded on first use, core/
 * budget_counters.hpp container_scope) are container work, and the profiler's synchronizations of
 * profiler_options::sync_stages are instrumentation (instrumentation_scope): both are counted and
 * reported, never held against the problem's budget.
 *
 * The measurement of the last update enactor run on the calling thread stays readable through
 * last_budget_report() (for the tests of C8), and the commit of the last run_update() on the
 * calling thread through last_commit_counts() (container growth: reported, never failed).
 * Counting is per calling thread, so updates on other threads are not charged to this one; only
 * the worker threads of OpenMP regions share one tally (core/budget_counters.hpp, "Attribution").
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
   * @return true if every bounded quantity is at most its bound. Container work
   *         (container_scope) does not count, and the allocation bound does not apply to a
   *         reserving phase (used.reservations > 0); see the file comment.
   */
  [[nodiscard]] constexpr bool allows(const budget_counters& used) const noexcept {
    return (allocations == unlimited || used.reservations > 0 ||
            used.own_allocations() <= allocations) &&
           (host_syncs == unlimited || used.own_host_syncs() <= host_syncs);
  }

  /**
   * @brief Whether both quantities are bounded.
   * @return true unless a quantity is unlimited.
   */
  [[nodiscard]] constexpr bool bounded() const noexcept {
    return allocations != unlimited && host_syncs != unlimited;
  }
};

/**
 * @brief The measurement of one algorithm phase.
 */
struct budget_report {
  bool measured = false;  ///< false: no phase ran, or counting is off (not a budgets build)
  budget limit;           ///< the problem's budget
  budget_counters used;   ///< what the phase did

  /**
   * @brief Whether the phase reserved (grew a reusable array on purpose).
   * @return used.reservations > 0.
   */
  [[nodiscard]] constexpr bool reserving() const noexcept {
    return used.reservations > 0;
  }
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
   * @return The differences of the calling thread's counters (all zero if counting is off).
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
 * @brief The counts of the commit of the last run_update() on the calling thread (graph::apply
 *        and the preparation of G_{t+1}): container growth, reported and never failed (PLAN
 *        4.5.5, I9).
 * @return The counts (all zero before the first update, or when counting is off).
 */
[[nodiscard]] budget_counters last_commit_counts() noexcept;

/**
 * @brief Record the counts of a commit (run_update()).
 * @param[in] used The counts of the commit.
 */
void record_commit_counts(const budget_counters& used) noexcept;

/**
 * @brief Store the report of an algorithm phase and, in a budgets build, check it (see the file
 *        comment, "An over-budget phase").
 * @param[in] algorithm The algorithm's name (for the message).
 * @param[in] limit     The problem's budget.
 * @param[in] used      The counts of the phase.
 * @throws internal_error if budgets_enabled(), strict_budgets() and the counts exceed the budget
 *         (otherwise an excess is logged at log_level::warn).
 */
void check_budget(std::string_view algorithm, const budget& limit, const budget_counters& used);

/**
 * @brief Whether an over-budget phase throws internal_error (strict) or is logged (the default).
 * @return The current setting (initially true only if the environment variable
 *         DYNG_STRICT_BUDGETS is set to a value other than 0).
 */
[[nodiscard]] bool strict_budgets() noexcept;

/**
 * @brief Make an over-budget phase throw (true) or log (false); process-wide. For tests.
 * @param[in] on The new setting.
 */
void set_strict_budgets(bool on) noexcept;

/**
 * @brief Arms strict budgets for its lifetime and restores the previous setting after (tests).
 */
class strict_budgets_scope {
 public:
  /// @brief Arm strict budgets.
  strict_budgets_scope() noexcept : previous_(strict_budgets()) {
    set_strict_budgets(true);
  }
  strict_budgets_scope(const strict_budgets_scope&) = delete;             ///< not copyable
  strict_budgets_scope& operator=(const strict_budgets_scope&) = delete;  ///< not copyable
  strict_budgets_scope(strict_budgets_scope&&) = delete;                  ///< not movable
  strict_budgets_scope& operator=(strict_budgets_scope&&) = delete;       ///< not movable
  /// @brief Restore the previous setting.
  ~strict_budgets_scope() {
    set_strict_budgets(previous_);
  }

 private:
  bool previous_;
};

}  // namespace dyng::detail::framework
