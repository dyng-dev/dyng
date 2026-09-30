// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budgets.cpp
 * @brief The report and the check of an algorithm phase's budget (invariant I9).
 */
#include "framework/budgets.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace dyng::detail::framework {

namespace {

thread_local budget_report last_report;
thread_local budget_counters last_commit;

/// DYNG_STRICT_BUDGETS set to a value other than "0".
bool strict_from_environment() noexcept {
  const char* value = std::getenv("DYNG_STRICT_BUDGETS");  // NOLINT(concurrency-mt-unsafe)
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

std::atomic<bool>& strict_setting() noexcept {
  static std::atomic<bool> strict{strict_from_environment()};
  return strict;
}

/// "unlimited" or the number, for the message.
std::string bound_text(std::int64_t bound) {
  return bound == budget::unlimited ? std::string("unlimited") : std::to_string(bound);
}

}  // namespace

budget_report last_budget_report() noexcept {
  return last_report;
}

budget_counters last_commit_counts() noexcept {
  return last_commit;
}

void record_commit_counts(const budget_counters& used) noexcept {
  last_commit = used;
}

bool strict_budgets() noexcept {
  return strict_setting().load(std::memory_order_relaxed);
}

void set_strict_budgets(bool on) noexcept {
  strict_setting().store(on, std::memory_order_relaxed);
}

void check_budget(std::string_view algorithm, const budget& limit, const budget_counters& used) {
  const bool counting = budgets_enabled();
  last_report.measured = counting;
  last_report.limit = limit;
  last_report.used = used;
  if (!counting || limit.allows(used)) {
    return;
  }
  const std::string message = concat_message(
      "framework: the algorithm work of ", algorithm, "::update made ", used.own_allocations(),
      " allocation(s) and ", used.own_host_syncs(),
      " host synchronization(s) besides container work and instrumentation (in all ",
      used.allocations, " allocation(s) of ", used.allocated_bytes, " bytes and ", used.host_syncs,
      " synchronization(s))", used.reservations > 0 ? " while reserving" : "",
      "; its budget allows ", bound_text(limit.allocations), " allocation(s) and ",
      bound_text(limit.host_syncs),
      " host synchronization(s) once reserved (invariant I9, DYNG_DEBUG_BUDGETS)");
  if (strict_budgets()) {
    DYNG_FAIL(message);
  }
  log_message(log_level::warn, concat_message("dyng: ", message,
                                              "; the update succeeded (set DYNG_STRICT_BUDGETS=1 "
                                              "to make an excess an error)"));
}

}  // namespace dyng::detail::framework
