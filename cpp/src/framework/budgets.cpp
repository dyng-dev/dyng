// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budgets.cpp
 * @brief The report and the check of an algorithm phase's budget (invariant I9).
 */
#include "framework/budgets.hpp"

#include <dyng/core/error.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace dyng::detail::framework {

namespace {

thread_local budget_report last_report;

/// "unlimited" or the number, for the message.
std::string bound_text(std::int64_t bound) {
  return bound == budget::unlimited ? std::string("unlimited") : std::to_string(bound);
}

}  // namespace

budget_report last_budget_report() noexcept {
  return last_report;
}

void check_budget(std::string_view algorithm, const budget& limit, const budget_counters& used) {
  const bool counting = budgets_enabled();
  last_report.measured = counting;
  last_report.limit = limit;
  last_report.used = used;
  if (counting && !limit.allows(used)) {
    DYNG_FAIL("framework: the algorithm phase of ", algorithm, "::update made ", used.allocations,
              " allocation(s) (", used.allocated_bytes, " bytes) and ", used.host_syncs,
              " host synchronization(s); its budget allows ", bound_text(limit.allocations),
              " allocation(s) and ", bound_text(limit.host_syncs),
              " host synchronization(s) once reserved (invariant I9, DYNG_DEBUG_BUDGETS)");
  }
}

}  // namespace dyng::detail::framework
