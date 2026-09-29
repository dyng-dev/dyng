// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budget_counters.cpp
 * @brief The process-wide counters of invariant I9 (counting only with DYNG_DEBUG_BUDGETS).
 */
#include "core/budget_counters.hpp"

#include <atomic>

namespace dyng::detail {

#if defined(DYNG_DEBUG_BUDGETS)
namespace {

std::atomic<std::int64_t> allocations{0};
std::atomic<std::int64_t> allocated_bytes{0};
std::atomic<std::int64_t> host_syncs{0};

}  // namespace

bool budgets_enabled() noexcept {
  return true;
}

budget_counters budget_snapshot() noexcept {
  budget_counters c;
  c.allocations = allocations.load(std::memory_order_relaxed);
  c.allocated_bytes = allocated_bytes.load(std::memory_order_relaxed);
  c.host_syncs = host_syncs.load(std::memory_order_relaxed);
  return c;
}

void note_allocation(std::size_t bytes) noexcept {
  allocations.fetch_add(1, std::memory_order_relaxed);
  allocated_bytes.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
}

void note_host_sync() noexcept {
  host_syncs.fetch_add(1, std::memory_order_relaxed);
}
#else
bool budgets_enabled() noexcept {
  return false;
}

budget_counters budget_snapshot() noexcept {
  return {};
}

void note_allocation(std::size_t /*bytes*/) noexcept {}

void note_host_sync() noexcept {}
#endif

}  // namespace dyng::detail
