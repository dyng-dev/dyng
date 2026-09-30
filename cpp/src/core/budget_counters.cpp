// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file budget_counters.cpp
 * @brief The per-thread counters of invariant I9 (counting only with DYNG_DEBUG_BUDGETS).
 */
#include "core/budget_counters.hpp"

#include <atomic>

#if defined(DYNG_DEBUG_BUDGETS) && defined(_OPENMP)
#include <omp.h>
#endif

namespace dyng::detail {

#if defined(DYNG_DEBUG_BUDGETS)
namespace {

/// The counts of one thread (only that thread writes and reads them).
struct thread_counts {
  std::int64_t allocations = 0;
  std::int64_t allocated_bytes = 0;
  std::int64_t host_syncs = 0;
  std::int64_t reservations = 0;
  std::int64_t container_allocations = 0;
  std::int64_t container_host_syncs = 0;
  std::int64_t instrumentation_allocations = 0;
  std::int64_t instrumentation_host_syncs = 0;
};

thread_local thread_counts mine;

/// The shared tally of OpenMP worker threads (see budget_counters.hpp, "Attribution"). Workers
/// only allocate (host heap, in the conformance executables) and note reservations.
std::atomic<std::int64_t> worker_allocations{0};
std::atomic<std::int64_t> worker_allocated_bytes{0};
std::atomic<std::int64_t> worker_reservations{0};

/// Whether the calling thread is a worker (not the primary thread) of an OpenMP parallel region.
bool on_openmp_worker() noexcept {
#if defined(_OPENMP)
  return omp_in_parallel() != 0 && omp_get_thread_num() != 0;
#else
  return false;
#endif
}

}  // namespace

bool budgets_enabled() noexcept {
  return true;
}

budget_counters budget_snapshot() noexcept {
  budget_counters c;
  c.allocations = mine.allocations + worker_allocations.load(std::memory_order_relaxed);
  c.allocated_bytes = mine.allocated_bytes + worker_allocated_bytes.load(std::memory_order_relaxed);
  c.host_syncs = mine.host_syncs;
  c.reservations = mine.reservations + worker_reservations.load(std::memory_order_relaxed);
  c.container_allocations = mine.container_allocations;
  c.container_host_syncs = mine.container_host_syncs;
  c.instrumentation_allocations = mine.instrumentation_allocations;
  c.instrumentation_host_syncs = mine.instrumentation_host_syncs;
  return c;
}

void note_allocation(std::size_t bytes) noexcept {
  if (on_openmp_worker()) {
    worker_allocations.fetch_add(1, std::memory_order_relaxed);
    worker_allocated_bytes.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    return;
  }
  mine.allocations += 1;
  mine.allocated_bytes += static_cast<std::int64_t>(bytes);
}

void note_host_sync() noexcept {
  // Library code synchronizes only outside parallel regions; a worker's sync is its own thread's.
  mine.host_syncs += 1;
}

void note_reservation() noexcept {
  if (on_openmp_worker()) {
    worker_reservations.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  mine.reservations += 1;
}

void note_container_work(std::int64_t allocations, std::int64_t host_syncs) noexcept {
  mine.container_allocations += allocations;
  mine.container_host_syncs += host_syncs;
}

void note_instrumentation_work(std::int64_t allocations, std::int64_t host_syncs) noexcept {
  mine.instrumentation_allocations += allocations;
  mine.instrumentation_host_syncs += host_syncs;
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

void note_reservation() noexcept {}

void note_container_work(std::int64_t /*allocations*/, std::int64_t /*host_syncs*/) noexcept {}

void note_instrumentation_work(std::int64_t /*allocations*/, std::int64_t /*host_syncs*/) noexcept {
}
#endif

}  // namespace dyng::detail
