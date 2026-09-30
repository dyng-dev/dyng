// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocation_counter.hpp
 * @brief Counting of host heap allocations for conformance check C8 (invariant I9).
 *
 * allocation_counter.cpp, linked into every conformance executable, replaces the global operator
 * new; while a host_allocation_counter is alive each allocation (on any thread, the library's
 * OpenMP workers included) is reported to the budget counters (detail::note_allocation), so the
 * budget of the algorithm phase covers std::vector growth too. The counting is off outside the
 * counter's lifetime, so no other check is affected. Under AddressSanitizer or ThreadSanitizer the
 * replacement is not compiled (the sanitizers replace operator new themselves) and C8 counts the
 * library's memory resources only.
 */
#pragma once

namespace dyng::conformance {

/// Whether this executable counts host heap allocations (the replacement is compiled in).
[[nodiscard]] bool host_allocation_counting_available() noexcept;

/// Switch the counting on or off (process-wide).
void count_host_allocations(bool on) noexcept;

/// Counts host heap allocations while alive (or until stop()).
class host_allocation_counter {
 public:
  /// Start counting.
  host_allocation_counter() noexcept {
    count_host_allocations(true);
  }
  host_allocation_counter(const host_allocation_counter&) = delete;             ///< not copyable
  host_allocation_counter& operator=(const host_allocation_counter&) = delete;  ///< not copyable
  host_allocation_counter(host_allocation_counter&&) = delete;                  ///< not movable
  host_allocation_counter& operator=(host_allocation_counter&&) = delete;       ///< not movable
  /// Stop counting.
  ~host_allocation_counter() {
    stop();
  }

  /// Stop counting now.
  void stop() noexcept {
    count_host_allocations(false);
  }

  /// Whether host allocations are counted in this executable.
  [[nodiscard]] bool counting() const noexcept {
    return host_allocation_counting_available();
  }
};

}  // namespace dyng::conformance
