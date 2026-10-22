// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file region_exceptions.hpp
 * @brief region_exceptions: carry an exception out of an OpenMP parallel region (ADR 0061, D13).
 *
 * An exception must not leave an OpenMP region or an iteration of a worksharing loop: one that
 * does ends the process (std::terminate), also when the region's `if` clause is false and it runs
 * on one thread. Work inside a region that can throw (it allocates) therefore runs through
 * run(), which catches every exception and keeps the first one; the remaining iterations are
 * skipped; and rethrow(), after the region, throws the kept exception on the calling thread,
 * where the entry point's handlers translate it (DYNG_TRANSLATE_ALLOCATION_FAILURE: a
 * std::bad_alloc becomes out_of_memory_error) and the container's store phase poisons the
 * hypergraph, as the headers document.
 *
 *     region_exceptions errors;
 *     #pragma omp parallel for num_threads(threads) if (parallel)
 *     for (std::int64_t k = 0; k < n; ++k) {
 *       errors.run([&] { rows[k].push_back(k); });
 *     }
 *     errors.rethrow();
 *
 * "First" is the first to be caught: with several threads, which one depends on timing (the
 * errors of the regions this guards are allocation failures, all reported alike). The same code
 * works without OpenMP (a plain loop) and costs one relaxed atomic load per call of run() when
 * nothing throws.
 */
#pragma once

#include <atomic>
#include <exception>
#include <utility>

namespace dyng::detail {

/// The first exception thrown by the work of a parallel region, rethrown after it.
class region_exceptions {
 public:
  region_exceptions() noexcept = default;
  region_exceptions(const region_exceptions&) = delete;
  region_exceptions& operator=(const region_exceptions&) = delete;
  region_exceptions(region_exceptions&&) = delete;
  region_exceptions& operator=(region_exceptions&&) = delete;
  ~region_exceptions() = default;

  /**
   * @brief Run `work()` unless an earlier call failed; keep its exception if it throws.
   * @tparam work_t A callable `void()`.
   * @param[in] work The work of one iteration (or of one thread).
   */
  template <typename work_t>
  void run(work_t&& work) noexcept {
    if (failed()) {
      return;
    }
    try {
      std::forward<work_t>(work)();
    } catch (...) {
      capture();
    }
  }

  /**
   * @brief Whether a call of run() threw.
   * @return True once an exception is kept.
   */
  [[nodiscard]] bool failed() const noexcept {
    return claimed_.load(std::memory_order_relaxed);
  }

  /**
   * @brief After the region: throw the kept exception, if any.
   * @throws The first exception a call of run() caught.
   */
  void rethrow() const {
    if (ready_.load(std::memory_order_acquire)) {
      std::rethrow_exception(first_);
    }
  }

 private:
  // The first thread to claim the slot stores its exception; the release store of `ready_`
  // publishes it to rethrow() (the region's closing barrier orders it as well).
  void capture() noexcept {
    bool expected = false;
    if (claimed_.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
      first_ = std::current_exception();
      ready_.store(true, std::memory_order_release);
    }
  }

  std::atomic<bool> claimed_{false};
  std::atomic<bool> ready_{false};
  std::exception_ptr first_;
};

}  // namespace dyng::detail
