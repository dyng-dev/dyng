// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file failing_allocator.hpp
 * @brief A replaceable global operator new that fails the n-th allocation, of the calling thread
 *        or of any thread (the OpenMP workers included), and a sweep that fails every allocation
 *        of a call in turn: the allocation-failure tests of ADR 0060 and ADR 0061 (D13).
 *
 * Include it in ONE translation unit of a test executable: it defines the global allocation
 * functions. Under AddressSanitizer or ThreadSanitizer, which replace them themselves, nothing is
 * defined and the tests skip (DYNG_TEST_SANITIZED).
 */
#pragma once

#include "support/gtest_helpers.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <new>
#include <string>

namespace dyng::test::alloc {

/// Allocations of this thread until one fails (0: none fails).
inline thread_local long long calling_thread_countdown = 0;
/// Allocations of every thread until one fails (0: none fails).
inline std::atomic<long long> any_thread_countdown{0};

/// Whose allocations a sweep counts.
enum class scope {
  calling_thread,  ///< the thread that makes the call (the host work of one thread)
  any_thread,      ///< every thread, OpenMP workers included (the order depends on timing)
};

/// Whether this allocation must fail (called by operator new).
inline bool fail_now() noexcept {
  if (calling_thread_countdown > 0 && --calling_thread_countdown == 0) {
    return true;
  }
  long long left = any_thread_countdown.load(std::memory_order_relaxed);
  while (left > 0 &&
         !any_thread_countdown.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
  }
  return left == 1;
}

/// Arm the countdown of `s` at k (0 disarms both).
inline void arm(scope s, long long k) noexcept {
  calling_thread_countdown = s == scope::calling_thread ? k : 0;
  any_thread_countdown.store(s == scope::any_thread ? k : 0, std::memory_order_relaxed);
}

/// Disarm; return whether the armed allocation was reached.
inline bool disarm() noexcept {
  const bool reached =
      calling_thread_countdown == 0 && any_thread_countdown.load(std::memory_order_relaxed) == 0;
  arm(scope::calling_thread, 0);
  return reached;
}

/**
 * @brief Fail the k-th allocation of `call` for k = 1, 2, ... until the call makes fewer than k
 *        allocations (of `s`). Before every call `prepare()` runs unarmed; after it `check(threw)`
 *        runs unarmed, `threw` saying whether the call threw `expected_t`. Any other exception
 *        is a test failure (a std::bad_alloc where `expected_t` is out_of_memory_error is one:
 *        it left the library untranslated); an exception that leaves an OpenMP region ends the
 *        process (the defect of D13).
 * @tparam expected_t The exception a failed allocation must produce.
 * @param[in] s       Whose allocations are failed.
 * @param[in] prepare The set-up of one call (unarmed).
 * @param[in] call    The call.
 * @param[in] check   The checks after one call (unarmed).
 * @return The number of calls that threw `expected_t`.
 */
template <typename expected_t>
int fail_each_allocation(
    scope s, const std::function<void()>& prepare, const std::function<void()>& call,
    const std::function<void(bool)>& check = [](bool) {}) {
  int failures = 0;
  for (long long k = 1; k < 1000000; ++k) {
    prepare();
    bool threw = false;
    std::string other;
    arm(s, k);
    try {
      call();
    } catch (const expected_t&) {
      threw = true;
    } catch (const std::exception& e) {
      other = e.what();
      if (other.empty()) other = "an exception";
    }
    const bool reached = disarm();
    if (!other.empty()) {
      ADD_FAILURE() << "allocation " << k << ": unexpected exception: " << other;
      return failures;
    }
    check(threw);
    if (!reached) break;  // the call made fewer than k allocations: every one was failed once
    failures += threw ? 1 : 0;
  }
  return failures;
}

}  // namespace dyng::test::alloc

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  if (dyng::test::alloc::fail_now()) {
    throw std::bad_alloc();
  }
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
// NOLINTEND(cppcoreguidelines-no-malloc)
#endif
