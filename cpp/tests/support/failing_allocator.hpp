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
 *
 * Added for sssp's OpenMP regions (M11b, ADR 0094): with DYNG_TEST_FAIL_ALIGNED_NEW defined to 1
 * before the include, the aligned forms (operator new(size, align_val_t), which the cache-line
 * lists of util/thread_list.hpp and dynG's host memory resource use) are replaced as well and
 * counted apart (allocation_kind); failure::from_then_on fails every allocation of every thread
 * once the countdown is reached (memory exhausted for the whole team, not one allocation); and
 * within_seconds() ends the process with a message when a call does not return (a thread that
 * missed a barrier). Without the define the executables of ADRs 0060-0061 and 0083 are unchanged.
 */
#pragma once

#include "support/gtest_helpers.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#ifndef DYNG_TEST_FAIL_ALIGNED_NEW
#define DYNG_TEST_FAIL_ALIGNED_NEW 0
#endif

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

/// What fails once the countdown is reached.
enum class failure {
  once,          ///< that allocation only (the others succeed)
  from_then_on,  ///< that one and every later allocation of every thread, until disarmed
};

/// The form of operator new an allocation used.
enum class allocation_kind : int {
  none = 0,     ///< no allocation failed yet
  plain = 1,    ///< operator new(size)
  aligned = 2,  ///< operator new(size, align_val_t) (DYNG_TEST_FAIL_ALIGNED_NEW)
};

/// The mode of the armed countdown.
inline std::atomic<failure> armed_failure{failure::once};
/// failure::from_then_on was reached: every allocation fails until disarm().
inline std::atomic<bool> exhausted{false};
/// The kind of the first allocation that failed since arm().
inline std::atomic<int> first_failed_kind{0};
/// Allocations made so far, by kind (counted whether armed or not).
inline std::atomic<long long> plain_allocations{0};
inline std::atomic<long long> aligned_allocations{0};  ///< see plain_allocations

/// Whether this allocation must fail (called by operator new).
inline bool fail_now(allocation_kind kind = allocation_kind::plain) noexcept {
  (kind == allocation_kind::aligned ? aligned_allocations : plain_allocations)
      .fetch_add(1, std::memory_order_relaxed);
  bool fail = exhausted.load(std::memory_order_relaxed);
  if (!fail && calling_thread_countdown > 0 && --calling_thread_countdown == 0) {
    fail = true;
  }
  if (!fail) {
    long long left = any_thread_countdown.load(std::memory_order_relaxed);
    while (left > 0 &&
           !any_thread_countdown.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
    }
    fail = left == 1;
  }
  if (fail) {
    int expected = 0;
    first_failed_kind.compare_exchange_strong(expected, static_cast<int>(kind),
                                              std::memory_order_relaxed);
    if (armed_failure.load(std::memory_order_relaxed) == failure::from_then_on) {
      exhausted.store(true, std::memory_order_relaxed);
    }
  }
  return fail;
}

/// Arm the countdown of `s` at k (0 disarms both), failing as `mode` says.
inline void arm(scope s, long long k, failure mode = failure::once) noexcept {
  exhausted.store(false, std::memory_order_relaxed);
  first_failed_kind.store(0, std::memory_order_relaxed);
  armed_failure.store(mode, std::memory_order_relaxed);
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

/// The kind of the first allocation that failed since the last arm().
inline allocation_kind failed_kind() noexcept {
  return static_cast<allocation_kind>(first_failed_kind.load(std::memory_order_relaxed));
}

/**
 * @brief Run `body`; end the process with a message if it has not returned after `seconds` (a
 *        deadlock: a thread of an OpenMP team that missed a barrier or a `single`). The watchdog
 *        thread is created before `body` runs and allocates nothing while it waits.
 * @param[in] seconds The limit.
 * @param[in] what    Named in the message.
 * @param[in] body    The work.
 */
inline void within_seconds(int seconds, const char* what, const std::function<void()>& body) {
  std::mutex mutex;
  std::condition_variable done_cv;
  bool done = false;
  std::thread watchdog([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (!done_cv.wait_for(lock, std::chrono::seconds(seconds), [&] { return done; })) {
      std::fprintf(stderr, "within_seconds: %s did not return within %d s (a deadlock)\n", what,
                   seconds);
      std::fflush(stderr);
      std::_Exit(124);
    }
  });
  try {
    body();
  } catch (...) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      done = true;
    }
    done_cv.notify_one();
    watchdog.join();
    throw;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex);
    done = true;
  }
  done_cv.notify_one();
  watchdog.join();
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
    scope s, failure mode, const std::function<void()>& prepare, const std::function<void()>& call,
    const std::function<void(bool)>& check = [](bool) {}) {
  int failures = 0;
  for (long long k = 1; k < 1000000; ++k) {
    prepare();
    bool threw = false;
    std::string other;
    arm(s, k, mode);
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

/// fail_each_allocation() with failure::once (one allocation fails, the others succeed).
template <typename expected_t>
int fail_each_allocation(
    scope s, const std::function<void()>& prepare, const std::function<void()>& call,
    const std::function<void(bool)>& check = [](bool) {}) {
  return fail_each_allocation<expected_t>(s, failure::once, prepare, call, check);
}

}  // namespace dyng::test::alloc

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions. Not
// inlined: GCC at -O2 and above would otherwise see free() paired with operator new's pointers
// in the callers and report -Wmismatched-new-delete.
[[gnu::noinline]] void* operator new(std::size_t size) {
  if (dyng::test::alloc::fail_now()) {
    throw std::bad_alloc();
  }
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
[[gnu::noinline]] void operator delete(void* p) noexcept {
  std::free(p);
}
[[gnu::noinline]] void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
#if DYNG_TEST_FAIL_ALIGNED_NEW
[[gnu::noinline]] void* operator new(std::size_t size, std::align_val_t alignment) {
  if (dyng::test::alloc::fail_now(dyng::test::alloc::allocation_kind::aligned)) {
    throw std::bad_alloc();
  }
  const auto align = static_cast<std::size_t>(alignment);
  const std::size_t bytes = ((size == 0 ? 1 : size) + align - 1) / align * align;
  void* p = std::aligned_alloc(align, bytes);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
[[gnu::noinline]] void operator delete(void* p, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
[[gnu::noinline]] void operator delete(void* p, std::size_t /*size*/,
                                       std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
#endif
// NOLINTEND(cppcoreguidelines-no-malloc)
#endif
