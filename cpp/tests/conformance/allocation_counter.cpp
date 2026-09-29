// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocation_counter.cpp
 * @brief The replacement of the global operator new of the conformance executables (see
 *        allocation_counter.hpp).
 */
#include "conformance/allocation_counter.hpp"

#include "core/budget_counters.hpp"
#include "support/gtest_helpers.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if !DYNG_TEST_SANITIZED && defined(__GLIBC__)
#include <execinfo.h>
#include <unistd.h>
#endif

namespace dyng::conformance {

namespace {

std::atomic<bool> counting{false};
/// DYNG_TEST_ALLOCATION_TRACE=1: print a backtrace of every counted allocation to stderr (to find
/// the allocation a failing C8 reports).
const bool trace = std::getenv("DYNG_TEST_ALLOCATION_TRACE") != nullptr;

}  // namespace

bool host_allocation_counting_available() noexcept {
  return DYNG_TEST_SANITIZED == 0;
}

void count_host_allocations(bool on) noexcept {
  counting.store(on, std::memory_order_seq_cst);
}

namespace {

#if !DYNG_TEST_SANITIZED
void* counted(std::size_t size, std::size_t alignment) {
  if (counting.load(std::memory_order_relaxed)) {
    detail::note_allocation(size);
#if defined(__GLIBC__)
    if (trace) {
      counting.store(false);  // backtrace() may allocate
      void* frames[32];
      const int depth = ::backtrace(frames, 32);
      ::backtrace_symbols_fd(frames, depth, STDERR_FILENO);
      static_cast<void>(::write(STDERR_FILENO, "----\n", 5));
      counting.store(true);
    }
#endif
  }
  const std::size_t bytes = size == 0 ? 1 : size;
  void* p = nullptr;
  if (alignment <= alignof(std::max_align_t)) {
    p = std::malloc(bytes);  // NOLINT(cppcoreguidelines-no-malloc): the global operator new
  } else {
    p = std::aligned_alloc(alignment, (bytes + alignment - 1) / alignment * alignment);
  }
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
#endif

}  // namespace

}  // namespace dyng::conformance

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  return dyng::conformance::counted(size, alignof(std::max_align_t));
}
void* operator new[](std::size_t size) {
  return dyng::conformance::counted(size, alignof(std::max_align_t));
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  return dyng::conformance::counted(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return dyng::conformance::counted(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return dyng::conformance::counted(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return dyng::conformance::counted(size, alignof(std::max_align_t));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
  std::free(p);
}
// NOLINTEND(cppcoreguidelines-no-malloc)
#endif
