// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocation_failure_test.cpp
 * @brief Host allocation failures in header-inline code: they leave the library as
 *        out_of_memory_error (never std::bad_alloc or std::length_error), and the builders keep
 *        their strong guarantee (a failed edge_batch::insert_edge / delete_edge or
 *        edge_list::add_edge leaves the arrays as they were).
 *
 * This executable replaces the global operator new with one that can be told to fail the n-th
 * allocation on the calling thread (not compiled under the sanitizers, which replace operator new
 * themselves; the injected tests then skip).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {

/// Allocations on this thread until one fails (0: none fails).
thread_local int fail_countdown = 0;

}  // namespace

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  if (fail_countdown > 0 && --fail_countdown == 0) {
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

namespace {

/// Fail the n-th allocation of the calling thread from now on.
struct failing_allocation {
  explicit failing_allocation(int n) noexcept {
    fail_countdown = n;
  }
  failing_allocation(const failing_allocation&) = delete;
  failing_allocation& operator=(const failing_allocation&) = delete;
  failing_allocation(failing_allocation&&) = delete;
  failing_allocation& operator=(failing_allocation&&) = delete;
  ~failing_allocation() {
    fail_countdown = 0;
  }
};

TEST(AllocationFailure, ToVectorThrowsOutOfMemoryError) {
  const dyng::resources res = dyng::resources::sequential();
  const std::int64_t x = 0;
  // More elements than a vector can hold: std::length_error inside, out_of_memory_error outside.
  const dyng::array_view<const std::int64_t> huge(&x, SIZE_MAX / 4);
  EXPECT_THROW((void)dyng::to_vector(res, huge), dyng::out_of_memory_error);
}

TEST(AllocationFailure, InsertEdgeIsStrong) {
  if (DYNG_TEST_SANITIZED) {
    GTEST_SKIP() << "operator new is the sanitizer's";
  }
  for (int n : {1, 2, 3}) {
    SCOPED_TRACE("allocation " + std::to_string(n) + " fails");
    dyng::edge_batch<std::int32_t, std::int32_t> b(2);
    {
      const failing_allocation fail(n);  // src, dst, then the weights grow
      EXPECT_THROW(b.insert_edge(1, 2, {10, 20}), dyng::out_of_memory_error);
    }
    EXPECT_EQ(b.num_insertions(), 0U);
    const auto v = b.view();
    EXPECT_EQ(v.insert_src.size(), 0U);
    EXPECT_EQ(v.insert_dst.size(), 0U);
    EXPECT_EQ(v.insert_weights.size(), 0U);
    b.insert_edge(3, 4, {1, 2});  // still usable
    EXPECT_EQ(b.view().insert_weights.size(), 2U);
  }
}

TEST(AllocationFailure, DeleteEdgeIsStrong) {
  if (DYNG_TEST_SANITIZED) {
    GTEST_SKIP() << "operator new is the sanitizer's";
  }
  for (int n : {1, 2}) {
    dyng::edge_batch<std::int32_t, std::int32_t> b(1);
    {
      const failing_allocation fail(n);
      EXPECT_THROW(b.delete_edge(1, 2), dyng::out_of_memory_error);
    }
    EXPECT_EQ(b.num_deletions(), 0U);
    EXPECT_EQ(b.view().delete_dst.size(), 0U);
  }
}

TEST(AllocationFailure, AddEdgeIsStrong) {
  if (DYNG_TEST_SANITIZED) {
    GTEST_SKIP() << "operator new is the sanitizer's";
  }
  for (int n : {1, 2, 3}) {
    dyng::edge_list<std::int32_t, std::int32_t> list;
    list.num_weights = 1;
    {
      const failing_allocation fail(n);
      EXPECT_THROW(list.add_edge(0, 1, {5}), dyng::out_of_memory_error);
    }
    EXPECT_EQ(list.src.size(), 0U);
    EXPECT_EQ(list.dst.size(), 0U);
    EXPECT_EQ(list.weights.size(), 0U);
  }
}

}  // namespace
