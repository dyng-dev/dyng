// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file thread_list_test.cpp
 * @brief Per-thread lists never share a cache line (util/thread_list.hpp).
 */
#include "util/thread_list.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using dyng::detail::cache_line_bytes;

std::uintptr_t address(const void* p) {
  return reinterpret_cast<std::uintptr_t>(p);
}

TEST(ThreadList, BlocksAreWholeAlignedCacheLines) {
  std::vector<dyng::detail::thread_list<std::int32_t>> lists(8);
  for (std::size_t i = 0; i < lists.size(); ++i) {
    for (std::size_t k = 0; k <= i * 7; ++k) {
      lists[i].push_back(static_cast<std::int32_t>(k));
    }
    EXPECT_EQ(address(lists[i].data()) % cache_line_bytes, 0U) << i;
  }
  // Two blocks never overlap a cache line: each spans whole lines from an aligned start.
  for (std::size_t i = 0; i < lists.size(); ++i) {
    const std::uintptr_t begin = address(lists[i].data());
    const std::uintptr_t end = begin + lists[i].capacity() * sizeof(std::int32_t);
    const std::uintptr_t last_line = (end - 1) / cache_line_bytes;
    for (std::size_t j = 0; j < lists.size(); ++j) {
      if (i != j) {
        const std::uintptr_t other = address(lists[j].data()) / cache_line_bytes;
        EXPECT_FALSE(other >= begin / cache_line_bytes && other <= last_line) << i << " " << j;
      }
    }
  }
}

TEST(ThreadList, PaddedHeadersHaveTheirOwnLines) {
  static_assert(alignof(dyng::detail::padded_thread_list<std::int64_t>) == cache_line_bytes);
  static_assert(sizeof(dyng::detail::padded_thread_list<std::int32_t>) % cache_line_bytes == 0);
  std::vector<dyng::detail::padded_thread_list<std::int32_t>> lists(4);
  for (const auto& list : lists) {
    EXPECT_EQ(address(&list) % cache_line_bytes, 0U);
  }
  lists[1].items.assign({3, 1, 2});
  lists[1].items.clear();
  EXPECT_GE(lists[1].items.capacity(), 3U);  // clear() keeps the capacity: no reallocation
}

}  // namespace
