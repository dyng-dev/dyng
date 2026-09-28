// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file thread_list.hpp
 * @brief Per-thread lists that never share a cache line with another thread's list.
 *
 * The OpenMP engines collect the vertices each thread finds in a thread-local list and
 * concatenate the lists afterwards (list_gather). MOSP creates those lists as `std::vector`s
 * inside every parallel region: every round then allocates small blocks from glibc's arenas, and
 * the blocks of two threads can lie next to each other, so that their push_back writes share a
 * cache line. A thread_list allocates whole cache lines at cache-line boundaries, and a
 * padded_thread_list keeps the vector's own header (written by push_back) on a line of its own.
 * Kept in a workspace, the lists also keep their capacity between rounds and runs, so a list
 * allocates only when its thread takes a larger share of a round than ever before (the dynamic
 * schedule decides the shares). (M1b measured no speed difference from this
 * alone; the near-far loop's time is in its barriers, see list_gather::gather_pair() and
 * parity/results/M1b.md section 9.)
 */
#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace dyng::detail {

/// Size of the cache line the lists are padded to (x86-64 and the usual AArch64 cores).
inline constexpr std::size_t cache_line_bytes = 64;

/**
 * @brief Allocator whose blocks start at a cache-line boundary and span whole cache lines.
 * @tparam value_t Element type.
 */
template <typename value_t>
struct cache_line_allocator {
  using value_type = value_t;  ///< element type

  cache_line_allocator() noexcept = default;

  /**
   * @brief Rebinding copy (allocators of any element type are interchangeable).
   * @tparam other_t The other element type.
   */
  template <typename other_t>
  explicit cache_line_allocator(const cache_line_allocator<other_t>& /*other*/) noexcept {}

  /**
   * @brief Allocate room for `count` elements.
   * @param count Number of elements.
   * @return The block (cache-line aligned, a whole number of cache lines).
   * @throws std::bad_alloc if the memory cannot be allocated.
   */
  [[nodiscard]] value_t* allocate(std::size_t count) {
    return static_cast<value_t*>(
        ::operator new(rounded(count), std::align_val_t{cache_line_bytes}));
  }

  /**
   * @brief Free a block from allocate().
   * @param block The block.
   * @param count The element count it was allocated for.
   */
  void deallocate(value_t* block, std::size_t count) noexcept {
    ::operator delete(block, rounded(count), std::align_val_t{cache_line_bytes});
  }

  /**
   * @brief All instances are interchangeable.
   * @return true.
   */
  template <typename other_t>
  bool operator==(const cache_line_allocator<other_t>& /*other*/) const noexcept {
    return true;
  }

  /**
   * @brief All instances are interchangeable.
   * @return false.
   */
  template <typename other_t>
  bool operator!=(const cache_line_allocator<other_t>& /*other*/) const noexcept {
    return false;
  }

 private:
  static std::size_t rounded(std::size_t count) noexcept {
    const std::size_t bytes = count * sizeof(value_t);
    return (bytes + cache_line_bytes - 1) / cache_line_bytes * cache_line_bytes;
  }
};

/// A per-thread list whose elements never share a cache line with another block.
template <typename value_t>
using thread_list = std::vector<value_t, cache_line_allocator<value_t>>;

/**
 * @brief A thread_list whose header (begin, end, capacity: written by push_back) fills a cache
 *        line of its own, so an array of them can be indexed by thread number.
 * @tparam value_t Element type.
 */
template <typename value_t>
struct alignas(cache_line_bytes) padded_thread_list {
  thread_list<value_t> items;  ///< the list
};

}  // namespace dyng::detail
