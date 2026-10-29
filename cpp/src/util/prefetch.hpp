// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file prefetch.hpp
 * @brief A read prefetch hint for the host loops that walk a batch's ids and vertices through
 *        id-indexed arrays (one cache miss per element otherwise; M11a step 3, ADR 0089), and the
 *        lowest set bit of a word.
 */
#pragma once

#include <cstdint>

namespace dyng::detail {

/// @brief Hint that `p` will be read soon (no effect where the compiler has no builtin).
/// @param[in] p The address (any; it is not dereferenced).
inline void prefetch_read(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 3);
#else
  (void)p;
#endif
}

/// @brief The index of the lowest set bit of a non-zero word. @param[in] x The word.
/// @return The index.
inline int lowest_bit(std::uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#else
  int bit = 0;
  while (((x >> bit) & 1U) == 0) ++bit;
  return bit;
#endif
}

}  // namespace dyng::detail
