// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file radix_sort.hpp
 * @brief A stable LSD radix sort of packed (key << 32 | payload) words by their key, for the
 *        host side of a slack_csr batch (M11a step 3, ADR 0089: the sorts of the touched ids and
 *        vertices of a batch, which std::sort made a visible part of its host cost).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail::hyper {

/**
 * @brief Sort `words` (each `key << 32 | payload`, the key below `key_bound` and at most 2^31)
 *        by their key, stably: words of equal key keep their order. When the payloads increase in
 *        the input order (an index), the result equals std::sort of the words.
 * @param[in,out] words     The words.
 * @param[in]     key_bound An exclusive bound of the keys.
 * @param[in,out] scratch   A buffer the sort reuses (its contents are undefined after).
 */
inline void radix_sort_by_key(std::vector<std::uint64_t>& words, std::uint64_t key_bound,
                              std::vector<std::uint64_t>& scratch) {
  constexpr int digit_bits = 11;
  constexpr std::size_t radix = std::size_t{1} << digit_bits;
  if (words.size() < 256) {
    std::stable_sort(words.begin(), words.end(),
                     [](std::uint64_t a, std::uint64_t b) { return (a >> 32) < (b >> 32); });
    return;
  }
  int key_bits = 0;
  while (key_bits < 32 && (std::uint64_t{1} << key_bits) < key_bound) ++key_bits;
  scratch.resize(words.size());
  std::array<std::size_t, radix> count{};
  for (int shift = 32; shift < 32 + key_bits; shift += digit_bits) {
    count.fill(0);
    for (const std::uint64_t w : words) ++count[(w >> shift) & (radix - 1)];
    std::size_t sum = 0;
    for (std::size_t& c : count) {
      const std::size_t n = c;
      c = sum;
      sum += n;
    }
    for (const std::uint64_t w : words) scratch[count[(w >> shift) & (radix - 1)]++] = w;
    words.swap(scratch);
  }
}

}  // namespace dyng::detail::hyper
