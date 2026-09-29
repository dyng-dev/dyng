// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file rng.hpp
 * @brief Library-owned reproductions of the random streams the original tools produced.
 *
 * The MOSP tools draw weights with std::mt19937 and std::uniform_int_distribution<int>. The
 * engine is fully specified by the C++ standard, but the distribution is implementation-defined.
 * legacy_uniform_int reproduces the algorithm of libstdc++ 11 and later (GCC 11+, the toolchain
 * the pinned originals were built with): Lemire's nearly divisionless method with 64-bit products
 * for a 32-bit engine. Because it does not call std::uniform_int_distribution, the streams are the
 * same under libc++ or MSVC. A unit test compares it with std::uniform_int_distribution when the
 * test is built against libstdc++.
 */
#pragma once

#include <dyng/core/error.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace dyng::detail {

/**
 * @brief Draw from [a, b] exactly as libstdc++ (>= 11) std::uniform_int_distribution does with a
 *        32-bit engine such as std::mt19937.
 * @tparam engine_t A uniform random bit generator with min() == 0 and max() == 2^32 - 1.
 * @param[in,out] engine The engine.
 * @param[in]     a      Lower bound (inclusive).
 * @param[in]     b      Upper bound (inclusive), a <= b, b - a < 2^32.
 * @return The drawn value.
 * @throws invalid_argument_error if the range is empty or wider than 2^32 values.
 */
template <typename engine_t>
std::int64_t legacy_uniform_int(engine_t& engine, std::int64_t a, std::int64_t b) {
  static_assert(engine_t::min() == 0 && engine_t::max() == 0xffffffffULL,
                "legacy_uniform_int reproduces the 32-bit engine path only");
  DYNG_EXPECTS(a <= b, "legacy_uniform_int: empty range [", a, ", ", b, "]");
  const std::uint64_t urange = static_cast<std::uint64_t>(b) - static_cast<std::uint64_t>(a);
  DYNG_EXPECTS(urange <= 0xffffffffULL, "legacy_uniform_int: range wider than 2^32 values");
  std::uint64_t ret = 0;
  if (urange < 0xffffffffULL) {
    // Downscaling: libstdc++ _S_nd<uint64_t>(urng, uint32_t(urange + 1)).
    const auto range = static_cast<std::uint32_t>(urange + 1);
    std::uint64_t product = static_cast<std::uint64_t>(engine()) * range;
    auto low = static_cast<std::uint32_t>(product);
    if (low < range) {
      const std::uint32_t threshold = static_cast<std::uint32_t>(-range) % range;
      while (low < threshold) {
        product = static_cast<std::uint64_t>(engine()) * range;
        low = static_cast<std::uint32_t>(product);
      }
    }
    ret = product >> 32;
  } else {
    ret = static_cast<std::uint64_t>(engine());  // the range equals the engine's range
  }
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + ret);
}

/// Unsigned 128-bit integer (GCC and Clang) for the products of legacy_uniform_int64().
__extension__ typedef unsigned __int128 legacy_uint128;  // NOLINT(modernize-use-using)

/**
 * @brief Draw from [a, b] exactly as libstdc++ (>= 11) std::uniform_int_distribution<uint64_t>
 *        does with a 64-bit engine such as std::mt19937_64.
 *
 * libstdc++ downscales with Lemire's nearly divisionless method on 128-bit products when the
 * engine produces exactly 64 bits; a range of all 2^64 values returns the engine's draw.
 * @tparam engine_t A uniform random bit generator with min() == 0 and max() == 2^64 - 1.
 * @param[in,out] engine The engine.
 * @param[in]     a      Lower bound (inclusive).
 * @param[in]     b      Upper bound (inclusive), a <= b.
 * @return The drawn value.
 * @throws invalid_argument_error if the range is empty.
 */
template <typename engine_t>
std::uint64_t legacy_uniform_int64(engine_t& engine, std::uint64_t a, std::uint64_t b) {
  static_assert(engine_t::min() == 0 && engine_t::max() == 0xffffffffffffffffULL,
                "legacy_uniform_int64 reproduces the 64-bit engine path only");
  DYNG_EXPECTS(a <= b, "legacy_uniform_int64: empty range [", a, ", ", b, "]");
  const std::uint64_t urange = b - a;
  if (urange == 0xffffffffffffffffULL) {
    return a + static_cast<std::uint64_t>(engine());  // the range equals the engine's range
  }
  // libstdc++ _S_nd<unsigned __int128>(urng, urange + 1).
  const std::uint64_t range = urange + 1;
  using u128 = legacy_uint128;
  u128 product = static_cast<u128>(static_cast<std::uint64_t>(engine())) * range;
  auto low = static_cast<std::uint64_t>(product);
  if (low < range) {
    const std::uint64_t threshold = (0 - range) % range;
    while (low < threshold) {
      product = static_cast<u128>(static_cast<std::uint64_t>(engine())) * range;
      low = static_cast<std::uint64_t>(product);
    }
  }
  return a + static_cast<std::uint64_t>(product >> 64U);
}

/**
 * @brief std::shuffle exactly as libstdc++ (GCC 9 to 12 at least) performs it with a 64-bit
 *        engine such as std::mt19937_64, so the order does not depend on the standard library.
 *
 * When the engine's range covers the square of the element count, libstdc++ draws the swap
 * positions of two successive elements from one uniform draw (after one single swap for an even
 * count); otherwise one draw per element.
 * @tparam value_t  Element type.
 * @tparam engine_t A 64-bit uniform random bit generator (min() == 0, max() == 2^64 - 1).
 * @param[in,out] values The elements, shuffled in place.
 * @param[in,out] engine The engine.
 */
template <typename value_t, typename engine_t>
void legacy_shuffle(std::vector<value_t>& values, engine_t& engine) {
  if (values.empty()) {
    return;
  }
  using std::swap;
  const std::uint64_t urngrange = engine_t::max() - engine_t::min();
  const auto urange = static_cast<std::uint64_t>(values.size());
  if (urngrange / urange >= urange) {
    std::uint64_t i = 1;
    if (urange % 2 == 0) {
      swap(values[i], values[legacy_uniform_int64(engine, 0, 1)]);
      ++i;
    }
    while (i != urange) {
      // __gen_two_uniform_ints(i + 1, i + 2): one draw in [0, (i + 1)(i + 2) - 1].
      const std::uint64_t b0 = i + 1;
      const std::uint64_t b1 = i + 2;
      const std::uint64_t x = legacy_uniform_int64(engine, 0, b0 * b1 - 1);
      swap(values[i], values[x / b1]);
      ++i;
      swap(values[i], values[x % b1]);
      ++i;
    }
    return;
  }
  for (std::uint64_t i = 1; i != urange; ++i) {
    swap(values[i], values[legacy_uniform_int64(engine, 0, i)]);
  }
}

}  // namespace dyng::detail
