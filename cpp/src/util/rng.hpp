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

}  // namespace dyng::detail
