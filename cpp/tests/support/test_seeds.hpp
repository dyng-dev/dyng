// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_seeds.hpp
 * @brief Seeds of the randomized tests and their replay (PLAN Section 8.1): every failure prints
 *        its seed, and `DYNG_TEST_SEED=<seed> ctest -R <test>` replays that one seed.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace dyng::test {

/**
 * @brief The seeds of a randomized test: `base`, `base + 1`, ... (`default_count` of them).
 *
 * `DYNG_TEST_SEED=<seed>` replays one seed; `DYNG_TEST_SEEDS=<n>` widens (or narrows) the
 * campaign to n seeds (for example in nightly runs).
 * @param[in] base          The first seed.
 * @param[in] default_count The number of seeds without DYNG_TEST_SEEDS.
 * @return The seeds.
 */
inline std::vector<std::uint64_t> test_seeds(std::uint64_t base, int default_count) {
  if (const char* one = std::getenv("DYNG_TEST_SEED")) {
    return {std::strtoull(one, nullptr, 10)};
  }
  int count = default_count;
  if (const char* n = std::getenv("DYNG_TEST_SEEDS")) {
    count = std::max(1, std::atoi(n));
  }
  std::vector<std::uint64_t> out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(base + static_cast<std::uint64_t>(i));
  }
  return out;
}

/**
 * @brief The trace line of one seed (for SCOPED_TRACE): the seed and its replay command.
 * @param[in] seed The seed.
 * @return "seed <s> (replay: DYNG_TEST_SEED=<s> ctest -R <test>)".
 */
inline std::string seed_trace(std::uint64_t seed) {
  return "seed " + std::to_string(seed) + " (replay: DYNG_TEST_SEED=" + std::to_string(seed) +
         " ctest -R <test>)";
}

}  // namespace dyng::test
