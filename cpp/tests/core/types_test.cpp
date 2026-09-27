// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/types.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

TEST(Types, InvalidIdIsMinusOne) {
  static_assert(dyng::invalid_id<std::int32_t>() == -1);
  static_assert(dyng::invalid_id<std::int64_t>() == -1);
  SUCCEED();
}

TEST(Types, InfiniteDistanceMatchesMospDistanceInf) {
  // MOSP-OpenMP/MOSP-CUDA: DISTANCE_INF = INT64_MAX / 4; kept for byte parity.
  static_assert(dyng::infinite_distance<std::int64_t>() ==
                std::numeric_limits<std::int64_t>::max() / 4);
  static_assert(dyng::infinite_distance<std::int64_t>() == 2305843009213693951LL);
  // Adding the largest valid weight to "infinity" must not overflow.
  static_assert(dyng::infinite_distance<std::int64_t>() + std::numeric_limits<std::int32_t>::max() >
                0);
  SUCCEED();
}

TEST(Types, EnumsAreDistinct) {
  EXPECT_NE(dyng::engine::fused, dyng::engine::operators);
  EXPECT_NE(dyng::determinism::bitwise, dyng::determinism::tolerance);
}
