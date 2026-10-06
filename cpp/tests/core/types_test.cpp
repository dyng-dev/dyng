// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/memory.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
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

// The 0.1 API review (ADR 0023): every core enumeration that a stats object, the registry or a
// log line reports has a to_string() giving the enumerator's own name (the manifests' spelling,
// which the CLI and the Python layer accept as strings).
TEST(Types, EveryCoreEnumerationHasItsNameAsString) {
  EXPECT_EQ(dyng::to_string(dyng::engine::automatic), "automatic");
  EXPECT_EQ(dyng::to_string(dyng::engine::fused), "fused");
  EXPECT_EQ(dyng::to_string(dyng::engine::operators), "operators");
  EXPECT_EQ(dyng::to_string(dyng::determinism::bitwise), "bitwise");
  EXPECT_EQ(dyng::to_string(dyng::determinism::exact_value), "exact_value");
  EXPECT_EQ(dyng::to_string(dyng::determinism::tolerance), "tolerance");
  EXPECT_EQ(dyng::to_string(dyng::memory_space::host), "host");
  EXPECT_EQ(dyng::to_string(dyng::memory_space::pinned_host), "pinned_host");
  EXPECT_EQ(dyng::to_string(dyng::memory_space::device), "device");
  EXPECT_EQ(dyng::to_string(dyng::memory_space::managed), "managed");
  EXPECT_EQ(dyng::to_string(dyng::copy_policy::allow), "allow");
  EXPECT_EQ(dyng::to_string(dyng::copy_policy::warn), "warn");
  EXPECT_EQ(dyng::to_string(dyng::copy_policy::error), "error");
  EXPECT_EQ(dyng::to_string(dyng::algorithm_family::fixed_point), "fixed_point");
  EXPECT_EQ(dyng::to_string(dyng::algorithm_family::aggregate_delta), "aggregate_delta");
  EXPECT_EQ(dyng::to_string(dyng::container_kind::graph), "graph");
  EXPECT_EQ(dyng::to_string(dyng::container_kind::hypergraph), "hypergraph");
  EXPECT_EQ(dyng::to_string(dyng::maturity_level::experimental), "experimental");
  EXPECT_EQ(dyng::to_string(dyng::maturity_level::stable), "stable");
  EXPECT_EQ(dyng::to_string(dyng::maturity_level::deprecated), "deprecated");
  EXPECT_EQ(dyng::to_string(dyng::maturity_level::tutorial), "tutorial");
  EXPECT_EQ(dyng::to_string(dyng::oracle_kind::compute), "compute");
  EXPECT_EQ(dyng::to_string(dyng::oracle_kind::reference), "reference");
}

TEST(Types, TheRegistryReportsItsEntriesByTheManifestNames) {
  for (const dyng::algorithm_info& info : dyng::algorithms()) {
    EXPECT_NE(dyng::to_string(info.family), "unknown") << info.name;
    EXPECT_NE(dyng::to_string(info.container), "unknown") << info.name;
    EXPECT_NE(dyng::to_string(info.maturity), "unknown") << info.name;
    EXPECT_NE(dyng::to_string(info.determinism_level), "unknown") << info.name;
    EXPECT_NE(dyng::to_string(info.oracle), "unknown") << info.name;
  }
}
