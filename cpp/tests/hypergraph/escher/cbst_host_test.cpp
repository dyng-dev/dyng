// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_host_test.cpp
 * @brief Host parts of the CBST store (M8a steps 3-4, ADR 0038, 0039): the payload policies
 *        (ESCHER-GPU's payloadCapacity and ensurePayloadCapacity) and flatten_rows (its
 *        flatten2DVector), on every build.
 */
#include "hypergraph/escher/flatten.hpp"
#include "hypergraph/escher/store_policy.hpp"

#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <climits>
#include <vector>

namespace dyng::detail::escher {
namespace {

TEST(CbstStorePolicy, EscherGpuInitialCapacity) {
  const store_policy p = store_policy::escher_gpu();
  EXPECT_EQ(p.name, "escher_gpu");
  EXPECT_EQ(p.alignment, 4);
  EXPECT_EQ(p.compaction, store_policy::compaction_rule::never);
  // In-place trees: max(requested, size + size/8 + 1024); merged trees: the exact size.
  EXPECT_EQ(p.initial_capacity(0, 8000, true), 8000 + 1000 + 1024);
  EXPECT_EQ(p.initial_capacity(1 << 20, 8000, true), 1 << 20);
  EXPECT_EQ(p.initial_capacity(1 << 20, 8000, false), 8000);
  EXPECT_EQ(p.initial_capacity(0, 0, true), 1024);
  EXPECT_THROW((void)p.initial_capacity(0, INT_MAX, true), capacity_error);
  EXPECT_EQ(p.initial_capacity(0, INT_MAX, false), INT_MAX);
}

TEST(CbstStorePolicy, EscherGpuGrowth) {
  const store_policy p = store_policy::escher_gpu();
  // max(needed + needed/4, 3 * capacity / 2), in integers.
  EXPECT_EQ(p.grown_capacity(101, 100, "t"), 150);
  EXPECT_EQ(p.grown_capacity(200, 100, "t"), 250);
  EXPECT_EQ(p.grown_capacity(7, 5, "t"), 8);
  EXPECT_EQ(p.grown_capacity(1, 0, "t"), 1);
  // Capped at INT_MAX; beyond it, capacity_error (ESCHER-GPU exited the process).
  EXPECT_EQ(p.grown_capacity(2000000000LL, 1900000000LL, "t"), INT_MAX);
  EXPECT_EQ(p.grown_capacity(INT_MAX, 1900000000LL, "t"), INT_MAX);
  EXPECT_THROW((void)p.grown_capacity(static_cast<long long>(INT_MAX) + 1, 100, "t"),
               capacity_error);
  store_policy fixed = p;
  fixed.growth = store_policy::growth_rule::none;
  EXPECT_THROW((void)fixed.grown_capacity(101, 100, "t"), capacity_error);
}

TEST(CbstStorePolicy, TheStagingPresetAndTheNames) {
  // MOSP_ESCHER's behaviours merged so far (M8a step 4, ADR 0039): a fixed capacity, no
  // compaction of its own yet.
  const store_policy p = store_policy::mosp_escher_staged();
  EXPECT_EQ(p.name, "mosp_escher_staged");
  EXPECT_EQ(p.growth, store_policy::growth_rule::none);
  EXPECT_EQ(p.compaction, store_policy::compaction_rule::never);
  EXPECT_EQ(p.alignment, 4);
  EXPECT_THROW((void)p.grown_capacity(101, 100, "t"), capacity_error);
  store_policy q;
  ASSERT_TRUE(store_policy::from_name("escher_gpu", q));
  EXPECT_EQ(q.name, "escher_gpu");
  ASSERT_TRUE(store_policy::from_name("mosp_escher_staged", q));
  EXPECT_EQ(q.name, "mosp_escher_staged");
  EXPECT_FALSE(store_policy::from_name("mosp_escher", q));  // M8a step 5
}

TEST(CbstFlatten, TheConstructLayout) {
  const auto [flat, starts] = flatten_rows({{1, 2, 3}, {}, {4, 5, 6, 7}, {8}});
  // Row i takes ((n + 4) / 4) * 4 slots: its values, zeros, INT_MIN in the last slot.
  EXPECT_EQ(starts, (std::vector<int>{0, 4, 8, 16}));
  EXPECT_EQ(flat, (std::vector<int>{1, 2, 3, INT_MIN, 0, 0,       0, INT_MIN, 4, 5,
                                    6, 7, 0, 0,       0, INT_MIN, 8, 0,       0, INT_MIN}));
  const auto [none, no_starts] = flatten_rows({});
  EXPECT_TRUE(none.empty());
  EXPECT_TRUE(no_starts.empty());
}

}  // namespace
}  // namespace dyng::detail::escher
