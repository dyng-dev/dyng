// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_host_test.cpp
 * @brief Host parts of the CBST store (M8a steps 3-5, ADR 0038, 0039): the payload policies
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

TEST(CbstStorePolicy, TheMospEscherPresetAndTheNames) {
  // MOSP_ESCHER@4b86159 (M8a step 5, ADR 0039): a fixed capacity of flat * 1.3 + extra, at most
  // 2.0e9 (DynamicHypergraph's constructFromRows), compaction at its trigger points.
  const store_policy p = store_policy::mosp_escher();
  EXPECT_EQ(p.name, "mosp_escher");
  EXPECT_EQ(p.growth, store_policy::growth_rule::none);
  EXPECT_EQ(p.compaction, store_policy::compaction_rule::mosp_escher);
  EXPECT_EQ(p.scratch, store_policy::scratch_rule::geometric);
  EXPECT_EQ(p.alignment, 4);
  EXPECT_THROW((void)p.grown_capacity(101, 100, "t"), capacity_error);
  EXPECT_EQ(p.initial_capacity(0, 1000, true), 1300 + (1 << 20));
  EXPECT_EQ(p.initial_capacity(1 << 30, 1000, false), 1300 + (1 << 20));  // no lower bound
  EXPECT_EQ(store_policy::mosp_escher(8).initial_capacity(0, 10, true), 13 + 8);
  EXPECT_EQ(store_policy::mosp_escher(0).initial_capacity(0, 3, true), 3);  // floor(3.9)
  EXPECT_EQ(p.initial_capacity(0, 1800000000LL, true), 2000000000LL);       // the cap
  EXPECT_THROW((void)p.initial_capacity(0, 2000000001LL, true), capacity_error);
  store_policy q;
  ASSERT_TRUE(store_policy::from_name("escher_gpu", q));
  EXPECT_EQ(q.name, "escher_gpu");
  ASSERT_TRUE(store_policy::from_name("mosp_escher", q));
  EXPECT_EQ(q.name, "mosp_escher");
  EXPECT_EQ(q.extra, 1 << 20);
  EXPECT_FALSE(store_policy::from_name("mosp_escher_staged", q));  // the staging preset of step 4
}

// dynG's default (M8a step 5, ADR 0039): ESCHER-GPU's sizing and growth after MOSP_ESCHER's
// compaction, capacity_error only beyond max_payload (INT_MAX: the int32 offsets).
TEST(CbstStorePolicy, TheDefaultPreset) {
  const store_policy d = store_policy::dyng_default();
  EXPECT_EQ(d.name, "dyng_default");
  EXPECT_EQ(d.growth, store_policy::growth_rule::escher_gpu);
  EXPECT_EQ(d.compaction, store_policy::compaction_rule::mosp_escher);
  EXPECT_EQ(d.scratch, store_policy::scratch_rule::geometric);
  EXPECT_EQ(d.alignment, 4);
  EXPECT_EQ(d.max_payload, INT_MAX);
  EXPECT_EQ(d.initial_capacity(0, 1000, true), 1000 + 125 + 1024);  // ESCHER-GPU's sizing
  EXPECT_EQ(d.initial_capacity(0, 1000, false), 1000);
  EXPECT_EQ(d.grown_capacity(101, 100, "t"), 150);
  EXPECT_EQ(d.grown_capacity(INT_MAX, INT_MAX - 1, "t"), INT_MAX);
  EXPECT_THROW((void)d.grown_capacity(static_cast<long long>(INT_MAX) + 1, 100, "t"),
               capacity_error);
  const store_policy small = store_policy::dyng_default(64);
  EXPECT_EQ(small.max_payload, 64);
  EXPECT_EQ(small.grown_capacity(60, 40, "t"), 64);
  EXPECT_THROW((void)small.grown_capacity(65, 40, "t"), capacity_error);
  EXPECT_THROW((void)small.initial_capacity(0, 100, true), capacity_error);
  EXPECT_EQ(store_policy::dyng_default(1LL << 40).max_payload, INT_MAX);
  store_policy q;
  ASSERT_TRUE(store_policy::from_name("dyng_default", q));
  EXPECT_EQ(q.name, "dyng_default");
}

// The growth of the operation scratch (M8a step 4, behaviour (g)): exact under escher_gpu()
// (ESCHER-GPU's reserveScratch), geometric under the MOSP_ESCHER presets (its
// ensureScratchCapacity: +n/2+1 steps from the current size, from 1 item or 4 values).
TEST(CbstStorePolicy, TheScratchGrowth) {
  const store_policy exact = store_policy::escher_gpu();
  EXPECT_EQ(exact.scratch, store_policy::scratch_rule::exact);
  EXPECT_EQ(exact.scratch_capacity(10, 4, 1, INT_MAX), 10);
  EXPECT_EQ(exact.scratch_capacity(3, 4, 1, INT_MAX), 4);
  const store_policy geometric = store_policy::mosp_escher();
  EXPECT_EQ(geometric.scratch, store_policy::scratch_rule::geometric);
  EXPECT_EQ(geometric.scratch_capacity(10, 4, 1, INT_MAX), 11);   // 4, 7, 11
  EXPECT_EQ(geometric.scratch_capacity(50, 12, 4, INT_MAX), 67);  // 12, 19, 29, 44, 67
  EXPECT_EQ(geometric.scratch_capacity(3, 4, 1, INT_MAX), 4);     // it fits: unchanged
  EXPECT_EQ(geometric.scratch_capacity(1, 0, 1, INT_MAX), 1);     // from 1
  EXPECT_EQ(geometric.scratch_capacity(5, 0, 4, INT_MAX), 7);     // from 4: 4, 7
  // Capped (MOSP_ESCHER's int loop overflowed near INT_MAX), never below the need.
  EXPECT_EQ(geometric.scratch_capacity(INT_MAX - 5, 1500000000, 1, INT_MAX), INT_MAX);
  EXPECT_EQ(geometric.scratch_capacity(100, 90, 1, 95), 100);
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

// A layout beyond the int32 offsets is a capacity_error (ESCHER-GPU truncated the offsets); the
// limit is lowered to reach it without 8 GB of rows (M8a review).
TEST(CbstFlatten, ALayoutBeyondTheLimitIsACapacityError) {
  const std::vector<std::vector<int>> rows = {{1, 2, 3}, {}, {4, 5, 6, 7}, {8}};  // 20 slots
  EXPECT_EQ(flatten_rows(rows, 20).first.size(), 20U);
  EXPECT_THROW((void)flatten_rows(rows, 19), capacity_error);
  EXPECT_THROW((void)flatten_rows({{1}}, 3), capacity_error);
  // The limit never exceeds INT_MAX.
  EXPECT_EQ(flatten_rows(rows, 1LL << 40).first.size(), 20U);
}

}  // namespace
}  // namespace dyng::detail::escher
