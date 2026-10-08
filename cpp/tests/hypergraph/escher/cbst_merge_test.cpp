// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_merge_test.cpp
 * @brief Unit tests of the MOSP_ESCHER@4b86159 behaviours merged into the dynG CBST store (M8a step
 *        4, ADR 0039), one section per behaviour: the fill clamps, ...
 */
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/cbst_test_support.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <climits>
#include <vector>

namespace dyng::detail::escher {
namespace {

using namespace test_support;

// --- (b) The fill clamps --------------------------------------------------------------------------

// Metadata with an occupancy above the tail capacity (which no operation of the store produces;
// only wrong caller counts could) is clamped by the fill kernels (MOSP_ESCHER's
// max(0, tail_capacity - occupancy), escher/kernel/payload.cu@4b86159:123, 178, 241; dynG clamps
// the write position too) and by the warp and block unfill kernels: nothing is written or read
// outside the row's segments (compute-sanitizer memcheck runs this test), and the other rows are
// intact.
TEST(CbstMergeFillClamps, OccupancyAboveTheTailCapacityIsClamped) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "clamps", 1 << 16);
  // Keys 1-3: full rows (3 values in 4 slots) appended to at the thread, warp and block level;
  // keys 4-5: full rows (7 values in 8 slots) unfilled at the warp and block level; key 6 last
  // (its row is the end of the payload).
  rows want = {iota_row(10, 3), iota_row(20, 3), iota_row(30, 3),
               iota_row(40, 7), iota_row(50, 7), iota_row(60, 3)};
  construct_rows(s, want);
  for (int key = 1; key <= 6; ++key) {
    cbst_node n = read_node(s, key);
    EXPECT_EQ(n.occupancy, n.tail_capacity);
    n.occupancy = n.tail_capacity + (key == 6 ? 1000 : 9);  // past the payload for key 6
    write_node(s, key, n);
  }
  const std::vector<int> small = iota_row(100, 5), medium = iota_row(200, 40),
                         large = iota_row(1000, 1100);
  const batch fill = make_batch({1, 2, 3}, {small, medium, large});
  s.append(fill.keys, fill.payload, fill.prefix);
  want[0].insert(want[0].end(), small.begin(), small.end());
  want[1].insert(want[1].end(), medium.begin(), medium.end());
  want[2].insert(want[2].end(), large.begin(), large.end());
  // Warp (40 removals) and block (1100 removals) unfill of single-segment rows: two values of
  // each go, the clamped occupancy keeps the compaction inside the segment.
  std::vector<int> drop_warp = iota_row(5000, 38), drop_block = iota_row(9000, 1098);
  drop_warp.insert(drop_warp.end(), {40, 46});
  drop_block.insert(drop_block.end(), {50, 53});
  const batch unfill = make_batch({4, 5}, {drop_warp, drop_block});
  s.remove_values(unfill.keys, unfill.payload, unfill.prefix);
  want[3] = {41, 42, 43, 44, 45};
  want[4] = {51, 52, 54, 55, 56};
  // Key 6 appended to past the end of the payload: clamped, it overflows into a new segment.
  const batch last = make_batch({6}, {{7, 8}});
  s.append(last.keys, last.payload, last.prefix);
  want[5].insert(want[5].end(), {7, 8});
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  // Every row reads back completely; the metadata is consistent again where the rows changed.
  EXPECT_EQ(violations(s, want), 0);
  EXPECT_FALSE(s.poisoned());
}

}  // namespace
}  // namespace dyng::detail::escher
