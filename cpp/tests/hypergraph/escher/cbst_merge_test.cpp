// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_merge_test.cpp
 * @brief Unit tests of the MOSP_ESCHER@4b86159 behaviours merged into the dynG CBST store (M8a step
 *        4, ADR 0039), one section per behaviour.
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

// --- (c) The caller's row counts -----------------------------------------------------------------

/// The node array of a store with its pointers as heap positions (comparable across stores).
std::vector<long long> nodes_of(const cbst_store& s) {
  std::vector<long long> out;
  for (const cbst_host_node& n : s.download().nodes)
    out.insert(out.end(), {n.index, n.value, n.length, n.size, n.occupancy, n.tail_base,
                           n.tail_capacity, n.left, n.right, n.parent});
  return out;
}

// MOSP_ESCHER's rowOccupancy (escher/structure/operations.cu@4b86159:123-132, 268-275) as a fast
// path of construct next to the scan, which stays the default: the same nodes; never its nullptr
// path (occupancy 0); counts that do not fit their rows are rejected, and in Debug builds counts
// that differ from the scan are an internal_error.
TEST(CbstMergeRowCounts, AFastPathWithTheNodesOfTheScan) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  const rows want = {iota_row(1, 3), {}, iota_row(10, 5), {7}, iota_row(20, 12)};
  layout l = make_layout(want);
  std::vector<int> counts;
  for (const auto& r : want) counts.push_back(static_cast<int>(r.size()));
  cbst_store scan(res, "scan", 1024), fast(res, "fast", 1024);
  scan.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(), static_cast<int>(l.flat.size()));
  fast.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(), static_cast<int>(l.flat.size()),
                 counts.data());
  EXPECT_EQ(nodes_of(fast), nodes_of(scan));
  EXPECT_EQ(violations(fast, want), 0);
  // Appends land after the constructed values (the reason for the counts).
  const batch b = make_batch({2, 4}, {{30, 31}, {8, 9, 10, 11}});
  fast.append(b.keys, b.payload, b.prefix);
  rows grown = want;
  grown[1] = {30, 31};
  grown[3] = {7, 8, 9, 10, 11};
  EXPECT_EQ(violations(fast, grown), 0);
  // No nullptr path, and counts that do not fit their rows are rejected before anything changes.
  cbst_store other(res, "other", 1024);
  EXPECT_THROW(other.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(),
                               static_cast<int>(l.flat.size()), nullptr),
               invalid_argument_error);
  std::vector<int> too_many = counts;
  too_many[0] = 4;  // a row of 4 slots holds 3 values
  EXPECT_THROW(other.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(),
                               static_cast<int>(l.flat.size()), too_many.data()),
               invalid_argument_error);
  std::vector<int> negative = counts;
  negative[2] = -1;
  EXPECT_THROW(other.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(),
                               static_cast<int>(l.flat.size()), negative.data()),
               invalid_argument_error);
  EXPECT_FALSE(other.poisoned());
#ifndef NDEBUG
  // Debug builds: counts that fit but differ from the payload are a broken invariant.
  std::vector<int> wrong = counts;
  wrong[2] = 2;
  EXPECT_THROW(other.construct(l.keys.data(), l.starts.data(), 5, l.flat.data(),
                               static_cast<int>(l.flat.size()), wrong.data()),
               internal_error);
#endif
}

}  // namespace
}  // namespace dyng::detail::escher
