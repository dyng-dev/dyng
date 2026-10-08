// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_merge_test.cpp
 * @brief Unit tests of the MOSP_ESCHER@4b86159 behaviours merged into the dynG CBST store (M8a step
 *        4, ADR 0039), one section per behaviour.
 */
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/cbst_test_support.hpp"
#include "hypergraph/escher/integrity.hpp"
#include "hypergraph/escher/row_view.cuh"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <sstream>
#include <string>
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

// --- (d) The bounded chain walk (chain.cuh) ------------------------------------------------------

// A row of 2,000 segments (each append of 3 values to a full tail adds one): more than
// MOSP_ESCHER's literal 1,024 hops, which would truncate it; the walk bounded by the used payload
// reads it completely, in the integrity checker, the row locator and the unfill writer.
TEST(CbstMergeChain, TwoThousandOverflowingAppendsReadBackCompletely) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "long chain", 64);
  rows want = {iota_row(1, 3), iota_row(100, 2)};
  construct_rows(s, want);
  int next = 1000;
  for (int i = 0; i < 2000; ++i, next += 3) {
    const std::vector<int> three = iota_row(next, 3);
    const batch b = make_batch({1}, {three});
    s.append(b.keys, b.payload, b.prefix);
    want[0].insert(want[0].end(), three.begin(), three.end());
  }
  ASSERT_EQ(want[0].size(), 6003U);
  EXPECT_EQ(violations(s, want), 0);  // in read order
  // The row has 2,001 segments: 2,000 chain hops.
  const cbst_host_state h = s.download();
  long long hops = 0;
  for (const cbst_host_node& n : h.nodes) {
    if (n.index != 1) continue;
    for (long long loc = n.value;;) {
      const int v = h.payload[static_cast<std::size_t>(loc)];
      if (v == 0 || v == INT_MIN) break;
      if (v < 0) {
        ++hops;
        loc = -static_cast<long long>(v);
      } else {
        ++loc;
      }
    }
  }
  EXPECT_EQ(hops, 2000);
  // The row locator's sorted copy (V2H rows are read so) has every value.
  row_locator view;
  view.build_sorted_copy(s.context(), 2);
  int length = 0;
  ASSERT_EQ(cudaMemcpy(&length, view.view().len + 1, sizeof length, cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(length, 6003);
  // The serial unfill walks the 2,000 segments: values of the first, a middle and the last go.
  const batch drop = make_batch({1}, {{2, 3001, next - 1}});
  s.remove_values(drop.keys, drop.payload, drop.prefix);
  for (int v : {2, 3001, next - 1}) want[0].erase(std::find(want[0].begin(), want[0].end(), v));
  s.check_errors();
  EXPECT_EQ(violations(s, want), 0);
  EXPECT_FALSE(s.poisoned());
}

/// A store whose row of key 2 (slots 4..7: 4, 5, 0, INT_MIN) is made a cyclic chain: slot 6
/// points back to slot 4.
void make_cyclic(cbst_store& s) {
  construct_rows(s, {iota_row(1, 3), {4, 5}});
  write_payload(s, 6, -4);
}

// A cyclic chain raises the store's device error word: the walks stop (no hang), nothing is
// truncated silently, the host throws internal_error at the next read point and the store is
// poisoned.
TEST(CbstMergeChain, ACyclicChainRaisesTheErrorWithoutAHang) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  const batch drop = make_batch({2}, {{5}});
  {
    // The unfill writer (remove_values has no read point of its own: check_errors()).
    cbst_store s(res, "cycle", 64);
    make_cyclic(s);
    s.remove_values(drop.keys, drop.payload, drop.prefix);
    EXPECT_THROW(s.check_errors(), internal_error);
    EXPECT_TRUE(s.poisoned());
    EXPECT_THROW(s.erase({1}), invalid_argument_error);
  }
  {
    // ... or the next operation's read point (here the fill's total).
    cbst_store s(res, "cycle", 64);
    make_cyclic(s);
    s.remove_values(drop.keys, drop.payload, drop.prefix);
    const batch more = make_batch({1}, {{9}});
    EXPECT_THROW(s.append(more.keys, more.payload, more.prefix), internal_error);
    EXPECT_TRUE(s.poisoned());
  }
  {
    // The row locator's chained reads.
    cbst_store s(res, "cycle", 64);
    make_cyclic(s);
    row_locator view;
    EXPECT_THROW(view.build_sorted_copy(s.context(), 2), internal_error);
  }
  {
    // The integrity checker (host) reports the row as unterminated.
    cbst_store s(res, "cycle", 64);
    make_cyclic(s);
    std::ostringstream log;
    EXPECT_GT(check_tree_rows(s.context(), {iota_row(1, 3), {4, 5}}, false, "cycle", log), 0);
    EXPECT_NE(log.str().find("unterminated"), std::string::npos) << log.str();
  }
  {
    // A chain pointer past the used payload is an overrun too.
    cbst_store s(res, "past the end", 64);
    construct_rows(s, {iota_row(1, 3), {4, 5}});
    write_payload(s, 6, -40);
    s.remove_values(drop.keys, drop.payload, drop.prefix);
    EXPECT_THROW(s.check_errors(), internal_error);
  }
}

// MOSP_ESCHER's unfill-chain scenario (tests/unit/test_cbst_ops.cu@4b86159:268-281): unfill on
// a chained row, then fill (the tail occupancy was wrong in the original ESCHER code).
TEST(CbstMergeChain, MospEschersUnfillChainScenario) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "unfill-chain", 4096);
  construct_rows(s, {{1, 2}, {50}});
  auto fill = [&](const std::vector<int>& v) {
    const batch b = make_batch({1}, {v});
    s.append(b.keys, b.payload, b.prefix);
  };
  auto unfill = [&](const std::vector<int>& v) {
    const batch b = make_batch({1}, {v});
    s.remove_values(b.keys, b.payload, b.prefix);
  };
  fill({3, 4, 5});  // overflows into a chained segment
  EXPECT_EQ(violations(s, {{1, 2, 3, 4, 5}, {50}}), 0);
  unfill({1});  // removal from the head segment
  EXPECT_EQ(violations(s, {{2, 3, 4, 5}, {50}}), 0);
  fill({6});
  EXPECT_EQ(violations(s, {{2, 3, 4, 5, 6}, {50}}), 0);
  unfill({5, 6});
  fill({7, 8, 9, 10, 11, 12, 13});
  EXPECT_EQ(violations(s, {{2, 3, 4, 7, 8, 9, 10, 11, 12, 13}, {50}}), 0);
  s.check_errors();
  EXPECT_FALSE(s.poisoned());
}

// --- (e) The integrity checks --------------------------------------------------------------------

/// A tree with chained, deleted, reused and surplus rows.
rows busy(cbst_store& s) {
  rows want = {iota_row(1, 3), {9, 7, 8}, iota_row(20, 5), {4}};
  construct_rows(s, want);
  const batch fill = make_batch({2, 4}, {iota_row(50, 9), {3, 2, 1}});
  s.append(fill.keys, fill.payload, fill.prefix);
  for (int v : iota_row(50, 9)) want[1].push_back(v);
  want[3].insert(want[3].end(), {3, 2, 1});
  s.erase({3});
  want[2].clear();
  const batch ins = make_batch({0, 0}, {{6, 5}, iota_row(70, 6)});
  (void)s.insert(ins.keys, ins.payload, ins.prefix);
  want[2] = {6, 5};
  want.push_back(iota_row(70, 6));
  return want;
}

TEST(CbstMergeIntegrity, EveryCheckIsCleanOnAValidTree) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "valid", 256);
  const rows want = busy(s);
  std::ostringstream log;
  EXPECT_EQ(violations(s, want), 0);
  EXPECT_EQ(check_subtree_avail(s.context(), "valid", log), 0) << log.str();
  EXPECT_EQ(check_tree_index(s.context(), "valid", log), 0) << log.str();
  s.erase({1, 4});
  EXPECT_EQ(check_subtree_avail(s.context(), "valid", log), 0) << log.str();
  EXPECT_EQ(check_tree_index(s.context(), "valid", log), 0) << log.str();
}

// Order-insensitive rows (V2H) are compared as sorted multisets on both sides
// (escher/structure/integrity.cu@4b86159:131-134): unsorted expected rows are no failure.
TEST(CbstMergeIntegrity, UnsortedExpectedRowsGiveNoFalseFailure) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "v2h", 256);
  rows want = busy(s);
  for (auto& r : want) std::reverse(r.begin(), r.end());
  std::ostringstream log;
  EXPECT_EQ(check_tree_rows(s.context(), want, true, "v2h", log), 0) << log.str();
  // In order they differ (the rows are read in their order).
  EXPECT_GT(check_tree_rows(s.context(), want, false, "v2h", log), 0);
  // A missing value is a failure as a multiset too.
  want[0].pop_back();
  EXPECT_GT(check_tree_rows(s.context(), want, true, "v2h", log), 0);
}

TEST(CbstMergeIntegrity, ACorruptedSubtreeAvailEntryIsCaught) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "corrupt", 256);
  (void)busy(s);
  s.erase({2});
  std::ostringstream log;
  ASSERT_EQ(check_subtree_avail(s.context(), "corrupt", log), 0) << log.str();
  int wrong = 7;
  ASSERT_EQ(
      cudaMemcpy(s.context().d_subtree_avail + 1, &wrong, sizeof wrong, cudaMemcpyHostToDevice),
      cudaSuccess);
  // Node 1 and its parent, the root, are inconsistent.
  EXPECT_EQ(check_subtree_avail(s.context(), "corrupt", log), 2);
  EXPECT_NE(log.str().find("subtree_avail[1] = 7"), std::string::npos) << log.str();
}

TEST(CbstMergeIntegrity, TheIndexArraysAreCheckedAgainstTheNodes) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  // A key, a row offset and an avail flag overwritten in turn.
  for (int which = 0; which < 3; ++which) {
    SCOPED_TRACE(which);
    cbst_store s(res, "index", 256);
    (void)busy(s);
    const cbst_context& c = s.context();
    int* array = which == 0 ? c.d_keys : (which == 1 ? c.d_start_offsets : c.d_avail);
    const int value = which == 0 ? 1000 : (which == 1 ? 3 : 2);
    ASSERT_EQ(cudaMemcpy(array + 2, &value, sizeof value, cudaMemcpyHostToDevice), cudaSuccess);
    std::ostringstream log;
    EXPECT_GT(check_tree_index(c, "index", log), 0) << log.str();
  }
}

}  // namespace
}  // namespace dyng::detail::escher
