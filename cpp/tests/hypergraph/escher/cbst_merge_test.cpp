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
  EXPECT_EQ(s.rows({1, 2}), want);    // the row reads of find (f)
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
    // The row reads of find (f).
    cbst_store s(res, "cycle", 64);
    make_cyclic(s);
    EXPECT_EQ(s.rows({1}), rows{iota_row(1, 3)});
    EXPECT_THROW((void)s.rows({1, 2}), internal_error);
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

// --- (f) The row reads of find ------------------------------------------------------------------

// ESCHER-GPU's and MOSP_ESCHER's findContents printed the rows of keys from the device; rows()
// returns them, read along the chain (chain.cuh): live keys in read order, absent and deleted keys
// empty, in the order of the keys asked for.
TEST(CbstMergeFind, TheRowsOfKeys) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "find", 256);
  rows want = {iota_row(1, 3), iota_row(10, 2), {7}};
  construct_rows(s, want);
  const batch fill = make_batch({1}, {iota_row(100, 40)});  // a chained row
  s.append(fill.keys, fill.payload, fill.prefix);
  for (int v : iota_row(100, 40)) want[0].push_back(v);
  s.erase({2});
  EXPECT_EQ(s.rows({3, 1, 2, 99, 1}), (rows{want[2], want[0], {}, {}, want[0]}));
  EXPECT_EQ(s.find({3, 2, 99})[2], -1);
  EXPECT_TRUE(s.rows({}).empty());
  cbst_store empty(res, "empty", 16);
  empty.construct(nullptr, nullptr, 0, nullptr, 0);
  EXPECT_EQ(empty.rows({1}), rows{{}});
}

// --- (g) The scratch growth -----------------------------------------------------------------------

// The operation scratch grows to an operation's size under escher_gpu() (ESCHER-GPU's
// reserveScratch, the scope of the memory gate) and geometrically under the MOSP_ESCHER presets
// (its ensureScratchCapacity, escher/structure/operations.cu@4b86159:139-165); device_bytes()
// counts what the store holds either way.
TEST(CbstMergeScratch, ExactUnderEscherGpuGeometricUnderMospEscher) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  for (const store_policy& policy :
       {store_policy::escher_gpu(), store_policy::mosp_escher_staged()}) {
    SCOPED_TRACE(policy.name);
    cbst_store s(res, "scratch", 1 << 12, 4, policy);
    construct_rows(s, {{1}, {2}, {3}, {4}});
    EXPECT_EQ(s.context().scratch_item_capacity, 4);
    EXPECT_EQ(s.context().scratch_payload_capacity, 12);
    // An append of 50 values on the 4 keys, then an insert of 10 items.
    const batch b = make_batch(
        {1, 2, 3, 4}, {iota_row(100, 13), iota_row(200, 13), iota_row(300, 12), iota_row(400, 12)});
    s.append(b.keys, b.payload, b.prefix);
    rows values;
    for (int i = 0; i < 10; ++i) values.push_back(iota_row(1000 + 5 * i, 5));
    const bool exact = policy.scratch == store_policy::scratch_rule::exact;
    EXPECT_EQ(s.context().scratch_item_capacity, 4);
    EXPECT_EQ(s.context().scratch_payload_capacity, exact ? 50 : 67);  // 12, 19, 29, 44, 67
    const batch more = make_batch({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, values);
    (void)s.insert(more.keys, more.payload, more.prefix);
    EXPECT_EQ(s.context().scratch_item_capacity, exact ? 10 : 11);  // 4, 7, 11
    EXPECT_EQ(s.device_bytes(), s.allocated_bytes());
  }
}

// --- (h) An explicit compact() ------------------------------------------------------------------

// MOSP_ESCHER's churn-small scenario (tests/unit/test_cbst_ops.cu@4b86159:351-377) under its fixed
// capacity of 2,000 slots: four rows of 8 values; every round each row gets 4 new values and loses
// its 4 oldest. Without compaction the bump pointer runs out after a few dozen rounds; with an
// explicit compact() whenever the next round might not fit (MOSP_ESCHER compacted inside the fill;
// the automatic compaction is M8a step 5), 500 rounds run in the fixed payload.
TEST(CbstMergeCompact, ChurnSmallInAFixedPayload) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "churn-small", 2000, 4, store_policy::mosp_escher_staged());
  rows want(4);
  int next = 1;
  for (auto& r : want) {
    r = iota_row(next, 8);
    next += 8;
  }
  construct_rows(s, want);
  int compactions = 0;
  for (int round = 0; round < 500; ++round) {
    if (s.context().initial_payload_size > 2000 - 4 * 16) {
      const int before = s.context().initial_payload_size;
      EXPECT_LT(s.compact(), before);
      ++compactions;
    }
    batch add, rem;
    rows adds(4), rems(4);
    for (int k = 1; k <= 4; ++k) {
      adds[static_cast<std::size_t>(k - 1)] = iota_row(next, 4);
      next += 4;
      std::vector<int>& row = want[static_cast<std::size_t>(k - 1)];
      rems[static_cast<std::size_t>(k - 1)].assign(row.begin(), row.begin() + 4);
    }
    add = make_batch({1, 2, 3, 4}, adds);
    rem = make_batch({1, 2, 3, 4}, rems);
    s.append(add.keys, add.payload, add.prefix);
    s.remove_values(rem.keys, rem.payload, rem.prefix);
    for (std::size_t k = 0; k < 4; ++k) {
      want[k].erase(want[k].begin(), want[k].begin() + 4);
      want[k].insert(want[k].end(), adds[k].begin(), adds[k].end());
    }
    if (round % 25 == 0 || round == 499) {
      ASSERT_EQ(violations(s, want, true), 0) << "round " << round;
      std::ostringstream log;
      ASSERT_EQ(check_subtree_avail(s.context(), "churn", log), 0) << log.str();
    }
  }
  EXPECT_GE(compactions, 3);  // 5: every ~100 rounds
  EXPECT_EQ(s.compactions(), compactions);
  EXPECT_GT(s.compaction_temp_bytes(), 0);
  EXPECT_FALSE(s.poisoned());
}

// The layout compact() leaves: live rows single segments of (count + 4) / 4 * 4 slots in key order
// from slot 0, deleted rows their first segment of `length` slots, emptied; the bump pointer at the
// compacted size; the keys, avail and subtree_avail unchanged.
TEST(CbstMergeCompact, TheCompactedLayout) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "layout", 256, 4, store_policy::mosp_escher_staged());
  construct_rows(s, {iota_row(1, 3), iota_row(10, 5), {7}});
  const batch fill = make_batch({1}, {iota_row(100, 6)});  // key 1 chained: 9 values
  s.append(fill.keys, fill.payload, fill.prefix);
  s.erase({2});  // key 2: length 8, deleted
  const std::vector<int> avail_before = s.download().avail;
  EXPECT_EQ(s.compact(), 12 + 8 + 4);  // 9 values: 12 slots; key 2: 8 slots; {7}: 4 slots
  const cbst_host_state h = s.download();
  EXPECT_EQ(h.initial_payload_size, 24);
  EXPECT_EQ(h.avail, avail_before);
  EXPECT_EQ(h.keys, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(h.start_offsets, (std::vector<int>{0, 12, 20}));
  for (const cbst_host_node& n : h.nodes) {
    const int want_value[] = {0, 0, 12, 20}, want_length[] = {0, 12, 8, 4};
    EXPECT_EQ(n.value, want_value[n.index]);
    EXPECT_EQ(n.length, want_length[n.index]);
    EXPECT_EQ(n.tail_base, n.value);
    EXPECT_EQ(n.tail_capacity, n.length - 1);
  }
  std::vector<int> row1 = iota_row(1, 3);
  for (int v : iota_row(100, 6)) row1.push_back(v);
  EXPECT_EQ(std::vector<int>(h.payload.begin(), h.payload.begin() + 12),
            (std::vector<int>{1, 2, 3, 100, 101, 102, 103, 104, 105, INT_MIN, 0, 0}));
  EXPECT_EQ(h.payload[12], INT_MIN);  // the deleted row, emptied
  EXPECT_EQ(violations(s, {row1, {}, {7}}), 0);
  // Its slot is reused by a best fit as before (capacity 7).
  const batch ins = make_batch({0}, {iota_row(50, 7)});
  EXPECT_EQ(s.insert(ins.keys, ins.payload, ins.prefix).item_to_key, (std::vector<int>{2}));
  EXPECT_EQ(violations(s, {row1, iota_row(50, 7), {7}}), 0);
  // An empty tree has nothing to compact.
  cbst_store empty(res, "empty", 16);
  empty.construct(nullptr, nullptr, 0, nullptr, 0);
  EXPECT_EQ(empty.compact(), 0);
}

// Compacted rows can need more slots than the payload has (a surplus row of 4 values takes 5
// slots, compacted 8): a fixed payload throws capacity_error before anything changes (the store is
// not poisoned); escher_gpu() grows it (A's growth rule) first.
TEST(CbstMergeCompact, CompactedRowsThatDoNotFit) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  for (const store_policy& policy :
       {store_policy::mosp_escher_staged(), store_policy::escher_gpu()}) {
    SCOPED_TRACE(policy.name);
    cbst_store s(res, "tight", 4 + 5 + 5, 4, policy);
    construct_rows(s, {{1}});
    const batch ins = make_batch({0, 0}, {iota_row(10, 4), iota_row(20, 4)});
    (void)s.insert(ins.keys, ins.payload, ins.prefix);  // two surplus rows of 5 slots: full
    ASSERT_EQ(s.context().initial_payload_size, 14);
    const rows want = {{1}, iota_row(10, 4), iota_row(20, 4)};
    if (policy.growth == store_policy::growth_rule::none) {
      EXPECT_THROW((void)s.compact(), capacity_error);
      EXPECT_FALSE(s.poisoned());
      EXPECT_EQ(s.context().initial_payload_size, 14);
      EXPECT_EQ(violations(s, want), 0);
    } else {
      EXPECT_EQ(s.compact(), 4 + 8 + 8);
      EXPECT_GE(s.context().fixed_size, 20);
      EXPECT_EQ(violations(s, want), 0);
    }
  }
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
