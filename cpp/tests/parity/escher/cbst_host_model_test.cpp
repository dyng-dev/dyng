// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_host_model_test.cpp
 * @brief dyng::testing::cbst_host_model on its own (M8a, ADR 0037): the cbst_test layout sizes of
 *        ESCHER-GPU@abcf9b2 on the model's layout function, and the store's documented rules
 *        (best fit, surplus keys and the rebuild, erase, fill overflow, unfill, growth, MOSP_ESCHER's
 *        compaction and failures, copies).
 */
#include <dyng/testing/cbst_host_model.hpp>

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <vector>

namespace dyng::testing {
namespace {

using ints = std::vector<std::int32_t>;

// The construct layout of both originals' flatten: row i padded to ((n + 4) / 4) * 4 slots, the
// values, zeros, INT_MIN in the last slot.
struct flat_rows {
  ints keys, starts, flat, counts;
};
flat_rows flatten(const std::vector<ints>& rows) {
  flat_rows f;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    f.keys.push_back(static_cast<std::int32_t>(i) + 1);
    f.starts.push_back(static_cast<std::int32_t>(f.flat.size()));
    f.counts.push_back(static_cast<std::int32_t>(rows[i].size()));
    const std::size_t padded = (rows[i].size() + 4) / 4 * 4;
    f.flat.insert(f.flat.end(), rows[i].begin(), rows[i].end());
    f.flat.resize(f.starts.back() + padded - 1, 0);
    f.flat.push_back(INT_MIN);
  }
  return f;
}

// Items as (payload, inclusive prefix sums).
struct items {
  ints keys, payload, prefix;
};
items items_of(const std::vector<ints>& rows, const ints& keys = {}) {
  items it;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    it.keys.push_back(keys.empty() ? 0 : keys[i]);
    it.payload.insert(it.payload.end(), rows[i].begin(), rows[i].end());
    it.prefix.push_back(static_cast<std::int32_t>(it.payload.size()));
  }
  return it;
}

cbst_host_model built(const cbst_host_policy& p, const std::vector<ints>& rows,
                      std::int64_t capacity) {
  cbst_host_model m(p, capacity, 4, "test");
  const flat_rows f = flatten(rows);
  const cbst_op_result r = m.construct(f.keys, f.starts, f.flat, &f.counts);
  EXPECT_EQ(r.status, cbst_op_status::ok) << r.message;
  return m;
}

ints row_of(const cbst_host_model& m, std::int32_t key) {
  cbst_key_record r;
  EXPECT_TRUE(m.record(key, r)) << key;
  return r.row;
}

cbst_key_record record_of(const cbst_host_model& m, std::int32_t key) {
  cbst_key_record r;
  EXPECT_TRUE(m.record(key, r)) << key;
  return r;
}

// --- The layout: ESCHER-GPU's tests/cbst_test.cu sizes ---------------------------------------------

TEST(CbstLayout, EverySizeUpTo1024) {
  for (std::int64_t n = 1; n <= 1024; ++n) {
    const cbst_layout_check c = check_cbst_layout(n);
    EXPECT_TRUE(c.ok()) << "n = " << n;
  }
}

TEST(CbstLayout, TheSeventeenLargeSizes) {
  // cbst_test.cu's list, 2,467,389 (coauth-DBLP's hyperedges) included, up to 2^24.
  const std::vector<std::int64_t> sizes = {4095,    4096,    4097,     32767,    32768,   65535,
                                           65536,   65537,   70000,    100000,   1000000, 2467389,
                                           8388607, 8388608, 16000000, 16777215, 16777216};
  ASSERT_EQ(sizes.size(), 17U);
  for (std::int64_t n : sizes) {
    const cbst_layout_check c = check_cbst_layout(n, n <= 2467389);
    EXPECT_TRUE(c.ok()) << "n = " << n << ": permutation " << c.not_a_permutation << ", in-order "
                        << c.not_in_order << ", not found " << c.not_found << ", offsets "
                        << c.wrong_offset << ", inverse " << c.inverse_mismatches;
  }
}

TEST(CbstLayout, TheFormula) {
  // n = 6: heap positions 0..5 hold ranks 3 1 5 0 2 4.
  const std::vector<std::int64_t> want = {3, 1, 5, 0, 2, 4};
  for (std::int64_t t = 0; t < 6; ++t) EXPECT_EQ(cbst_rank_of_position(t, 6), want[t]) << t;
  EXPECT_EQ(cbst_rank_of_position(0, 1), 0);
  EXPECT_EQ(cbst_rank_of_position(-1, 4), -1);
  EXPECT_EQ(cbst_rank_of_position(4, 4), -1);
  EXPECT_EQ(cbst_rank_of_position(0, 0), -1);
  // The 64-bit evaluation, n above 65,535 (the original's int32 product overflowed): the root of
  // 70,000 nodes has 32,767 + 4,465 nodes on its left (15 full levels and the whole last level).
  EXPECT_EQ(cbst_rank_of_position(0, 70000), 37232);
  EXPECT_EQ(cbst_rank_of_position(69999, 70000), 8928);  // the 4,465th leaf: rank 2 * 4464
}

// --- Construct ---------------------------------------------------------------------------------

TEST(CbstHostModel, ConstructLaysOutTheFlattenedRows) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{10}, {20, 21}, {30, 31, 32, 33}}, 64);
  EXPECT_EQ(m.keys(), (ints{1, 2, 3}));
  EXPECT_EQ(m.bump(), 16);
  EXPECT_EQ(m.num_records(), 3);
  const cbst_key_record r = record_of(m, 3);
  EXPECT_EQ(r.length, 8);  // ((4 + 4) / 4) * 4
  EXPECT_EQ(r.value, 8);
  EXPECT_EQ(r.occupancy, 4);
  EXPECT_EQ(r.tail_capacity, 7);
  EXPECT_EQ(r.row, (ints{30, 31, 32, 33}));
  EXPECT_EQ(m.subtree_avail(), (ints{0, 0, 0}));
  EXPECT_EQ(m.reusable_slots(), 0);
  EXPECT_EQ(m.scratch_items(), 3);
  EXPECT_EQ(m.scratch_payload(), 9);
  EXPECT_EQ(m.device_bytes(), 3 * 72 + 64 * 4 + 3 * 20 + 9 * 4);
}

TEST(CbstHostModel, ConstructOccupancyFollowsThePolicy) {
  const flat_rows f = flatten({{1, 2}, {3}});
  // ESCHER-GPU scans the payload whatever the caller passes.
  cbst_host_model a(cbst_host_policy::escher_gpu(), 32);
  ints wrong = {0, 0};
  (void)a.construct(f.keys, f.starts, f.flat, &wrong);
  EXPECT_EQ(record_of(a, 1).occupancy, 2);
  // MOSP_ESCHER takes the caller's counts, or 0 without them (the legacy behaviour).
  cbst_host_model b(cbst_host_policy::mosp_escher(), 32);
  (void)b.construct(f.keys, f.starts, f.flat, nullptr);
  EXPECT_EQ(record_of(b, 1).occupancy, 0);
  cbst_host_model c(cbst_host_policy::mosp_escher(), 32);
  (void)c.construct(f.keys, f.starts, f.flat, &f.counts);
  EXPECT_EQ(record_of(c, 1).occupancy, 2);
  // The dynG default scans when the caller has no counts.
  cbst_host_model d(cbst_host_policy::dyng_default(), 32);
  (void)d.construct(f.keys, f.starts, f.flat, nullptr);
  EXPECT_EQ(record_of(d, 2).occupancy, 1);
}

TEST(CbstHostModel, ConstructBelowThePayloadSize) {
  const flat_rows f = flatten({{1, 2}, {3}});
  // ESCHER-GPU returns without building the tree; MOSP_ESCHER throws.
  cbst_host_model a(cbst_host_policy::escher_gpu(), 4);
  EXPECT_EQ(a.construct(f.keys, f.starts, f.flat).status, cbst_op_status::ok);
  EXPECT_EQ(a.num_nodes(), 0);
  EXPECT_EQ(a.num_records(), 2);
  EXPECT_EQ(a.bump(), 8);
  cbst_host_model b(cbst_host_policy::mosp_escher(), 4);
  const cbst_op_result r = b.construct(f.keys, f.starts, f.flat, &f.counts);
  EXPECT_EQ(r.status, cbst_op_status::error);
  EXPECT_EQ(r.error_type, "escher::EscherError");
  EXPECT_EQ(b.num_records(), 0);
}

// --- Insert: best fit, surplus keys, the rebuild --------------------------------------------------

TEST(CbstHostModel, InsertReusesTheSmallestFittingSlotAndIgnoresRequestedKeys) {
  // Rows of lengths 4, 8, 4, 8 (capacities 3, 7, 3, 7).
  cbst_host_model m =
      built(cbst_host_policy::escher_gpu(), {{1}, {2, 2, 2, 2}, {3}, {4, 4, 4, 4}}, 128);
  (void)m.erase({1, 2, 3, 4});
  EXPECT_EQ(m.reusable_slots(), 4);
  // Items of sizes 5, 2, 3 (requested keys 99: ignored). Ascending size: 2 -> key 1 (capacity 3,
  // the lower key of the tie), 3 -> key 3, 5 -> key 2 (capacity 7, the lower key of the tie).
  const items it = items_of({{50, 51, 52, 53, 54}, {60, 61}, {70, 71, 72}}, {99, 99, 99});
  const cbst_op_result r = m.insert(it.keys, it.payload, it.prefix);
  ASSERT_EQ(r.status, cbst_op_status::ok) << r.message;
  EXPECT_EQ(r.reusable_slots, 4);
  EXPECT_EQ(r.mapping, (ints{2, 1, 3}));
  EXPECT_EQ(row_of(m, 1), (ints{60, 61}));
  EXPECT_EQ(row_of(m, 2), (ints{50, 51, 52, 53, 54}));
  EXPECT_EQ(row_of(m, 3), (ints{70, 71, 72}));
  // No surplus: no rebuild, the unmatched dead slot (key 4) persists.
  EXPECT_EQ(m.keys(), (ints{1, 2, 3, 4}));
  EXPECT_EQ(record_of(m, 4).avail, 1);
  EXPECT_EQ(m.reusable_slots(), 1);
}

TEST(CbstHostModel, OnlyTheFirstDItemsAreEligible) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1}, {2}, {3}}, 128);
  (void)m.erase({2});
  // D = 1: only item 0 may reuse; it does not fit (4 values, capacity 3), so both are surplus.
  const items it = items_of({{9, 9, 9, 9}, {8}});
  const cbst_op_result r = m.insert(it.keys, it.payload, it.prefix);
  EXPECT_EQ(r.mapping, (ints{4, 5}));
  // The surplus rebuild keeps only live nodes: the dead slot of key 2 is dropped.
  EXPECT_EQ(m.keys(), (ints{1, 3, 4, 5}));
  EXPECT_EQ(m.reusable_slots(), 0);
  const cbst_key_record r4 = record_of(m, 4);
  EXPECT_EQ(r4.length, 5);  // ceil(4 / 4) * 4 + 1
  EXPECT_EQ(r4.tail_capacity, 4);
  EXPECT_EQ(record_of(m, 5).length, 5);  // ceil(1 / 4) * 4 + 1
  EXPECT_EQ(r4.size, static_cast<std::int32_t>(m.bump()));
}

TEST(CbstHostModel, SurplusKeysFollowTheLastLiveKey) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1}, {2}, {3}}, 128);
  (void)m.erase({3});
  // Key 3 is dead and not reused (the item is too large): the surplus item gets key 3 again.
  const items it = items_of({{7, 7, 7, 7, 7}});
  const cbst_op_result r = m.insert(it.keys, it.payload, it.prefix);
  EXPECT_EQ(r.mapping, (ints{3}));
  EXPECT_EQ(row_of(m, 3), (ints{7, 7, 7, 7, 7}));
  EXPECT_EQ(record_of(m, 3).avail, 0);
}

TEST(CbstHostModel, AnEmptyInsertChangesNothing) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1}, {2}}, 64);
  const cbst_op_result r = m.insert({}, {}, {});
  EXPECT_EQ(r.status, cbst_op_status::ok);
  EXPECT_TRUE(r.mapping.empty());
  EXPECT_EQ(m.keys(), (ints{1, 2}));
}

// --- Erase ---------------------------------------------------------------------------------------

TEST(CbstHostModel, EraseKeepsKeyAndValuesIsIdempotentAndIgnoresMissingKeys) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1, 2}, {3}}, 64);
  (void)m.erase({1, 77});
  (void)m.erase({1});
  EXPECT_EQ(m.keys(), (ints{1, 2}));
  const cbst_key_record r = record_of(m, 1);
  EXPECT_EQ(r.avail, 1);
  EXPECT_EQ(r.row, (ints{1, 2}));  // the payload keeps the dead row's values
  EXPECT_EQ(r.occupancy, 0);
  EXPECT_EQ(m.reusable_slots(), 1);
}

// --- Append (fill) and remove_values (unfill) ------------------------------------------------------

TEST(CbstHostModel, AppendFillsTheTailThenAddsAnOverflowSegment) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1, 2}}, 64);
  // Capacity 3, two values in it: one more fits, five overflow into ((5 + 4) / 4) * 4 = 8 slots.
  const items it = items_of({{3, 4, 5, 6, 7, 8}}, {1});
  (void)m.append(it.keys, it.payload, it.prefix);
  const cbst_key_record r = record_of(m, 1);
  EXPECT_EQ(r.row, (ints{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(r.segments, 2);
  EXPECT_EQ(r.tail_base, 4);
  EXPECT_EQ(r.tail_capacity, 7);
  EXPECT_EQ(r.occupancy, 5);
  EXPECT_EQ(r.length, 4);
  EXPECT_EQ(m.bump(), 12);
}

TEST(CbstHostModel, RemoveValuesRemovesEveryOccurrenceAcrossSegments) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1, 2}}, 64);
  const items more = items_of({{3, 2, 5, 2, 7}}, {1});
  (void)m.append(more.keys, more.payload, more.prefix);
  ASSERT_EQ(row_of(m, 1), (ints{1, 2, 3, 2, 5, 2, 7}));
  const items rm = items_of({{2, 7}}, {1});
  (void)m.remove_values(rm.keys, rm.payload, rm.prefix);
  const cbst_key_record r = record_of(m, 1);
  EXPECT_EQ(r.row, (ints{1, 3, 5}));
  EXPECT_EQ(r.segments, 2);   // the chain stays
  EXPECT_EQ(r.occupancy, 1);  // the values left in the tail segment
}

TEST(CbstHostModel, RemoveValuesOnALongRemovalListUsesTheOccupancy) {
  // 40 removals: the warp-level kernel compacts the first `occupancy` slots of a one-segment row.
  std::vector<ints> rows = {{}};
  for (int v = 1; v <= 60; ++v) rows[0].push_back(v);
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), rows, 256);
  ints remove;
  for (int v = 1; v <= 40; ++v) remove.push_back(v);
  const items rm = items_of({remove}, {1});
  const cbst_op_result r = m.remove_values(rm.keys, rm.payload, rm.prefix);
  ints left;
  for (int v = 41; v <= 60; ++v) left.push_back(v);
  EXPECT_EQ(row_of(m, 1), left);
  EXPECT_EQ(r.bin_launches[6 + 1], 1);  // one warp-level unfill launch
}

// --- Capacity: ESCHER-GPU grows, MOSP_ESCHER compacts then throws ----------------------------------

TEST(CbstHostModel, EscherGpuGrowsByItsRule) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1, 2, 3}}, 8);
  ints big;
  for (int v = 10; v < 50; ++v) big.push_back(v);
  const items it = items_of({big}, {1});
  const cbst_op_result r = m.append(it.keys, it.payload, it.prefix);
  ASSERT_EQ(r.status, cbst_op_status::ok);
  EXPECT_TRUE(r.grew);
  // needed = 4 + ((40 + 4) / 4) * 4 = 48: max(48 + 12, 3 * 8 / 2) = 60.
  EXPECT_EQ(m.payload_capacity(), 60);
  ASSERT_EQ(m.events().size(), 1U);
  EXPECT_EQ(m.events()[0].what, cbst_model_event::kind::growth);
  EXPECT_EQ(m.events()[0].op_index, 1);
}

TEST(CbstHostModel, MospEscherCompactsThenRefillsThenThrows) {
  // Two rows; the first gets an overflow segment, then loses values; a fill that does not fit
  // behind the bump pointer compacts the payload first.
  cbst_host_model m = built(cbst_host_policy::mosp_escher(), {{1, 2, 3}, {4, 5, 6}}, 23);
  const items grow = items_of({{7, 8, 9, 10}}, {1});
  ASSERT_EQ(m.append(grow.keys, grow.payload, grow.prefix).status, cbst_op_status::ok);
  EXPECT_EQ(m.bump(), 16);
  const items rm = items_of({{7, 8, 9, 10}}, {1});
  (void)m.remove_values(rm.keys, rm.payload, rm.prefix);
  const items next = items_of({{11, 12, 13, 14, 15}}, {2});
  const cbst_op_result r = m.append(next.keys, next.payload, next.prefix);
  ASSERT_EQ(r.status, cbst_op_status::ok) << r.message;
  EXPECT_EQ(r.compactions, 1);
  EXPECT_FALSE(r.grew);
  EXPECT_EQ(row_of(m, 1), (ints{1, 2, 3}));
  EXPECT_EQ(row_of(m, 2), (ints{4, 5, 6, 11, 12, 13, 14, 15}));
  EXPECT_EQ(record_of(m, 1).segments, 1);
  // Nothing more fits: the next overflow throws (after compacting once more).
  ints big;
  for (int v = 100; v < 140; ++v) big.push_back(v);
  const items huge = items_of({big}, {1});
  const cbst_op_result e = m.append(huge.keys, huge.payload, huge.prefix);
  EXPECT_EQ(e.status, cbst_op_status::error);
  EXPECT_EQ(e.error_type, "escher::EscherError");
  EXPECT_EQ(e.compactions, 1);
  EXPECT_EQ(m.payload_capacity(), 23);
}

TEST(CbstHostModel, DyngDefaultCompactsFirstThenGrows) {
  cbst_host_model m = built(cbst_host_policy::dyng_default(), {{1, 2, 3}, {4, 5, 6}}, 24);
  ints big;
  for (int v = 100; v < 140; ++v) big.push_back(v);
  const items huge = items_of({big}, {1});
  const cbst_op_result r = m.append(huge.keys, huge.payload, huge.prefix);
  ASSERT_EQ(r.status, cbst_op_status::ok) << r.message;
  EXPECT_EQ(r.compactions, 1);
  EXPECT_TRUE(r.grew);
  std::vector<std::int32_t> want = {1, 2, 3};
  want.insert(want.end(), big.begin(), big.end());
  EXPECT_EQ(row_of(m, 1), want);
}

TEST(CbstHostModel, ExplicitCompactOnlyWhereThePolicyHasIt) {
  cbst_host_model a = built(cbst_host_policy::escher_gpu(), {{1}}, 16);
  EXPECT_EQ(a.compact().status, cbst_op_status::undefined);
  cbst_host_model b = built(cbst_host_policy::mosp_escher(), {{1}, {2}}, 16);
  (void)b.erase({1});
  const cbst_op_result r = b.compact();
  EXPECT_EQ(r.status, cbst_op_status::ok);
  EXPECT_EQ(r.compact_return, 8);          // the dead row keeps its first segment of 4 slots
  EXPECT_EQ(record_of(b, 1).row, ints{});  // a dead row is emptied
  EXPECT_EQ(record_of(b, 2).row, (ints{2}));
}

TEST(CbstHostModel, AFailedInsertLeavesItsInPlaceWrites) {
  // MOSP_ESCHER: the reuse is applied, then the surplus row does not fit: the exception leaves the
  // reused slot written and subtree_avail stale (it is recomputed only at the end of insert).
  cbst_host_model m = built(cbst_host_policy::mosp_escher(), {{1}, {2}}, 8);
  (void)m.erase({1});
  const items it = items_of({{9}, {5, 5, 5, 5, 5, 5, 5, 5, 5}});
  const cbst_op_result r = m.insert(it.keys, it.payload, it.prefix);
  EXPECT_EQ(r.status, cbst_op_status::error);
  EXPECT_EQ(row_of(m, 1), (ints{9}));
  EXPECT_EQ(record_of(m, 1).avail, 0);
  EXPECT_EQ(m.reusable_slots(), 1);  // stale
  EXPECT_FALSE(m.check().empty());
}

TEST(CbstHostModel, CopiesAreIndependent) {
  cbst_host_model a = built(cbst_host_policy::escher_gpu(), {{1}, {2}}, 64);
  cbst_host_model b = a;
  (void)b.erase({1});
  EXPECT_EQ(record_of(a, 1).avail, 0);
  EXPECT_EQ(record_of(b, 1).avail, 1);
  cbst_host_model c(cbst_host_policy::escher_gpu(), 8);
  c = b;
  EXPECT_EQ(c.reusable_slots(), 1);
  cbst_host_model d = std::move(c);
  EXPECT_EQ(d.keys(), (ints{1, 2}));
}

TEST(CbstHostModel, TwoItemsOnOneRowAreUndefined) {
  cbst_host_model m = built(cbst_host_policy::escher_gpu(), {{1}}, 64);
  const items it = items_of({{2}, {3}}, {1, 1});
  EXPECT_EQ(m.append(it.keys, it.payload, it.prefix).status, cbst_op_status::undefined);
}

TEST(CbstHostModel, OperationsOnAnEmptyTree) {
  // A tree of 0 records: append and remove_values find no row; insert and erase hit the
  // original's CUDA error (a read of D from no tree, a launch of 0 blocks).
  cbst_host_model a(cbst_host_policy::mosp_escher(), 64);
  (void)a.construct({}, {}, {}, nullptr);
  const items it = items_of({{1}}, {1});
  EXPECT_EQ(a.append(it.keys, it.payload, it.prefix).status, cbst_op_status::ok);
  EXPECT_EQ(a.remove_values(it.keys, it.payload, it.prefix).status, cbst_op_status::ok);
  EXPECT_EQ(a.insert(it.keys, it.payload, it.prefix).status, cbst_op_status::error);
  EXPECT_EQ(a.erase({1}).status, cbst_op_status::error);
  cbst_host_model b(cbst_host_policy::escher_gpu(), 64);
  (void)b.construct({}, {}, {}, nullptr);
  EXPECT_EQ(b.insert(it.keys, it.payload, it.prefix).status, cbst_op_status::process_exit);
  EXPECT_TRUE(b.exited());
  EXPECT_EQ(b.erase({1}).status, cbst_op_status::undefined);
}

TEST(CbstHostPolicy, Names) {
  cbst_host_policy p;
  for (const char* name : {"escher_gpu", "mosp_escher", "dyng_default"}) {
    ASSERT_TRUE(cbst_host_policy::from_name(name, p)) << name;
    EXPECT_EQ(p.name, name);
  }
  EXPECT_FALSE(cbst_host_policy::from_name("other", p));
}

}  // namespace
}  // namespace dyng::testing
