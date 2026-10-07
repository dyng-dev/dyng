// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_store_test.cpp
 * @brief Unit tests of the dynG CBST store (cpp/src/hypergraph/escher; M8a step 3, ADR 0038): the
 *        dynG additions (the empty tree, ...), the error policy and the resources the store runs
 *        on. Rows are checked with the store's own integrity checker (ESCHER-GPU's checkTreeRows:
 *        every expected row found by a BST search from the root, reachability, tail metadata,
 *        disjoint segments).
 */
#include "hypergraph/escher/cbst_store.hpp"

#include "hypergraph/escher/integrity.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::detail::escher {
namespace {

using rows = std::vector<std::vector<int>>;

/// The violations the integrity checker finds against `expected` (rows of keys 1..n).
int violations(const cbst_store& s, const rows& expected, bool order_insensitive = false) {
  std::ostringstream log;
  const int v = check_tree_rows(s.context(), expected, order_insensitive, "test", log);
  if (v != 0) ADD_FAILURE() << log.str();
  return v;
}

/// Keys, payload and prefix sizes of a batch of rows.
struct batch {
  std::vector<int> keys, payload, prefix;
};

batch make_batch(const std::vector<int>& keys, const rows& values) {
  batch b;
  b.keys = keys;
  for (const auto& r : values) {
    b.payload.insert(b.payload.end(), r.begin(), r.end());
    b.prefix.push_back(static_cast<int>(b.payload.size()));
  }
  return b;
}

/// An empty tree: construct with 0 records.
void construct_empty(cbst_store& s) {
  std::vector<int> none;
  s.construct(none.data(), none.data(), 0, none.data(), 0);
}

// --- The empty tree (dynG addition) -----------------------------------------------------------

TEST(CbstStoreEmptyTree, InsertBuildsTheTree) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  for (int capacity : {0, 4, 64}) {
    SCOPED_TRACE(capacity);
    cbst_store s(res, "empty", capacity);
    construct_empty(s);
    EXPECT_EQ(violations(s, {}), 0);
    const batch b = make_batch({1, 2, 3}, {{5, 6}, {7}, {8, 9, 10}});
    const insert_mapping m = s.insert(b.keys, b.payload, b.prefix);
    EXPECT_EQ(m.item_to_key, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(s.context().num_records, 3);
    EXPECT_EQ(violations(s, {{5, 6}, {7}, {8, 9, 10}}), 0);
    // The tree is an ordinary tree from now on.
    const batch more = make_batch({2}, {{11, 12, 13, 14, 15}});
    s.append(more.keys, more.payload, more.prefix);
    EXPECT_EQ(violations(s, {{5, 6}, {7, 11, 12, 13, 14, 15}, {8, 9, 10}}), 0);
    s.erase({1});
    EXPECT_EQ(violations(s, {{}, {7, 11, 12, 13, 14, 15}, {8, 9, 10}}), 0);
    const batch again = make_batch({0}, {{20}});
    EXPECT_EQ(s.insert(again.keys, again.payload, again.prefix).item_to_key,
              (std::vector<int>{1}));  // the deleted slot of key 1, reused
    EXPECT_EQ(violations(s, {{20}, {7, 11, 12, 13, 14, 15}, {8, 9, 10}}), 0);
    EXPECT_FALSE(s.poisoned());
  }
}

TEST(CbstStoreEmptyTree, AppendEraseAndRemoveValuesChangeNothing) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "empty", 16);
  construct_empty(s);
  const batch b = make_batch({1, 2}, {{1, 2, 3}, {4}});
  s.append(b.keys, b.payload, b.prefix);
  s.remove_values(b.keys, b.payload, b.prefix);
  s.erase({1, 2});
  EXPECT_EQ(s.context().num_records, 0);
  EXPECT_EQ(s.context().initial_payload_size, 0);
  EXPECT_EQ(violations(s, {}), 0);
  // ... and an insert still works afterwards.
  EXPECT_EQ(s.insert(b.keys, b.payload, b.prefix).item_to_key, (std::vector<int>{1, 2}));
  EXPECT_EQ(violations(s, {{1, 2, 3}, {4}}), 0);
}

TEST(CbstStoreEmptyTree, OperationsBeforeConstructThrow) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "unconstructed", 16);
  const batch b = make_batch({1}, {{1}});
  EXPECT_THROW((void)s.insert(b.keys, b.payload, b.prefix), invalid_argument_error);
  EXPECT_THROW(s.append(b.keys, b.payload, b.prefix), invalid_argument_error);
  EXPECT_THROW(s.erase({1}), invalid_argument_error);
  EXPECT_THROW(s.remove_values(b.keys, b.payload, b.prefix), invalid_argument_error);
  EXPECT_FALSE(s.poisoned());
}

}  // namespace
}  // namespace dyng::detail::escher

namespace dyng::detail::escher {
namespace {

/// A tree with reused, chained (fill overflow), grown and deleted rows.
rows build_busy_tree(cbst_store& s) {
  rows want = {{1, 2, 3}, {4}, {5, 6, 7, 8, 9}, {10}};
  std::vector<int> keys = {1, 2, 3, 4};
  std::vector<int> flat, starts;
  for (const auto& r : want) {
    starts.push_back(static_cast<int>(flat.size()));
    const std::size_t padded = (r.size() + 4) / 4 * 4;
    flat.insert(flat.end(), r.begin(), r.end());
    flat.insert(flat.end(), padded - r.size() - 1, 0);
    flat.push_back(INT_MIN);
  }
  s.construct(keys.data(), starts.data(), 4, flat.data(), static_cast<int>(flat.size()));
  std::vector<int> big(40);
  for (int i = 0; i < 40; ++i) big[static_cast<std::size_t>(i)] = 100 + i;
  const batch fill = make_batch({2, 4}, {big, {11, 12}});
  s.append(fill.keys, fill.payload, fill.prefix);  // key 2 overflows into a chained segment
  want[1].insert(want[1].end(), big.begin(), big.end());
  want[3].insert(want[3].end(), {11, 12});
  s.erase({3});
  want[2].clear();
  const batch ins = make_batch({0, 0}, {{50, 51}, {52, 53, 54, 55, 56, 57, 58}});
  const insert_mapping m = s.insert(ins.keys, ins.payload, ins.prefix);
  EXPECT_EQ(m.item_to_key, (std::vector<int>{3, 5}));  // reuse of key 3, then a surplus row
  want[2] = {50, 51};
  want.push_back({52, 53, 54, 55, 56, 57, 58});
  s.remove_values({2}, {100, 4}, {2});
  want[1].erase(want[1].begin(), want[1].begin() + 2);
  return want;
}

// --- Store clone (dynG addition) -----------------------------------------------------------------

TEST(CbstStoreClone, EqualsItsSourceAndIsIndependent) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store source(res, "source", 64);
  const rows want = build_busy_tree(source);
  ASSERT_EQ(violations(source, want, true), 0);
  cbst_store copy = source.clone();
  EXPECT_EQ(violations(copy, want, true), 0);
  EXPECT_EQ(copy.context().num_records, source.context().num_records);
  EXPECT_EQ(copy.context().initial_payload_size, source.context().initial_payload_size);
  EXPECT_EQ(copy.context().fixed_size, source.context().fixed_size);
  EXPECT_EQ(copy.device_bytes(), source.device_bytes());
  EXPECT_NE(copy.context().d_nodes, source.context().d_nodes);
  // Changing the copy leaves the source as it was, and the other way round.
  const batch more = make_batch({1, 5}, {{200, 201, 202, 203, 204, 205}, {206}});
  copy.append(more.keys, more.payload, more.prefix);
  copy.erase({4});
  rows changed = want;
  changed[0].insert(changed[0].end(), {200, 201, 202, 203, 204, 205});
  changed[4].push_back(206);
  changed[3].clear();
  EXPECT_EQ(violations(copy, changed, true), 0);
  EXPECT_EQ(violations(source, want, true), 0);
  source.erase({1});
  rows source_rows = want;
  source_rows[0].clear();
  EXPECT_EQ(violations(source, source_rows, true), 0);
  EXPECT_EQ(violations(copy, changed, true), 0);
}

TEST(CbstStoreClone, OntoAnotherStream) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store source(res, "source", 64);
  const rows want = build_busy_tree(source);
  cudaStream_t other = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), cudaSuccess);
  {
    const resources res2 = resources::cuda(res.device(), stream_ref(other));
    cbst_store copy = source.clone(res2);
    EXPECT_EQ(violations(copy, want, true), 0);
    cbst_store copy_of_copy = copy.clone(res);
    EXPECT_EQ(violations(copy_of_copy, want, true), 0);
  }
  ASSERT_EQ(cudaStreamSynchronize(other), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(other), cudaSuccess);
}

}  // namespace
}  // namespace dyng::detail::escher
