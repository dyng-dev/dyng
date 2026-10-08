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
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::detail::escher {
namespace {

using rows = std::vector<std::vector<int>>;

/// The violations the integrity checker finds against `expected` (rows of keys 1..n).
long long violations(const cbst_store& s, const rows& expected, bool order_insensitive = false) {
  std::ostringstream log;
  const long long v = check_tree_rows(s.context(), expected, order_insensitive, "test", log);
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

TEST(CbstStoreEmptyTree, EraseChangesNothingAppendAndRemoveValuesNameAbsentKeys) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  const batch b = make_batch({1, 2}, {{1, 2, 3}, {4}});
  {
    cbst_store s(res, "empty", 16);
    construct_empty(s);
    s.erase({1, 2});  // absent keys: erase ignores them (both originals)
    s.check_errors();
    EXPECT_EQ(s.context().num_records, 0);
    EXPECT_EQ(s.context().initial_payload_size, 0);
    EXPECT_EQ(violations(s, {}), 0);
    // ... and an insert still works afterwards.
    EXPECT_EQ(s.insert(b.keys, b.payload, b.prefix).item_to_key, (std::vector<int>{1, 2}));
    EXPECT_EQ(violations(s, {{1, 2, 3}, {4}}), 0);
  }
  // Appending to or removing values from absent keys is the error of M8a step 4 (ADR 0039; the
  // originals ignored the items silently): nothing is written, and the read point throws.
  {
    cbst_store s(res, "empty", 16);
    construct_empty(s);
    EXPECT_THROW(s.append(b.keys, b.payload, b.prefix), internal_error);
    EXPECT_EQ(s.stats().rejected_items, 2);
    EXPECT_EQ(s.context().initial_payload_size, 0);
  }
  {
    cbst_store s(res, "empty", 16);
    construct_empty(s);
    s.remove_values(b.keys, b.payload, b.prefix);
    EXPECT_THROW(s.check_errors(), internal_error);
  }
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

namespace dyng::detail::escher {
namespace {

// --- Download of the state (dynG addition) -------------------------------------------------------

/// The rows of a downloaded tree by key (BST search from the root, the chain followed).
std::vector<std::vector<int>> rows_of(const cbst_host_state& h, int max_key) {
  std::vector<std::vector<int>> out(static_cast<std::size_t>(max_key));
  for (int key = 1; key <= max_key; ++key) {
    long long t = h.nodes.empty() ? -1 : 0;
    while (t >= 0 && h.nodes[static_cast<std::size_t>(t)].index != key)
      t = h.nodes[static_cast<std::size_t>(t)].index > key
              ? h.nodes[static_cast<std::size_t>(t)].left
              : h.nodes[static_cast<std::size_t>(t)].right;
    if (t < 0 || h.avail[static_cast<std::size_t>(t)] != 0) continue;
    long long loc = h.nodes[static_cast<std::size_t>(t)].value;
    for (;;) {
      const int v = h.payload[static_cast<std::size_t>(loc)];
      if (v < 0 && v != INT_MIN) {
        loc = -static_cast<long long>(v);
        continue;
      }
      if (v == 0 || v == INT_MIN) break;
      out[static_cast<std::size_t>(key - 1)].push_back(v);
      ++loc;
    }
  }
  return out;
}

TEST(CbstStoreDownload, HoldsTheWholeState) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "download", 64);
  rows want = build_busy_tree(s);
  const cbst_host_state h = s.download();
  const cbst_context& c = s.context();
  EXPECT_EQ(h.num_records, c.num_records);
  EXPECT_EQ(h.fixed_size, c.fixed_size);
  EXPECT_EQ(h.initial_payload_size, c.initial_payload_size);
  EXPECT_EQ(h.alignment, 4);
  EXPECT_EQ(h.scratch_item_capacity, c.scratch_item_capacity);
  EXPECT_EQ(h.scratch_payload_capacity, c.scratch_payload_capacity);
  ASSERT_EQ(h.nodes.size(), static_cast<std::size_t>(c.num_records));
  EXPECT_EQ(h.payload.size(), static_cast<std::size_t>(c.initial_payload_size));
  EXPECT_EQ(h.keys, (std::vector<int>{1, 2, 3, 4, 5}));
  // The heap layout of the pointers, as positions.
  const auto n = static_cast<long long>(h.nodes.size());
  for (long long t = 0; t < n; ++t) {
    const cbst_host_node& x = h.nodes[static_cast<std::size_t>(t)];
    EXPECT_EQ(x.left, 2 * t + 1 < n ? 2 * t + 1 : -1);
    EXPECT_EQ(x.right, 2 * t + 2 < n ? 2 * t + 2 : -1);
    EXPECT_EQ(x.parent, t == 0 ? -1 : (t - 1) / 2);
  }
  // subtree_avail: no deleted slot is left (the one erased was reused).
  for (int v : h.subtree_avail) EXPECT_EQ(v, 0);
  rows got = rows_of(h, 5);
  for (auto& r : got) std::sort(r.begin(), r.end());
  for (auto& r : want) std::sort(r.begin(), r.end());
  EXPECT_EQ(got, want);
}

}  // namespace
}  // namespace dyng::detail::escher

namespace dyng::detail::escher {
namespace {

/// A device memory resource that counts the live bytes and their peak (and can be told to fail
/// an allocation).
class counting_resource {
 public:
  explicit counting_resource(int device) : upstream_(default_device_memory_resource(device)) {}
  void* allocate(stream_ref s, std::size_t bytes, std::size_t alignment) {
    if (fail_after_ >= 0 && allocations_ >= fail_after_)
      throw out_of_memory_error("counting_resource: injected allocation failure");
    void* p = upstream_.allocate(s, bytes, alignment);
    ++allocations_;
    live_ += static_cast<long long>(bytes);
    peak_ = std::max(peak_, live_);
    return p;
  }
  void deallocate(stream_ref s, void* p, std::size_t bytes, std::size_t alignment) noexcept {
    live_ -= static_cast<long long>(bytes);
    upstream_.deallocate(s, p, bytes, alignment);
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocate(stream_ref(), bytes, alignment);
  }
  void deallocate_sync(void* p, std::size_t bytes, std::size_t alignment) noexcept {
    deallocate(stream_ref(), p, bytes, alignment);
  }
  [[nodiscard]] memory_space space() const noexcept {
    return memory_space::device;
  }
  long long live() const {
    return live_;
  }
  long long peak() const {
    return peak_;
  }
  long long allocations() const {
    return allocations_;
  }
  /// Fail every allocation after the next `n` (-1: never).
  void fail_after(long long n) {
    fail_after_ = n < 0 ? -1 : allocations_ + n;
  }

 private:
  memory_resource_ref upstream_;
  long long live_ = 0, peak_ = 0, allocations_ = 0, fail_after_ = -1;
};

// --- device_bytes() in ESCHER-GPU's accounting scope (dynG addition) ------------------------------

TEST(CbstStoreDeviceBytes, IsWhatTheStoreHolds) {
  DYNG_SKIP_IF_NO_CUDA();
  resources res = resources::cuda();
  counting_resource counter(res.device());
  res.set_memory_resource(counter);
  {
    cbst_store s(res, "bytes", 64);
    // Created, not constructed: ESCHER-GPU's deviceBytes() counts the payload capacity already.
    EXPECT_EQ(s.device_bytes(), 64 * sizeof(int));
    EXPECT_EQ(s.allocated_bytes(), 0U);
    auto check = [&](const char* what) {
      SCOPED_TRACE(what);
      ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(res.stream().get())), cudaSuccess);
      EXPECT_EQ(s.allocated_bytes(), s.device_bytes());
      // The temporaries of an operation are gone when it returns: the store holds exactly its
      // arrays.
      // (The store's control block, the device error word and its counters of M8a step 4, is
      // outside ESCHER-GPU's accounting scope.)
      EXPECT_EQ(counter.live(), static_cast<long long>(s.allocated_bytes() +
                                                       control_words * sizeof(std::uint32_t)));
    };
    // construct, fill with an overflow, erase, reuse and surplus, unfill
    (void)build_busy_tree(s);
    check("after the busy tree");
    const batch more = make_batch({1, 2, 3, 4, 5}, {{7}, {8}, {9}, {10}, {11}});
    s.append(more.keys, more.payload, more.prefix);
    check("after append");
    s.erase({1, 2});
    check("after erase");
    const batch big = make_batch({0, 0, 0}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1}, {2}});
    (void)s.insert(big.keys, big.payload, big.prefix);
    check("after insert");
    EXPECT_GT(counter.peak(), counter.live());  // the temporaries existed during the operations
  }
  EXPECT_EQ(counter.live(), 0);
}

}  // namespace
}  // namespace dyng::detail::escher

namespace dyng::detail::escher {
namespace {

// --- The error policy -----------------------------------------------------------------------------

TEST(CbstStoreErrors, ACapacityBelowThePayloadIsRejectedBeforeAnythingChanges) {
  DYNG_SKIP_IF_NO_CUDA();
  resources res = resources::cuda();
  counting_resource counter(res.device());
  res.set_memory_resource(counter);
  cbst_store s(res, "small", 8);
  std::vector<int> keys = {1, 2}, starts = {0, 4};
  std::vector<int> flat = {1, 2, 3, INT_MIN, 4, 0, 0, INT_MIN};
  cbst_store small(res, "smaller", 7);
  const long long created = counter.allocations();  // the stores' control blocks
  EXPECT_THROW(small.construct(keys.data(), starts.data(), 2, flat.data(), 8),
               invalid_argument_error);
  EXPECT_EQ(counter.allocations(), created);  // nothing was allocated
  EXPECT_FALSE(small.poisoned());
  EXPECT_EQ(small.context().num_records, 0);
  s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
  EXPECT_EQ(violations(s, {{1, 2, 3}, {4}}), 0);
}

// MOSP_ESCHER's construct validation (escher/structure/operations.cu@4b86159:196-211, M8a step
// 4): every invalid argument is rejected before anything is allocated or changed.
TEST(CbstStoreErrors, TheConstructValidation) {
  DYNG_SKIP_IF_NO_CUDA();
  resources res = resources::cuda();
  counting_resource counter(res.device());
  res.set_memory_resource(counter);
  std::vector<int> keys = {1, 2}, starts = {0, 4};
  std::vector<int> flat = {1, 2, 3, INT_MIN, 4, 0, 0, INT_MIN};
  cbst_store s(res, "validated", 8);
  // Negative sizes.
  EXPECT_THROW(s.construct(keys.data(), starts.data(), -1, flat.data(), 8), invalid_argument_error);
  EXPECT_THROW(s.construct(keys.data(), starts.data(), 2, flat.data(), -8), invalid_argument_error);
  cbst_store negative(res, "negative", -1);
  EXPECT_THROW(negative.construct(keys.data(), starts.data(), 2, flat.data(), 8),
               invalid_argument_error);
  // A capacity below the payload, also for a tree of 0 records.
  cbst_store small(res, "small", 4);
  const long long created = counter.allocations();  // the stores' control blocks
  EXPECT_THROW(small.construct(nullptr, nullptr, 0, flat.data(), 8), invalid_argument_error);
  // Missing arrays.
  EXPECT_THROW(s.construct(nullptr, starts.data(), 2, flat.data(), 8), invalid_argument_error);
  EXPECT_THROW(s.construct(keys.data(), nullptr, 2, flat.data(), 8), invalid_argument_error);
  EXPECT_THROW(s.construct(keys.data(), starts.data(), 2, nullptr, 8), invalid_argument_error);
  // A record count beyond the int32 indexing of the operation scratch (3 ints per record).
  EXPECT_THROW(s.construct(keys.data(), starts.data(), INT_MAX / 3 + 1, flat.data(), 8),
               capacity_error);
  EXPECT_EQ(counter.allocations(), created);  // nothing was allocated
  for (const cbst_store* t : {&s, &negative, &small}) {
    EXPECT_FALSE(t->poisoned());
    EXPECT_EQ(t->context().num_records, 0);
  }
  // Valid arguments still construct (and an empty tree needs no arrays).
  s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
  EXPECT_EQ(violations(s, {{1, 2, 3}, {4}}), 0);
  cbst_store empty(res, "empty", 0);
  empty.construct(nullptr, nullptr, 0, nullptr, 0);
  EXPECT_EQ(violations(empty, {}), 0);
}

TEST(CbstStoreErrors, AFailureAfterInPlaceWritesPoisonsTheStore) {
  DYNG_SKIP_IF_NO_CUDA();
  resources res = resources::cuda();
  counting_resource counter(res.device());
  res.set_memory_resource(counter);
  std::vector<int> big(100);
  for (int i = 0; i < 100; ++i) big[static_cast<std::size_t>(i)] = i + 1;
  // An append that overflows key 1 and grows the payload: its kernels write in place before the
  // growth allocates. Fail the k-th allocation of the operation, for every k until it succeeds.
  bool failed_late = false;
  for (int k = 0; k < 32; ++k) {
    SCOPED_TRACE(k);
    cbst_store s(res, "poison", 8);
    std::vector<int> keys = {1, 2}, starts = {0, 4};
    std::vector<int> flat = {1, 2, 3, INT_MIN, 4, 0, 0, INT_MIN};
    s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
    const batch b = make_batch({1}, {big});
    counter.fail_after(k);
    bool threw = false;
    try {
      s.append(b.keys, b.payload, b.prefix);
    } catch (const out_of_memory_error&) {
      threw = true;
    }
    counter.fail_after(-1);
    if (!threw) {
      EXPECT_FALSE(s.poisoned());
      ASSERT_GT(k, 0);
      break;
    }
    failed_late = true;
    // Every later operation throws, until the store is constructed again.
    EXPECT_TRUE(s.poisoned());
    EXPECT_THROW(s.erase({1}), invalid_argument_error);
    EXPECT_THROW(s.append(b.keys, b.payload, b.prefix), invalid_argument_error);
    EXPECT_THROW((void)s.insert(b.keys, b.payload, b.prefix), invalid_argument_error);
    EXPECT_THROW(s.remove_values(b.keys, b.payload, b.prefix), invalid_argument_error);
    EXPECT_THROW((void)s.clone(), invalid_argument_error);
    s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
    EXPECT_FALSE(s.poisoned());
    EXPECT_EQ(violations(s, {{1, 2, 3}, {4}}), 0);
  }
  EXPECT_TRUE(failed_late);
}

// --- Two stores on two streams ------------------------------------------------------------------

/// The comparable part of a downloaded state (everything but the device addresses).
std::vector<long long> fingerprint(const cbst_host_state& h) {
  std::vector<long long> f = {h.num_records,           h.fixed_size,
                              h.initial_payload_size,  h.alignment,
                              h.scratch_item_capacity, h.scratch_payload_capacity};
  for (const auto& n : h.nodes)
    f.insert(f.end(), {n.index, n.value, n.length, n.size, n.occupancy, n.tail_base,
                       n.tail_capacity, n.left, n.right, n.parent});
  for (const auto* v : {&h.keys, &h.start_offsets, &h.avail, &h.subtree_avail, &h.payload})
    f.insert(f.end(), v->begin(), v->end());
  return f;
}

/// A sequence of operations with every path (applied to a store step by step).
void run_step(cbst_store& s, int step) {
  std::vector<int> rowa(70), rowb(1500);
  for (int i = 0; i < 70; ++i) rowa[static_cast<std::size_t>(i)] = 1000 + i;
  for (int i = 0; i < 1500; ++i) rowb[static_cast<std::size_t>(i)] = 5000 + i;
  switch (step) {
    case 0: {
      std::vector<int> keys, starts, flat;
      for (int k = 1; k <= 300; ++k) {
        keys.push_back(k);
        starts.push_back(static_cast<int>(flat.size()));
        flat.insert(flat.end(), {k, k + 1, k + 2, INT_MIN});
      }
      s.construct(keys.data(), starts.data(), 300, flat.data(), static_cast<int>(flat.size()));
      break;
    }
    case 1: {
      const batch b = make_batch({3, 7, 9}, {rowa, rowb, {1, 2}});
      s.append(b.keys, b.payload, b.prefix);
      break;
    }
    case 2:
      s.erase({5, 6, 7, 100, 200});
      break;
    case 3: {
      const batch b = make_batch({0, 0, 0, 0}, {{1}, {2, 3}, rowa, rowb});
      (void)s.insert(b.keys, b.payload, b.prefix);
      break;
    }
    case 4: {
      std::vector<int> drop(rowb.begin(), rowb.begin() + 40);
      const batch b = make_batch({3, 9}, {drop, {1}});
      s.remove_values(b.keys, b.payload, b.prefix);
      break;
    }
    default:
      break;
  }
}

TEST(CbstStoreStreams, TwoStoresOnTwoStreamsEqualSequentialRuns) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  // Sequential reference on the default resources.
  cbst_store reference(res, "reference", 4096);
  for (int step = 0; step < 5; ++step) run_step(reference, step);
  const std::vector<long long> want = fingerprint(reference.download());
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  {
    const resources ra = resources::cuda(res.device(), stream_ref(a));
    const resources rb = resources::cuda(res.device(), stream_ref(b));
    cbst_store sa(ra, "a", 4096), sb(rb, "b", 4096);
    for (int step = 0; step < 5; ++step) {  // interleaved, no host synchronization in between
      run_step(sa, step);
      run_step(sb, step);
    }
    std::vector<long long> got_a = fingerprint(sa.download());
    std::vector<long long> got_b = fingerprint(sb.download());
    EXPECT_EQ(got_a, want);
    EXPECT_EQ(got_b, want);
  }
  ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

// --- Host synchronizations per operation (ADR 0038, the read points) ------------------------------

/// The host synchronizations of one operation.
template <typename op_t>
long long syncs_of(cbst_store& s, op_t&& op) {
  const long long before = s.host_syncs();
  op();
  return s.host_syncs() - before;
}

TEST(CbstStoreSyncs, OnlyAtEscherGpusReadPointsAndNeverMoreThanEscherGpu) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "syncs", 4096);
  // ESCHER-GPU's blocking calls on the same paths (operations.cu@abcf9b2, ADR 0038): construct 3
  // (the cudaMemcpy uploads); rebuild 7 (6 cudaFree, thrust::sequence); fill 2 (inclusive_scan,
  // the total appended) + 1 with growth (cudaFree); erase and unfill 0; insert 1 (D) + 6 with
  // reuse (2 sequence, 2 sort_by_key, inclusive_scan, count_if) + 3 with matches (the three
  // copies) + 7 with surplus (copy_if, the last key, 5 cudaFree).
  EXPECT_EQ(syncs_of(s, [&] { run_step(s, 0); }), 0);                  // A: 3
  EXPECT_EQ(syncs_of(s, [&] { run_step(s, 1); }), 1);                  // A: 3 (growth)
  EXPECT_EQ(syncs_of(s, [&] { run_step(s, 2); }), 0);                  // A: 0
  EXPECT_EQ(syncs_of(s, [&] { run_step(s, 3); }), 1 + 1 + 1 + 1 + 1);  // A: 1 + 6 + 3 + 7
  EXPECT_EQ(syncs_of(s, [&] { run_step(s, 4); }), 0);                  // A: 0
  // An insert without deleted slots: D and the surplus rebuild only.
  const batch only_surplus = make_batch({0}, {{1, 2}});
  EXPECT_EQ(
      syncs_of(
          s, [&] { (void)s.insert(only_surplus.keys, only_surplus.payload, only_surplus.prefix); }),
      1 + 1 + 1);  // A: 1 + 7
  // An insert whose items all fit deleted slots: no rebuild.
  s.erase({1, 2});
  const batch only_reuse = make_batch({0, 0}, {{1}, {2}});
  EXPECT_EQ(
      syncs_of(s, [&] { (void)s.insert(only_reuse.keys, only_reuse.payload, only_reuse.prefix); }),
      1 + 1 + 1);  // A: 1 + 6 + 3
  // A fill within the capacity.
  const batch small = make_batch({4}, {{9}});
  EXPECT_EQ(syncs_of(s, [&] { s.append(small.keys, small.payload, small.prefix); }), 1);  // A: 2
}

}  // namespace
}  // namespace dyng::detail::escher

namespace dyng::detail::escher {
namespace {

// --- The payload policy ------------------------------------------------------------------------

TEST(CbstStorePolicyOnDevice, EscherGpuGrowsAFixedPolicyThrows) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  std::vector<int> keys = {1, 2}, starts = {0, 4};
  std::vector<int> flat = {1, 2, 3, INT_MIN, 4, 0, 0, INT_MIN};
  std::vector<int> big(100);
  for (int i = 0; i < 100; ++i) big[static_cast<std::size_t>(i)] = i + 1;
  const batch b = make_batch({1}, {big});
  {
    cbst_store s(res, "grows", 8);
    EXPECT_EQ(s.policy().name, "escher_gpu");
    s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
    s.append(b.keys, b.payload, b.prefix);
    // 8 used + one overflow segment of 100 - 0 values: (100 + 4) / 4 * 4 = 104 slots: 112 needed,
    // grown to max(112 + 28, 12) = 140.
    EXPECT_EQ(s.context().fixed_size, 140);
    rows want = {{1, 2, 3}, {4}};
    want[0].insert(want[0].end(), big.begin(), big.end());
    EXPECT_EQ(violations(s, want), 0);
  }
  {
    store_policy fixed = store_policy::escher_gpu();
    fixed.name = "fixed";
    fixed.growth = store_policy::growth_rule::none;
    cbst_store s(res, "fixed", 8, 4, fixed);
    s.construct(keys.data(), starts.data(), 2, flat.data(), 8);
    EXPECT_THROW(s.append(b.keys, b.payload, b.prefix), capacity_error);
    EXPECT_TRUE(s.poisoned());  // the fill wrote in place before the payload was full
  }
}

// --- The presets of M8a step 5 (ADR 0039) ------------------------------------------------------

/// Values first, first + 1, ..., first + n - 1.
std::vector<int> iota_row(int first, int n) {
  std::vector<int> r(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) r[static_cast<std::size_t>(i)] = first + i;
  return r;
}

/// Constructs `s` over rows of keys 1..n (ESCHER-GPU's flatten2DVector layout).
void construct_rows(cbst_store& s, const rows& values) {
  std::vector<int> keys, starts, flat;
  for (std::size_t i = 0; i < values.size(); ++i) {
    keys.push_back(static_cast<int>(i) + 1);
    starts.push_back(static_cast<int>(flat.size()));
    const std::size_t padded = (values[i].size() + 4) / 4 * 4;
    flat.insert(flat.end(), values[i].begin(), values[i].end());
    flat.insert(flat.end(), padded - values[i].size() - 1, 0);
    flat.push_back(INT_MIN);
  }
  s.construct(keys.data(), starts.data(), static_cast<int>(keys.size()), flat.data(),
              static_cast<int>(flat.size()));
}

// dynG's default compacts first and grows only when the compacted rows do not fit; escher_gpu()
// grows; mosp_escher() compacts and then throws.
TEST(CbstStorePresets, DefaultCompactsFirstThenGrows) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  const store_policy d = store_policy::dyng_default();
  EXPECT_EQ(d.name, "dyng_default");
  EXPECT_EQ(d.growth, store_policy::growth_rule::escher_gpu);
  EXPECT_EQ(d.compaction, store_policy::compaction_rule::mosp_escher);
  EXPECT_EQ(d.scratch, store_policy::scratch_rule::geometric);
  EXPECT_EQ(d.max_payload, INT_MAX);
  store_policy named;
  ASSERT_TRUE(store_policy::from_name("dyng_default", named));
  EXPECT_EQ(named.name, "dyng_default");
  for (const store_policy& policy :
       {store_policy::escher_gpu(), store_policy::mosp_escher(), store_policy::dyng_default()}) {
    SCOPED_TRACE(policy.name);
    cbst_store s(res, "churn", 40, 4, policy);
    construct_rows(s, {iota_row(1, 8), iota_row(100, 8)});  // 24 slots
    rows want = {iota_row(1, 8), iota_row(100, 8)};
    int next = 1000;
    bool threw = false;
    // Churn that fits once compacted: 8 live values in key 1.
    for (int round = 0; round < 20 && !threw; ++round) {
      const batch add = make_batch({1}, {iota_row(next, 4)});
      try {
        s.append(add.keys, add.payload, add.prefix);
      } catch (const capacity_error&) {
        threw = true;
        break;
      }
      want[0].insert(want[0].end(), add.payload.begin(), add.payload.end());
      const batch rem = make_batch({1}, {std::vector<int>(want[0].begin(), want[0].begin() + 4)});
      s.remove_values(rem.keys, rem.payload, rem.prefix);
      want[0].erase(want[0].begin(), want[0].begin() + 4);
      next += 4;
    }
    EXPECT_FALSE(threw);
    EXPECT_EQ(violations(s, want, true), 0);
    const cbst_store_stats st = s.stats();
    if (policy.name == "escher_gpu") {
      EXPECT_EQ(s.compactions(), 0);
      EXPECT_GT(st.growths, 0);
    } else {
      EXPECT_GT(s.compactions(), 0);
      EXPECT_EQ(st.growths, 0);
      EXPECT_EQ(s.context().fixed_size, 40);
    }
    // Live rows that no longer fit, compacted: mosp_escher() throws, the others grow.
    const batch big = make_batch({2}, {iota_row(5000, 40)});
    if (policy.name == "mosp_escher") {
      EXPECT_THROW(s.append(big.keys, big.payload, big.prefix), capacity_error);
      EXPECT_TRUE(s.poisoned());  // the tail was written before the payload was full
    } else {
      const long long compactions = s.compactions();
      s.append(big.keys, big.payload, big.prefix);
      want[1].insert(want[1].end(), big.payload.begin(), big.payload.end());
      EXPECT_GT(s.context().fixed_size, 40);
      EXPECT_EQ(violations(s, want, true), 0);
      if (policy.name == "dyng_default") {
        EXPECT_GT(s.compactions(), compactions);  // it compacted first
      }
    }
  }
}

// A forced small max_payload: the default preset raises capacity_error at the limit (where the
// int32 offsets would end), the store is poisoned after the in-place writes, and every later
// operation throws until it is constructed again.
TEST(CbstStorePresets, ASmallMaxPayloadRaisesCapacityError) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  EXPECT_EQ(store_policy::dyng_default(64).max_payload, 64);
  EXPECT_EQ(store_policy::dyng_default(1LL << 40).max_payload, INT_MAX);
  EXPECT_THROW((void)store_policy::dyng_default(64).grown_capacity(65, 40, "t"), capacity_error);
  EXPECT_EQ(store_policy::dyng_default(64).grown_capacity(60, 40, "t"), 64);  // min(75, 64)
  EXPECT_THROW((void)store_policy::dyng_default(64).initial_capacity(0, 100, true), capacity_error);
  cbst_store s(res, "limit", 40, 4, store_policy::dyng_default(64));
  construct_rows(s, {iota_row(1, 8), iota_row(100, 8)});
  // 24 used + an overflow segment of 30 values (32 slots) = 56: grows to min(70, 64) = 64.
  const batch fits = make_batch({1}, {iota_row(1000, 30)});
  s.append(fits.keys, fits.payload, fits.prefix);
  EXPECT_EQ(s.context().fixed_size, 64);
  EXPECT_EQ(violations(s, {[] {
                             std::vector<int> r = iota_row(1, 8);
                             const std::vector<int> more = iota_row(1000, 30);
                             r.insert(r.end(), more.begin(), more.end());
                             return r;
                           }(),
                           iota_row(100, 8)}),
            0);
  const batch over = make_batch({2}, {iota_row(2000, 40)});
  EXPECT_THROW(s.append(over.keys, over.payload, over.prefix), capacity_error);
  EXPECT_TRUE(s.poisoned());
  EXPECT_THROW(s.erase({1}), invalid_argument_error);
  EXPECT_THROW((void)s.compact(), invalid_argument_error);
  construct_rows(s, {{1}});
  EXPECT_FALSE(s.poisoned());
}

}  // namespace
}  // namespace dyng::detail::escher
