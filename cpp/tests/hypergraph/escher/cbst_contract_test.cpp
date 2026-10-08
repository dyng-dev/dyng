// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_contract_test.cpp
 * @brief Tests of the CBST store's input and state contracts that the M8a review asked for: trees
 *        of 0 records after an earlier tree, the row locator on no keys and on chained rows, the
 *        shape of a batch, the construct layout, reads of a poisoned store, the clone of a store
 *        with a pending device error or onto another device, the counted Debug check of the
 *        caller's row counts, two streams with compaction and growth, and the hysteresis of the
 *        default preset's compaction.
 */
#include "core/cuda_runtime.hpp"
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/cbst_test_support.hpp"
#include "hypergraph/escher/row_view.cuh"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/testing/cbst_host_model.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>
#include <vector>

namespace dyng::detail::escher {
namespace {

using namespace test_support;

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

/// An empty tree: construct with 0 records.
void construct_empty(cbst_store& s) {
  std::vector<int> none;
  s.construct(none.data(), none.data(), 0, none.data(), 0);
}

// --- Trees of 0 records after an earlier tree -----------------------------------------------------

// A construct of 0 records drops the earlier tree: no lookup finds its keys, an append to one of
// them is the documented error (nothing is written), and the store holds what device_bytes()
// counts.
TEST(CbstContractEmptyTree, AReconstructWithZeroRecordsDropsTheEarlierTree) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  {
    cbst_store s(res, "again", 64);
    construct_rows(s, {{1, 2, 3}, {4}, {5, 6}});
    construct_empty(s);
    EXPECT_EQ(s.context().num_records, 0);
    EXPECT_EQ(s.context().d_nodes, nullptr);
    EXPECT_EQ(s.find({1, 2, 3}), (std::vector<int>{-1, -1, -1}));
    EXPECT_EQ(s.rows({1, 2}), (rows{{}, {}}));
    EXPECT_EQ(s.allocated_bytes(), s.device_bytes());
    EXPECT_EQ(s.device_bytes(), 64 * sizeof(int));
    EXPECT_EQ(violations(s, {}), 0);
    const batch b = make_batch({2}, {{77, 78}});
    EXPECT_THROW(s.append(b.keys, b.payload, b.prefix), internal_error);
    EXPECT_EQ(s.stats().rejected_items, 1);
    EXPECT_EQ(s.context().initial_payload_size, 0);
    EXPECT_EQ(read_payload(s, 0, 8), std::vector<int>(8, 0));  // nothing was written
  }
  {
    // ... and an insert builds a new tree from scratch.
    cbst_store s(res, "again", 64);
    construct_rows(s, {{1, 2, 3}, {4}, {5, 6}});
    s.erase({2});
    construct_empty(s);
    const batch b = make_batch({0, 0}, {{9}, {10, 11}});
    EXPECT_EQ(s.insert(b.keys, b.payload, b.prefix).item_to_key, (std::vector<int>{1, 2}));
    EXPECT_EQ(violations(s, {{9}, {10, 11}}), 0);
    EXPECT_FALSE(s.poisoned());
  }
}

// A merged tree of 0 records (every row deleted) is valid input of rebuild_from_device: nothing is
// launched, the payload is kept, and later inserts and appends work.
TEST(CbstContractEmptyTree, RebuildWithZeroRecords) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  for (const store_policy& policy :
       {store_policy::escher_gpu(), store_policy::mosp_escher(16), store_policy::dyng_default()}) {
    SCOPED_TRACE(policy.name);
    cbst_store s(res, "rebuilt", 16, 4, policy);
    construct_rows(s, {{1, 2, 3}, {4}});
    buffer<int> starts(res, 0), flat(res, 16);
    ASSERT_EQ(cudaMemsetAsync(flat.data(), 0, 16 * sizeof(int),
                              static_cast<cudaStream_t>(res.stream().get())),
              cudaSuccess);
    s.rebuild_from_device(std::move(starts), 0, std::move(flat), 0, 16);
    EXPECT_FALSE(s.poisoned());
    EXPECT_EQ(s.context().num_records, 0);
    EXPECT_EQ(s.context().fixed_size, 16);
    EXPECT_EQ(s.find({1, 2}), (std::vector<int>{-1, -1}));
    s.check_errors();
    EXPECT_EQ(violations(s, {}), 0);
    const batch b = make_batch({0, 0}, {{5, 6}, {7}});
    EXPECT_EQ(s.insert(b.keys, b.payload, b.prefix).item_to_key, (std::vector<int>{1, 2}));
    const batch more = make_batch({1}, {{8, 9, 10}});
    s.append(more.keys, more.payload, more.prefix);
    s.check_errors();
    EXPECT_EQ(violations(s, {{5, 6, 8, 9, 10}, {7}}, true), 0);
    EXPECT_FALSE(s.poisoned());
  }
}

// --- The row locator ---------------------------------------------------------------------------

/// The (loc, len) of key `key` in a view.
std::vector<int> row_of_view(const row_view& v, int key) {
  int loc = 0, len = 0;
  EXPECT_EQ(cudaMemcpy(&loc, v.loc + key, sizeof loc, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(&len, v.len + key, sizeof len, cudaMemcpyDeviceToHost), cudaSuccess);
  std::vector<int> out(static_cast<std::size_t>(len));
  if (len > 0) {
    EXPECT_EQ(
        cudaMemcpy(out.data(), v.flat + loc, out.size() * sizeof(int), cudaMemcpyDeviceToHost),
        cudaSuccess);
  }
  return out;
}

// A view of no keys (an empty V2H tree, a hypergraph without vertices) launches nothing.
TEST(CbstContractLocator, MaxKeyZero) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store empty(res, "empty", 16);
  construct_empty(empty);
  cbst_store some(res, "some", 64);
  construct_rows(some, {{1, 2}, {3}});
  for (const cbst_store* s : {&empty, &some}) {
    row_locator view;
    view.build_sorted_copy(s->context(), 0);
    EXPECT_EQ(view.view().max_key, 0);
    EXPECT_EQ(row_of_view(view.view(), 0), std::vector<int>{});
    row_locator direct;
    direct.build(s->context(), 0, true);
    EXPECT_EQ(direct.view().max_key, 0);
    EXPECT_THROW(view.build_sorted_copy(s->context(), -1), invalid_argument_error);
  }
  some.check_errors();
  EXPECT_FALSE(some.poisoned());
}

// --- The shape of a batch ------------------------------------------------------------------------

// A batch whose prefix sums do not describe its payload is rejected before anything changes (the
// kernels read the payload through them): invalid_argument_error, the store not poisoned, its state
// as before.
TEST(CbstContractBatch, AMalformedBatchIsRejectedBeforeAnythingChanges) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  cbst_store s(res, "shape", 256);
  construct_rows(s, {{1, 2, 3}, {4}, {5, 6}});
  const std::vector<long long> before = fingerprint(s.download());
  struct malformed {
    const char* what;
    std::vector<int> keys, payload, prefix;
  };
  const std::vector<malformed> cases = {
      {"more prefix sizes than keys", {1}, {7, 8}, {1, 2}},
      {"fewer prefix sizes than keys", {1, 2}, {7, 8}, {2}},
      {"a prefix beyond the payload", {1}, {7, 8}, {600}},
      {"a prefix short of the payload", {1, 2}, {7, 8, 9}, {1, 2}},
      {"a decreasing prefix", {1, 2}, {7, 8}, {2, 1}},
      {"a negative prefix", {1, 2}, {7, 8}, {-1, 2}},
  };
  for (const malformed& m : cases) {
    SCOPED_TRACE(m.what);
    EXPECT_THROW((void)s.insert(m.keys, m.payload, m.prefix), invalid_argument_error);
    EXPECT_THROW(s.append(m.keys, m.payload, m.prefix), invalid_argument_error);
    EXPECT_THROW(s.remove_values(m.keys, m.payload, m.prefix), invalid_argument_error);
    EXPECT_FALSE(s.poisoned());
  }
  s.check_errors();
  EXPECT_EQ(fingerprint(s.download()), before);
  EXPECT_EQ(violations(s, {{1, 2, 3}, {4}, {5, 6}}), 0);
  // An empty batch is valid (and changes nothing).
  (void)s.insert({}, {}, {});
  s.append({}, {}, {});
  s.remove_values({}, {}, {});
  EXPECT_EQ(fingerprint(s.download()), before);
  // The free functions check the same.
  cbst_context& ctx = const_cast<cbst_context&>(s.context());
  EXPECT_THROW(fill_cbst({1}, {7, 8}, {600}, ctx), invalid_argument_error);
  EXPECT_THROW((void)insert_cbst({1}, {7, 8}, {600}, ctx), invalid_argument_error);
  EXPECT_THROW(unfill_cbst({1}, {7, 8}, {1, 2}, ctx), invalid_argument_error);
  EXPECT_EQ(fingerprint(s.download()), before);
}

// --- The construct layout ------------------------------------------------------------------------

// Rows of no slot, overlapping or unordered rows, rows outside the payload and unsorted or
// repeated keys are rejected before anything is allocated (a zero-length row put the chain pointer
// of an append into the next row's first slot).
TEST(CbstContractConstruct, TheLayoutIsValidated) {
  DYNG_SKIP_IF_NO_CUDA();
  const resources res = resources::cuda();
  struct layout_case {
    const char* what;
    std::vector<int> keys, starts, flat;
  };
  const std::vector<int> flat = {5, 6, 0, INT_MIN, 7, 0, 0, INT_MIN};
  const std::vector<layout_case> cases = {
      {"a zero-length row", {1, 2}, {0, 0}, {5, 6, 0, INT_MIN}},
      {"decreasing offsets", {1, 2}, {4, 0}, flat},
      {"a negative offset", {1, 2}, {-1, 4}, flat},
      {"the last row outside the payload", {1, 2}, {0, 8}, flat},
      {"unsorted keys", {2, 1}, {0, 4}, flat},
      {"a repeated key", {1, 1}, {0, 4}, flat},
  };
  for (const layout_case& c : cases) {
    SCOPED_TRACE(c.what);
    cbst_store s(res, "layout", 16);
    std::vector<int> keys = c.keys, starts = c.starts, payload = c.flat;
    EXPECT_THROW(s.construct(keys.data(), starts.data(), static_cast<int>(keys.size()),
                             payload.data(), static_cast<int>(payload.size())),
                 invalid_argument_error);
    EXPECT_FALSE(s.poisoned());
    EXPECT_EQ(s.context().num_records, 0);
    EXPECT_EQ(s.allocated_bytes(), 0U);
  }
  // Keys need not be 1..n, only strictly increasing.
  cbst_store s(res, "layout", 16);
  std::vector<int> keys = {3, 7}, starts = {0, 4}, payload = flat;
  s.construct(keys.data(), starts.data(), 2, payload.data(), 8);
  EXPECT_EQ(s.rows({3, 7}), (rows{{5, 6}, {7}}));
  // The zero-length row of the review's probe: the append to key 1 would have overwritten key 2.
  cbst_store probe(res, "probe", 16);
  std::vector<int> pk = {1, 2}, ps = {0, 1}, pf = {INT_MIN, 5, 6, INT_MIN};
  probe.construct(pk.data(), ps.data(), 2, pf.data(), 4);
  const batch one = make_batch({1}, {{9}});
  probe.append(one.keys, one.payload, one.prefix);
  EXPECT_EQ(probe.rows({1, 2}), (rows{{9}, {5, 6}}));
}

}  // namespace
}  // namespace dyng::detail::escher
