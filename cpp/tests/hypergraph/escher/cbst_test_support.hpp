// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_test_support.hpp
 * @brief Helpers of the CBST store's unit tests (M8a steps 3-4): batches of rows, the construct
 *        layout, the integrity checker as a test assertion, reads and writes of the device
 *        arrays.
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/integrity.hpp"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <sstream>
#include <vector>

namespace dyng::detail::escher::test_support {

using rows = std::vector<std::vector<int>>;

/// Keys, payload and prefix sizes of a batch of rows.
struct batch {
  std::vector<int> keys, payload, prefix;
};

/// The batch of `values[i]` for `keys[i]`.
inline batch make_batch(const std::vector<int>& keys, const rows& values) {
  batch b;
  b.keys = keys;
  for (const auto& r : values) {
    b.payload.insert(b.payload.end(), r.begin(), r.end());
    b.prefix.push_back(static_cast<int>(b.payload.size()));
  }
  return b;
}

/// The construct layout of rows of keys 1..n (ESCHER-GPU's flatten2DVector).
struct layout {
  std::vector<int> keys, starts, flat;
};

inline layout make_layout(const rows& values) {
  layout l;
  for (std::size_t i = 0; i < values.size(); ++i) {
    l.keys.push_back(static_cast<int>(i) + 1);
    l.starts.push_back(static_cast<int>(l.flat.size()));
    const std::size_t padded = (values[i].size() + 4) / 4 * 4;
    l.flat.insert(l.flat.end(), values[i].begin(), values[i].end());
    l.flat.insert(l.flat.end(), padded - values[i].size() - 1, 0);
    l.flat.push_back(INT_MIN);
  }
  return l;
}

/// Constructs `s` over the rows of keys 1..n.
inline void construct_rows(cbst_store& s, const rows& values) {
  layout l = make_layout(values);
  s.construct(l.keys.data(), l.starts.data(), static_cast<int>(l.keys.size()), l.flat.data(),
              static_cast<int>(l.flat.size()));
}

/// The violations the integrity checker finds against `expected` (rows of keys 1..n).
inline long long violations(const cbst_store& s, const rows& expected,
                            bool order_insensitive = false) {
  std::ostringstream log;
  const long long v = check_tree_rows(s.context(), expected, order_insensitive, "test", log);
  if (v != 0) ADD_FAILURE() << log.str();
  return v;
}

/// Values first, first + 1, ..., first + n - 1.
inline std::vector<int> iota_row(int first, int n) {
  std::vector<int> r(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) r[static_cast<std::size_t>(i)] = first + i;
  return r;
}

/// The node of `key` (its heap position) read from the device.
inline cbst_node read_node(const cbst_store& s, int key) {
  const std::vector<int> pos = s.find({key});
  cbst_node n{};
  EXPECT_GE(pos[0], 0) << "key " << key;
  if (pos[0] >= 0) {
    EXPECT_EQ(cudaMemcpy(&n, s.context().d_nodes + pos[0], sizeof n, cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return n;
}

/// Overwrites the node of `key` on the device (tests of inconsistent metadata).
inline void write_node(const cbst_store& s, int key, const cbst_node& n) {
  const std::vector<int> pos = s.find({key});
  ASSERT_GE(pos[0], 0) << "key " << key;
  ASSERT_EQ(cudaMemcpy(s.context().d_nodes + pos[0], &n, sizeof n, cudaMemcpyHostToDevice),
            cudaSuccess);
}

/// Payload slots [begin, begin + n) read from the device.
inline std::vector<int> read_payload(const cbst_store& s, long long begin, long long n) {
  std::vector<int> out(static_cast<std::size_t>(n));
  if (n > 0) {
    EXPECT_EQ(cudaMemcpy(out.data(), s.context().d_flat_payload + begin, out.size() * sizeof(int),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return out;
}

/// Overwrites payload slot `at` on the device.
inline void write_payload(const cbst_store& s, long long at, int value) {
  ASSERT_EQ(
      cudaMemcpy(s.context().d_flat_payload + at, &value, sizeof value, cudaMemcpyHostToDevice),
      cudaSuccess);
}

}  // namespace dyng::detail::escher::test_support
