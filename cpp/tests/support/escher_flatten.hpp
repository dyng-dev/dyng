// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/flatten.cpp
// Derived from ESCHER-GPU@abcf9b2:include/utils.hpp
/**
 * @file escher_flatten.hpp
 * @brief flatten_rows: the CBST construct layout of a list of rows (ESCHER-GPU's
 *        flatten2DVector), a test helper.
 *
 * The port of M8a (ADR 0038) put it in the library (cpp/src/hypergraph/escher/flatten.cpp), but
 * nothing there calls it: the stores build their construct layout themselves. It moved here in
 * 0.4 (ADR 0084, the carried item (c)), with its tests (hypergraph/escher/cbst_host_test.cpp).
 * ESCHER-GPU writes the rows in an OpenMP region; the layout does not depend on it, and the helper
 * writes them in order.
 */
#pragma once

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test {

/**
 * @brief The construct layout of `rows` (rows of keys 1..n): the payload (row i padded to
 *        ((n_i + 4) / 4) * 4 slots: its values, zeros, INT_MIN in the last slot) and the row
 *        offsets.
 * @param[in] rows      The rows.
 * @param[in] max_slots The largest layout (the int32 offsets, INT_MAX; tests lower it to reach
 *                      the limit; never above INT_MAX).
 * @return The payload and the offset of every row.
 * @throws capacity_error when the layout needs more than `max_slots` slots.
 */
inline std::pair<std::vector<int>, std::vector<int>> flatten_rows(
    const std::vector<std::vector<int>>& rows, long long max_slots = INT_MAX) {
  const auto padded = [](long long n) { return ((n + 4) / 4) * 4; };
  max_slots = std::min<long long>(max_slots, INT_MAX);
  const auto num_rows = static_cast<long long>(rows.size());
  std::vector<int> starts(static_cast<std::size_t>(num_rows));
  long long total = 0;
  for (long long i = 0; i < num_rows; ++i) {
    starts[static_cast<std::size_t>(i)] = static_cast<int>(total);
    total += padded(static_cast<long long>(rows[static_cast<std::size_t>(i)].size()));
  }
  // The payload offsets are int32 (PLAN 4.4.2): ESCHER-GPU truncated them beyond 2^31 - 1.
  if (total > max_slots) {
    throw capacity_error("dyng: flatten_rows: the CBST layout of " + std::to_string(num_rows) +
                         " rows needs " + std::to_string(total) + " payload slots, more than " +
                         (max_slots == INT_MAX ? std::string("the int32 offsets address (2^31 - 1)")
                                               : std::to_string(max_slots)));
  }
  std::vector<int> flat(static_cast<std::size_t>(total), 0);
  for (long long i = 0; i < num_rows; ++i) {
    const std::vector<int>& values = rows[static_cast<std::size_t>(i)];
    int* row = flat.data() + starts[static_cast<std::size_t>(i)];
    std::copy(values.begin(), values.end(), row);
    row[padded(static_cast<long long>(values.size())) - 1] = INT_MIN;
  }
  return {flat, starts};
}

}  // namespace dyng::test
