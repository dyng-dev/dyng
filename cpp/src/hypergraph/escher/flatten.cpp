// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/flatten.cpp
/**
 * @file flatten.cpp
 * @brief flatten_rows: the CBST construct layout of a list of rows (padded rows, INT_MIN
 *        terminators, 64-bit total, OpenMP).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#include "hypergraph/escher/flatten.hpp"

#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

namespace dyng::detail::escher {

// Padded length of a row: the next multiple of 4 that leaves at least one
// slot for the INT_MIN terminator (same rule as the device-side
// next_multiple_of_4 in kernel/device_utils.cuh).
static inline long long padded_row_size(long long num) {
  return ((num + 4) / 4) * 4;
}

// Row i occupies [start_offsets[i], start_offsets[i] + padded): its values,
// zero padding, and INT_MIN in the last slot. Rows are written in parallel.
std::pair<std::vector<int>, std::vector<int>> flatten_rows(
    const std::vector<std::vector<int>>& vec2d) {
  const long long num_rows = static_cast<long long>(vec2d.size());
  std::vector<int> start_offsets(num_rows);
  long long total = 0;
  for (long long i = 0; i < num_rows; ++i) {
    start_offsets[i] = static_cast<int>(total);
    total += padded_row_size(static_cast<long long>(vec2d[i].size()));
  }
  std::vector<int> flat_values(total, 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1024)
#endif
  for (long long i = 0; i < num_rows; ++i) {
    int* row = flat_values.data() + start_offsets[i];
    const std::vector<int>& values = vec2d[i];
    for (size_t j = 0; j < values.size(); ++j) row[j] = values[j];
    row[padded_row_size(static_cast<long long>(values.size())) - 1] = INT_MIN;
  }
  return {flat_values, start_offsets};
}

}  // namespace dyng::detail::escher
