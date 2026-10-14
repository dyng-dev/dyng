// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/utils.hpp
/**
 * @file flatten.hpp
 * @brief flatten_rows: the CBST construct layout of a list of rows (ESCHER-GPU's flatten2DVector,
 *        declared in its include/utils.hpp).
 */
#pragma once

#include <climits>
#include <utility>
#include <vector>

namespace dyng::detail::escher {

// The construct layout of `vec2d` (rows of keys 1..n): the payload (every row padded to
// (n + 4) / 4 * 4 slots, its last slot INT_MIN) and the row offsets. Throws capacity_error when it
// needs more than `max_slots` slots (the int32 offsets, INT_MAX; tests lower it to reach the
// limit, M8a review).
std::pair<std::vector<int>, std::vector<int>> flatten_rows(
    const std::vector<std::vector<int>>& vec2d, long long max_slots = INT_MAX);

}  // namespace dyng::detail::escher
