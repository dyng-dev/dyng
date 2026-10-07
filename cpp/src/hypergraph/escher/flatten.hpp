// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/utils.hpp
/**
 * @file flatten.hpp
 * @brief flatten_rows: the CBST construct layout of a list of rows (ESCHER-GPU's flatten2DVector,
 *        declared in its include/utils.hpp).
 */
#pragma once

#include <utility>
#include <vector>

namespace dyng::detail::escher {

std::pair<std::vector<int>, std::vector<int>> flatten_rows(
    const std::vector<std::vector<int>>& vec2d);

}  // namespace dyng::detail::escher
