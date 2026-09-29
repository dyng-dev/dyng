// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief The algorithm registry. The table is generated from the manifests by scripts/regen.py
 *        (registry_table.inc); an entry is compiled in when CMake built its algorithm
 *        (DYNG_ALGORITHM_<NAME>, set by cpp/CMakeLists.txt for every dyng_add_algorithm()).
 */
#include <dyng/core/backend.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/types.hpp>

#include <string_view>
#include <vector>

namespace dyng {

namespace {

std::vector<algorithm_info> make_table() {
  std::vector<algorithm_info> table;
#include "core/registry_table.inc"
  return table;
}

}  // namespace

const std::vector<algorithm_info>& algorithms() {
  static const std::vector<algorithm_info> table = make_table();
  return table;
}

const algorithm_info* find_algorithm(std::string_view name) {
  for (const algorithm_info& info : algorithms()) {
    if (info.name == name) {
      return &info;
    }
  }
  return nullptr;
}

}  // namespace dyng
