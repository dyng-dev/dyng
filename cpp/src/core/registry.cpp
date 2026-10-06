// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief The algorithm registry. The table is generated from the manifests by scripts/regen.py
 *        (registry_table.inc); an entry is compiled in when CMake built its algorithm
 *        (DYNG_ALGORITHM_<NAME>, set by cpp/CMakeLists.txt for every dyng_add_algorithm()).
 */
#include "util/allocation.hpp"

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

const std::vector<algorithm_info>& algorithms() try {
  static const std::vector<algorithm_info> table = make_table();
  return table;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("dyng::algorithms() (the registry table)")

const algorithm_info* find_algorithm(std::string_view name) {
  for (const algorithm_info& info : algorithms()) {
    if (info.name == name) {
      return &info;
    }
  }
  return nullptr;
}

std::string_view to_string(algorithm_family family) noexcept {
  switch (family) {
    case algorithm_family::fixed_point:
      return "fixed_point";
    case algorithm_family::aggregate_delta:
      return "aggregate_delta";
  }
  return "unknown";
}

std::string_view to_string(container_kind container) noexcept {
  switch (container) {
    case container_kind::graph:
      return "graph";
    case container_kind::hypergraph:
      return "hypergraph";
  }
  return "unknown";
}

std::string_view to_string(maturity_level maturity) noexcept {
  switch (maturity) {
    case maturity_level::experimental:
      return "experimental";
    case maturity_level::stable:
      return "stable";
    case maturity_level::deprecated:
      return "deprecated";
    case maturity_level::tutorial:
      return "tutorial";
  }
  return "unknown";
}

std::string_view to_string(oracle_kind oracle) noexcept {
  switch (oracle) {
    case oracle_kind::compute:
      return "compute";
    case oracle_kind::reference:
      return "reference";
  }
  return "unknown";
}

}  // namespace dyng
