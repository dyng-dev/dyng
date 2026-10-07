// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu (ensurePayloadCapacity)
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (payloadCapacity)
/**
 * @file store_policy.cpp
 * @brief The CBST store's payload policy and its preset escher_gpu() (M8a, ADR 0038).
 */
#include "hypergraph/escher/store_policy.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <string>

namespace dyng::detail::escher {

store_policy store_policy::escher_gpu() {
  store_policy p;
  p.name = "escher_gpu";
  p.growth = growth_rule::escher_gpu;
  p.compaction = compaction_rule::never;
  p.alignment = 4;
  return p;
}

long long store_policy::initial_capacity(long long requested, long long size, bool in_place) const {
  // ESCHER-GPU: trees updated in place get 1/8 of slack for appends and grow on demand; trees
  // rebuilt by the delta merge are sized exactly; a requested capacity is a lower bound for the
  // in-place trees.
  const long long capacity = in_place ? std::max(requested, size + size / 8 + 1024) : size;
  if (capacity > INT_MAX) {
    throw capacity_error("dyng: CBST payload capacity " + std::to_string(capacity) +
                         " exceeds the int32 offsets of the CBST (at most " +
                         std::to_string(INT_MAX) + ")");
  }
  return capacity;
}

long long store_policy::grown_capacity(long long needed, long long capacity,
                                       const std::string& name) const {
  if (growth == growth_rule::none) {
    throw capacity_error("dyng: CBST " + name + ": " + std::to_string(needed) +
                         " payload entries do not fit its capacity of " + std::to_string(capacity) +
                         " (policy " + this->name + " does not grow the payload)");
  }
  long long grown = std::max(needed + needed / 4, 3LL * capacity / 2);
  grown = std::min<long long>(grown, INT_MAX);
  if (needed > grown) {
    throw capacity_error("dyng: CBST " + name + ": " + std::to_string(needed) +
                         " payload entries exceed the int32 offsets of the CBST (at most " +
                         std::to_string(INT_MAX) + ")");
  }
  return grown;
}

}  // namespace dyng::detail::escher
