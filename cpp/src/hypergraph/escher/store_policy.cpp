// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu (ensurePayloadCapacity)
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (payloadCapacity)
// Derived from MOSP_ESCHER@4b86159:escher/structure/operations.cu (ensureScratchCapacity)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (constructFromRows)
/**
 * @file store_policy.cpp
 * @brief The CBST store's payload policy and its presets escher_gpu() (M8a, ADR 0038),
 *        and mosp_escher() (M8a step 5, ADR 0039).
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
  p.scratch = scratch_rule::exact;
  p.alignment = 4;
  return p;
}

store_policy store_policy::mosp_escher(long long extra) {
  store_policy p;
  p.name = "mosp_escher";
  p.growth = growth_rule::none;  // MOSP_ESCHER's fixed capacity
  // escher/structure/operations.cu@4b86159:409-444 (fillImpl), 694-721 (insertCBST)
  p.compaction = compaction_rule::mosp_escher;
  p.scratch = scratch_rule::geometric;  // its ensureScratchCapacity
  p.alignment = 4;
  // hypergraph/src/DynamicHypergraph.cpp@4b86159:79-107 (headroomFactor 1.3, extraPayloadInts)
  p.headroom = 1.3;
  p.extra = extra;
  return p;
}

bool store_policy::from_name(const std::string& name, store_policy& out) {
  if (name == "escher_gpu") {
    out = escher_gpu();
  } else if (name == "mosp_escher") {
    out = mosp_escher();
  } else {
    return false;
  }
  return true;
}

long long store_policy::initial_capacity(long long requested, long long size, bool in_place) const {
  if (headroom > 0) {
    // MOSP_ESCHER's constructFromRows (hypergraph/src/DynamicHypergraph.cpp@4b86159:79-107): the
    // flat size times the headroom plus the extra entries, capped at 2.0e9; a payload above the
    // cap is an error (escher::EscherError there).
    constexpr long long max_initial = 2000000000LL;
    long long capacity =
        static_cast<long long>(static_cast<double>(size) * headroom) + std::max(extra, 0LL);
    capacity = std::min(capacity, max_initial);
    if (capacity < size) {
      throw capacity_error("dyng: CBST initial payload of " + std::to_string(size) +
                           " values exceeds the cap of " + std::to_string(max_initial) +
                           " values per CBST (policy " + name + ")");
    }
    return capacity;
  }
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

long long store_policy::scratch_capacity(long long needed, long long capacity, long long empty,
                                         long long limit) const {
  if (scratch == scratch_rule::exact) return std::max(capacity, needed);
  if (needed <= capacity) return capacity;
  // MOSP_ESCHER's ensureScratchCapacity (escher/structure/operations.cu@4b86159:145-149), in 64
  // bits and capped at `limit` (its int loop overflowed near INT_MAX).
  long long grown = capacity > 0 ? capacity : empty;
  while (grown < needed) grown += grown / 2 + 1;
  return std::min(grown, std::max(limit, needed));
}

}  // namespace dyng::detail::escher
