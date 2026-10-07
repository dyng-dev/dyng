// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu (ensurePayloadCapacity)
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (payloadCapacity)
/**
 * @file store_policy.hpp
 * @brief How a CBST store sizes and grows its payload, and whether it compacts (M8a, ADR 0038):
 *        the internal store_policy and its preset escher_gpu(), which reproduces ESCHER-GPU@abcf9b2.
 *        MOSP_ESCHER's preset and dynG's default follow in M8a steps 4-5.
 *
 * Host-only (no CUDA): the hypergraph container (M8b) sizes its trees with initial_capacity()
 * before it creates them; the store grows its payload with grown_capacity().
 */
#pragma once

#include <cstdint>
#include <string>

namespace dyng::detail::escher {

/// The payload policy of a CBST store.
struct store_policy {
  /// How a payload that does not fit grows.
  enum class growth_rule : std::uint8_t {
    none,        ///< it does not: capacity_error
    escher_gpu,  ///< to max(needed * 1.25, capacity * 1.5) in integers, at most INT_MAX
  };
  /// When a store compacts its payload.
  enum class compaction_rule : std::uint8_t {
    never,  ///< ESCHER-GPU
  };

  std::string name = "escher_gpu";                      ///< the preset's name
  growth_rule growth = growth_rule::escher_gpu;         ///< growth of a full payload
  compaction_rule compaction = compaction_rule::never;  ///< compaction
  int alignment = 4;  ///< the padding of a surplus row (rows of `alignment` slots + terminator)

  /**
   * @brief ESCHER-GPU@abcf9b2's policy: in-place trees (H2V, V2H) start at max(requested,
   *        size + size/8 + 1024), trees rebuilt by the delta merge (H2H, V2V) at their exact
   *        size; a full payload grows to max(needed + needed/4, 3 * capacity / 2) entries, at
   *        most INT_MAX; alignment 4; never compacts.
   * @return The preset.
   */
  static store_policy escher_gpu();

  /**
   * @brief The initial payload capacity of a tree (ESCHER-GPU's payloadCapacity).
   * @param[in] requested The caller's lower bound (A's --payload-capacity; 0 for none).
   * @param[in] size      The payload entries of the tree's construct layout.
   * @param[in] in_place  The tree is updated in place (H2V, V2H), not rebuilt by a merge.
   * @return The capacity.
   * @throws capacity_error if it exceeds the int32 offsets (INT_MAX).
   */
  [[nodiscard]] long long initial_capacity(long long requested, long long size,
                                           bool in_place) const;

  /**
   * @brief The capacity a full payload grows to (ESCHER-GPU's ensurePayloadCapacity).
   * @param[in] needed   The entries that must fit (> capacity).
   * @param[in] capacity The current capacity.
   * @param[in] name     The tree's name, for the message.
   * @return The new capacity (>= needed).
   * @throws capacity_error if the policy does not grow or `needed` exceeds INT_MAX.
   */
  [[nodiscard]] long long grown_capacity(long long needed, long long capacity,
                                         const std::string& name) const;
};

}  // namespace dyng::detail::escher
