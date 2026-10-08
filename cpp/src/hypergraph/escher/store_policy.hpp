// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu (ensurePayloadCapacity)
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (payloadCapacity)
// Derived from MOSP_ESCHER@4b86159:escher/structure/operations.cu (ensureScratchCapacity)
/**
 * @file store_policy.hpp
 * @brief How a CBST store sizes and grows its payload, and whether it compacts (M8a, ADR 0038):
 *        the internal store_policy, its preset escher_gpu(), which reproduces ESCHER-GPU@abcf9b2,
 *        and the staging preset mosp_escher_staged() of MOSP_ESCHER@4b86159's behaviours merged
 *        so far (M8a step 4). MOSP_ESCHER's full preset and dynG's default follow in M8a step 5.
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

  /// How the operation scratch (the per-operation input buffers) grows (M8a step 4, ADR 0039).
  enum class scratch_rule : std::uint8_t {
    exact,      ///< to the operation's size (ESCHER-GPU's reserveScratch; the memory gate's)
    geometric,  ///< in steps of +n/2+1 from the current size, until it fits (MOSP_ESCHER's
                ///< ensureScratchCapacity, escher/structure/operations.cu@4b86159:139-165)
  };

  std::string name = "escher_gpu";                      ///< the preset's name
  growth_rule growth = growth_rule::escher_gpu;         ///< growth of a full payload
  compaction_rule compaction = compaction_rule::never;  ///< compaction
  scratch_rule scratch = scratch_rule::exact;           ///< growth of the operation scratch
  int alignment = 4;  ///< the padding of a surplus row (rows of `alignment` slots + terminator)

  /**
   * @brief ESCHER-GPU@abcf9b2's policy: in-place trees (H2V, V2H) start at max(requested,
   *        size + size/8 + 1024), trees rebuilt by the delta merge (H2H, V2V) at their exact
   *        size; a full payload grows to max(needed + needed/4, 3 * capacity / 2) entries, at
   *        most INT_MAX; alignment 4; never compacts; the operation scratch grows to the size of
   *        an operation (exact).
   * @return The preset.
   */
  static store_policy escher_gpu();

  /**
   * @brief The staging preset of M8a step 4 (ADR 0039): MOSP_ESCHER@4b86159's behaviours merged
   *        so far, before its automatic compaction (step 5, which replaces this preset by
   *        mosp_escher()). A fixed capacity (MOSP_ESCHER's: a full payload is a capacity_error,
   *        where MOSP_ESCHER would compact first), alignment 4, never compacts on its own, the
   *        operation scratch grows geometrically (MOSP_ESCHER's ensureScratchCapacity). The
   *        trace replay runs MOSP_ESCHER's traces under it after every merge step.
   * @return The preset.
   */
  static store_policy mosp_escher_staged();

  /**
   * @brief The preset of a name ("escher_gpu", "mosp_escher_staged").
   * @param[in] name The name.
   * @param[out] out The preset.
   * @return false if the name is unknown.
   */
  [[nodiscard]] static bool from_name(const std::string& name, store_policy& out);

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

  /**
   * @brief The capacity of one operation scratch array after an operation that needs `needed`
   *        entries (scratch_rule): exact, max(capacity, needed); geometric, `capacity` (1 for
   *        the item arrays and 4 for the payload array when it is 0) grown by +capacity/2+1
   *        until it holds `needed`, unchanged when it already does.
   * @param[in] needed   The entries the operation needs.
   * @param[in] capacity The current capacity.
   * @param[in] empty    The start of a geometric growth from 0 (1 items, 4 payload values).
   * @param[in] limit    The largest capacity (INT_MAX for the item arrays).
   * @return The new capacity.
   */
  [[nodiscard]] long long scratch_capacity(long long needed, long long capacity, long long empty,
                                           long long limit) const;
};

}  // namespace dyng::detail::escher
