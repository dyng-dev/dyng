// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file id_allocator.hpp
 * @brief The host best-fit id allocator: the ids, deleted slots and payload accounting of the
 *        ESCHER store's H2V tree, reproduced on the host (ADR 0041, section 5).
 *
 * Written from the store's documented contract (cpp/src/hypergraph/escher/README.md) and the
 * originals' algorithms, independently of dyng::testing::cbst_host_model (which checks it in
 * lockstep): the host backends give the same ids as the CUDA store for every batch. Keys are
 * the store's (1-based, key = id + 1); the allocator keeps, per key, whether it is in the tree,
 * live or a deleted slot, its slot length, its row's value count and its tail segment, and for
 * the tree the bump pointer and the capacity of the payload, under the store policy of the
 * hypergraph's capacity_policy.
 */
#pragma once

#include "hypergraph/escher/store_policy.hpp"
#include "hypergraph/store.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail::hyper {

/// The best-fit id allocator of one tree.
class id_allocator {
 public:
  /// The state of a key.
  enum class key_state : std::uint8_t {
    absent,  ///< not in the tree
    live,    ///< a live row
    dead,    ///< a deleted slot (reusable by best fit until a surplus rebuild drops it)
  };

  /**
   * @brief An empty tree.
   * @param[in] policy   The store policy (growth, compaction, alignment, hysteresis).
   * @param[in] capacity The payload capacity.
   * @param[in] name     The tree's name (messages).
   */
  id_allocator(escher::store_policy policy, std::int64_t capacity, std::string name = "h2v");

  /**
   * @brief The tree over rows of `counts[i]` values for the keys 1..n (padding rule 1: a row of
   *        c values takes `((c + 4) / 4) * 4` slots), replacing any earlier tree.
   * @param[in] counts The value count of every row.
   * @throws invalid_argument_error if the rows need more than the capacity.
   */
  void construct(const std::vector<std::int32_t>& counts);

  /**
   * @brief Restore a state (a snapshot of any backend's store).
   * @param[in] state The state.
   */
  void restore(const allocator_state& state);

  /**
   * @brief The state (for a copy to another backend).
   * @return The state.
   */
  [[nodiscard]] allocator_state state() const;

  /**
   * @brief Insert rows of the given value counts: best-fit reuse of the deleted slots, then the
   *        surplus rows after the last live key (ESCHER+'s insert).
   * @param[in] counts The value count of every item.
   * @return The key of every item.
   * @throws capacity_error if the payload cannot hold the surplus rows (the allocator is then
   *         undefined, as the store's tree).
   */
  [[nodiscard]] std::vector<std::int32_t> insert(const std::vector<std::int32_t>& counts);

  /**
   * @brief Mark keys as deleted slots (keys not in the tree are ignored).
   * @param[in] keys The keys.
   */
  void erase(const std::vector<std::int32_t>& keys);

  /**
   * @brief Append values to live rows (a fill): into the tail segment, the rest into an overflow
   *        segment of `((k + 4) / 4) * 4` slots at the bump pointer (padding rule 3).
   * @param[in] keys   Distinct live keys.
   * @param[in] counts The values appended to each.
   * @throws capacity_error if the payload cannot hold the overflow segments (undefined after).
   */
  void append(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& counts);

  /**
   * @brief Compact the payload: every row one segment, in key order; a live row of c values
   *        takes `((c + 4) / 4) * 4` slots, a deleted slot keeps its length.
   * @return The new bump pointer.
   * @throws capacity_error if the compacted rows exceed the capacity and the policy does not
   *         grow (nothing changes then).
   */
  std::int64_t compact();

  /// @brief The state of a key. @param[in] key The key. @return Its state.
  [[nodiscard]] key_state state_of(std::int64_t key) const noexcept;
  /// @brief The slot length of a key in the tree. @param[in] key The key. @return The length.
  [[nodiscard]] std::int32_t length(std::int64_t key) const noexcept;
  /// @brief The value count of a live key. @param[in] key The key. @return The count.
  [[nodiscard]] std::int32_t count(std::int64_t key) const noexcept;
  /// @brief The largest key in the tree (0: empty). @return The key.
  [[nodiscard]] std::int64_t max_key() const noexcept {
    return max_key_;
  }
  /// @brief Live keys. @return The count.
  [[nodiscard]] std::int64_t live() const noexcept {
    return live_;
  }
  /// @brief Deleted slots in the tree (D). @return The count.
  [[nodiscard]] std::int64_t dead() const noexcept {
    return dead_;
  }
  /// @brief The bump pointer. @return Used payload entries.
  [[nodiscard]] std::int64_t bump() const noexcept {
    return bump_;
  }
  /// @brief The payload capacity. @return Entries.
  [[nodiscard]] std::int64_t capacity() const noexcept {
    return capacity_;
  }
  /// @brief Payload growths. @return The count.
  [[nodiscard]] std::int64_t growths() const noexcept {
    return growths_;
  }
  /// @brief Payload compactions. @return The count.
  [[nodiscard]] std::int64_t compactions() const noexcept {
    return compactions_;
  }
  /// @brief The reuse path counters (thread, warp, block) of the inserts so far.
  /// @return The counters.
  [[nodiscard]] const std::array<std::int64_t, 3>& reuse_paths() const noexcept {
    return reuse_paths_;
  }
  /// @brief The append path counters of the fills so far. @return The counters.
  [[nodiscard]] const std::array<std::int64_t, 3>& fill_paths() const noexcept {
    return fill_paths_;
  }
  /// @brief The policy. @return The store policy.
  [[nodiscard]] const escher::store_policy& policy() const noexcept {
    return policy_;
  }

  /**
   * @brief The allocator's own consistency check: the counters, the slot lengths of live rows,
   *        the payload accounting.
   * @return The problems (empty: none).
   */
  [[nodiscard]] std::vector<std::string> check() const;

 private:
  struct slot {
    key_state state = key_state::absent;
    std::int32_t length = 0;         // slots of the first segment, terminator included
    std::int32_t count = 0;          // values of the row
    std::int32_t tail_capacity = 0;  // usable slots of the tail segment
    std::int32_t occupancy = 0;      // values in the tail segment
  };

  void ensure_capacity(std::int64_t needed);
  void keep_headroom_after_compaction(std::int64_t compactions_before);
  void fill_impl(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& counts,
                 bool allow_compact);
  slot& at(std::int64_t key);

  escher::store_policy policy_;
  std::string name_;
  std::vector<slot> slots_;  // index = key (0 unused)
  std::int64_t max_key_ = 0;
  std::int64_t live_ = 0;
  std::int64_t dead_ = 0;
  std::int64_t bump_ = 0;
  std::int64_t capacity_ = 0;
  std::int64_t growths_ = 0;
  std::int64_t compactions_ = 0;
  std::array<std::int64_t, 3> reuse_paths_{};
  std::array<std::int64_t, 3> fill_paths_{};
};

}  // namespace dyng::detail::hyper
