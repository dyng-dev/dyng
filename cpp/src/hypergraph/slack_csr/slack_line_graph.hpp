// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/include/hsosp.cuh (DeviceH2H)
/**
 * @file slack_line_graph.hpp
 * @brief The slack CSR line graph of slack_csr storage on the host (MOSP_ESCHER@4b86159's
 *        DeviceH2H, its build and its delta apply, kept by the host store of the sequential and
 *        openmp backends; ADR 0086).
 *
 * Node r (hyperedge id r, r < node_capacity) has deg[r] neighbour ids at
 * col_ind[row_start[r] .. row_start[r] + deg[r]) in a slot of cap[r] entries; node_weight[r] is
 * its weight (a dead id keeps the last weight it had, as on the device). The rows are sets in
 * no particular order after a batch (swap-remove deletions, appended insertions; a fresh build
 * sorts them). The layout rules are slack_layout.hpp's.
 */
#pragma once

#include "hypergraph/line_graph/incidence_delta.hpp"
#include "hypergraph/slack_csr/slack_layout.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail::hyper {

/// The slack CSR line graph (host).
class slack_line_graph {
 public:
  /**
   * @brief Build the CSR from the incidence lists (MOSP_ESCHER's buildCsr over its lineGraph()):
   *        the rows of the ids [0, num_nodes) (a dead id: degree 0, a slot of
   *        row_capacity_for(0)), the entry capacity entry_capacity(cursor, entry_headroom).
   * @tparam exec_t operators::sequential_exec or openmp_exec.
   * @param[in] exec           The executor.
   * @param[in] rows           The members of every id (sorted; dead ids empty).
   * @param[in] v2h            The ids of every vertex.
   * @param[in] num_nodes      The nodes with rows (the largest id bound so far).
   * @param[in] node_capacity  The node-indexed arrays' size (>= num_nodes).
   * @param[in] entry_headroom The entry headroom (>= 1).
   * @param[in] weights        The weight of every id below num_nodes (others keep theirs).
   * @throws invalid_argument_error for an entry headroom below 1 or NaN.
   * @throws capacity_error if the entries exceed 2^31 - 1 (the int32 offsets of the device).
   */
  template <typename exec_t>
  void build(const exec_t& exec, const std::vector<std::vector<std::int32_t>>& rows,
             const std::vector<std::vector<std::int64_t>>& v2h, std::int64_t num_nodes,
             std::int64_t node_capacity, double entry_headroom,
             const std::vector<std::int64_t>& weights);

  /**
   * @brief Apply a batch's pair delta (applyDeltaPairs / applyRowsWarp): every row's deletions
   *        (swap-remove of the first occurrence), then its insertions appended, a row that
   *        outgrows its slot moved to the tail; the weights of the new ids. num_entries changes by
   *        2 x (inserted - deleted) pairs either way.
   * @tparam exec_t operators::sequential_exec or openmp_exec.
   * @param[in] exec      The executor.
   * @param[in] delta     The pairs.
   * @param[in] num_nodes The nodes after the batch (<= node_capacity()).
   * @param[in] new_ids   The ids inserted by the batch.
   * @param[in] new_weights Their weights.
   * @return false if the relocations do not fit the tail (the caller rebuilds: the rows are then
   *         undefined).
   */
  template <typename exec_t>
  bool apply(const exec_t& exec, const line_graph_delta& delta, std::int64_t num_nodes,
             const std::vector<std::int64_t>& new_ids,
             const std::vector<std::int64_t>& new_weights);

  /**
   * @brief Grow the node-indexed arrays to at least `needed` nodes (by half of the capacity at
   *        least; the new nodes have no row and no slot).
   * @param[in] needed The nodes needed.
   */
  void grow_nodes(std::int64_t needed);

  /// @brief Set the weight of an id. @param[in] id The id. @param[in] w The weight.
  void set_weight(std::int64_t id, std::int64_t w) {
    node_weight_[static_cast<std::size_t>(id)] = w;
  }

  /**
   * @brief The rows sorted (the public h2h view): neighbours of node r, increasing, for r below
   *        `num_rows`.
   * @param[in]  num_rows Rows (the id bound).
   * @param[out] row_ptr  Offsets.
   * @param[out] col_ind  Neighbours.
   * @param[out] out      Per row: the offset of its first neighbour above r.
   */
  void sorted_rows(std::int64_t num_rows, std::vector<std::int64_t>& row_ptr,
                   std::vector<std::int64_t>& col_ind, std::vector<std::int64_t>& out) const;

  /**
   * @brief The layout's consistency and the rows against the incidence lists: every row of an id
   *        below num_nodes equals (as a set) the hyperedges that share a vertex with it.
   * @param[in] rows The members of every id.
   * @param[in] v2h  The ids of every vertex.
   * @return The problems (empty: none).
   */
  [[nodiscard]] std::vector<std::string> check(
      const std::vector<std::vector<std::int32_t>>& rows,
      const std::vector<std::vector<std::int64_t>>& v2h) const;

  /// @brief Nodes with rows. @return num_nodes.
  [[nodiscard]] std::int64_t num_nodes() const noexcept {
    return num_nodes_;
  }
  /// @brief The node capacity. @return It.
  [[nodiscard]] std::int64_t node_capacity() const noexcept {
    return static_cast<std::int64_t>(deg_.size());
  }
  /// @brief Entries (both directions of every pair). @return num_entries.
  [[nodiscard]] std::int64_t num_entries() const noexcept {
    return num_entries_;
  }
  /// @brief Used slots (the tail cursor). @return used_entries.
  [[nodiscard]] std::int64_t used_entries() const noexcept {
    return used_entries_;
  }
  /// @brief The entry capacity. @return cap_entries.
  [[nodiscard]] std::int64_t cap_entries() const noexcept {
    return static_cast<std::int64_t>(col_ind_.size());
  }
  /// @brief Rows relocated to the tail so far. @return The count.
  [[nodiscard]] std::int64_t relocations() const noexcept {
    return relocations_;
  }
  /// @brief Slots taken at the tail by moved rows so far. @return The count.
  [[nodiscard]] std::int64_t tail_entries() const noexcept {
    return tail_entries_;
  }
  /// @brief Node-capacity growths so far. @return The count.
  [[nodiscard]] std::int64_t node_growths() const noexcept {
    return node_growths_;
  }
  /// @brief The arrays (the slack-row accessor). @return Pointers into them.
  [[nodiscard]] const std::int64_t* row_start() const noexcept {
    return row_start_.data();
  }
  /// @brief Degrees. @return deg.
  [[nodiscard]] const std::int32_t* deg() const noexcept {
    return deg_.data();
  }
  /// @brief Slots. @return cap.
  [[nodiscard]] const std::int32_t* cap() const noexcept {
    return cap_.data();
  }
  /// @brief Neighbour ids. @return col_ind.
  [[nodiscard]] const std::int32_t* col_ind() const noexcept {
    return col_ind_.data();
  }
  /// @brief Weights by node. @return node_weight.
  [[nodiscard]] const std::int64_t* node_weight() const noexcept {
    return node_weight_.data();
  }
  /// @brief Host bytes of the arrays. @return Bytes.
  [[nodiscard]] std::int64_t bytes() const noexcept;

 private:
  std::int64_t num_nodes_ = 0;
  std::int64_t used_entries_ = 0;
  std::int64_t num_entries_ = 0;
  std::int64_t relocations_ = 0;
  std::int64_t tail_entries_ = 0;
  std::int64_t node_growths_ = 0;
  std::vector<std::int64_t> row_start_;
  std::vector<std::int32_t> deg_, cap_;
  std::vector<std::int32_t> col_ind_;
  std::vector<std::int64_t> node_weight_;
};

}  // namespace dyng::detail::hyper
