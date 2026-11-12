// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slack_rows.cuh
 * @brief The device-side reading of slack CSR rows (rows with gaps: row v is
 *        col_ind[row_start[v] .. row_start[v] + deg[v]), each row in a slot of its own capacity),
 *        written once for every algorithm that reads such rows on the device (M11b, O12; ADR
 *        0095).
 *
 * slack_csr storage's line graph is such a CSR (hyper::slack_row_view, the host-side slack-row
 * accessor of M11a, hypergraph/slack_csr/slack_rows.hpp, gives its pointers); hyper_sssp's shared
 * engine reads it through this view (algorithms/hyper_sssp/operators.cuh). A slotted layout of a
 * graph (M10b's label_propagation) can read its rows through the same view. CUDA translation
 * units only.
 */
#pragma once

#include <cstdint>

namespace dyng::detail {

/**
 * @brief Read-only slack CSR rows in device memory.
 * @tparam vertex_t Row and neighbour id type.
 * @tparam offset_t Entry offset type (the slot starts).
 * @tparam degree_t Row length type.
 */
template <typename vertex_t, typename offset_t, typename degree_t = std::int32_t>
struct slack_rows {
  const offset_t* row_start = nullptr;  ///< per row: the offset of its slot
  const degree_t* deg = nullptr;        ///< per row: its entries (a dead row: 0)
  const vertex_t* col_ind = nullptr;    ///< the neighbour ids

  /// @brief First entry of row v. @param[in] v The row. @return Its offset.
  __device__ __forceinline__ offset_t begin(vertex_t v) const {
    return row_start[v];
  }
  /// @brief One past the last entry of row v. @param[in] v The row. @return The offset.
  __device__ __forceinline__ offset_t end(vertex_t v) const {
    return row_start[v] + static_cast<offset_t>(deg[v]);
  }
  /// @brief The neighbour of entry e. @param[in] e The entry. @return Its id.
  __device__ __forceinline__ vertex_t neighbour(offset_t e) const {
    return col_ind[e];
  }
};

}  // namespace dyng::detail
