// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file csr.hpp
 * @brief csr<V,E,W> (owning host CSR) and csr_view<V,E,W> (non-owning, any memory space).
 * @ingroup graph
 *
 * Weights are stored OBJECTIVE-MAJOR: column k (objective k) holds one weight per edge, and the
 * weight of edge e in column k is `weights[k * num_edges() + e]`. The engines read one column at
 * a time, so a column is a contiguous array. (The MOSP text files are edge-major; the readers and
 * writers in dyng/io convert.)
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>

#include <cstddef>
#include <vector>

namespace dyng {

/**
 * @brief A non-owning description of a CSR graph: row offsets, neighbours, weight columns.
 *
 * Rows are indexed by source for out-edges and by destination for in-edges (the transposed
 * graph). The arrays may live in any memory space; element access needs a host-accessible space.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct csr_view {
  array_view<const edge_t> row_ptr;    ///< num_vertices() + 1 offsets, row_ptr[0] == 0
  array_view<const vertex_t> col_ind;  ///< the neighbour of each edge, in row order
  array_view<const weight_t> weights;  ///< num_weights columns of num_edges() values each
  int num_weights = 0;                 ///< number of weight columns (0: unweighted)

  /**
   * @brief Number of rows.
   * @return row_ptr.size() - 1, or 0 for an empty view.
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept {
    return row_ptr.empty() ? vertex_t{0} : static_cast<vertex_t>(row_ptr.size() - 1);
  }

  /**
   * @brief Number of edges.
   * @return col_ind.size().
   */
  [[nodiscard]] edge_t num_edges() const noexcept {
    return static_cast<edge_t>(col_ind.size());
  }

  /**
   * @brief The weights of one objective, one per edge in row order.
   * @param[in] k Objective in [0, num_weights).
   * @return A view of num_edges() weights.
   * @throws invalid_argument_error if `k` is out of range.
   */
  [[nodiscard]] array_view<const weight_t> weight_column(int k) const {
    DYNG_EXPECTS(k >= 0 && k < num_weights, "weight column ", k, " out of range [0, ", num_weights,
                 ")");
    const std::size_t m = col_ind.size();
    return weights.subview(static_cast<std::size_t>(k) * m, m);
  }
};

/**
 * @brief An owning host CSR graph with objective-major weight columns.
 *
 * A plain aggregate: the fields may be filled directly. graph::from_csr() validates it.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct csr {
  std::vector<edge_t> row_ptr;    ///< num_vertices() + 1 offsets, row_ptr[0] == 0
  std::vector<vertex_t> col_ind;  ///< the neighbour of each edge, in row order
  std::vector<weight_t> weights;  ///< num_weights columns of num_edges() values each
  int num_weights = 0;            ///< number of weight columns (0: unweighted)

  /**
   * @brief Number of rows.
   * @return row_ptr.size() - 1, or 0 if row_ptr is empty.
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept {
    return row_ptr.empty() ? vertex_t{0} : static_cast<vertex_t>(row_ptr.size() - 1);
  }

  /**
   * @brief Number of edges.
   * @return col_ind.size().
   */
  [[nodiscard]] edge_t num_edges() const noexcept {
    return static_cast<edge_t>(col_ind.size());
  }

  /**
   * @brief The weight of one edge in one column (unchecked).
   * @param[in] e Edge index in [0, num_edges()).
   * @param[in] k Objective in [0, num_weights).
   * @return weights[k * num_edges() + e].
   */
  [[nodiscard]] const weight_t& weight(edge_t e, int k) const noexcept {
    return weights[static_cast<std::size_t>(k) * col_ind.size() + static_cast<std::size_t>(e)];
  }

  /**
   * @brief A read-only view of the arrays.
   * @return A csr_view in host memory; valid while this object is alive and unmodified.
   */
  [[nodiscard]] csr_view<vertex_t, edge_t, weight_t> view() const noexcept {
    return csr_view<vertex_t, edge_t, weight_t>{host_view(row_ptr), host_view(col_ind),
                                                host_view(weights), num_weights};
  }
};

}  // namespace dyng
