// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_view.hpp
 * @brief graph_view<V,E,W>: a read-only, non-owning description of a graph's storage.
 * @ingroup graph
 */
#pragma once

#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstdint>

namespace dyng {

/**
 * @brief A read-only description of the storage of a graph at one version.
 *
 * A view is invalidated by the next graph::apply() (compare `version` with graph::version()).
 * For row_layout::compact, `out` is the exact out-edge CSR and `in` the in-edge CSR (the
 * transposed graph, rows indexed by destination, each row listing sources in increasing order of
 * their out-edge position).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct graph_view {
  csr_view<vertex_t, edge_t, weight_t> out;  ///< out-edges (rows indexed by source)
  csr_view<vertex_t, edge_t, weight_t> in;   ///< in-edges; empty unless has_transposed
  bool has_transposed = false;               ///< whether `in` describes the in-edges
  bool directed = true;                      ///< false: every edge is stored in both directions
  row_layout layout = row_layout::compact;   ///< the storage layout of the rows
  std::uint64_t version = 0;                 ///< the graph version this view describes

  /**
   * @brief Number of vertices.
   * @return out.num_vertices().
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept {
    return out.num_vertices();
  }

  /**
   * @brief Number of stored (directed) edges.
   * @return out.num_edges().
   */
  [[nodiscard]] edge_t num_edges() const noexcept {
    return out.num_edges();
  }

  /**
   * @brief Number of weight columns.
   * @return out.num_weights.
   */
  [[nodiscard]] int num_weights() const noexcept {
    return out.num_weights;
  }
};

}  // namespace dyng
