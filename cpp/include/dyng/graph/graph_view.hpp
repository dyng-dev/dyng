// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_view.hpp
 * @brief graph_view<V,E,W>: a read-only, non-owning description of a graph's storage.
 * @ingroup graph
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstdint>

namespace dyng {

/**
 * @brief The state of a vertex of a graph with vertex batches (row_layout::slotted; 0.4).
 *
 * A vertex of graph::with_capacity() starts inactive; a batch's insertion makes it active, a
 * deletion makes it deleted. Ids are single-use: a deleted vertex never becomes active again,
 * and its row keeps its entries (rows are never compacted), which every reader skips (an entry
 * counts only when both of its endpoints are active). The enumerators are compared by name: O's
 * (LabelPropagation-CUDA's) VertexStatus maps by name, never by value (ADR 0069), and a seeded
 * vertex (O's VERTEX_GROUND_TRUTH) is active (its class is the result's seed label).
 * @ingroup graph
 */
enum class vertex_status : std::uint8_t {
  active,    ///< in the graph
  deleted,   ///< removed by a batch (or retired by a deletion before it was ever inserted)
  inactive,  ///< never inserted yet
};

/**
 * @brief A read-only description of the storage of a graph at one version.
 *
 * A view is invalidated by the next graph::apply() (compare `version` with graph::version()).
 * For row_layout::compact, `out` is the exact out-edge CSR and `in` the in-edge CSR (the
 * transposed graph, rows indexed by destination, each row listing sources in increasing order of
 * their out-edge position).
 *
 * For row_layout::slotted (graph::with_capacity()), `out` describes the slot storage: vertex v
 * owns the slots [out.row_ptr[v], out.row_ptr[v + 1]) and its entries are the slots up to
 * row_end[v]; the unused slots hold column -1 and weight 0. An entry is an edge of the graph only
 * when both of its endpoints are active (`status`): a deleted vertex keeps its entries and the
 * entries pointing to it stay in place. `in` is empty (the graph is undirected and stores each
 * edge in both rows). graph::to_csr() returns the compacted edges.
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
  /** @brief Per vertex its state (row_layout::slotted; empty for row_layout::compact, whose
   *         vertices are all active). */
  array_view<const vertex_status> status;
  /** @brief Per vertex the end of its entries in `out` (row_layout::slotted; empty for
   *         row_layout::compact, whose rows end at out.row_ptr[v + 1]). */
  array_view<const edge_t> row_end;

  /**
   * @brief Number of vertices.
   * @return out.num_vertices().
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept {
    return out.num_vertices();
  }

  /**
   * @brief Number of stored (directed) edges.
   * @return out.num_edges(): for row_layout::slotted the number of slots (graph::num_edges()
   *         counts the edges).
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
