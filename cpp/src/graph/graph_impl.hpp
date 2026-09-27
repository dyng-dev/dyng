// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_impl.hpp
 * @brief The private state of graph<V,E,W> and the internal access used by the algorithms.
 *
 * Algorithms (cpp/src/algorithms/<algo>) reach the storage through detail::graph_access. The
 * update of an algorithm applies the batch with graph_access::apply(), which also returns the
 * per-edge classification (apply_delta) the incremental engines need.
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstdint>
#include <vector>

namespace dyng::detail {

/**
 * @brief The effective changes of one applied batch, in the order they were applied.
 *
 * This is the per-edge classification of MOSP's applyChangeBatch (its weightIncreaseMask),
 * widened to any number of objectives: one byte per (insertion, objective).
 *
 * - insertions: every insertion that reached the rows (after self-loop dropping; for an
 *   undirected graph each direction), in batch order, including upserts and ignored ones;
 * - weight_increased[i * num_weights + k] = 1 if insertion i names an edge that existed in the
 *   graph before the batch and the final objective-k weight of the first (u,v) in row u is larger
 *   than the objective-k weight of the first (u,v) of row u before the batch (MOSP semantics);
 * - deletions: every requested deletion with both ends in range (after self-loop dropping; for an
 *   undirected graph each direction), in batch order, whether or not it matched an edge.
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct apply_delta {
  int num_weights = 0;                         ///< objectives per insertion
  std::vector<vertex_t> insert_src;            ///< tails of the insertions
  std::vector<vertex_t> insert_dst;            ///< heads of the insertions
  std::vector<std::uint8_t> weight_increased;  ///< num_weights flags per insertion
  std::vector<vertex_t> delete_src;            ///< tails of the deletions
  std::vector<vertex_t> delete_dst;            ///< heads of the deletions
};

/**
 * @brief The state behind graph<V,E,W>: properties, version and host CSR storage.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_impl {
 public:
  graph_properties props;               ///< properties (num_weights as stored)
  std::uint64_t version = 0;            ///< +1 per applied batch
  csr<vertex_t, edge_t, weight_t> out;  ///< out-edges
  csr<vertex_t, edge_t, weight_t> in;   ///< in-edges (valid if props.store_transposed)
};

/**
 * @brief Internal access to a graph's storage (for the algorithms and the tests).
 */
struct graph_access {
  /**
   * @brief The state of a graph.
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] g The graph.
   * @return Its implementation object.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static const graph_impl<vertex_t, edge_t, weight_t>& impl(
      const graph<vertex_t, edge_t, weight_t>& g) {
    return g.impl();
  }

  /**
   * @brief Apply a batch and report the effective changes (graph::apply plus classification).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in]     res   Execution resources.
   * @param[in,out] g     The graph (version + 1).
   * @param[in]     batch The batch.
   * @param[out]    delta The effective changes (may be nullptr).
   * @return What the batch did.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static apply_summary apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                             const edge_batch_view<vertex_t, weight_t>& batch,
                             apply_delta<vertex_t>* delta);
};

}  // namespace dyng::detail
