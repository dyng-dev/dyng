// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/directed_graph.cpp (prepare_batch,
// has_edge) and include/cycle_enum/dynamic/edge_change.hpp (EdgeChange, its order)
/**
 * @file structural_change.hpp
 * @brief Step 0 for algorithms that read G_t: the net structural change of a batch, computed on
 *        the graph BEFORE the batch is applied (without applying it).
 *
 * An aggregate-delta algorithm (cycle_count) subtracts what the deleted edges contribute on G_t
 * before the commit and adds what the inserted edges contribute on G_{t+1} after it (PLAN Section
 * 4.5.1, invariant I1). It therefore needs the deletions of the batch, normalized, while G_t still
 * exists. This is CycleEnumeration-GPU's prepare_batch() generalized to every batch_semantics:
 *
 * - both lists are sorted by (source, destination) without repeats (EdgeChange's order); the
 *   position of a change in its list is its ownership id;
 * - self-loops are dropped (they lie on no cycle of length >= 2), as are ids < 0 and, without
 *   vertex growth, insertions naming a vertex >= n (the commit rejects those batches anyway);
 * - an undirected graph changes both directions of every batch edge;
 * - deletions: the requested edges that exist in G_t;
 * - insertions: the requested edges that do not exist in G_t; with deletions_first, an edge the
 *   batch both deletes and inserts stays in both lists (removed, then added back); without it
 *   (insertions first), an insertion of a new edge that the batch also deletes changes nothing
 *   and leaves the insertions.
 *
 * Under batch_semantics::set() the two lists equal the normalized batch that graph::apply()
 * computes (apply_delta), and CycleEnumeration-GPU's prepare_batch() exactly. Under the other
 * semantics of a simple graph (upsert, ignore, error) they are the net change of the edge SET:
 * weight-only upserts are no-ops.
 */
#pragma once

#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <vector>

namespace dyng::detail {

/**
 * @brief One directed edge change (CycleEnumeration-GPU's EdgeChange).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct edge_change {
  vertex_t source;  ///< tail of the edge
  vertex_t target;  ///< head of the edge

  /**
   * @brief Lexicographic order by source, then target (the order of the ownership ids).
   * @param[in] other The other change.
   * @return true if this change sorts first.
   */
  [[nodiscard]] bool operator<(const edge_change& other) const noexcept {
    return source != other.source ? source < other.source : target < other.target;
  }

  /**
   * @brief Equality by source and target.
   * @param[in] other The other change.
   * @return true if both name the same edge.
   */
  [[nodiscard]] bool operator==(const edge_change& other) const noexcept {
    return source == other.source && target == other.target;
  }
};

/**
 * @brief The net structural change of a batch: two sorted, duplicate-free change lists.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct structural_change {
  std::vector<edge_change<vertex_t>> deletions;   ///< edges of G_t the batch removes
  std::vector<edge_change<vertex_t>> insertions;  ///< edges the batch adds (or re-adds)
  std::vector<edge_change<vertex_t>> requested;   ///< scratch: the requested deletions
};

/**
 * @brief Compute the net structural change of `batch` on the graph `g` (G_t), without changing
 *        anything.
 *
 * The capacity of the vectors in `out` is reused (no allocation once they are large enough).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored).
 * @param[in]  g     The out-edge CSR of G_t (host memory; sorted rows without parallel edges).
 * @param[in]  batch The batch (host memory).
 * @param[in]  props The graph's properties (direction and batch semantics).
 * @param[out] out   The change lists.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void compute_structural_change(const csr_view<vertex_t, edge_t, weight_t>& g,
                               const edge_batch_view<vertex_t, weight_t>& batch,
                               const graph_properties& props, structural_change<vertex_t>& out);

}  // namespace dyng::detail
