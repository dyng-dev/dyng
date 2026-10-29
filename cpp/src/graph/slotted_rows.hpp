// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:headers/dynamicGraph.h (DeviceGraph: the fixed slot
// layout, rowEnd, the input contract) and src/dynLP.cu (the host net effect of a batch)
/**
 * @file slotted_rows.hpp
 * @brief row_layout::slotted on the host (ADR 0069): the per-vertex state and row ends next to
 *        the slot storage, and the functions of graph::with_capacity(), Step 0 of
 *        batch_semantics::net_effect(), the host apply, to_csr() and check_integrity().
 *
 * The slot storage is the graph's host CSR (graph_impl::host_edges()): row_ptr holds the fixed
 * slot ranges (the prefix sum of the capacities), col_ind and weights every slot, -1 / 0 in the
 * unused ones (LabelPropagation-CUDA's createDeviceGraph clears them the same way). The rest of a
 * slotted graph's state is slotted_rows below: the end of each row's entries, the state of each
 * vertex, the counts the queries return in O(1), and two scratch arrays of Step 0.
 *
 * The rules (dynamicGraph.h's INPUT CONTRACT and runDynLP's host pass b1, ADR 0069):
 *   - vertex ids are single-use: a vertex is inserted at most once, and an id named in a deletion
 *     is never inserted afterwards; Ins \ Del (in list order) become active, every id of Del ends
 *     deleted (an id in both never enters the graph; deleting an id never inserted retires it);
 *   - a batch's edges are appended to the rows of both endpoints, in batch order (the original's
 *     stable radix sort by source), each edge whose two endpoints are active after the batch;
 *     rows only grow and never move, so a deleted vertex keeps its entries and the entries
 *     pointing to it stay, and every reader skips them;
 *   - unlike the original, which drops an append past the end of its row and sets a sticky flag
 *     that it reads after the batch, the host path refuses such a batch with capacity_error before
 *     anything changes, and the other contract violations (an id out of range, inserted twice,
 *     re-inserted) are refused before anything changes too, as invalid_argument_error.
 */
#pragma once

#include "graph/normalized_batch.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail {

template <typename vertex_t>
struct apply_delta;

/**
 * @brief The state of a slotted graph beyond its slot storage.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct slotted_rows {
  std::vector<edge_t> row_end;        ///< per vertex: the end of its entries (a slot index)
  std::vector<vertex_status> status;  ///< per vertex: its state
  edge_t live_entries = 0;            ///< entries whose two endpoints are active (num_edges())
  vertex_t active_vertices = 0;       ///< vertices whose state is active
  /// Scratch of Step 0: per vertex, bit 0 = in this batch's deletions, bit 1 = in its
  /// insertions. All 0 between calls (Step 0 clears what it set, also when it throws).
  mutable std::vector<std::uint8_t> marks;
  /// Scratch of the capacity check: per row, the entries this batch appends (saturating). All 0
  /// between calls.
  mutable std::vector<std::uint32_t> counts;
};

/**
 * @brief Whether `props` are the slotted rows of 0.4, and if not why: not_supported_error naming
 *        the property (directed, order, parallel_edges, semantics) and the value 0.4 needs.
 * @param[in] props The properties given to graph::with_capacity().
 * @throws invalid_argument_error if props.layout is compact (with_capacity() builds slotted rows).
 * @throws not_supported_error    for row_layout::slack (M10b) or an unsupported property.
 */
void expect_slotted_properties(const graph_properties& props);

/**
 * @brief Build the empty slotted storage: `n` vertices, all inactive, row v with
 *        capacity[v] slots, every slot -1 / 0.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  n           The vertex count (>= 0).
 * @param[in]  capacity    The slots of each vertex (host memory; n values >= 0).
 * @param[in]  num_weights 1 (0 for weight_t = unweighted).
 * @param[out] slots       The slot storage.
 * @param[out] rows        The row ends (empty rows), the states (inactive) and the scratch.
 * @param[in]  fill_slots  Whether to allocate and clear the slot columns and weights (false: the
 *                         rows are built on the device, ADR 0074; the host slots stay empty).
 * @throws invalid_argument_error if the sizes or a capacity are wrong.
 * @throws capacity_error         if the sum of the capacities does not fit edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_slotted(std::int64_t n, array_view<const edge_t> capacity, int num_weights,
                   csr<vertex_t, edge_t, weight_t>& slots, slotted_rows<vertex_t, edge_t>& rows,
                   bool fill_slots = true);

/**
 * @brief Step 0 of batch_semantics::net_effect() against G_t: every check of a vertex batch
 *        (graph::apply()'s list), then the net effect (Ins_eff, its labels, Del_eff) into `out`.
 *
 * Nothing of the graph changes (the scratch arrays are cleared again, also when it throws).
 * Sets out.vertex_step; out.state_id is not set (the caller records the state).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  slots The slot storage of G_t.
 * @param[in]  rows  The row ends and states of G_t.
 * @param[in]  batch The batch (host memory).
 * @param[out] out   Receives the net effect (its vectors are resized).
 * @param[in]  check_capacity Whether to refuse an append past the end of its row here (the host
 *                   apply; the device apply records it on the device instead, ADR 0074, and
 *                   needs no host row ends).
 * @throws invalid_argument_error for a malformed batch or a contract violation (see graph::apply).
 * @throws not_supported_error    for edge deletions.
 * @throws capacity_error         if an append would pass the end of its row (only with
 *                                `check_capacity`; the device apply checks it on the device).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_vertex_batch(const csr<vertex_t, edge_t, weight_t>& slots,
                            const slotted_rows<vertex_t, edge_t>& rows,
                            const edge_batch_view<vertex_t, weight_t>& batch,
                            normalized_batch<vertex_t>& out, bool check_capacity = true);

/**
 * @brief The host part of the device apply of a vertex batch (ADR 0074): the vertex states and
 *        the active count (as apply_vertex_batch_host() changes them; the device copy changes the
 *        rows and counts the live entries), and the applied record without the edge lists.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in,out] rows       The states and counts (row_end is not touched).
 * @param[in]     deleted    The batch's deletions.
 * @param[in]     normalized Its Step 0.
 * @param[out]    delta      The applied record (may be nullptr): Ins_eff with its labels, Del_eff,
 *                           edges_on_device set.
 * @return The counters of the batch (the edge counts are completed when the report is settled).
 * @throws out_of_memory_error if the record cannot be allocated (before anything changes).
 */
template <typename vertex_t, typename edge_t>
apply_summary apply_vertex_states_host(slotted_rows<vertex_t, edge_t>& rows,
                                       array_view<const vertex_t> deleted,
                                       const normalized_batch<vertex_t>& normalized,
                                       apply_delta<vertex_t>* delta);

/**
 * @brief Apply a batch whose Step 0 is `normalized` to the slotted storage (in place): the
 *        deletions, the effective insertions, then the appends in batch order.
 *
 * Precondition: `normalized` is normalize_vertex_batch() of this state and batch. Nothing here
 * throws once the record's arrays are reserved (they are reserved first), so the graph is changed
 * completely or not at all.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in,out] slots      The slot storage.
 * @param[in,out] rows       The row ends, states and counts.
 * @param[in]     batch      The batch (host memory).
 * @param[in]     normalized Its Step 0.
 * @param[out]    delta      The applied record (may be nullptr): Ins_eff with its labels,
 *                           Del_eff, and the appended edges, each once in batch order with its
 *                           position in the batch.
 * @return The counters of the batch.
 * @throws out_of_memory_error if the record cannot be allocated (before anything changes).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_vertex_batch_host(csr<vertex_t, edge_t, weight_t>& slots,
                                      slotted_rows<vertex_t, edge_t>& rows,
                                      const edge_batch_view<vertex_t, weight_t>& batch,
                                      const normalized_batch<vertex_t>& normalized,
                                      apply_delta<vertex_t>* delta);

/**
 * @brief The compacted edges of a slotted graph (graph::to_csr()): per row, its entries whose two
 *        endpoints are active, in slot order.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  slots The slot storage.
 * @param[in]  rows  The row ends and states.
 * @param[out] out   The CSR (num_vertices() rows).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void compact_slotted(const csr<vertex_t, edge_t, weight_t>& slots,
                     const slotted_rows<vertex_t, edge_t>& rows,
                     csr<vertex_t, edge_t, weight_t>& out);

/**
 * @brief The invariants of a slotted graph (graph::check_integrity()); a description of the first
 *        violation.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] slots       The slot storage.
 * @param[in] rows        The row ends, states and counts.
 * @param[in] num_weights The graph's weight columns.
 * @return An empty string if every invariant holds.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::string slotted_integrity_violation(const csr<vertex_t, edge_t, weight_t>& slots,
                                        const slotted_rows<vertex_t, edge_t>& rows,
                                        int num_weights);

}  // namespace dyng::detail
