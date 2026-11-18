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
  /// An apply on the device overflowed a row (its appends past the end of the row were dropped;
  /// ADR 0074): the graph is unusable. Kept with the host state, so that clone(), to_backend()
  /// and an apply on any resources see it (M10a review).
  bool poisoned = false;
  /// Scratch of Step 0: per vertex, bit 0 = in deletions, bit 1 = in insertions, bits 2-7 the
  /// tag of the Step 0 that set them; one more byte, the last tag (step_marks; no clearing).
  mutable std::vector<std::uint8_t> marks;
  /// Scratch of the capacity check: per row, the entries this batch appends (saturating). All 0
  /// between calls.
  mutable std::vector<std::uint32_t> counts;

  // ---- row_layout::slack (ADR 0103; empty / zero for slotted rows) ----
  /// Whether the rows are slack rows: row v owns the slots [slots.row_ptr[v], row_limit[v]), its
  /// entries are [slots.row_ptr[v], row_end[v]); slots.row_ptr[n] is the size of the slot arena,
  /// and slots.row_ptr is not monotone once a row moved to the tail.
  bool slack = false;
  double headroom = 0.125;         ///< graph_properties::headroom
  bool allow_growth = false;       ///< batch_semantics::allow_vertex_growth
  std::vector<edge_t> row_limit;   ///< per vertex: the end of its slots
  edge_t dead_slots = 0;           ///< arena slots no row owns (left by moved rows), -1 / 0
  std::int64_t relocations = 0;    ///< rows moved to the tail since construction
  std::int64_t compactions = 0;    ///< compactions since construction
  std::uint64_t layout_epoch = 0;  ///< +1 whenever a row's start or capacity changes
};

/**
 * @brief The slots a slack row gets for `entries` entries: ceil(entries * (1 + headroom)), 0 for
 *        none (ADR 0103).
 * @param[in] entries  The entries the row must hold (>= 0).
 * @param[in] headroom graph_properties::headroom (>= 0).
 * @return The capacity (>= entries).
 */
[[nodiscard]] std::int64_t slack_capacity(std::int64_t entries, double headroom) noexcept;

/**
 * @brief The slots row v owns in a slotted or slack graph: its capacity.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] slots The slot storage.
 * @param[in] rows  The rows.
 * @param[in] v     The vertex.
 * @return row_limit[v] - row_ptr[v] (slack), row_ptr[v + 1] - row_ptr[v] (slotted).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] std::int64_t row_capacity_of(const csr<vertex_t, edge_t, weight_t>& slots,
                                           const slotted_rows<vertex_t, edge_t>& rows,
                                           std::size_t v) noexcept {
  const auto end = rows.slack ? rows.row_limit[v] : slots.row_ptr[v + 1];
  return static_cast<std::int64_t>(end) - static_cast<std::int64_t>(slots.row_ptr[v]);
}

/**
 * @brief Build the empty slack storage (ADR 0103): `n` vertices, all inactive, row v with
 *        slack_capacity(capacity[v]) slots (0 where `capacity` has no entry), every slot -1 / 0.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  n           The vertex count (>= 0).
 * @param[in]  capacity    The expected entries of the first rows (host memory; at most n values
 *                         >= 0; empty or partial).
 * @param[in]  num_weights 1 (0 for weight_t = unweighted).
 * @param[in]  props       The properties (headroom, vertex growth).
 * @param[out] slots       The slot storage.
 * @param[out] rows        The rows.
 * @param[in]  fill_slots  Whether to allocate and clear the slot columns and weights.
 * @throws invalid_argument_error if the sizes or a capacity are wrong.
 * @throws capacity_error         if the slots do not fit edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_slack(std::int64_t n, array_view<const edge_t> capacity, int num_weights,
                 const graph_properties& props, csr<vertex_t, edge_t, weight_t>& slots,
                 slotted_rows<vertex_t, edge_t>& rows, bool fill_slots = true);

/**
 * @brief Step 0 of a vertex batch on slack rows: the checks of normalize_vertex_batch() (ids in
 *        range of the vertex count after a growth under allow_vertex_growth), no capacity check
 *        (a row that would overflow moves to the tail), and the layout step of the batch
 *        (normalized_batch::slack_moves, slack_compact, vertices_after).
 *
 * Nothing of the graph changes (its scratch may grow to the vertex count after the batch).
 * @throws invalid_argument_error for a malformed batch or a contract violation.
 * @throws not_supported_error    for edge deletions.
 * @throws capacity_error         if the rows would need more slots than edge_t holds.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_vertex_batch_slack(const csr<vertex_t, edge_t, weight_t>& slots,
                                  const slotted_rows<vertex_t, edge_t>& rows,
                                  const edge_batch_view<vertex_t, weight_t>& batch,
                                  normalized_batch<vertex_t>& out);

/**
 * @brief Grow the slack rows to `n_after` vertices (inactive, empty rows of no slots at the end
 *        of the arena); the scratch too.
 * @param[in,out] slots      The slot storage (row_ptr grows; the arena does not).
 * @param[in,out] rows       The rows.
 * @param[in]     n_after    The new vertex count (>= the current one).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void grow_slack_rows(csr<vertex_t, edge_t, weight_t>& slots, slotted_rows<vertex_t, edge_t>& rows,
                     std::int64_t n_after);

/**
 * @brief The per-row appends of a batch: for each edge whose two endpoints are active after the
 *        batch, both endpoints' rows.
 * @param[in]  status The vertex states after the batch.
 * @param[in]  batch  The batch.
 * @param[out] counts Per vertex (resized to the vertex count; zero elsewhere).
 */
template <typename vertex_t, typename weight_t>
void slack_append_counts(const std::vector<vertex_status>& status,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         std::vector<std::int64_t>& counts);

/**
 * @brief The capacities a compaction gives the rows (ADR 0103): a row with entries to keep or to
 *        receive gets slack_capacity(kept + appended); a deleted vertex's row gets none; every
 *        other row without entries (an inactive vertex's, or an active vertex's whose edges have
 *        not arrived or are gone) keeps its capacity (a hint of with_capacity() for a row never
 *        written); a row grown in this batch (beyond the rows before it) starts without slots.
 * @param[in]  slots    The slot storage (row starts and the arena) before the batch.
 * @param[in]  rows     The rows before the batch (their capacities).
 * @param[in]  status   The vertex states after the batch (its vertex count after a growth).
 * @param[in]  kept     Per vertex the entries the compaction keeps.
 * @param[in]  appended Per vertex the entries the batch appends.
 * @param[out] starts   The new row starts (n + 1: the last is the new arena size).
 * @param[out] limits   The new row limits.
 * @throws capacity_error if the arena does not fit edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void slack_compaction_layout(const csr<vertex_t, edge_t, weight_t>& slots,
                             const slotted_rows<vertex_t, edge_t>& rows,
                             const std::vector<vertex_status>& status,
                             const std::vector<std::int64_t>& kept,
                             const std::vector<std::int64_t>& appended, std::vector<edge_t>& starts,
                             std::vector<edge_t>& limits);

/**
 * @brief Apply a vertex batch to slack rows on the host (in place): the growth, the deletions,
 *        the effective insertions, the layout step (the planned moves, or a compaction), then the
 *        appends in batch order (rows keep their entries' order; a moved row keeps all of its
 *        entries, a compaction keeps the entries whose endpoint is not deleted).
 *
 * Precondition: `normalized` is normalize_vertex_batch_slack() of this state and batch. Nothing
 * throws once the record's arrays and the arena are reserved (first).
 * @return The counters of the batch (with apply_summary::relocated_rows and compactions).
 * @throws out_of_memory_error if the record or the arena cannot be allocated (before anything
 *         changes).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_vertex_batch_slack_host(csr<vertex_t, edge_t, weight_t>& slots,
                                            slotted_rows<vertex_t, edge_t>& rows,
                                            const edge_batch_view<vertex_t, weight_t>& batch,
                                            const normalized_batch<vertex_t>& normalized,
                                            apply_delta<vertex_t>* delta);

/**
 * @brief Compact the slack rows now (graph-level maintenance; the tests force it): every row
 *        rewritten in vertex order with the entries whose endpoint is not deleted, capacities by
 *        slack_compaction_layout() (nothing appended).
 * @param[in,out] slots The slot storage (host slots filled).
 * @param[in,out] rows  The rows.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void compact_slack_host(csr<vertex_t, edge_t, weight_t>& slots,
                        slotted_rows<vertex_t, edge_t>& rows);

/**
 * @brief The invariants of a slack graph (graph::check_integrity()); a description of the first
 *        violation, empty if every invariant holds.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::string slack_integrity_violation(const csr<vertex_t, edge_t, weight_t>& slots,
                                      const slotted_rows<vertex_t, edge_t>& rows, int num_weights);

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
 * Nothing of the graph changes (its scratch reads as clear again, also when it throws: the
 * marks by their tag, ADR 0079; the append counts are cleared).
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
