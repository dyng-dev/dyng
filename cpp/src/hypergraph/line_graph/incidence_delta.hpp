// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu (deriveDelta: the order-free rule)
/**
 * @file incidence_delta.hpp
 * @brief incidence_delta: the line-graph change of one batch from its incidence changes, exact
 *        and independent of the order of the batch's operations (PLAN 4.5.3's `translate`,
 *        MOSP_ESCHER@4b86159's deriveDelta; ADR 0086). The host form, one source on
 *        operators::sequential_exec and openmp_exec; the CUDA store's is M11a step 3's.
 *
 * The rule (hsospDelta.cu:19-25): a pair of hyperedges (a, b) can only change if a vertex was
 * added to or removed from one of them while it belonged to the other, and that incidence change
 * lists the pair as a candidate. So:
 *   1. the batch's (vertex, hyperedge, +-1) changes are reduced to a net sign per incidence;
 *   2. for every incidence (v, h) whose net sign is not 0, every o in v's hyperedge list before
 *      the batch (pass 0) and after it (pass 1) gives the candidate (min(h, o), max(h, o)) (o == h
 *      gives none; it still counts as a candidate, as in MOSP_ESCHER's count);
 *   3. the candidates are sorted and deduplicated, and each pair is classified by whether the
 *      vertex lists of its two hyperedges overlap before and after the batch: inserted if only
 *      after, deleted if only before.
 * The output depends only on the states before and after the batch, never on the order of the
 * operations or on the thread count.
 */
#pragma once

#include "hypergraph/store.hpp"

#include <cstdint>
#include <vector>

namespace dyng::detail::hyper {

/// The rows the delta reads, by hyperedge id and by vertex.
struct incidence_delta_rows {
  /// The rows of every id after the batch (sorted members; a dead id is empty).
  const std::vector<std::vector<std::int32_t>>* post_rows = nullptr;
  /// Per id: the index of its row before the batch in `pre_rows` (a hyperedge the batch touched:
  /// deleted, edited or inserted), or -1 (untouched: its row is the same before and after).
  /// Covers the ids of `post_rows` at least.
  const std::vector<std::int32_t>* touched_slot = nullptr;
  const member_rows* pre_rows = nullptr;  ///< the rows before the batch of the touched ids
  /// The hyperedges of the changed vertices before the batch: vertex `pre_vertices[i]` (the
  /// vertices increasing) had the ids pre_v2h_ids[pre_v2h_offsets[i] .. pre_v2h_offsets[i + 1]).
  const std::vector<std::int32_t>* pre_vertices = nullptr;
  const std::vector<std::int64_t>* pre_v2h_offsets = nullptr;
  const std::vector<std::int64_t>* pre_v2h_ids = nullptr;
  /// The hyperedges of every vertex after the batch.
  const std::vector<std::vector<std::int64_t>>* post_v2h = nullptr;
};

/**
 * @brief The net incidence changes of a batch: the (vertex, id, sign) changes reduced to one net
 *        sign per incidence, the incidences whose net sign is not 0 as keys
 *        `(vertex << 32) | id`, increasing.
 * @param[in]  vertices The vertex of every change.
 * @param[in]  ids      The hyperedge of every change (in [0, 2^31 - 1)).
 * @param[in]  signs    -1 (removed) or +1 (added).
 * @param[out] keys     The keys.
 * @param[out] net      Their net signs.
 */
void net_incidence_changes(const std::vector<std::int32_t>& vertices,
                           const std::vector<std::int64_t>& ids,
                           const std::vector<std::int8_t>& signs, std::vector<std::uint64_t>& keys,
                           std::vector<std::int8_t>& net);

/**
 * @brief The line-graph change of a batch (see the file comment).
 * @tparam exec_t operators::sequential_exec or operators::openmp_exec.
 * @param[in]  exec           The executor.
 * @param[in]  keys           The net incidence changes (net_incidence_changes()).
 * @param[in]  rows           The rows before and after the batch.
 * @param[in]  max_candidates The largest candidate count (INT_MAX: MOSP_ESCHER's limit).
 * @param[out] out            The delta.
 * @throws capacity_error if the batch has more than `max_candidates` candidate entries (nothing
 *         else changes).
 */
template <typename exec_t>
void incidence_delta(const exec_t& exec, const std::vector<std::uint64_t>& keys,
                     const incidence_delta_rows& rows, std::int64_t max_candidates,
                     line_graph_delta& out);

}  // namespace dyng::detail::hyper
