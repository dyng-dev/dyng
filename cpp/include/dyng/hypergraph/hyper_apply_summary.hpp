// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_apply_summary.hpp
 * @brief hyper_apply_summary: what applying one batch did to a hypergraph, and the ids it gave.
 * @ingroup hypergraph
 */
#pragma once

#include <cstdint>
#include <vector>

namespace dyng {

/**
 * @brief Returned by hypergraph::apply(): the counts of one batch and the id of every inserted
 *        hyperedge.
 *
 * The ids come from the store's best-fit reuse and are authoritative: read them here, never
 * predict them (a slot freed earlier, or in the same batch under id_reuse::erase_first, may be
 * reused, and the surplus rule may give out an id that was dead before the batch).
 * @ingroup hypergraph
 */
struct hyper_apply_summary {
  std::int64_t inserted_hyperedges = 0;  ///< hyperedges inserted
  std::int64_t deleted_hyperedges = 0;   ///< live hyperedges deleted
  std::int64_t inserted_incidences = 0;  ///< (hyperedge, vertex) incidences added by edits (0.4)
  std::int64_t deleted_incidences = 0;   ///< (hyperedge, vertex) incidences removed by edits (0.4)
  /// Every skipped operation: in 0.3 ignored_deletions + dropped_members; incidence edits (0.4)
  /// add the edits they skip (on a dead hyperedge, of a non-member, ...).
  std::int64_t skipped_ops = 0;
  std::int64_t ignored_deletions = 0;      ///< deletions of dead, out-of-range or repeated ids
  std::int64_t dropped_members = 0;        ///< repeated members of inserted hyperedges dropped
  std::int64_t num_hyperedges_after = 0;   ///< live hyperedges after the batch
  std::int64_t id_bound_after = 0;         ///< hyperedge_id_bound() after the batch
  std::vector<std::int64_t> inserted_ids;  ///< the id of every inserted hyperedge, batch order
};

}  // namespace dyng
