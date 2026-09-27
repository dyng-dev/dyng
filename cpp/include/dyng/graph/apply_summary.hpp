// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file apply_summary.hpp
 * @brief apply_summary: what applying one batch did to a graph.
 * @ingroup graph
 */
#pragma once

#include <cstdint>

namespace dyng {

/**
 * @brief Counters returned by graph::apply() and contained in every algorithm's stats.
 *
 * All counters are deterministic. For an undirected graph every batch edge counts once per
 * stored direction.
 * @ingroup graph
 */
struct apply_summary {
  std::int64_t inserted_edges = 0;      ///< insertions that added a new edge to a row
  std::int64_t updated_edges = 0;       ///< insertions that overwrote an existing edge (upsert)
  std::int64_t deleted_edges = 0;       ///< deletions that removed an edge
  std::int64_t ignored_deletions = 0;   ///< deletions of edges that did not exist
  std::int64_t dropped_self_loops = 0;  ///< self-loop operations skipped (self_loop::drop)
  std::int64_t cancelled_pairs = 0;     ///< insert/delete pairs cancelled (set semantics; from M2)
  std::int64_t inserted_vertices = 0;   ///< vertices added (explicitly or by vertex growth)
  std::int64_t deleted_vertices = 0;    ///< vertices removed
  std::int64_t num_vertices_after = 0;  ///< the vertex count after the batch
  std::int64_t ignored_insertions = 0;  ///< insertions of existing edges left unchanged (ignore)
};

}  // namespace dyng
