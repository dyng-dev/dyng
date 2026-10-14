// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/row_view.hpp (struct RowView)
/**
 * @file row_views.hpp
 * @brief The internal row views of a hypergraph state: ESCHER-GPU's key-indexed row view
 *        (escher::row_view, A's RowView) of h2v and of the line graph h2h, in the store's memory
 *        space (ADR 0042).
 *
 * The algorithms of M9 (triad_count) and M11 read a hypergraph through these views (reached with
 * detail::hypergraph_access::views()): rows by CBST key (key = hyperedge id + 1), the values as
 * the store keeps them (h2v: vertex + 1; h2h: neighbour key = neighbour id + 1), sorted and
 * contiguous, and `out` (h2h) the offset of the first entry greater than the row's key. The views
 * of the CUDA store point into device memory (the trees' payloads and A's locator arrays), those
 * of the host stores into host memory with the same layout, so one kernel or loop serves every
 * backend. They are rebuilt once per commit and invalidated by the next apply(). The public
 * views (incidence_view, line_graph_view) are decoded copies of them (0-based, no sentinels),
 * built on request. Host-only: no CUDA header.
 */
#pragma once

#include <dyng/core/memory.hpp>

#include <cstdint>

namespace dyng::detail::escher {

// Contiguous, key-indexed view of the rows of a CBST ("locator"). The
// counting kernels read rows through this offset table instead of walking
// the BST for every lookup; the CBST stays the structure that is maintained
// (delete / insert / fill / unfill) and the view is rebuilt from its nodes
// whenever the tree changes (one pass over the nodes).
struct row_view {
  const int* flat = nullptr;  // payload of the tree
  const int* loc = nullptr;   // key -> offset of the row in flat
  const int* len = nullptr;   // key -> number of entries (0 for absent keys)
  const int* out = nullptr;   // key -> offset of the first entry greater than key
                              //        (sorted adjacency rows only)
  int max_key = 0;            // arrays cover keys 0..max_key
};

}  // namespace dyng::detail::escher

namespace dyng::detail::hyper {

/**
 * @brief The internal row views of one hypergraph state (ESCHER-GPU's Snapshot rows: h2vRows,
 *        h2hRows), valid until the next apply().
 */
struct row_views {
  /// h2v by key 1..id_bound: the members + 1 of every live id, sorted; dead or absent keys have
  /// loc = len = 0 (A's locator); `out` is null.
  escher::row_view h2v;
  /// h2h by key 1..line_graph_keys (has_h2h only): the neighbour keys of every key, sorted; `out`
  /// set. line_graph_keys is ESCHER-GPU's numHyperedgeKeys, the largest id bound the hypergraph
  /// has had, so it can exceed id_bound (the rows above are empty).
  escher::row_view h2h;
  bool has_h2h = false;                     ///< whether the line graph is maintained
  memory_space space = memory_space::host;  ///< where the arrays are
  int device = -1;                          ///< the CUDA device (space == device)
  std::int64_t id_bound = 0;                ///< hyperedge_id_bound()
  std::int32_t num_vertices = 0;            ///< num_vertices()
};

}  // namespace dyng::detail::hyper
