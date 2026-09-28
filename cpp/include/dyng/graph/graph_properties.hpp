// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_properties.hpp
 * @brief graph_properties: layout, row order, multigraph switch and batch semantics of a graph.
 * @ingroup graph
 *
 * The properties belong to the GRAPH, so every result maintained on one graph agrees on how a
 * batch is interpreted (PLAN Section 5.2, ADR 0010).
 */
#pragma once

#include <cstdint>

namespace dyng {

/**
 * @brief Storage layout of the rows of a graph.
 * @ingroup graph
 */
enum class row_layout : std::uint8_t {
  compact,  ///< exact CSR; a batch rebuilds the touched rows (the only layout in 0.1)
  slotted,  ///< fixed per-vertex slot ranges (DynLP); planned for 0.3
  slack,    ///< per-row headroom with a tail region; planned for 0.3
};

/**
 * @brief Order of the neighbours inside a row.
 * @ingroup graph
 */
enum class row_order : std::uint8_t {
  sorted,  ///< rows sorted by neighbour id (stable: parallel edges keep their relative order)
  append,  ///< surviving edges keep their order; new edges are appended to their row (MOSP)
};

/**
 * @brief Whether parallel edges (several (u,v) in one row) may exist.
 * @ingroup graph
 */
enum class multi_edges : std::uint8_t {
  forbid,  ///< a simple graph: duplicate (u,v) in the input are merged (the last one wins)
  allow,   ///< parallel edges are kept; deletions and upserts act on the first (u,v) in the row
};

/**
 * @brief How a batch of edge changes is interpreted.
 *
 * The default-constructed value equals upsert_last_wins(), the semantics of MOSP's
 * applyChangeBatch():
 * 1. deletions first, in batch order; each removes the FIRST remaining (u,v) of row u;
 * 2. then insertions in batch order; an insertion of an existing (u,v) overwrites the weights of
 *    the first remaining (u,v) (so the last insertion of a pair wins), otherwise the edge is
 *    appended to row u (or merged into it under row_order::sorted).
 * @ingroup graph
 */
struct batch_semantics {
  /**
   * @brief What an insertion of an edge that already exists does.
   */
  enum class existing_insert : std::uint8_t {
    upsert,  ///< overwrite the weights of the first existing (u,v)
    error,   ///< throw invalid_argument_error
    ignore,  ///< leave the existing edge unchanged
  };

  /**
   * @brief What a deletion of an edge that does not exist does.
   */
  enum class missing_delete : std::uint8_t {
    ignore,  ///< nothing (counted in apply_summary::ignored_deletions)
    error,   ///< throw invalid_argument_error
  };

  /**
   * @brief What an insertion or deletion of a self-loop (u,u) does.
   */
  enum class self_loop : std::uint8_t {
    keep,   ///< treat it like any other edge (MOSP)
    drop,   ///< skip it (counted in apply_summary::dropped_self_loops)
    error,  ///< throw invalid_argument_error
  };

  existing_insert on_existing_insert = existing_insert::upsert;  ///< insertion of an existing edge
  missing_delete on_missing_delete = missing_delete::ignore;     ///< deletion of a missing edge
  self_loop on_self_loop = self_loop::keep;                      ///< self-loops in the batch
  bool deletions_first = true;      ///< true: deletions, then insertions; false: the reverse
  bool allow_vertex_growth = true;  ///< an insertion naming v >= num_vertices grows the vertex set

  /**
   * @brief true: the deletions and the insertions are read as two SETS (Step 0 of set()).
   *
   * Before anything is applied, each list is sorted by (source, destination) and deduplicated
   * (the first insertion of a pair in batch order supplies its weights); deletions of edges that
   * do not exist and insertions of edges that exist are no-ops and leave the lists, except that an
   * edge the batch both deletes and inserts stays in both lists (it is removed, then added back;
   * apply_summary::cancelled_pairs counts these pairs). The two lists, in this order, are the net
   * structural change the algorithms see (for example the change edges of cycle_count, whose
   * position in the sorted list is their ownership id). The switches above keep their meaning:
   * `on_missing_delete` and `on_existing_insert` decide what the no-ops do (ignore or error),
   * `on_self_loop` what self-loops do, and `allow_vertex_growth` whether insertions may name new
   * vertices.
   *
   * Not supported with as_sets: `on_existing_insert = upsert` (an upsert of the weights of an
   * existing edge is not a structural change; delete and re-insert the edge to change its
   * weights), `deletions_first = false`, and rows that are not row_order::sorted with
   * multi_edges::forbid (the lists are merged into the sorted rows of a simple graph). A graph
   * constructed with such properties throws not_supported_error at construction.
   */
  bool as_sets = false;

  /**
   * @brief MOSP applyChangeBatch() / updateGraphCSR() semantics (the default).
   * @return Upsert, ignore missing deletions, keep self-loops, deletions first, vertex growth.
   */
  [[nodiscard]] static constexpr batch_semantics upsert_last_wins() noexcept {
    return batch_semantics{};
  }

  /**
   * @brief CycleEnumeration-GPU prepare_batch() / apply_batch() semantics: a batch is two sets.
   *
   * as_sets with: insertions of existing edges and deletions of missing edges ignored (dropped
   * from the normalized batch), self-loops dropped, deletions first, vertex growth. The graph
   * after the batch has the edges (E minus Del) plus Ins, without self-loops; the normalized batch equals the
   * original's prepare_batch() exactly.
   * @return The set semantics.
   */
  [[nodiscard]] static constexpr batch_semantics set() noexcept {
    batch_semantics s;
    s.on_existing_insert = existing_insert::ignore;
    s.on_missing_delete = missing_delete::ignore;
    s.on_self_loop = self_loop::drop;
    s.deletions_first = true;
    s.allow_vertex_growth = true;
    s.as_sets = true;
    return s;
  }
};

/**
 * @brief The properties of a graph: direction, stored in-edges, weights, layout, row order,
 *        multigraph switch and batch semantics.
 * @ingroup graph
 */
struct graph_properties {
  bool directed = true;  ///< false: stored symmetric; a batch edge (u,v) changes both
  bool store_transposed =
      true;  ///< keep the in-edges (CSC, built on first use); sssp, mosp and cycle_count need them
  /** @brief Number of weight columns (objectives); 0 for an unweighted graph. graph::from_edges()
   *         and graph::from_csr() take the number of columns from their input and overwrite this
   *         field. */
  int num_weights = 1;
  row_layout layout = row_layout::compact;           ///< storage layout of the rows
  double headroom = 0.125;                           ///< spare capacity for row_layout::slack
  row_order order = row_order::sorted;               ///< the general default: sorted rows
  multi_edges parallel_edges = multi_edges::forbid;  ///< the general default: a simple graph
  batch_semantics semantics = batch_semantics::upsert_last_wins();  ///< how batches are applied

  /**
   * @brief Byte parity with MOSP's CSR: append order, parallel edges allowed, upsert_last_wins.
   * @return Directed, transposed stored, compact rows, row_order::append, multi_edges::allow.
   */
  [[nodiscard]] static constexpr graph_properties mosp_compatible() noexcept {
    graph_properties props;
    props.order = row_order::append;
    props.parallel_edges = multi_edges::allow;
    props.semantics = batch_semantics::upsert_last_wins();
    return props;
  }

  /**
   * @brief Parity with CycleEnumeration-GPU's DirectedGraph: sorted rows of a simple graph and
   *        set() batch semantics.
   *
   * A graph built with these properties from the same edges has the CSR of the original's
   * build_directed_graph(), and every applied batch gives the CSR of its apply_batch() (byte for
   * byte, row offsets and neighbours).
   * @return Directed, transposed stored, compact rows, row_order::sorted, multi_edges::forbid,
   *         batch_semantics::set(), no weight columns.
   */
  [[nodiscard]] static constexpr graph_properties cycle_enum_compatible() noexcept {
    graph_properties props;
    props.num_weights = 0;
    props.order = row_order::sorted;
    props.parallel_edges = multi_edges::forbid;
    props.semantics = batch_semantics::set();
    return props;
  }
};

}  // namespace dyng
