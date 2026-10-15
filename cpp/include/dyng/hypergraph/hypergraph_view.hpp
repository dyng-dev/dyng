// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_view.hpp
 * @brief incidence_view (h2v, v2h), line_graph_view (h2h) and storage_stats of a hypergraph.
 * @ingroup hypergraph
 */
#pragma once

#include <dyng/core/array_view.hpp>

#include <cstdint>

namespace dyng {

/**
 * @brief A read-only CSR of one incidence relation of a hypergraph at one version: hyperedge to
 *        member vertices (h2v) or vertex to hyperedge ids (v2h).
 *
 * Row r lists `col_ind[row_ptr[r] .. row_ptr[r + 1])` in increasing order. Rows of h2v are
 * indexed by hyperedge id in [0, hyperedge_id_bound()); a dead id has an empty row. Rows of v2h
 * are indexed by vertex. The arrays live in the container's memory space (device memory for a
 * CUDA hypergraph) and are invalidated by the next hypergraph::apply() (compare `version` with
 * hypergraph::version()).
 * @tparam id_t The type of the listed ids (vertex_t for h2v, std::int64_t for v2h).
 * @ingroup hypergraph
 */
template <typename id_t>
struct incidence_view {
  array_view<const std::int64_t> row_ptr;  ///< num_rows() + 1 offsets, row_ptr[0] == 0
  array_view<const id_t> col_ind;          ///< the ids of every row, rows back to back
  std::uint64_t version = 0;               ///< the hypergraph version this view describes

  /**
   * @brief Number of rows.
   * @return row_ptr.size() - 1, or 0 for an empty view.
   */
  [[nodiscard]] std::int64_t num_rows() const noexcept {
    return row_ptr.empty() ? 0 : static_cast<std::int64_t>(row_ptr.size()) - 1;
  }

  /**
   * @brief Number of entries (incidences).
   * @return col_ind.size().
   */
  [[nodiscard]] std::int64_t num_entries() const noexcept {
    return static_cast<std::int64_t>(col_ind.size());
  }
};

/**
 * @brief A read-only view of the line graph (h2h) of a hypergraph at one version: every live
 *        hyperedge to the other hyperedges it shares at least one vertex with.
 *
 * Rows are indexed by hyperedge id, sorted and contiguous; a dead id has an empty row. `out[r]`
 * is the offset of the first neighbour of row r with an id greater than r. `row_ptr`, `col_ind`,
 * `out` and `overlaps` live in the container's memory space; `weights` is host memory on every
 * backend (each array_view carries its space). All are invalidated by the next
 * hypergraph::apply().
 * @tparam weight_t The hyperedge weight type (unweighted: no weights).
 * @ingroup hypergraph
 */
template <typename weight_t>
struct line_graph_view {
  array_view<const std::int64_t> row_ptr;   ///< num_rows() + 1 offsets, row_ptr[0] == 0
  array_view<const std::int64_t> col_ind;   ///< the neighbour ids of every row, sorted
  array_view<const std::int64_t> out;       ///< per row: the first neighbour above the row's id
  array_view<const std::int32_t> overlaps;  ///< shared vertices per entry; empty under escher
  array_view<const weight_t> weights;       ///< weights by id (host memory); empty if unweighted
  std::uint64_t version = 0;                ///< the hypergraph version this view describes

  /**
   * @brief Number of rows.
   * @return row_ptr.size() - 1, or 0 for an empty view.
   */
  [[nodiscard]] std::int64_t num_rows() const noexcept {
    return row_ptr.empty() ? 0 : static_cast<std::int64_t>(row_ptr.size()) - 1;
  }

  /**
   * @brief Number of entries (each adjacent pair counts in both rows).
   * @return col_ind.size().
   */
  [[nodiscard]] std::int64_t num_entries() const noexcept {
    return static_cast<std::int64_t>(col_ind.size());
  }
};

/**
 * @brief Which kernel paths an operation family took: the degree-binned thread, warp and block
 *        paths (an operation counts once per path it used).
 * @ingroup hypergraph
 */
struct storage_path_counters {
  std::int64_t thread = 0;  ///< operations that used the thread-per-item path
  std::int64_t warp = 0;    ///< operations that used the warp-per-item path
  std::int64_t block = 0;   ///< operations that used the block-per-item path
};

/**
 * @brief The storage counters of a hypergraph (hypergraph::store_stats()).
 *
 * `device_bytes` is the scope of ESCHER-GPU's `DynamicHypergraph::deviceBytes()`: the stores of
 * the incidence relations and of the line graph with their operation buffers, and the row views
 * the container keeps; the views built on request (h2v(), v2h()) are counted in `view_bytes`.
 * The counters are those of the last completed operation (reading them does not synchronize).
 * @ingroup hypergraph
 */
struct storage_stats {
  std::int64_t device_bytes = 0;      ///< device memory of the stores and kept row views
  std::int64_t view_bytes = 0;        ///< memory of the views built on request
  std::int64_t host_bytes = 0;        ///< host memory of the storage (CPU backends)
  std::int64_t payload_capacity = 0;  ///< h2v payload entries allocated
  std::int64_t payload_used = 0;      ///< h2v payload entries in use (the bump pointer)
  std::int64_t growths = 0;           ///< payload growths (all trees)
  std::int64_t compactions = 0;       ///< payload compactions (all trees)
  storage_path_counters fill;         ///< appends of values to rows (v2h)
  storage_path_counters reuse;        ///< inserted rows that reused a deleted slot (h2v)
  storage_path_counters unfill;       ///< removals of values from rows (v2h)
};

}  // namespace dyng
