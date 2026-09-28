// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file edge_list_io.hpp
 * @brief Edge-list text files: read_edge_list() and write_edge_list() (`src dst [w1..wK] [ts]`,
 *        TUDataset `*_A.txt`, SNAP lists, and Matrix Market coordinate files).
 * @ingroup io
 *
 * With the default options, read_edge_list() reproduces the parser of CycleEnumeration-GPU
 * (read_temporal_graph() / read_graph_view() of commit 0a976ad): the same vertices, in the same
 * compact order, and the same edges, so that a graph built from the result under
 * graph_properties::cycle_enum_compatible() has the original's CSR.
 */
#pragma once

#include <dyng/graph/edge_list.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::io {

/**
 * @brief How the vertex ids of an edge-list file become graph ids.
 * @ingroup io
 */
enum class vertex_ids : std::uint8_t {
  /// The ids that occur in an edge (after self-loops are dropped) are numbered 0, 1, ... in
  /// ascending order of their value in the file (CycleEnumeration-GPU); ids may be any 64-bit
  /// integers, and edge_list_info::external_ids returns the mapping.
  compact,
  /// The id in the file minus edge_list_options::index_base; num_vertices is the largest id + 1,
  /// so vertices without edges are kept.
  as_is,
};

/**
 * @brief What happens to rows that repeat a (source, destination) pair.
 * @ingroup io
 */
enum class duplicate_edges : std::uint8_t {
  /// One edge per pair (the weights of the last such row in file order).
  merge,
  /// Every row is an edge (CycleEnumeration-GPU's grouped temporal edges, flattened); the
  /// timestamps are returned in edge_list_info::timestamps.
  keep,
};

/**
 * @brief Options of read_edge_list().
 * @ingroup io
 */
struct edge_list_options {
  /// Weight columns after `src dst` (integers). 0 (the default): a third column, if present, is a
  /// timestamp, and a fourth column is an error (CycleEnumeration-GPU). With K > 0 a row holds
  /// `src dst w1..wK [ts]`. Must be 0 for weight_t = unweighted and for Matrix Market files.
  int num_weights = 0;
  vertex_ids ids = vertex_ids::compact;  ///< how file ids become graph ids
  std::int64_t index_base = 0;  ///< vertex_ids::as_is only: subtracted from every id (1: 1-based)
  bool symmetrize = false;      ///< also add (dst, src) for every row (an undirected input)
  bool drop_self_loops = true;  ///< skip rows with src == dst (CycleEnumeration-GPU)
  duplicate_edges duplicates = duplicate_edges::merge;  ///< rows with a repeated pair
  int threads = 0;  ///< parser threads (std::thread); 0 = the hardware concurrency
};

/**
 * @brief What read_edge_list() found besides the edges.
 * @ingroup io
 */
struct edge_list_info {
  /// vertex_ids::compact: the id in the file of graph vertex i (ascending). Empty for as_is.
  std::vector<std::int64_t> external_ids;
  /// duplicate_edges::keep: the timestamp of every returned edge (0 where the row has none).
  /// Empty for duplicate_edges::merge.
  std::vector<std::int64_t> timestamps;
  bool matrix_market = false;  ///< the file had a `%%MatrixMarket` banner
  bool symmetric = false;      ///< its symmetry was symmetric, skew-symmetric or hermitian
};

/**
 * @brief Read a directed edge list.
 *
 * Text format: one edge per line, `src dst [w1..wK] [ts]`, integers separated by spaces, tabs or
 * commas (a TUDataset `*_A.txt` file loads directly). Blank lines and lines whose first non-blank
 * character is `#` or `%` are skipped. An integer is an optional sign followed by decimal digits
 * and must fit 64 bits. A file whose first line starts with `%%MatrixMarket` is read as a Matrix
 * Market `coordinate` file: its dimensions line is skipped and its value columns are ignored; a
 * `general` matrix gives one edge per entry, and a `symmetric`, `skew-symmetric` or `hermitian`
 * matrix gives both directions of every entry; the `array` format and unknown symmetries are
 * rejected.
 *
 * The result is sorted by (source, destination) (and by timestamp, then file order, among the
 * rows of one pair with duplicate_edges::keep). The file is split into line-aligned parts parsed
 * on `options.threads` threads; the result does not depend on the thread count.
 *
 * @tparam vertex_t Vertex id type of the result.
 * @tparam weight_t Weight type of the result: `unweighted`, or an integer type.
 * @param[in]  path    The file.
 * @param[in]  options Weight columns, id mapping, symmetrization, self-loops, duplicates, threads.
 * @param[out] info    Optional: the id mapping, the timestamps and the kind of file.
 * @return The edges; num_vertices is the number of compact ids (compact) or the largest id + 1
 *         (as_is).
 * @throws io_error               if the file cannot be read or is malformed (path and line), an id
 *                                is out of range for as_is, or there are more vertices than
 *                                vertex_t holds.
 * @throws invalid_argument_error if the options are invalid (negative num_weights, weights for an
 *                                unweighted result).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] edge_list<vertex_t, weight_t> read_edge_list(const std::string& path,
                                                           const edge_list_options& options = {},
                                                           edge_list_info* info = nullptr);

/**
 * @brief Write an edge list as text, one line `src dst [w1..wK]` per edge (0-based ids, single
 *        spaces), in edge order.
 *
 * read_edge_list() with `ids = vertex_ids::as_is`, `num_weights = edges.num_weights` and
 * `drop_self_loops = false` reads it back (sorted by (source, destination); duplicates merged
 * unless `duplicates = keep`).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (`unweighted` or integral).
 * @param[in] path  The file (its parent directory is created).
 * @param[in] edges The edges (host memory).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if the arrays disagree in size or are not host-accessible.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_edge_list(const std::string& path, const edge_list_view<vertex_t, weight_t>& edges);

}  // namespace dyng::io
