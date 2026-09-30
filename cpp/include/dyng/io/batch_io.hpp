// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_io.hpp
 * @brief Legacy MOSP batch files: `insert.txt` ("u v w1 .. wK") and `delete.txt` ("u v").
 * @ingroup io
 */
#pragma once

#include <dyng/graph/edge_batch.hpp>

#include <cstdint>
#include <string>

namespace dyng::io {

/**
 * @brief Options of read_legacy_batch().
 * @ingroup io
 */
struct legacy_batch_options {
  int num_weights = 1;             ///< weights per insertion line (K of the graph), >= 0
  std::int64_t num_vertices = -1;  ///< ids must be < num_vertices; -1: only ids >= 0 are checked
  /**
   * @brief MOSP's accept/reject rules instead of dynG's stricter ones.
   *
   * As readChangeBatch() of MOSP-OpenMP\@c352151 / MOSP-CUDA\@e220ee2: integers after the first
   * 2 + K of an insertion line and after the first 2 of a deletion line are read and ignored,
   * and a deletion line with a single integer is skipped. The compatibility driver
   * dyng-compat-mosp sets it (the parity harness needs the original's input-rejection
   * semantics); ADR 0010 records why the default is strict.
   */
  bool mosp_lenient = false;
};

/**
 * @brief Read a MOSP batch: insertions from `insert_path`, deletions from `delete_path`.
 *
 * Each non-blank line of the insert file holds exactly 2 + K integers "u v w1 .. wK", and each
 * non-blank line of the delete file exactly 2 integers "u v"; ids are 0-based and must lie in
 * [0, num_vertices), weights in [1, max(weight_t)] (MOSP readChangeBatch()). By default dynG is
 * stricter than MOSP in three places: extra integers on an insertion or a deletion line, and a
 * deletion line with a single integer, are errors (MOSP ignores them);
 * `options.mosp_lenient` restores MOSP's decisions. In both modes every token must be a whole
 * decimal integer ("12abc" is an error, as in MOSP). Operations keep file order.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral).
 * @param[in] insert_path The insertion file.
 * @param[in] delete_path The deletion file.
 * @param[in] options     K and the vertex count used for validation.
 * @return The batch.
 * @throws io_error               if a file is missing or malformed (with path, line and column).
 * @throws invalid_argument_error if the options are invalid.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] edge_batch<vertex_t, weight_t> read_legacy_batch(
    const std::string& insert_path, const std::string& delete_path,
    const legacy_batch_options& options = {});

/**
 * @brief Write a batch as MOSP `insert.txt` / `delete.txt` (byte-identical to MOSP's
 *        writeChangeBatch()).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral).
 * @param[in] insert_path The insertion file (parent directories are created).
 * @param[in] delete_path The deletion file (parent directories are created).
 * @param[in] batch       The batch (host memory; vertex operations are not representable).
 * @throws io_error               if a file cannot be written.
 * @throws invalid_argument_error if the batch has vertex operations or inconsistent sizes.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_legacy_batch(const std::string& insert_path, const std::string& delete_path,
                        const edge_batch_view<vertex_t, weight_t>& batch);

}  // namespace dyng::io
