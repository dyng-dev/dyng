// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file csr_triplet.hpp
 * @brief The MOSP text CSR: `<prefix>RowPtr.txt`, `<prefix>ColInd.txt`, `<prefix>Values.txt`.
 * @ingroup io
 *
 * Format (MOSP-OpenMP / MOSP-CUDA): RowPtr holds n + 1 offsets and ColInd one 0-based
 * neighbour per edge, one integer per line; Values holds one line per edge with its K weights
 * separated by single spaces. The writer produces byte-identical files to MOSP's writeCsrGraph().
 */
#pragma once

#include <dyng/graph/csr.hpp>

#include <string>

namespace dyng::io {

/**
 * @brief Options of read_csr_triplet().
 * @ingroup io
 */
struct csr_triplet_options {
  /// Weight columns expected in Values.txt. 0: infer K from the file (the first non-blank line);
  /// a graph without edges then needs an explicit value (MOSP's `-k`).
  int num_weights = 0;
};

/**
 * @brief Read a MOSP text CSR.
 *
 * Validation follows MOSP's readCsrGraph(): offsets are integers in [0, max(edge_t)] starting at
 * 0 and non-decreasing, with at least one vertex; neighbours lie in [0, n) and number exactly
 * row_ptr[n]; every non-blank line of Values.txt holds the same number K of weights, each in
 * [1, max(weight_t)], and there is one such line per edge. Every token must be a complete decimal
 * integer.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (integral).
 * @param[in] prefix  Path prefix of the three files.
 * @param[in] options The expected number of weight columns.
 * @return The CSR (objective-major weights).
 * @throws io_error               if a file is missing or malformed (with path, line and column).
 * @throws invalid_argument_error if `options.num_weights` is negative.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] csr<vertex_t, edge_t, weight_t> read_csr_triplet(
    const std::string& prefix, const csr_triplet_options& options = {});

/**
 * @brief Write a CSR in the MOSP text format (byte-identical to MOSP's writeCsrGraph()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (integral).
 * @param[in] prefix Path prefix of the three files (parent directories are created).
 * @param[in] graph  The CSR (host memory), e.g. graph::view().out or csr::view().
 * @throws io_error               if a file cannot be written.
 * @throws invalid_argument_error if the view is not host-accessible or its sizes disagree.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void write_csr_triplet(const std::string& prefix,
                       const csr_view<vertex_t, edge_t, weight_t>& graph);

}  // namespace dyng::io
