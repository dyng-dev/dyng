// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file matrix_market.hpp
 * @brief Matrix Market coordinate files: read_matrix_market() and write_matrix_market().
 * @ingroup io
 *
 * Reading with `weights = matrix_market_weights::random` and the defaults of random_weights
 * reproduces `mospPrep mtx2csr <in.mtx> <prefix> K 1 100 12345` of MOSP-OpenMP / MOSP-CUDA
 * bit-exactly (same edges, same order, same weights): symmetric matrices get both directions,
 * self-loops and duplicate entries are dropped, the edges are sorted by (source, destination),
 * and K weights per edge are drawn in edge order from std::mt19937(seed) with the libstdc++
 * uniform_int_distribution algorithm (reproduced by the library, so the weights do not depend on
 * the C++ standard library in use).
 */
#pragma once

#include <dyng/graph/edge_list.hpp>

#include <cstdint>
#include <string>

/**
 * @defgroup io Input and output
 * @brief Readers and writers: graphs, batches and results in the originals' formats.
 */

namespace dyng::io {

/**
 * @brief Where the edge weights of a Matrix Market graph come from.
 * @ingroup io
 */
enum class matrix_market_weights : std::uint8_t {
  automatic,  ///< `integer` field: one column from the file; `pattern`: none; others: error
  none,       ///< no weight columns (num_weights = 0); values in the file are checked but ignored
  from_file,  ///< one column from the file's values (`integer` field only in this release)
  random,     ///< random_weights: K seeded uniform integer columns (mospPrep mtx2csr)
};

/**
 * @brief Seeded random weights (the `mospPrep mtx2csr` defaults: K = 1, [1, 100], seed 12345).
 * @ingroup io
 */
struct random_weights {
  int num_weights = 1;         ///< K, the number of weight columns (>= 1)
  std::int64_t min = 1;        ///< smallest weight (inclusive)
  std::int64_t max = 100;      ///< largest weight (inclusive); max - min < 2^32
  std::uint32_t seed = 12345;  ///< seed of std::mt19937
};

/**
 * @brief Options of read_matrix_market().
 * @ingroup io
 */
struct matrix_market_options {
  matrix_market_weights weights = matrix_market_weights::automatic;  ///< weight source
  random_weights random{};      ///< used with matrix_market_weights::random
  bool drop_self_loops = true;  ///< skip diagonal entries (mospPrep)
  bool sort_and_dedupe = true;  ///< sort edges by (source, destination), drop duplicates (mospPrep)
};

/**
 * @brief Read a Matrix Market `coordinate` file as a directed edge list (0-based ids).
 *
 * The header must be `%%MatrixMarket matrix coordinate <field> <symmetry>` with field `real`,
 * `double`, `integer`, `complex` or `pattern` and symmetry `general`, `symmetric`,
 * `skew-symmetric` or `hermitian`; `array` files are rejected. Comment lines (`%`) may follow the
 * header; blank lines are ignored. The matrix must be square (it is an adjacency matrix), and the
 * file must hold exactly the announced number of entries, each with 1-based indices in range and
 * the number of value tokens of its field. Symmetric, skew-symmetric and hermitian entries (i, j)
 * with i != j give both edges i -> j and j -> i (skew-symmetric: the mirrored value is negated).
 *
 * With `sort_and_dedupe`, duplicate edges keep the value of their LAST entry. With random
 * weights, the weights are drawn after sorting and deduplication, K per edge in edge order.
 * (mospPrep treats `hermitian` files as general; dynG follows the Matrix Market specification.)
 *
 * @tparam vertex_t Vertex id type of the result.
 * @tparam weight_t Weight type of the result (integral).
 * @param[in] path    The file.
 * @param[in] options Weight source, self-loop and duplicate handling.
 * @return The edges; num_vertices is the matrix dimension.
 * @throws io_error                if the file cannot be read or is malformed (with path, line and
 *                                 column).
 * @throws invalid_argument_error  if the options are invalid.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] edge_list<vertex_t, weight_t> read_matrix_market(
    const std::string& path, const matrix_market_options& options = {});

/**
 * @brief Write an edge list as a `general` coordinate Matrix Market file (1-based ids).
 *
 * The field is `pattern` if `edges.num_weights == 0` and `integer` otherwise; with several
 * weight columns only column `weight_column` is written.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (integral).
 * @param[in] path          The file (its parent directory is created).
 * @param[in] edges         The edges (host memory).
 * @param[in] weight_column The weight column to write (ignored for num_weights == 0).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if `weight_column` is out of range or the arrays disagree.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_matrix_market(const std::string& path, const edge_list_view<vertex_t, weight_t>& edges,
                         int weight_column = 0);

}  // namespace dyng::io
