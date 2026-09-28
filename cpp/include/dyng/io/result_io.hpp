// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file result_io.hpp
 * @brief Results in the originals' formats: shortest-path distance files ("v d" / "v INF") and
 *        SSSP-tree files ("v p", p = -1 for none) of MOSP, and cycle histograms in the CSV of
 *        CycleEnumeration-GPU ("# cycle_size, num_of_cycles" ... "Total, N").
 * @ingroup io
 */
#pragma once

#include <dyng/core/array_view.hpp>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace dyng::io {

/**
 * @brief Write distances, one line "v d" per vertex; d >= infinite_distance() / 2 is "INF".
 *
 * Byte-identical to MOSP's writeDistances().
 *
 * @tparam distance_t Signed integer distance type.
 * @param[in] path      The file (parent directories are created).
 * @param[in] distances One distance per vertex (host memory).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if the array is not host-accessible.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename distance_t>
void write_distances(const std::string& path, array_view<const distance_t> distances);

/**
 * @brief Write a parent array (an SSSP tree), one line "v p" per vertex.
 *
 * Byte-identical to MOSP's writeParents().
 *
 * @tparam vertex_t Signed vertex id type.
 * @param[in] path    The file (parent directories are created).
 * @param[in] parents One parent per vertex, -1 for none (host memory).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if the array is not host-accessible.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t>
void write_parents(const std::string& path, array_view<const vertex_t> parents);

/**
 * @brief Read a distance file: one line "v d" or "v INF" per vertex, each vertex exactly once.
 *
 * Validation follows MOSP's readDistances(): ids in [0, num_vertices), d >= 0, every vertex
 * listed exactly once; "INF" reads as infinite_distance<distance_t>().
 *
 * @tparam distance_t Signed integer distance type.
 * @param[in] path         The file.
 * @param[in] num_vertices Number of vertices.
 * @return The distances, indexed by vertex.
 * @throws io_error if the file is missing or malformed (with path, line and column).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename distance_t>
[[nodiscard]] std::vector<distance_t> read_distances(const std::string& path,
                                                     std::int64_t num_vertices);

/**
 * @brief Read an SSSP-tree file: one line "v p" per vertex, p in [-1, num_vertices).
 *
 * Validation follows MOSP's readParents(): every vertex listed exactly once.
 *
 * @tparam vertex_t Signed vertex id type.
 * @param[in] path         The file.
 * @param[in] num_vertices Number of vertices.
 * @return The parents, indexed by vertex (-1 for none).
 * @throws io_error if the file is missing or malformed (with path, line and column).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t>
[[nodiscard]] std::vector<vertex_t> read_parents(const std::string& path,
                                                 std::int64_t num_vertices);

/**
 * @brief Write a cycle histogram as CycleEnumeration-GPU's CSV (CycleHistogram::to_csv()).
 *
 * The header line "# cycle_size, num_of_cycles", one line "len, count" per length with a non-zero
 * count in increasing order, then (with `include_total`) "Total, N". This is the standard output
 * of the original's `cycle-enum`. The text is built first, so on an error nothing is written.
 * @param[in,out] out           The stream (e.g. std::cout, or a std::ofstream).
 * @param[in]     counts        counts[len] = cycles of length len (host memory; e.g.
 *                              cycle_count::result::counts()).
 * @param[in]     include_total Append the "Total, N" line.
 * @throws invalid_argument_error if the array is not host-accessible.
 * @throws capacity_error         if the total exceeds 2^64 - 1.
 * @throws io_error               if the stream fails.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
void write_histogram_csv(std::ostream& out, array_view<const std::uint64_t> counts,
                         bool include_total = true);

}  // namespace dyng::io
