// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file result_io.hpp
 * @brief Results in the originals' formats: shortest-path distance files ("v d" / "v INF"),
 *        SSSP-tree files ("v p", p = -1 for none) and path-cost files ("v c1 .. cK") of MOSP, and
 *        cycle histograms in the CSV of
 *        CycleEnumeration-GPU ("# cycle_size, num_of_cycles" ... "Total, N"), and triad bins
 *        (dyng::triad_count) as a JSON document of dynG's own (format "dyng.triad_bins").
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
 * @brief Write MOSP path costs, one line "v c1 .. cK" per vertex; a cost >= infinite_distance() /
 *        2 is "INF" (mosp::result::path_costs(), MOSP's combinedGraph/mospCosts.txt).
 *
 * Byte-identical to the writeCosts() of MOSP's `mosp` driver.
 *
 * @tparam distance_t     Signed integer distance type.
 * @param[in] path           The file (parent directories are created).
 * @param[in] costs          n * num_objectives costs, vertex-major (host memory).
 * @param[in] num_objectives K (>= 1; the length of `costs` must be a multiple of it).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if the array is not host-accessible or K does not divide its
 *         length.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename distance_t>
void write_path_costs(const std::string& path, array_view<const distance_t> costs,
                      int num_objectives);

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

/**
 * @brief The bins of a triad count (a dyng::triad_count::result), as a triad-bins JSON file holds
 *        them (docs/api/file_formats.md, "Triad bins").
 *
 * `kind` names the pattern as triad_count::to_string() spells it; `counts` has one value per bin
 * of that pattern (30 for "h_motif", 1 for the StatHyper types). `deltas` is optional: the net
 * change of every bin per applied batch, in batch order (what triad_count::stats::delta holds
 * for the first num_bins bins), so that `counts` is the baseline plus the sum of the deltas.
 * Filling one from a result: `{std::string(to_string(r.kind())), r.graph_version(), {counts
 * of r.counts()}, {}}`.
 * @ingroup io
 */
struct triad_bins {
  std::string kind;                   ///< the pattern ("h_motif", "stathyper_type1", ...)
  std::uint64_t graph_version = 0;    ///< the hypergraph version the counts match
  std::vector<std::uint64_t> counts;  ///< one count per bin
  std::vector<std::vector<std::int64_t>> deltas;  ///< optional: one row of num_bins per batch
};

/**
 * @brief The number of bins of a triad pattern named as triad_count::to_string() spells it.
 * @param[in] kind The pattern's name.
 * @return 30 for "h_motif", 1 for "stathyper_type1", "stathyper_type2" and "stathyper_type3".
 * @throws invalid_argument_error for any other name.
 * @ingroup io
 */
[[nodiscard]] int triad_bins_size(const std::string& kind);

/**
 * @brief Write triad bins as the triad-bins JSON document (format "dyng.triad_bins", version 1):
 *        one object with the keys format, version, kind, num_bins, graph_version, counts and,
 *        when there are deltas, deltas; two-space indentation, one array per line, a final
 *        newline. The text is built first, so on an error nothing is written.
 * @param[in,out] out  The stream (e.g. std::cout, or a std::ofstream).
 * @param[in]     bins The bins.
 * @throws invalid_argument_error if the kind is unknown or `counts` or a row of `deltas` does not
 *                                have the kind's number of bins.
 * @throws io_error               if the stream fails.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
void write_triad_bins_json(std::ostream& out, const triad_bins& bins);

/**
 * @brief Write triad bins to a file as write_triad_bins_json(std::ostream&, const triad_bins&)
 *        does.
 * @param[in] path The file (parent directories are created).
 * @param[in] bins The bins.
 * @throws invalid_argument_error as the stream overload.
 * @throws io_error               if the file cannot be written.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
void write_triad_bins_json(const std::string& path, const triad_bins& bins);

/**
 * @brief Read a triad-bins JSON document (write_triad_bins_json()'s format).
 *
 * The file must be one JSON object (RFC 8259) with "format": "dyng.triad_bins", "version": 1,
 * a known "kind", "num_bins" equal to the kind's bin count and "counts" with that many integers
 * in [0, 2^64 - 1]; "graph_version" (an integer in [0, 2^64 - 1], default 0) and "deltas" (an
 * array of rows of num_bins integers in the signed 64-bit range) are optional, and other keys
 * are ignored. A repeated key, a number with a fraction or an exponent where an integer is
 * expected, nesting deeper than 64 levels or anything after the object is an error.
 * @param[in] path The file.
 * @return The bins.
 * @throws io_error            if the file is missing or malformed (with path, line and column).
 * @throws out_of_memory_error if host memory cannot be allocated.
 * @ingroup io
 */
[[nodiscard]] triad_bins read_triad_bins_json(const std::string& path);

}  // namespace dyng::io
