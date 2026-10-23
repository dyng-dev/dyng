// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_io.hpp
 * @brief Hypergraph files (`.hg`, one hyperedge per line): read_hypergraph() and
 *        write_hypergraph(), with the preprocessing of ESCHER-GPU and of MOSP_ESCHER's H-SOSP as
 *        option sets.
 * @ingroup io
 */
#pragma once

#include <dyng/hypergraph/hyperedge_list.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dyng::io {

/**
 * @brief How the vertex ids of a `.hg` file become hypergraph vertex ids.
 * @ingroup io
 */
enum class vertex_relabel : std::uint8_t {
  /// The id in the file minus hypergraph_read_options::index_base (ESCHER-GPU's loader: vertex v
  /// of the file is vertex v - 1); the vertex count is the largest id + 1 unless
  /// hypergraph_read_options::num_vertices is given, so unused ids stay isolated vertices.
  none,
  /// The distinct ids of the kept hyperedges, any 64-bit integers, numbered 0, 1, ... in
  /// ascending order (ESCHER-GPU's bench/datasets/convert_coauth.py and MOSP_ESCHER's
  /// scripts/benson_to_hg.py, which write ids 1..V this way).
  dense_sorted,
  /// The distinct ids of the kept hyperedges, any 64-bit integers, numbered 0, 1, ... in the
  /// order in which they first appear, hyperedge by hyperedge and within a hyperedge in
  /// ascending id order (MOSP_ESCHER's loadHypergraphFile, the H-SOSP preprocessing).
  first_appearance,
};

/**
 * @brief The lower-case name of a vertex_relabel ("none", "dense_sorted", "first_appearance").
 * @param[in] relabel The value.
 * @return Its name.
 * @ingroup io
 */
[[nodiscard]] std::string_view to_string(vertex_relabel relabel) noexcept;

/**
 * @brief Seeded random hyperedge weights (hypergraph_read_options::random_weights): one weight
 *        per kept hyperedge, in file order, uniform in [lo, hi].
 *
 * The weights are drawn from one std::mt19937_64(seed) with the algorithm of libstdc++'s
 * std::uniform_int_distribution, which dynG reproduces itself (so they are the same with any
 * standard library): MOSP_ESCHER's loader draws them this way (U[1, 100]).
 * @ingroup io
 */
struct weight_spec {
  std::int64_t lo = 1;     ///< smallest weight (>= 1)
  std::int64_t hi = 100;   ///< largest weight (>= lo, <= max of the weight type)
  std::uint64_t seed = 1;  ///< seed of the std::mt19937_64
  /**
   * @brief Before the weights, draw (and discard) one vertex in [0, num_vertices - 1] from the
   *        same engine, as MOSP_ESCHER's loadHypergraphFile does for the target vertex of its
   *        virtual target hyperedge. The virtual source and target hyperedges themselves are not
   *        part of the file's hypergraph: hyper_sssp adds them (0.4, M11).
   */
  bool hsosp_target_draw = false;
};

/**
 * @brief Options of read_hypergraph().
 *
 * Every line's ids are sorted and repeated ids dropped first (both originals do this); then, in
 * file order, hyperedges above max_cardinality are dropped, repeated hyperedges are dropped if
 * dedupe_hyperedges is set, and the ids of the kept ones are relabelled. escher_gpu() and
 * hsosp() are the two originals' option sets.
 * @ingroup io
 */
struct hypergraph_read_options {
  /// vertex_relabel::none only: subtracted from every id of the file (1: ids 1..n, the
  /// convention of both originals' files; 0: ids 0..n-1).
  std::int64_t index_base = 1;
  /// Keep only the first hyperedge of every distinct member set (in file order).
  bool dedupe_hyperedges = false;
  /// Drop hyperedges with more than this many distinct members; -1: no limit.
  std::int64_t max_cardinality = -1;
  vertex_relabel relabel = vertex_relabel::none;  ///< how file ids become vertex ids
  /**
   * @brief For a weighted read: draw the weights instead of reading them from the file (then the file must not have weights). std::nullopt: a weighted read takes `; w` from every line.
   */
  std::optional<weight_spec> random_weights;
  /// vertex_relabel::none only: the vertex count (every id must lie below it after the
  /// index_base shift); -1: the largest id + 1.
  std::int64_t num_vertices = -1;

  /**
   * @brief ESCHER-GPU's loader (loadHypergraph of ESCHER-GPU\@abcf9b2): ids 1..n kept as they
   *        are (vertex v becomes v - 1, the vertex count is the largest id), every line a
   *        hyperedge (repeated hyperedges kept), no cardinality limit, no weights.
   *
   * Two differences, both on input ESCHER-GPU rejects: the largest id is 2^31 - 1 (the largest
   * that keeps the vertex count in int32; ESCHER-GPU stops at 2^31 - 2), and a file without
   * hyperedges gives an empty list (ESCHER-GPU fails).
   * @return The option set.
   */
  [[nodiscard]] static hypergraph_read_options escher_gpu() noexcept {
    return hypergraph_read_options{};
  }

  /**
   * @brief MOSP_ESCHER's H-SOSP loader (loadHypergraphFile of MOSP_ESCHER\@4b86159, hsospBench
   *        --hg): hyperedges of at most `max_cardinality` vertices, vertices numbered by first
   *        appearance, weights U[1, 100] from std::mt19937_64(seed) after its target draw.
   * @param[in] seed            The seed (hsospBench's --seed, default 20260725).
   * @param[in] max_cardinality The limit (hsospBench's --maxcard, default 25).
   * @return The option set.
   */
  [[nodiscard]] static hypergraph_read_options hsosp(std::uint64_t seed = 20260725U,
                                                     std::int64_t max_cardinality = 25) noexcept {
    hypergraph_read_options o;
    o.max_cardinality = max_cardinality;
    o.relabel = vertex_relabel::first_appearance;
    o.random_weights = weight_spec{1, 100, seed, true};
    return o;
  }
};

/**
 * @brief What read_hypergraph() found besides the hyperedges.
 * @ingroup io
 */
struct hypergraph_file_info {
  /// vertex_relabel::dense_sorted and first_appearance: the id in the file of vertex i. Empty for
  /// vertex_relabel::none.
  std::vector<std::int64_t> external_ids;
  std::int64_t lines = 0;                ///< non-empty, non-comment lines (hyperedges in the file)
  std::int64_t repeated_members = 0;     ///< ids dropped because they repeat within a line
  std::int64_t over_cardinality = 0;     ///< hyperedges dropped by max_cardinality
  std::int64_t repeated_hyperedges = 0;  ///< hyperedges dropped by dedupe_hyperedges
};

/**
 * @brief Read a `.hg` hypergraph file: one hyperedge per line (docs/api/file_formats.md).
 *
 * A line lists the hyperedge's vertex ids, decimal integers separated by spaces, tabs, carriage
 * returns or commas; a weighted file ends every line with `; w`. Blank lines (and lines of
 * separators only) are skipped; a line whose first character is `#` or `%` is a comment. Each
 * line's ids are sorted and deduplicated; hyperedge i of the result is the i-th kept line. Under
 * hypergraph_read_options::escher_gpu() the result is ESCHER-GPU's loaded hypergraph (hyperedge
 * id = its key - 1, vertex id = its id - 1), under hsosp() MOSP_ESCHER's (without its virtual
 * source and target hyperedges; vertex ids and weights bit for bit).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral with weights >= 1).
 * @param[in]  path    The file.
 * @param[in]  options The preprocessing.
 * @param[out] info    Optional: the relabelling and what was dropped.
 * @return The hyperedges and the vertex count.
 * @throws io_error               if the file is missing or malformed (path, line and column): a
 *                                token that is not a decimal integer, an id out of range, a
 *                                weight missing, misplaced or out of range, more vertices than
 *                                vertex_t holds; or if it is a file of another format (a `.dgb`
 *                                or `.dgt` batch file, a Matrix Market file: its first line).
 * @throws invalid_argument_error if the options are invalid (index_base < 0, max_cardinality
 *                                < -1, a weight range outside [1, max(weight_t)], random_weights
 *                                for an unweighted type).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyperedge_list<vertex_t, weight_t> read_hypergraph(
    const std::string& path, const hypergraph_read_options& options = {},
    hypergraph_file_info* info = nullptr);

/**
 * @brief Write hyperedges as a `.hg` file, read back by read_hypergraph() with the same
 *        index_base.
 *
 * One line per hyperedge: its members plus `index_base`, separated by single spaces, in list
 * order, then ` ; w` for a weighted list. The vertex count is not written (read_hypergraph()
 * takes the largest id + 1 unless told otherwise).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] path       The file (parent directories are created).
 * @param[in] hyperedges The hyperedges (host memory).
 * @param[in] index_base Added to every vertex id (1: the originals' convention).
 * @throws io_error               if the file cannot be written.
 * @throws invalid_argument_error if a hyperedge is empty (a `.hg` file cannot hold it), a member
 *                                is outside [0, num_vertices), the offsets or weights do not
 *                                match, index_base is negative, or an array is not in
 *                                host-accessible memory.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup io
 */
template <typename vertex_t, typename weight_t>
void write_hypergraph(const std::string& path,
                      const hyperedge_list_view<vertex_t, weight_t>& hyperedges,
                      std::int64_t index_base = 1);

}  // namespace dyng::io
