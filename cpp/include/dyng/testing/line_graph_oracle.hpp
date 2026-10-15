// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file line_graph_oracle.hpp
 * @brief dyng::testing::rebuild_line_graph: the line graph of a hypergraph rebuilt from its
 *        hyperedges by MOSP_ESCHER\@4b86159's HypergraphOracle algorithm, the oracle of the
 *        maintained line graph (M8b).
 * @ingroup testing
 *
 * The line graph (h2h) of a hypergraph has one row per hyperedge id: the ids of the other live
 * hyperedges that share at least one vertex with it, in increasing order. The hypergraph
 * container maintains it batch by batch (line_graph_policy::sorted_merge: ESCHER-GPU's sorted
 * delta merge on CUDA, its host form on the CPU backends); this oracle recomputes it from scratch
 * the way MOSP_ESCHER's test oracle does (hypergraph/src/HypergraphOracle.cpp@4b86159,
 * rebuildLineGraph): the hyperedges of every vertex, every ordered pair (a, b), a != b, of them
 * emitted per vertex, the pairs sorted and deduplicated, the rows read off the sorted pairs.
 *
 * It is an oracle (PLAN Section 5.8): it uses the C++ standard library only, shares no code with
 * the container (cpp/src/hypergraph) and includes no other project header; an include-graph and
 * link test (line_graph_oracle_isolation) checks this. It is quadratic in the vertex degrees and
 * meant for test sizes.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief A line graph as a CSR: row i (hyperedge id i) lists
 *        `neighbours[offsets[i] .. offsets[i + 1])`, sorted.
 * @ingroup testing
 */
struct line_graph_rows {
  std::vector<std::int64_t> offsets{0};  ///< num_ids() + 1 offsets
  std::vector<std::int64_t> neighbours;  ///< the neighbour ids of every row, rows back to back

  /**
   * @brief Number of rows (ids).
   * @return offsets.size() - 1.
   */
  [[nodiscard]] std::int64_t num_ids() const noexcept {
    return static_cast<std::int64_t>(offsets.size()) - 1;
  }

  /**
   * @brief The row of an id.
   * @param[in] id An id in [0, num_ids()).
   * @return Its neighbours, sorted.
   */
  [[nodiscard]] std::vector<std::int64_t> row(std::int64_t id) const {
    return std::vector<std::int64_t>(
        neighbours.begin() + offsets[static_cast<std::size_t>(id)],
        neighbours.begin() + offsets[static_cast<std::size_t>(id) + 1]);
  }
};

/**
 * @brief The line graph of a hypergraph, rebuilt from its hyperedges (MOSP_ESCHER's
 *        rebuildLineGraph).
 *
 * A hyperedge with an empty row is dead: it has an empty line-graph row and is nobody's
 * neighbour. Repeated members of a row count once.
 * @param[in] h2v_offsets  The hyperedges by id: id i has the members
 *                         `h2v_members[h2v_offsets[i] .. h2v_offsets[i + 1])` (the layout of
 *                         hypergraph::h2v() copied to the host).
 * @param[in] h2v_members  The members, in [0, num_vertices).
 * @param[in] num_vertices Vertices.
 * @return One row per id (h2v_offsets.size() - 1 rows).
 * @throws std::invalid_argument for malformed offsets or a member out of range.
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] line_graph_rows rebuild_line_graph(const std::vector<std::int64_t>& h2v_offsets,
                                                 const std::vector<std::int32_t>& h2v_members,
                                                 std::int32_t num_vertices);

}  // namespace dyng::testing
