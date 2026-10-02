// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_oracle.hpp
 * @brief The host references of mosp: combined_graph_reference() (the combined graph of K trees)
 *        and mosp_path_costs_reference() (the objective values along a tree).
 * @ingroup testing
 */
#pragma once

#include <dyng/graph/csr.hpp>

#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief The combined graph of K shortest-path trees (a port of MOSP's combinedGraphReference()).
 *
 * Edge (p, v) is in the combined graph iff p is the parent of v in some tree T_i; its weight is
 * L * (K + 1) - sum_{i : Parent_i[v] == p} L / Pref_i, with L = lcm(preferences) (1 for no
 * preferences). Rows list their edges in the order of the heads v, and for one head in objective
 * order of the first tree that contains the edge. Solving it with dijkstra() from `source` gives
 * the canonical MOSP tree (its distances in units of 1/L).
 * @tparam vertex_t Vertex id type.
 * @param[in] parents     K parent arrays of n entries each (-1: none).
 * @param[in] source      The source (it has no in-edge).
 * @param[in] preferences K preferences >= 1, or none (all ones).
 * @return The combined graph (int64 edge offsets, one int32 weight column).
 * @throws invalid_argument_error if the arrays differ in length, a parent is out of range, or the
 *         preferences are not K values >= 1 with lcm <= 2^20, or K is not in [1, 64].
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t>
[[nodiscard]] csr<vertex_t, std::int64_t, std::int32_t> combined_graph_reference(
    const std::vector<std::vector<vertex_t>>& parents, vertex_t source,
    const std::vector<std::int32_t>& preferences = {});

/**
 * @brief The objective values of the path from `source` to every vertex along a tree (MOSP's
 *        mospPathCosts(), written independently): the weights of the tree edge (p, v) are those of
 *        the first edge from p to v in p's row; columns 0..num_objectives-1 are summed.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] out            The graph's out-edges (host memory).
 * @param[in] parent         The tree (n entries, -1 for the source and unreachable vertices).
 * @param[in] source         The source.
 * @param[in] num_objectives Number of weight columns to sum (0: every column of `out`).
 * @return n * K values, vertex-major; infinite_distance<int64_t>() for vertices the tree does not
 *         connect to the source.
 * @throws invalid_argument_error if a tree edge is not an edge of `out`, the tree has a cycle,
 *         num_objectives is out of range, or the view is not in host memory.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] std::vector<std::int64_t> mosp_path_costs_reference(
    const csr_view<vertex_t, edge_t, weight_t>& out, const std::vector<vertex_t>& parent,
    vertex_t source, int num_objectives = 0);

}  // namespace dyng::testing
