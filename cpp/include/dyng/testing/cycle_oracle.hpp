// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_oracle.hpp
 * @brief Host oracles for cycle counting: a subset dynamic program, a brute-force DFS and the
 *        edge-set recount of a batch (CycleEnumeration-GPU's test oracles).
 * @ingroup testing
 *
 * The subset-DP oracle deliberately shares no code with the enumerators: it counts cycles with a
 * dynamic program over vertex subsets instead of a depth-first search, so a defect in the DFS style
 * of the enumerators cannot hide in the reference as well. It is exponential in the vertex count
 * and meant for graphs of at most 16 vertices. The brute force is an exhaustive DFS for small and
 * medium graphs. The time-window and temporal oracles follow with those modes (0.4).
 */
#pragma once

#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>

#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief A cycle histogram: `counts[len]` is the number of directed simple cycles of length
 *        `len` (entries 0 and 1 are always 0).
 * @ingroup testing
 */
using cycle_histogram = std::vector<std::uint64_t>;

/**
 * @brief Directed simple cycles by length, each counted once, by a dynamic program over vertex
 *        subsets (CycleEnumeration-GPU's oracle_simple_cycles).
 *
 * For every start vertex s the program counts simple paths s -> ... -> v over vertex subsets whose
 * other members are all greater than s, and closes them with an edge v -> s; each cycle is produced
 * once, from its minimum vertex. Self-loops and repeated neighbours are ignored; 2-cycles
 * (u -> v -> u) count.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored).
 * @param[in] graph      The out-edge CSR (host memory), at most 16 vertices.
 * @param[in] max_length Longest counted cycle (>= 2), or -1 for no bound.
 * @return The histogram, of size max_length + 1 (no bound: max(num_vertices, 2) + 1).
 * @throws invalid_argument_error if the graph has more than 16 vertices, max_length is neither -1
 *         nor >= 2, or the view is not on the host.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] cycle_histogram oracle_simple_cycles(
    const csr_view<vertex_t, edge_t, weight_t>& graph, int max_length = -1);

/**
 * @brief Directed simple cycles by length by exhaustive DFS (CycleEnumeration-GPU's
 *        count_simple_cycles_bruteforce): a cycle is counted from its minimum vertex as the root.
 *
 * Every row entry is followed, so a graph with parallel edges counts a cycle once per choice of
 * parallel edges (a simple graph, e.g. under multi_edges::forbid, counts each cycle once).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored).
 * @param[in] graph      The out-edge CSR (host memory).
 * @param[in] max_length Longest counted cycle (>= 2), or -1 for no bound.
 * @return The histogram, of size max_length + 1 (no bound: max(num_vertices, 2) + 1).
 * @throws invalid_argument_error if max_length is neither -1 nor >= 2 or the view is not on the
 *         host.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] cycle_histogram brute_force_simple_cycles(
    const csr_view<vertex_t, edge_t, weight_t>& graph, int max_length = -1);

/**
 * @brief The edge set after a batch, computed without the library's apply: (E minus Del) plus
 *        Ins, self-loop insertions ignored, the vertex count grown to the largest inserted id + 1
 *        (the reference graph of CycleEnumeration-GPU's arbitrary-batch tests).
 *
 * Deletions are applied before insertions, both as sets. The result has sorted rows without
 * repeats and no weight columns.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored).
 * @param[in] graph The out-edge CSR before the batch (host memory).
 * @param[in] batch The batch (host memory; ids >= 0).
 * @return The out-edge CSR after the batch.
 * @throws invalid_argument_error if an id is negative or an array is not on the host.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] csr<vertex_t, edge_t, weight_t> edge_set_after_batch(
    const csr_view<vertex_t, edge_t, weight_t>& graph,
    const edge_batch_view<vertex_t, weight_t>& batch);

}  // namespace dyng::testing
