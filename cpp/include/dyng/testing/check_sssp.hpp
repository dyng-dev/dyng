// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file check_sssp.hpp
 * @brief dyng::testing::check_sssp_tree(): validate a shortest-path tree against Dijkstra.
 * @ingroup testing
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/sssp.hpp>

#include <cstdint>
#include <string>

namespace dyng::testing {

/**
 * @brief Options of check_sssp_tree().
 * @ingroup testing
 */
struct check_sssp_options {
  int objective = 0;              ///< the weight column
  bool require_canonical = true;  ///< parents must be the lowest-id tight in-neighbours
};

/**
 * @brief The findings of check_sssp_tree() (MOSP's TreeCheck).
 * @ingroup testing
 */
struct sssp_tree_check {
  std::int64_t distance_mismatches = 0;    ///< dist[v] differs from Dijkstra
  std::int64_t inconsistent_parents = 0;   ///< no edge (p,v) with w > 0 and d[p] + w == d[v]
  std::int64_t non_canonical_parents = 0;  ///< consistent, but not the lowest id
  std::int64_t parent_mismatches = 0;      ///< parent differs from Dijkstra (canonical mode only)
  bool require_canonical = true;           ///< the mode the check ran in

  /**
   * @brief Whether the tree passed.
   * @return true if nothing was found (non-canonical parents count only in canonical mode).
   */
  [[nodiscard]] bool ok() const noexcept {
    return distance_mismatches == 0 && inconsistent_parents == 0 && parent_mismatches == 0 &&
           (!require_canonical || non_canonical_parents == 0);
  }

  /**
   * @brief A one-line summary of the counters.
   * @return The summary.
   */
  [[nodiscard]] std::string summary() const;
};

/**
 * @brief Check distances and parents of a shortest-path tree (a port of MOSP's checkSospTree()).
 *
 * The reference distances (and, in canonical mode, parents) come from dijkstra(). Parent
 * consistency: the source and unreachable vertices have parent -1; every other vertex v has a
 * parent p with an edge (p,v) of positive weight and dist[p] + w(p,v) == dist[v]. Canonical: p is
 * the lowest id among all such in-neighbours. The in-edges are rebuilt from `out` here, so the
 * check does not rely on the graph's own transposition.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] out       The out-edge CSR the tree belongs to (host memory).
 * @param[in] source    The source vertex.
 * @param[in] distances One distance per vertex (host memory).
 * @param[in] parents   One parent per vertex (host memory).
 * @param[in] opt       Objective and mode.
 * @return The findings.
 * @throws invalid_argument_error if the array sizes differ from the vertex count or the objective
 *         is out of range.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp_tree_check check_sssp_tree(const csr_view<vertex_t, edge_t, weight_t>& out,
                                              vertex_t source,
                                              array_view<const std::int64_t> distances,
                                              array_view<const vertex_t> parents,
                                              const check_sssp_options& opt = {});

/**
 * @brief Check a tree of a graph (see the csr_view overload).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] g         The graph.
 * @param[in] source    The source vertex.
 * @param[in] distances One distance per vertex (host memory).
 * @param[in] parents   One parent per vertex (host memory).
 * @param[in] opt       Objective and mode.
 * @return The findings.
 * @throws invalid_argument_error on size or objective mismatches.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp_tree_check check_sssp_tree(const graph<vertex_t, edge_t, weight_t>& g,
                                              vertex_t source,
                                              array_view<const std::int64_t> distances,
                                              array_view<const vertex_t> parents,
                                              const check_sssp_options& opt = {});

/**
 * @brief Check an sssp result against its graph (source and objective taken from the result).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] g                 The graph the result matches.
 * @param[in] r                 The result.
 * @param[in] require_canonical Parents must be the lowest-id tight in-neighbours.
 * @return The findings.
 * @throws invalid_argument_error on size mismatches or a moved-from result.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp_tree_check check_sssp_tree(const graph<vertex_t, edge_t, weight_t>& g,
                                              const sssp::result<vertex_t>& r,
                                              bool require_canonical = true);

}  // namespace dyng::testing
