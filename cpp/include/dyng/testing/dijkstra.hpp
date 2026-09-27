// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file dijkstra.hpp
 * @brief dyng::testing::dijkstra(): the host reference for sssp (lowest-id ties).
 * @ingroup testing
 */
#pragma once

#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph.hpp>

#include <cstdint>
#include <vector>

/**
 * @defgroup testing Testing oracles
 * @brief Host-only reference implementations (target dyng::testing) that check the algorithms'
 *        results. They share no code with the algorithms they check.
 */

namespace dyng::testing {

/**
 * @brief A shortest-path tree on the host.
 * @tparam vertex_t Vertex id type.
 * @ingroup testing
 */
template <typename vertex_t>
struct sssp_tree {
  std::vector<std::int64_t> distances;  ///< infinite_distance<int64_t>() if unreachable
  std::vector<vertex_t> parents;        ///< -1 for the source and unreachable vertices
};

/**
 * @brief Dijkstra from `source` with ties broken towards the lowest parent id (the canonical
 *        tree); a port of MOSP's dijkstraCsrGraph().
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] out       The out-edge CSR (host memory, objective-major weights, weights >= 1).
 * @param[in] source    The source vertex.
 * @param[in] objective The weight column.
 * @return The canonical tree; all vertices unreachable if `source` is out of range.
 * @throws invalid_argument_error if `objective` is out of range or the view is not on the host.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp_tree<vertex_t> dijkstra(const csr_view<vertex_t, edge_t, weight_t>& out,
                                           vertex_t source, int objective = 0);

/**
 * @brief Dijkstra on a graph's out-edges (see the csr_view overload).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] g         The graph (host storage).
 * @param[in] source    The source vertex.
 * @param[in] objective The weight column.
 * @return The canonical tree.
 * @throws invalid_argument_error if `objective` is out of range.
 * @sync
 * @ingroup testing
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp_tree<vertex_t> dijkstra(const graph<vertex_t, edge_t, weight_t>& g,
                                           vertex_t source, int objective = 0);

}  // namespace dyng::testing
