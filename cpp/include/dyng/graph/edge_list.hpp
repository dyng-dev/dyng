// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file edge_list.hpp
 * @brief edge_list<V,W> (owning COO edge list) and edge_list_view<V,W>.
 * @ingroup graph
 *
 * Weights of an edge list are EDGE-MAJOR: edge i carries `num_weights` consecutive values
 * `weights[i * num_weights + k]` (the layout of the text formats).
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace dyng {

/**
 * @brief A non-owning edge list: parallel arrays of sources, destinations and weights.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t, typename weight_t>
struct edge_list_view {
  vertex_t num_vertices = 0;           ///< ids lie in [0, num_vertices)
  array_view<const vertex_t> src;      ///< source of each edge
  array_view<const vertex_t> dst;      ///< destination of each edge
  array_view<const weight_t> weights;  ///< num_weights per edge, edge-major
  int num_weights = 0;                 ///< weights per edge (0: unweighted)

  /**
   * @brief Number of edges.
   * @return src.size().
   */
  [[nodiscard]] std::size_t num_edges() const noexcept {
    return src.size();
  }
};

/**
 * @brief An owning host edge list (COO).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t, typename weight_t>
struct edge_list {
  vertex_t num_vertices = 0;      ///< ids lie in [0, num_vertices)
  int num_weights = 0;            ///< weights per edge (0: unweighted)
  std::vector<vertex_t> src;      ///< source of each edge
  std::vector<vertex_t> dst;      ///< destination of each edge
  std::vector<weight_t> weights;  ///< num_weights per edge, edge-major

  /**
   * @brief Number of edges.
   * @return src.size().
   */
  [[nodiscard]] std::size_t num_edges() const noexcept {
    return src.size();
  }

  /**
   * @brief Append one edge.
   * @param[in] u Source.
   * @param[in] v Destination.
   * @param[in] w Its num_weights weights.
   * @throws invalid_argument_error if `w` does not hold num_weights values.
   */
  void add_edge(vertex_t u, vertex_t v, std::initializer_list<weight_t> w = {}) {
    DYNG_EXPECTS(w.size() == static_cast<std::size_t>(num_weights), "edge_list::add_edge got ",
                 w.size(), " weights, expected ", num_weights);
    src.push_back(u);
    dst.push_back(v);
    weights.insert(weights.end(), w.begin(), w.end());
  }

  /**
   * @brief A read-only view of the arrays.
   * @return An edge_list_view in host memory; valid while this object is alive and unmodified.
   */
  [[nodiscard]] edge_list_view<vertex_t, weight_t> view() const noexcept {
    return edge_list_view<vertex_t, weight_t>{num_vertices, host_view(src), host_view(dst),
                                              host_view(weights), num_weights};
  }
};

}  // namespace dyng
