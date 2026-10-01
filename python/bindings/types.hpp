// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file types.hpp
 * @brief The native types shared by several binding files: graph and result holders and the
 *        batch of arrays, with the lists of their instantiations.
 */
#pragma once

#include "common.hpp"

#include <dyng/cycle_count.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/sssp.hpp>

#include <optional>

namespace dyng::python {

/// The graph holder of graph<V,E,W>.
template <typename vertex_t, typename edge_t, typename weight_t>
using graph_holder = holder<graph<vertex_t, edge_t, weight_t>>;

/// The holder of an sssp result.
template <typename vertex_t>
using sssp_holder = result_holder<sssp::result<vertex_t>>;

/// The holder of a cycle_count result.
using cycle_count_holder = result_holder<cycle_count::result>;

/**
 * @brief A batch of edge changes whose arrays are Python arrays (kept alive, never copied).
 *
 * The typed layer passes arrays of exactly the vertex and weight types of the graph the batch is
 * used with (checked conversions happen there). The object is immutable, so it needs no lock.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (dyng::unweighted: no weights).
 */
template <typename vertex_t, typename weight_t>
struct batch_arrays {
  in_array<vertex_t> insert_src;  ///< sources of the insertions
  in_array<vertex_t> insert_dst;  ///< destinations of the insertions
  std::optional<in_array<weight_scalar_t<weight_t>>> insert_weights;  ///< insertion-major weights
  in_array<vertex_t> delete_src;                                      ///< sources of the deletions
  in_array<vertex_t> delete_dst;                              ///< destinations of the deletions
  std::optional<in_array<vertex_t>> insert_vertices;          ///< vertex insertions (0.3)
  std::optional<in_array<std::int8_t>> insert_vertex_labels;  ///< their labels (0.3)
  std::optional<in_array<vertex_t>> delete_vertices;          ///< vertex deletions (0.3)
  int num_weights = 0;                                        ///< weights per insertion

  /**
   * @brief The library's view of the arrays.
   * @return The view (valid while this object lives).
   */
  [[nodiscard]] edge_batch_view<vertex_t, weight_t> view() const {
    edge_batch_view<vertex_t, weight_t> v;
    v.insert_src = view_of(insert_src);
    v.insert_dst = view_of(insert_dst);
    if constexpr (!is_unweighted_v<weight_t>) {
      if (insert_weights) {
        v.insert_weights = view_of(*insert_weights);
      }
    }
    v.delete_src = view_of(delete_src);
    v.delete_dst = view_of(delete_dst);
    if (insert_vertices) {
      v.insert_vertices = view_of(*insert_vertices);
    }
    if (insert_vertex_labels) {
      v.insert_vertex_labels = view_of(*insert_vertex_labels);
    }
    if (delete_vertices) {
      v.delete_vertices = view_of(*delete_vertices);
    }
    v.num_weights = num_weights;
    return v;
  }
};

/**
 * @brief Move an edge_batch (from a reader or a generator) into NumPy arrays.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @param[in] b The batch.
 * @return (insert_src, insert_dst, insert_weights of shape (n, K) or None, delete_src, delete_dst,
 *         K).
 */
template <typename vertex_t, typename weight_t>
nb::tuple batch_to_numpy(const edge_batch<vertex_t, weight_t>& b) {
  const std::size_t n = b.num_insertions();
  const int k = b.num_weights();
  nb::object weights = nb::none();
  if constexpr (!is_unweighted_v<weight_t>) {
    std::vector<weight_t> w = b.insert_weights();
    weights = nb::cast(to_numpy_2d(std::move(w), n, static_cast<std::size_t>(k)));
  }
  std::vector<vertex_t> is = b.insert_src();
  std::vector<vertex_t> id = b.insert_dst();
  std::vector<vertex_t> ds = b.delete_src();
  std::vector<vertex_t> dd = b.delete_dst();
  return nb::make_tuple(to_numpy(std::move(is)), to_numpy(std::move(id)), weights,
                        to_numpy(std::move(ds)), to_numpy(std::move(dd)), k);
}

/**
 * @brief Call `X(V, E, W)` for every graph type the Python package binds (the library's
 *        instantiations, PLAN Section 4.4.3).
 */
#define DYNG_PY_FOR_EACH_GRAPH_TYPE(X)              \
  X(std::int32_t, std::int32_t, std::int32_t)       \
  X(std::int32_t, std::int64_t, std::int32_t)       \
  X(std::int64_t, std::int64_t, std::int32_t)       \
  X(std::int32_t, std::int32_t, ::dyng::unweighted) \
  X(std::int32_t, std::int64_t, ::dyng::unweighted)

/// Call `X(V, W)` for every batch type (the (vertex, weight) pairs of the graph types).
#define DYNG_PY_FOR_EACH_BATCH_TYPE(X) \
  X(std::int32_t, std::int32_t)        \
  X(std::int64_t, std::int32_t)        \
  X(std::int32_t, ::dyng::unweighted)

}  // namespace dyng::python
