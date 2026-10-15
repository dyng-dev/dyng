// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file types.hpp
 * @brief The native types shared by several binding files: graph, hypergraph and result holders
 *        and the batches of arrays, with the lists of their instantiations.
 */
#pragma once

#include "common.hpp"

#include <dyng/cycle_count.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/mosp.hpp>
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

/// The holder of a mosp result.
template <typename vertex_t>
using mosp_holder = result_holder<mosp::result<vertex_t>>;

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
  std::optional<in_array<vertex_t>> insert_vertices;          ///< vertex insertions (0.4)
  std::optional<in_array<std::int8_t>> insert_vertex_labels;  ///< their labels (0.4)
  std::optional<in_array<vertex_t>> delete_vertices;          ///< vertex deletions (0.4)
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

/// The holder of hypergraph<V,W>.
template <typename vertex_t, typename weight_t>
using hypergraph_holder = holder<hypergraph<vertex_t, weight_t>>;

/**
 * @brief A batch of hypergraph changes whose arrays are Python arrays (kept alive, never copied).
 *
 * The typed layer passes arrays of exactly the hypergraph's types (int64 hyperedge ids and
 * offsets, the vertex type, the weight type), converted with range checks there. Immutable, so it
 * needs no lock.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted: no weights).
 */
template <typename vertex_t, typename weight_t>
struct hyper_batch_arrays {
  in_array<std::int64_t> delete_hyperedges;  ///< ids of the deleted hyperedges
  in_array<std::int64_t> insert_offsets;     ///< insertions + 1 offsets (or empty)
  in_array<vertex_t> insert_members;         ///< members of the inserted hyperedges
  std::optional<in_array<weight_scalar_t<weight_t>>> insert_weights;  ///< one per insertion
  std::optional<in_array<std::int64_t>> delete_incidence_hyperedge;   ///< (h, v) removals (0.4)
  std::optional<in_array<vertex_t>> delete_incidence_vertex;          ///< (h, v) removals (0.4)
  std::optional<in_array<std::int64_t>> insert_incidence_hyperedge;   ///< (h, v) additions (0.4)
  std::optional<in_array<vertex_t>> insert_incidence_vertex;          ///< (h, v) additions (0.4)

  /**
   * @brief The library's view of the arrays.
   * @return The view (valid while this object lives).
   */
  [[nodiscard]] hyperedge_batch_view<vertex_t, weight_t> view() const {
    hyperedge_batch_view<vertex_t, weight_t> v;
    v.delete_hyperedges = view_of(delete_hyperedges);
    v.insert_offsets = view_of(insert_offsets);
    v.insert_members = view_of(insert_members);
    if constexpr (!is_unweighted_v<weight_t>) {
      if (insert_weights) {
        v.insert_weights = view_of(*insert_weights);
      }
    }
    if (delete_incidence_hyperedge) {
      v.delete_incidence_hyperedge = view_of(*delete_incidence_hyperedge);
    }
    if (delete_incidence_vertex) {
      v.delete_incidence_vertex = view_of(*delete_incidence_vertex);
    }
    if (insert_incidence_hyperedge) {
      v.insert_incidence_hyperedge = view_of(*insert_incidence_hyperedge);
    }
    if (insert_incidence_vertex) {
      v.insert_incidence_vertex = view_of(*insert_incidence_vertex);
    }
    return v;
  }
};

/**
 * @brief Move a hyperedge_batch (from a reader or a generator) into NumPy arrays.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] b The batch (its arrays are copied: the owning batch keeps them).
 * @return (delete_hyperedges int64, insert_offsets int64 (at least [0]), insert_members,
 *         insert_weights int32 or None).
 */
template <typename vertex_t, typename weight_t>
nb::tuple hyper_batch_to_numpy(const hyperedge_batch<vertex_t, weight_t>& b) {
  std::vector<std::int64_t> deletions = b.delete_hyperedges();
  std::vector<std::int64_t> offsets = b.insert_offsets();
  if (offsets.empty()) {
    offsets.push_back(0);
  }
  std::vector<vertex_t> members = b.insert_members();
  nb::object weights = nb::none();
  if constexpr (!is_unweighted_v<weight_t>) {
    std::vector<weight_t> w = b.insert_weights();
    weights = nb::cast(to_numpy(std::move(w)));
  }
  return nb::make_tuple(to_numpy(std::move(deletions)), to_numpy(std::move(offsets)),
                        to_numpy(std::move(members)), weights);
}

/**
 * @brief Move a hyperedge_list into NumPy arrays.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] list The list (moved from).
 * @return (num_vertices, offsets int64 (at least [0]), members, weights int32 or None).
 */
template <typename vertex_t, typename weight_t>
nb::tuple hyperedge_list_to_numpy(hyperedge_list<vertex_t, weight_t>&& list) {
  if (list.offsets.empty()) {
    list.offsets.push_back(0);
  }
  nb::object weights = nb::none();
  if constexpr (!is_unweighted_v<weight_t>) {
    weights = nb::cast(to_numpy(std::move(list.weights)));
  }
  return nb::make_tuple(static_cast<std::int64_t>(list.num_vertices),
                        to_numpy(std::move(list.offsets)), to_numpy(std::move(list.members)),
                        weights);
}

/**
 * @brief The hyperedge_list_view of arrays passed by the typed layer (host memory, no copy).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] num_vertices The vertex count (checked against vertex_t).
 * @param[in] offsets      Hyperedges + 1 offsets.
 * @param[in] members      The members.
 * @param[in] weights      One weight per hyperedge, or none.
 * @return The view (valid while the arrays live).
 * @throws invalid_argument_error if num_vertices does not fit vertex_t.
 */
template <typename vertex_t, typename weight_t>
hyperedge_list_view<vertex_t, weight_t> hyperedge_list_view_of(
    std::int64_t num_vertices, const in_array<std::int64_t>& offsets,
    const in_array<vertex_t>& members,
    const std::optional<in_array<weight_scalar_t<weight_t>>>& weights) {
  DYNG_EXPECTS(num_vertices >= 0 &&
                   num_vertices <= static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max()),
               "num_vertices ", num_vertices, " does not fit the vertex id type");
  hyperedge_list_view<vertex_t, weight_t> v;
  v.num_vertices = static_cast<vertex_t>(num_vertices);
  v.offsets = view_of(offsets);
  v.members = view_of(members);
  if constexpr (!is_unweighted_v<weight_t>) {
    if (weights) {
      v.weights = view_of(*weights);
    }
  }
  return v;
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

/// Call `X(V, W)` for every hypergraph type (the library's instantiations, PLAN Section 4.4.3).
#define DYNG_PY_FOR_EACH_HYPERGRAPH_TYPE(X) \
  X(std::int32_t, ::dyng::unweighted)       \
  X(std::int32_t, std::int32_t)

}  // namespace dyng::python
