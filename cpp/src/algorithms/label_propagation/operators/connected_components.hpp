// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/connectedComponents.cu (the component numbering:
// a root is its component's smallest vertex, its id the rank of the root in the vertex list)
/**
 * @file connected_components.hpp
 * @brief connected_components (PLAN 4.5.3): the components of an edge list over a vertex list,
 *        with canonical ids; its host half (union-find). The CUDA half (Shiloach-Vishkin) is
 *        connected_components.cuh.
 *
 * The vertices are a list (every endpoint of an edge must be in it); the result gives each listed
 * vertex v the id of its component: every component's root is its smallest vertex, and the ids
 * number the roots in the order of the list (0, 1, ...), the original's numbering. Both backends
 * give the same ids, so they are compared by value. The host uses union-find (union by the smaller
 * root, path halving); CUDA hooks the larger root under the smaller with atomicMin and compresses,
 * one round after the other until a hook changes nothing (one host synchronization per round),
 * then flags the roots, scans the flags and numbers them.
 *
 * Rule of two (ADR 0102): label_propagation is the only user (the named second user is a WCC
 * algorithm); the operator stays in this folder until a second algorithm uses it.
 */
#pragma once

#include "operators/execution.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <cstddef>
#include <cstdint>

namespace dyng::detail::lp_ops {

/// The scratch of connected_components (the caller's memory, of the executor's backend).
struct components_scratch {
  std::int32_t* parent = nullptr;     ///< per vertex id: the parent (only listed ids are touched)
  std::int32_t* root_id = nullptr;    ///< per vertex id: a root's component id
  std::int32_t* root_flag = nullptr;  ///< per list position (CUDA): whether it is a root
  std::int32_t* root_scan = nullptr;  ///< per list position (CUDA): the roots before it
  void* scan_temp = nullptr;          ///< CUDA: the scan's temporary storage
  std::size_t scan_temp_bytes = 0;    ///< its size
  std::int32_t* words = nullptr;      ///< CUDA: two device words (the round's flag, the count)
};

/// What connected_components found.
struct components_result {
  std::int32_t count = 0;  ///< the components (one per root)
  int rounds = 0;          ///< Shiloach-Vishkin rounds on CUDA (0 on the host)
};

/**
 * @brief connected_components on one backend (specialised per executor: the host here, cuda_exec
 *        in connected_components.cuh).
 * @tparam exec_t The executor.
 */
template <typename exec_t>
struct components_impl;

/// connected_components on the host: union-find.
template <>
struct components_impl<operators::sequential_exec> {
  /// See connected_components().
  static components_result run(const resources& /*res*/, const std::int32_t* src,
                               const std::int32_t* dst, const std::int32_t* num_edges,
                               int /*max_edges*/, const std::int32_t* vertices, int num_vertices,
                               const components_scratch& s, std::int32_t* component_id,
                               int /*max_rounds*/) {
    std::int32_t* parent = s.parent;
    for (int i = 0; i < num_vertices; ++i) {
      parent[vertices[i]] = vertices[i];
    }
    const auto find = [parent](std::int32_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const std::int32_t edges = *num_edges;
    for (std::int32_t e = 0; e < edges; ++e) {
      const std::int32_t a = find(src[e]);
      const std::int32_t b = find(dst[e]);
      if (a < b) {
        parent[b] = a;
      } else if (b < a) {
        parent[a] = b;
      }
    }
    components_result r;
    for (int i = 0; i < num_vertices; ++i) {
      const std::int32_t v = vertices[i];
      if (find(v) == v) {
        s.root_id[v] = r.count++;
      }
    }
    for (int i = 0; i < num_vertices; ++i) {
      component_id[vertices[i]] = s.root_id[find(vertices[i])];
    }
    return r;
  }
};

/**
 * @brief The components of the edges (src[e], dst[e]), e < *num_edges, over the listed vertices,
 *        with canonical ids (see the file comment).
 * @tparam exec_t The executor (operators::sequential_exec, or operators::cuda_exec with
 *                connected_components.cuh).
 * @param[in]  res          The resources (the executor's backend).
 * @param[in]  src          The edges' first endpoints.
 * @param[in]  dst          Their second endpoints.
 * @param[in]  num_edges    The number of edges (one word of the backend's memory: on CUDA it may
 *                          be written by an earlier kernel on the stream).
 * @param[in]  max_edges    An upper bound of *num_edges (CUDA: the hook's grid).
 * @param[in]  vertices     The vertices (every endpoint among them; distinct).
 * @param[in]  num_vertices Their number.
 * @param[in]  s            The scratch.
 * @param[out] component_id Per listed vertex its component's id (indexed by vertex id).
 * @param[in]  max_rounds   CUDA: the rounds before internal_error (a malfunction guard).
 * @return The count of components and the rounds.
 * @throws internal_error on CUDA if the rounds reach max_rounds.
 */
template <typename exec_t>
components_result connected_components(const resources& res, const std::int32_t* src,
                                       const std::int32_t* dst, const std::int32_t* num_edges,
                                       int max_edges, const std::int32_t* vertices,
                                       int num_vertices, const components_scratch& s,
                                       std::int32_t* component_id, int max_rounds = 64) {
  if (num_vertices <= 0) {
    return {};
  }
  return components_impl<exec_t>::run(res, src, dst, num_edges, max_edges, vertices, num_vertices,
                                      s, component_id, max_rounds);
}

}  // namespace dyng::detail::lp_ops
