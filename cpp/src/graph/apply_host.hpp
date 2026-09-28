// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:headers/csrGraph.h (applyChangeBatch, transposeCsrGraph)
/**
 * @file apply_host.hpp
 * @brief Host batch application: a new compact CSR per batch (MOSP applyChangeBatch semantics).
 */
#pragma once

#include "graph/graph_impl.hpp"

#include <dyng/core/error.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstdint>
#include <limits>
#include <string>

namespace dyng::detail {

/**
 * @brief Converts an edge count to edge_t: the checked construction of ADR 0009.
 * @tparam edge_t Edge offset type.
 * @param count The number of edges (>= 0).
 * @return `count` as edge_t.
 * @throws capacity_error if `count` does not fit edge_t; the message names the int64 edge_t
 *         instantiation.
 */
template <typename edge_t>
edge_t checked_edge_count(std::int64_t count) {
  if (count > static_cast<std::int64_t>(std::numeric_limits<edge_t>::max())) {
    throw capacity_error("dyng: " + std::to_string(count) +
                         " edges do not fit the edge offset type; use a graph with 64-bit "
                         "edge_t (int64)");
  }
  return static_cast<edge_t>(count);
}

/**
 * @brief Apply `batch` to `original` under `props`, writing the updated out-edge CSR.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  original The out-edge CSR before the batch.
 * @param[in]  batch    The batch (host memory).
 * @param[in]  props    Properties of the graph (direction, row order, semantics).
 * @param[out] updated  The out-edge CSR after the batch (must not alias `original`).
 * @param[out] delta    The effective changes and the weight-increase flags (may be nullptr).
 * @param[in]  threads  OpenMP threads for assembling the updated CSR (1 = sequential; the result
 *                      is the same for every thread count).
 * @return The counters of the batch.
 * @throws invalid_argument_error on malformed batches or a semantics rule that says error.
 * @throws not_supported_error    for vertex insertions or deletions.
 * @throws capacity_error         if the edge count no longer fits edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_batch_host(const csr<vertex_t, edge_t, weight_t>& original,
                               const edge_batch_view<vertex_t, weight_t>& batch,
                               const graph_properties& props,
                               csr<vertex_t, edge_t, weight_t>& updated,
                               apply_delta<vertex_t>* delta, int threads = 1);

/**
 * @brief The reverse graph: row v lists u for every edge u -> v (with its weights).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  graph   The out-edge CSR.
 * @param[out] reverse The in-edge CSR; inside a row, sources appear in out-edge order.
 * @param[in]  threads OpenMP threads (> 1: a parallel fill with the same, deterministic result).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void transpose_host(const csr<vertex_t, edge_t, weight_t>& graph,
                    csr<vertex_t, edge_t, weight_t>& reverse, int threads = 1);

/**
 * @brief Build a compact out-edge CSR from an edge list (see graph::from_edges).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  edges The edges (host memory).
 * @param[in]  props The properties of the graph.
 * @param[out] out   The CSR.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_from_edges_host(const edge_list_view<vertex_t, weight_t>& edges,
                           const graph_properties& props, csr<vertex_t, edge_t, weight_t>& out);

/**
 * @brief Build a compact out-edge CSR from a CSR (see graph::from_csr).
 *
 * A well-formed CSR that already is what the properties ask for (directed, self-loops kept or
 * absent, rows in the requested order and without forbidden parallel edges) is taken as it is:
 * copied in parallel, or moved from `movable`. Otherwise the rows are rebuilt.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     input   The input CSR (host memory).
 * @param[in]     props   The properties of the graph.
 * @param[out]    out     The CSR.
 * @param[in]     threads OpenMP threads for the checks and the copy (1 = sequential).
 * @param[in,out] movable The owner of the arrays `input` views, or nullptr; if given and the CSR
 *                        is taken as it is, its arrays are moved into `out` (it is left empty).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_from_csr_host(const csr_view<vertex_t, edge_t, weight_t>& input,
                         const graph_properties& props, csr<vertex_t, edge_t, weight_t>& out,
                         int threads = 1, csr<vertex_t, edge_t, weight_t>* movable = nullptr);

/**
 * @brief Check the invariants of a stored graph; returns a description of the first violation.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] impl The graph state.
 * @return An empty string if every invariant holds.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::string integrity_violation(const graph_impl<vertex_t, edge_t, weight_t>& impl);

}  // namespace dyng::detail
