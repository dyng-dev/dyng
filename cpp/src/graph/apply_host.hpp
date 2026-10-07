// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:headers/csrGraph.h (applyChangeBatch, transposeCsrGraph)
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/dynamic/directed_graph.hpp
// (apply_batch, prepare_batch)
/**
 * @file apply_host.hpp
 * @brief Host batch application: a new compact CSR per batch (MOSP applyChangeBatch semantics, or
 *        CycleEnumeration-GPU's set semantics under batch_semantics::as_sets).
 */
#pragma once

#include "graph/graph_impl.hpp"
#include "graph/normalized_batch.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
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
 * @brief Throws unless every array of `batch` is readable on the host. Checked before any host
 *        read of the batch (the public entries stage device arrays first, core/staging.hpp; this
 *        guards the internal paths).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @param[in] batch The batch.
 * @param[in] what  The calling function, for the message.
 * @throws invalid_argument_error naming the first array that is not host-accessible.
 */
template <typename vertex_t, typename weight_t>
void expect_host_batch(const edge_batch_view<vertex_t, weight_t>& batch, const char* what) {
  const auto check = [what](const auto& view, const char* name) {
    DYNG_EXPECTS(view.empty() || is_host_accessible(view.space()), what,
                 ": edge_batch_view::", name, " must be in host-accessible memory here");
  };
  check(batch.insert_src, "insert_src");
  check(batch.insert_dst, "insert_dst");
  check(batch.insert_weights, "insert_weights");
  check(batch.delete_src, "delete_src");
  check(batch.delete_dst, "delete_dst");
  check(batch.insert_vertices, "insert_vertices");
  check(batch.insert_vertex_labels, "insert_vertex_labels");
  check(batch.delete_vertices, "delete_vertices");
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
 * @param[out] delta    The requested changes and the weight-increase flags (apply_delta; may
 *                      be nullptr).
 * @param[in]  threads  OpenMP threads for assembling the updated CSR (1 = sequential; the result
 *                      is the same for every thread count).
 * @param[in]  normalized Under batch_semantics::as_sets: Step 0 already computed for `original`
 *                      and `batch` (normalize_set_batch), or nullptr; ignored otherwise.
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
                               apply_delta<vertex_t>* delta, int threads = 1,
                               const normalized_batch<vertex_t>* normalized = nullptr);

/**
 * @brief apply_batch_host() for batch_semantics::as_sets (apply_set_host.cpp): Step 0 reduces
 *        the batch to two sorted, duplicate-free lists of structural changes (CycleEnumeration-GPU
 *        prepare_batch), which are merged into the sorted rows (its apply_batch).
 *
 * `delta` receives the normalized batch: the deletions and the insertions, each sorted by
 * (source, destination), with zero weight-increase flags (every weight change is a deletion
 * followed by an insertion of the edge).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  original The out-edge CSR before the batch (sorted rows without parallel edges).
 * @param[in]  batch    The batch (host memory; its shape already validated).
 * @param[in]  props    Properties of the graph (semantics.as_sets set).
 * @param[out] updated  The out-edge CSR after the batch (must not alias `original`).
 * @param[out] delta    The normalized batch (may be nullptr).
 * @param[in]  threads  OpenMP threads for assembling the updated CSR (the result is the same for
 *                      every thread count).
 * @param[in]  normalized Step 0 already computed for `original` and `batch` (normalize_set_batch;
 *                      run_update computes it once per update), or nullptr to compute it here.
 * @return The counters of the batch.
 * @throws invalid_argument_error on malformed batches or a semantics rule that says error.
 * @throws not_supported_error    if the properties do not allow set semantics (see as_sets).
 * @throws capacity_error         if the edge count no longer fits edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_set_batch_host(const csr<vertex_t, edge_t, weight_t>& original,
                                   const edge_batch_view<vertex_t, weight_t>& batch,
                                   const graph_properties& props,
                                   csr<vertex_t, edge_t, weight_t>& updated,
                                   apply_delta<vertex_t>* delta, int threads = 1,
                                   const normalized_batch<vertex_t>* normalized = nullptr);

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
