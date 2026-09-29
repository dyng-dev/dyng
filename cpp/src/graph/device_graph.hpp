// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:headers/deviceGraph.cuh (DeviceGraph, uploadDeviceGraph)
/**
 * @file device_graph.hpp
 * @brief The resident device copy of a graph built with CUDA resources: out- and in-edge CSR with
 *        one contiguous weight column per objective (MOSP-CUDA's DeviceGraph).
 *
 * Host-only declarations (buffers, no CUDA headers); the functions live in device_graph.cu and
 * apply_set_device.cu. A batch is applied on the host, as MOSP-CUDA applies it (applyChangeBatch),
 * and the updated out-edges are uploaded once per graph state; the in-edges are built on the
 * device (count, scan, fill), as uploadDeviceGraph() does, when an engine first needs them. A
 * resident graph under batch_semantics::as_sets without weight columns keeps its arrays on the
 * device across batches instead: apply_set_batch_device() merges the normalized batch into the
 * sorted rows on the device (CycleEnumeration-GPU's build_next_rows_kernel; PLAN Section 6.4.1).
 */
#pragma once

#include "graph/normalized_batch.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

/// The value of device_graph::insertion_ids at an edge the last batch did not insert (the byte
/// pattern 0x7f of cudaMemset; larger than every change id, as CycleEnumeration-GPU's kNoOwner).
inline constexpr std::int32_t no_change_id = 0x7f7f7f7f;

/**
 * @brief One graph state on the device: out-edges and, once an engine asked for them, in-edges;
 *        objective-major weights.
 *
 * The row order of the in-edges is unspecified (the fill appends with atomics, as MOSP-CUDA's
 * fillReverseKernel does); the device engines only take minima over in-neighbours, which do not
 * depend on it. The host CSR of the graph is the authoritative copy unless a device apply
 * produced this state (graph_impl).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct device_graph {
  int device = -1;               ///< the device the arrays live on
  vertex_t num_vertices = 0;     ///< n
  edge_t num_edges = 0;          ///< m
  int num_weights = 0;           ///< K
  buffer<edge_t> out_row_ptr;    ///< n + 1 out-edge offsets
  buffer<vertex_t> out_col_ind;  ///< m out-neighbours
  buffer<weight_t> out_weights;  ///< K columns of m weights, out-edge order
  buffer<edge_t> in_row_ptr;     ///< n + 1 in-edge offsets
  buffer<vertex_t> in_col_ind;   ///< m in-neighbours
  buffer<weight_t> in_weights;   ///< K columns of m weights, in-edge order
  bool has_in_edges = false;     ///< the in-edge arrays are built
  /// Per out-edge of this state: the id (position in normalized_batch::insertions) of the
  /// insertion that added it, if a device apply produced this state (apply_set_batch_device),
  /// else no_change_id; empty for an uploaded state. The cycle_count insert phase reads it as its
  /// ownership array, as CycleEnumeration-GPU's count_update_cycles_device reads next_owner.
  buffer<std::int32_t> insertion_ids;

  /**
   * @brief The weight column of objective `k`, out-edge order (MOSP's DeviceGraph::out(k)).
   * @param[in] k Objective in [0, num_weights).
   * @return Device pointer to m weights.
   */
  [[nodiscard]] const weight_t* out_column(int k) const noexcept {
    return out_weights.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(num_edges);
  }

  /**
   * @brief The weight column of objective `k`, in-edge order (MOSP's DeviceGraph::in(k)).
   * @param[in] k Objective in [0, num_weights).
   * @return Device pointer to m weights.
   */
  [[nodiscard]] const weight_t* in_column(int k) const noexcept {
    return in_weights.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(num_edges);
  }

  /**
   * @brief The device memory held.
   * @return Bytes of all arrays.
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return out_row_ptr.size() * sizeof(edge_t) + out_col_ind.size() * sizeof(vertex_t) +
           out_weights.size() * sizeof(weight_t) + in_row_ptr.size() * sizeof(edge_t) +
           in_col_ind.size() * sizeof(vertex_t) + in_weights.size() * sizeof(weight_t) +
           insertion_ids.size() * sizeof(std::int32_t);
  }
};

/**
 * @brief Upload the out-edges of `host` (MOSP-CUDA's uploadDeviceGraph(): only the out-edge CSR
 *        crosses PCIe). The weights are uploaded as they are stored, one column per objective, so
 *        MOSP's split kernel (edge-major to objective-major) is not needed. The in-edges are built
 *        separately (build_device_in_edges) when an engine needs them.
 *
 * Ordered on the stream of `res` (no synchronization; graph_impl::device_edges() waits for it);
 * memory from `res.memory()`.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  res  Resources of the CUDA backend.
 * @param[in]  host The out-edge CSR (host memory, objective-major weights).
 * @param[out] out  Receives the device graph (its previous arrays are released first).
 * @throws out_of_memory_error if device memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_graph(const resources& res, const csr<vertex_t, edge_t, weight_t>& host,
                        device_graph<vertex_t, edge_t, weight_t>& out);

/**
 * @brief Build the in-edges of a device graph on the device (MOSP-CUDA's uploadDeviceGraph():
 *        in-degrees, exclusive scan and fill); no-op if they are built.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources of the CUDA backend (the graph's device).
 * @param[in,out] g   The device graph.
 * @throws out_of_memory_error if device memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_in_edges(const resources& res, device_graph<vertex_t, edge_t, weight_t>& g);

/**
 * @brief Copy the out-edges of a device graph into a host CSR (the download of a stale host copy,
 *        for a host read of a graph whose state a device apply produced): asynchronous copies on
 *        the stream the state's buffers are ordered on (the stream of the resources that built it;
 *        never the legacy default stream), then a synchronization of that stream. The copies go
 *        straight into the CSR's pageable vectors (a pinned staging copy would add a host copy of
 *        the whole graph); a vector that must grow is released first and reserved with room for
 *        growth, so the stale content is never copied.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  g    The device graph.
 * @param[out] host The out-edge CSR (resized).
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void download_device_graph(const device_graph<vertex_t, edge_t, weight_t>& g,
                           csr<vertex_t, edge_t, weight_t>& host);

/**
 * @brief Whether apply_set_batch_device() can apply batches to graphs of these types.
 * @tparam vertex_t Vertex id type.
 * @return true for 32-bit vertex ids (the merge kernels read vertex ids as 32-bit values).
 */
template <typename vertex_t>
inline constexpr bool device_set_apply_supported_v = sizeof(vertex_t) == 4;

/**
 * @brief Apply a normalized batch (batch_semantics::as_sets) to a resident graph on the device:
 *        the next graph state, built from `base` by CycleEnumeration-GPU's device row merge
 *        (change_rows_kernel, next_degree_kernel, an exclusive scan and build_next_rows_kernel),
 *        with its insertion_ids.
 *
 * Precondition: `base` has no weight columns and `normalized` was computed for its state. The
 * rows of the result are sorted and equal (row offsets and neighbours) to the host set apply's.
 * Synchronizes the stream of `res` at the end (the host copy of the state may be downloaded on
 * another stream).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  res        Resources of the CUDA backend (the graph's device).
 * @param[in]  base       The resident graph G_t.
 * @param[in]  normalized The normalized batch of G_t.
 * @param[out] next       Receives G_{t+1} (out-edges and insertion_ids; no in-edges).
 * @throws out_of_memory_error if device memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void apply_set_batch_device(const resources& res,
                            const device_graph<vertex_t, edge_t, weight_t>& base,
                            const normalized_batch<vertex_t>& normalized,
                            device_graph<vertex_t, edge_t, weight_t>& next);

/**
 * @brief The deletion marks of a normalized batch on the resident G_t (CycleEnumeration-GPU's
 *        mark_owners_kernel on a cudaMemset 0x7f array): computed once per update, then shared by
 *        the device apply and the cycle_count delete phase (no-op while
 *        `normalized.deletions_marked`).
 *
 * Precondition: `normalized` was computed for the state of `base`. The array (m ints) is kept in
 * `normalized` and reused by the next update of the same size.
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (the weight columns are not read).
 * @param[in] res        Resources of the CUDA backend (the graph's device).
 * @param[in] base       The resident graph G_t.
 * @param[in] normalized The normalized batch of G_t (its device fields are written).
 * @return Device pointer to m marks (normalized.deletion_owner).
 * @throws out_of_memory_error if device memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
const int* mark_normalized_deletions(const resources& res,
                                     const device_graph<vertex_t, edge_t, weight_t>& base,
                                     const normalized_batch<vertex_t>& normalized);

/**
 * @brief Step 0 of batch_semantics::as_sets (normalize_set_batch()) against the resident device
 *        copy of G_t, for a graph whose host copy a device apply left stale: the requested lists
 *        are built, sorted and deduplicated on the host as normalize_set_batch() does, and the
 *        membership of each requested change in G_t is answered on the device
 *        (device_edge_membership), so a chain of updates never downloads the graph (ADR 0020).
 *
 * The result equals normalize_set_batch() on the host CSR of the same state.
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  res   Resources of the CUDA backend (the graph's device).
 * @param[in]  base  The resident graph G_t.
 * @param[in]  batch The batch (host memory).
 * @param[in]  props Properties of the graph (semantics.as_sets set).
 * @param[out] out   The normalized batch.
 * @throws invalid_argument_error, not_supported_error, capacity_error as normalize_set_batch().
 * @throws cuda_error if the runtime reports an error.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_set_batch_device(const resources& res,
                                const device_graph<vertex_t, edge_t, weight_t>& base,
                                const edge_batch_view<vertex_t, weight_t>& batch,
                                const graph_properties& props, normalized_batch<vertex_t>& out);

/**
 * @brief Whether each requested change names an edge of the resident G_t (has_edge on the device:
 *        one binary search per change in its sorted row; false for a source >= n).
 *
 * The pairs are uploaded through the pinned staging of `nb` and the flags are read back through
 * pinned memory, on the stream of `res` (synchronized before the return).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (the weight columns are not read).
 * @param[in]  res        Resources of the CUDA backend (the graph's device).
 * @param[in]  base       The resident graph G_t.
 * @param[in]  deletions  The requested deletions.
 * @param[in]  insertions The requested insertions.
 * @param[in]  nb         The normalized batch whose device buffers are used as scratch.
 * @param[out] present    present[i] = 1 if change i (deletions first) is an edge of G_t, else 0
 *                        (deletions.size() + insertions.size() entries).
 * @throws capacity_error      if the lists hold 2^31 or more changes.
 * @throws out_of_memory_error if memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void device_edge_membership(const resources& res,
                            const device_graph<vertex_t, edge_t, weight_t>& base,
                            const std::vector<set_change<vertex_t>>& deletions,
                            const std::vector<set_change<vertex_t>>& insertions,
                            const normalized_batch<vertex_t>& nb,
                            std::vector<std::uint8_t>& present);

}  // namespace dyng::detail
