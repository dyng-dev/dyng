// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:headers/deviceGraph.cuh (DeviceGraph, uploadDeviceGraph)
/**
 * @file device_graph.hpp
 * @brief The resident device copy of a graph built with CUDA resources: out- and in-edge CSR with
 *        one contiguous weight column per objective (MOSP-CUDA's DeviceGraph).
 *
 * Host-only declarations (buffers, no CUDA headers); build_device_graph() lives in
 * device_graph.cu. In this release the batch is still applied on the host, as MOSP-CUDA applies
 * it (applyChangeBatch), and the updated out-edges are uploaded once per graph state; the in-edges
 * are built on the device (count, scan, fill), as uploadDeviceGraph() does. A device apply that
 * keeps the resident arrays across batches comes later (PLAN Section 6.4.1).
 */
#pragma once

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>

#include <cstddef>

namespace dyng::detail {

/**
 * @brief One graph state on the device: out- and in-edges, objective-major weights.
 *
 * The row order of the in-edges is unspecified (the fill appends with atomics, as MOSP-CUDA's
 * fillReverseKernel does); the device engines only take minima over in-neighbours, which do not
 * depend on it. The host CSR of the graph stays the authoritative copy (view(), to_csr()).
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
           in_col_ind.size() * sizeof(vertex_t) + in_weights.size() * sizeof(weight_t);
  }
};

/**
 * @brief Upload the out-edges of `host` and build the in-edges on the device (MOSP-CUDA's
 *        uploadDeviceGraph(): only the out-edge CSR crosses PCIe; in-degrees, exclusive scan and
 *        fill run on the device). The weights are uploaded as they are stored, one column per
 *        objective, so MOSP's split kernel (edge-major to objective-major) is not needed.
 *
 * Ordered on the stream of `res` (no synchronization); memory from `res.memory()`.
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

}  // namespace dyng::detail
