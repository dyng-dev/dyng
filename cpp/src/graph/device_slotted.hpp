// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:headers/dynamicGraph.h (DeviceGraph, ErrorFlag,
// createDeviceGraph, launchApplyVertexChanges, launchAppendEdges, launchReadErrorFlags)
/**
 * @file device_slotted.hpp
 * @brief row_layout::slotted on the CUDA backend (ADR 0074): the resident device copy of a
 *        slotted graph (LabelPropagation-CUDA's DeviceGraph) and the device apply of a vertex
 *        batch (its launchApplyVertexChanges and launchAppendEdges).
 *
 * Host-only declarations (buffers, no CUDA headers); the functions live in
 * apply_slotted_device.cu. A slotted graph built with CUDA resources keeps its rows on the device:
 * the slot ranges (row_ptr, fixed), the end of each row's entries (row_end), the slot columns and
 * weights, and the state of every vertex. Only the batch crosses PCIe; the host copy of the rows
 * is stale until something reads it (graph_impl::host_edges() downloads it once, ADR 0020's
 * rule). The vertex states are kept current on the host as well (Step 0 reads them; the original
 * keeps its host status array the same way), and so are the slot ranges (the capacities).
 *
 * One batch, in the original's order (dynLP.cu, runDynLP, steps b1-b3):
 *   1. Step 0 on the host (normalize_vertex_batch, every check but the capacity one), then
 *      upload_vertex_batch(): the deleted ids, Ins \ Del with their seed labels and the edges, from
 *      pinned memory directly or through the graph's pinned staging (graph_access::normalize);
 *   2. the algorithm's work on G_t (label_propagation: the deletion marks on the rows before the
 *      batch, markDeletionsKernel);
 *   3. apply_vertex_batch_device() (the commit): the removed live entries are counted, the deleted
 *      ids become deleted, Ins \ Del active (a vertex that was not inactive is recorded), then the
 *      edges are appended to the rows of both endpoints in batch order: half-edge keys (key = n for
 *      an edge whose endpoints are not both active, and for an endpoint out of range, which is
 *      recorded: a documented fix of the original, which would read out of bounds), CUB's stable
 *      radix sort of (key, half-edge), an exclusive sum by key (the rank in the row), the append
 *      (a half-edge past the end of its row is dropped and recorded), the advance of the row ends;
 *      then the report (the error word and the counts) is copied to its pinned mirror;
 *   4. settle(): after a synchronization of the stream that the caller makes anyway (the first of
 *      the algorithm's), the report is read: the error word (the overflow becomes capacity_error,
 *      a re-insertion invalid_argument_error) and the live entries (graph::num_edges()). An
 *      overflow poisons the graph: the graph keeps the dropped state (the basic guarantee), and
 *      every later apply throws.
 *
 * The kernels move vertex ids and weight bits only: no floating-point arithmetic.
 */
#pragma once

#include "core/cuda_runtime.hpp"
#include "graph/normalized_batch.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail {

/**
 * @brief Whether a slotted graph of these types keeps its rows on the device with CUDA resources:
 *        32-bit vertex ids (the original's int keys of the radix sort) and at most one weight
 *        column.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
inline constexpr bool device_slotted_supported_v = sizeof(vertex_t) == 4;

/// The entries of the report of one device apply (int64 each; LabelPropagation-CUDA's ErrorFlag,
/// widened to the counts the graph keeps).
enum device_slotted_report : int {
  report_error_bits = 0,    ///< the device_error bits (util/device_error_flags.hpp)
  report_reinserted = 1,    ///< 1 + the largest re-inserted id, 0 if none
  report_bad_endpoint = 2,  ///< 1 + the index of an edge with an endpoint out of range, 0 if none
  report_overflow_row = 3,  ///< 1 + the largest row an append overflowed, 0 if none
  report_dropped = 4,       ///< half-edges not appended (an endpoint not active after the batch)
  report_removed = 5,       ///< live entries the deletions removed
  report_entries = 6,       ///< the number of entries
};

/**
 * @brief What settle() read from the report of a device apply.
 */
struct device_slotted_settled {
  std::uint32_t error_bits = 0;    ///< device_error bits (0: none)
  std::int64_t reinserted = -1;    ///< a re-inserted id (-1: none)
  std::int64_t bad_endpoint = -1;  ///< an edge with an endpoint out of range (-1: none)
  std::int64_t overflow_row = -1;  ///< a row an append overflowed (-1: none)
  std::int64_t appended = 0;       ///< half-edges appended (both rows count)
  std::int64_t dropped = 0;        ///< half-edges dropped (an endpoint not active)
  std::int64_t removed = 0;        ///< live entries the deletions removed
};

/**
 * @brief The resident device copy of a slotted graph and the scratch of its device apply.
 *
 * The buffers are allocated from the memory resource of the resources that built the copy and
 * ordered on its stream; the scratch grows (with headroom) outside the kernels when a batch is
 * larger than the reserved size (graph::reserve() sizes it up front).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct device_slotted {
  int device = -1;              ///< the device the arrays live on
  vertex_t num_vertices = 0;    ///< n
  edge_t num_slots = 0;         ///< the total row capacity
  int num_weights = 0;          ///< 0 or 1
  int key_bits = 1;             ///< radix-sort bits: the keys lie in [0, n]
  buffer<edge_t> row_ptr;       ///< n + 1 slot offsets (fixed)
  buffer<edge_t> row_end;       ///< n row ends
  buffer<vertex_t> col_ind;     ///< the slot columns (-1 in the unused slots)
  buffer<weight_t> weights;     ///< the slot weights (0 in the unused slots; empty if K = 0)
  buffer<std::int32_t> status;  ///< n vertex states (vertex_status's values as int32)

  // ---- The current batch on the device (upload_vertex_batch) ----
  buffer<vertex_t> deleted;                ///< the batch's deletions (every listed id)
  buffer<vertex_t> deleted_effective;      ///< Del_eff (unused when it equals `deleted`)
  buffer<vertex_t> inserted;               ///< Ins_eff = Ins \ Del, in list order
  buffer<std::int8_t> inserted_labels;     ///< the seed label of each inserted vertex (-1: none)
  buffer<vertex_t> edge_src;               ///< the batch's edge sources
  buffer<vertex_t> edge_dst;               ///< the batch's edge destinations
  buffer<weight_t> edge_weight;            ///< the batch's edge weights (empty if K = 0)
  std::int64_t num_deleted = 0;            ///< |Del| of the current batch
  std::int64_t num_deleted_effective = 0;  ///< |Del_eff|
  std::int64_t num_inserted = 0;           ///< |Ins_eff|
  std::int64_t num_edges = 0;              ///< the batch's edges
  bool has_labels = false;                 ///< the batch has seed labels
  const vertex_t* deleted_effective_ptr = nullptr;  ///< Del_eff on the device
  bool batch_current = false;                       ///< the arrays hold the batch of the next apply

  // ---- The append scratch (16 bytes per half-edge, as the original's) ----
  buffer<std::int32_t> keys[2];   ///< half-edge sources (double buffer)
  buffer<std::int32_t> items[2];  ///< half-edge ids (double buffer)
  buffer<unsigned char> temp;     ///< CUB temporary storage
  std::int64_t max_edges = 0;     ///< edges the scratch holds

  // ---- The report and the staging ----
  buffer<std::int64_t> report;       ///< report_entries values on the device
  buffer<std::int64_t> report_host;  ///< their pinned mirror
  buffer<unsigned char> staging;     ///< pinned staging of pageable batch arrays
  cuda_stream_fence staging_fence;   ///< the end of the last copy out of `staging`
  mutable bool pending = false;      ///< an apply's report is not settled yet
  /// The last apply's report once settled: every participant of one dyng::update() reads the
  /// same counts (the first one to synchronize settles it; M10a step 5, ADR 0076).
  mutable device_slotted_settled last_settled;

  /**
   * @brief The device memory held.
   * @return Bytes of the device arrays (not the pinned mirror and staging).
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return row_ptr.size() * sizeof(edge_t) + row_end.size() * sizeof(edge_t) +
           col_ind.size() * sizeof(vertex_t) + weights.size() * sizeof(weight_t) +
           status.size() * sizeof(std::int32_t) +
           (deleted.size() + deleted_effective.size() + inserted.size() + edge_src.size() +
            edge_dst.size()) *
               sizeof(vertex_t) +
           inserted_labels.size() + edge_weight.size() * sizeof(weight_t) +
           (keys[0].size() + keys[1].size() + items[0].size() + items[1].size()) *
               sizeof(std::int32_t) +
           temp.size() + report.size() * sizeof(std::int64_t);
  }
};

/**
 * @brief The empty device copy of with_capacity(): `n` vertices, all inactive, row v with the
 *        slots [row_ptr[v], row_ptr[v + 1]), every slot -1 / 0 (createDeviceGraph).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  res         Resources of the CUDA backend.
 * @param[in]  row_ptr     The n + 1 slot offsets (host memory).
 * @param[in]  num_weights 0 or 1.
 * @param[out] out         The device copy.
 * @throws out_of_memory_error if device memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void build_device_slotted(const resources& res, const std::vector<edge_t>& row_ptr, int num_weights,
                          device_slotted<vertex_t, edge_t, weight_t>& out);

/**
 * @brief Upload the host state of a slotted graph (a host graph cloned to CUDA): the slots, row
 *        ends and states.
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  res     Resources of the CUDA backend.
 * @param[in]  slots   The slot storage (host).
 * @param[in]  row_end The row ends (host).
 * @param[in]  status  The vertex states (host).
 * @param[out] out     The device copy.
 * @throws out_of_memory_error, cuda_error, not_supported_error.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void upload_device_slotted(const resources& res, const csr<vertex_t, edge_t, weight_t>& slots,
                           const std::vector<edge_t>& row_end,
                           const std::vector<vertex_status>& status,
                           device_slotted<vertex_t, edge_t, weight_t>& out);

/**
 * @brief Size the batch scratch for batches of up to `max_edges` edges (graph::reserve()).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res       Resources of the CUDA backend (the copy's device).
 * @param[in,out] g         The device copy.
 * @param[in]     max_edges The largest number of edges of a batch.
 * @throws capacity_error      if a batch of that size cannot be appended (2^30 edges or more).
 * @throws out_of_memory_error, cuda_error, not_supported_error.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void reserve_device_slotted(const resources& res, device_slotted<vertex_t, edge_t, weight_t>& g,
                            std::int64_t max_edges);

/**
 * @brief Copy a batch whose Step 0 is `normalized` to the device (the original's transfer of the
 *        ids, then of the edges): arrays in pinned host memory directly, pageable ones through the
 *        copy's pinned staging; Ins \ Del and Del_eff from `normalized` (the batch's own arrays when
 *        they are equal).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res        Resources of the CUDA backend (the copy's device).
 * @param[in,out] g          The device copy.
 * @param[in]     batch      The batch (host-accessible memory).
 * @param[in]     normalized Its Step 0.
 * @throws capacity_error, out_of_memory_error, cuda_error, not_supported_error.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void upload_vertex_batch(const resources& res, device_slotted<vertex_t, edge_t, weight_t>& g,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         const normalized_batch<vertex_t>& normalized);

/**
 * @brief The device apply of the uploaded batch (see the file comment, item 3), ending with the
 *        copy of its report to the pinned mirror; no synchronization (settle() reads it).
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources of the CUDA backend (the copy's device).
 * @param[in,out] g   The device copy (its batch uploaded).
 * @throws cuda_error, not_supported_error.
 * @async
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void apply_vertex_batch_device(const resources& res, device_slotted<vertex_t, edge_t, weight_t>& g);

/**
 * @brief Read the report of the last device apply (precondition: the stream of the apply was
 *        synchronized after it). Clears `pending`.
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] g The device copy.
 * @return The report.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
device_slotted_settled settle_device_slotted(const device_slotted<vertex_t, edge_t, weight_t>& g);

/**
 * @brief Copy the rows of the device copy into the host storage (the download of a stale host
 *        copy): the slot columns and weights and the row ends; ordered on the copy's stream, then
 *        a synchronization of it.
 * @tparam vertex_t Vertex id type (32-bit).
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     g       The device copy.
 * @param[in,out] slots   The slot storage (its row_ptr is kept; col_ind and weights resized).
 * @param[out]    row_end The row ends (resized).
 * @throws cuda_error, not_supported_error.
 * @sync
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void download_device_slotted(const device_slotted<vertex_t, edge_t, weight_t>& g,
                             csr<vertex_t, edge_t, weight_t>& slots, std::vector<edge_t>& row_end);

}  // namespace dyng::detail
