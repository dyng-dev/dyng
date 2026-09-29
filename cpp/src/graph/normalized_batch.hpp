// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/directed_graph.cpp (prepare_batch) and
// src/dynamic/edge_change.cpp (sort_and_dedup, normalize)
/**
 * @file normalized_batch.hpp
 * @brief Step 0 of batch_semantics::as_sets, computed once per update: the normalized batch (two
 *        sorted, duplicate-free lists of structural changes) and the counters of the batch.
 *
 * The set apply (apply_set_host.cpp on the host, apply_set_device.cu on the device) merges these
 * lists into the sorted rows; the algorithms that read the net structural change (cycle_count)
 * take the same lists, so an update under set semantics normalizes its batch exactly once
 * (framework::run_update computes it before the participants' before_apply; ADR 0020).
 */
#pragma once

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

/**
 * @brief One requested change: the edge and its position in the batch (for the weights of an
 *        insertion and for error messages).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct set_change {
  vertex_t source;    ///< tail of the edge
  vertex_t target;    ///< head of the edge
  std::size_t index;  ///< position of the operation in the batch
};

/**
 * @brief The normalized batch of batch_semantics::as_sets (CycleEnumeration-GPU's prepare_batch()
 *        result) and what applying it does.
 *
 * `deletions` are the edges of G_t the batch removes and `insertions` the edges it adds (an edge
 * that the batch deletes and inserts is in both lists), each sorted by (source, target) without
 * repeats; the position of an edge in its list is its id (the ownership id of cycle_count). The
 * vectors keep their capacity when the object is reused.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct normalized_batch {
  std::vector<set_change<vertex_t>> deletions;   ///< the deletions of existing edges, sorted
  std::vector<set_change<vertex_t>> insertions;  ///< the insertions of new edges, sorted
  apply_summary summary;                         ///< the counters of the batch
  std::int64_t vertices_before = 0;              ///< n of G_t
  std::int64_t vertices_after = 0;               ///< n of G_{t+1}
  std::int64_t edges_before = 0;                 ///< m of G_t
  std::int64_t edges_after = 0;                  ///< m of G_{t+1}
  std::uint64_t state_id = 0;  ///< the graph state (graph_impl::state_id) it was computed for
  std::vector<set_change<vertex_t>> requested_deletions;   ///< scratch: requested deletions
  std::vector<set_change<vertex_t>> requested_insertions;  ///< scratch: requested insertions
  std::vector<set_change<vertex_t>> sort_scratch;          ///< scratch: the radix sort's copy
  std::vector<std::uint32_t> sort_counters;                ///< scratch: the radix sort's counters
  /// Scratch: per requested change (deletions, then insertions) 1 if it names an edge of G_t.
  std::vector<std::uint8_t> present;

  /// Device copy of the lists (CUDA backend, 32-bit ids): the deletions, then the insertions, as
  /// (source, target) pairs, uploaded once per update by upload_normalized_batch() and read by
  /// the device apply and the cycle_count phases. Valid while `device_current`.
  mutable buffer<std::uint32_t> device_lists;
  mutable buffer<std::uint32_t> staging;  ///< pinned host staging of device_lists
  mutable bool device_current = false;    ///< device_lists holds these lists

  /// Device deletion marks of G_t (CUDA backend, 32-bit ids): for every position of G_t's CSR the
  /// id of the deletion that removes it, else no_change_id (CycleEnumeration-GPU's owner array of
  /// the delete phase). Computed once per update by mark_normalized_deletions() and read by the
  /// device apply and the cycle_count delete phase. Valid while `deletions_marked`.
  mutable buffer<int> deletion_owner;
  mutable bool deletions_marked = false;  ///< deletion_owner holds the marks of these lists

  /// Step 0 against a resident graph whose host copy is stale (normalize_set_batch_device): the
  /// membership flags of the requested changes on the device and their pinned read-back.
  mutable buffer<std::uint8_t> device_present;
  mutable buffer<std::uint8_t> present_staging;  ///< pinned host staging of device_present

  /**
   * @brief The memory held (for workspace reports).
   * @return Bytes of the vectors' capacity and of the device and staging buffers.
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return (deletions.capacity() + insertions.capacity() + requested_deletions.capacity() +
            requested_insertions.capacity() + sort_scratch.capacity()) *
               sizeof(set_change<vertex_t>) +
           sort_counters.capacity() * sizeof(std::uint32_t) + present.capacity() +
           (device_lists.size() + staging.size()) * sizeof(std::uint32_t) +
           deletion_owner.size() * sizeof(int) + device_present.size() + present_staging.size();
  }
};

/**
 * @brief Upload the lists of a normalized batch to the device (once per update; no-op while
 *        `device_current`): one copy from pinned staging on the stream of `res`.
 *
 * Precondition: 32-bit vertex ids. The buffers keep their capacity across updates.
 * @tparam vertex_t Vertex id type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in,out] nb  The normalized batch (its device fields).
 * @return Device pointer to the deletions as (source, target) pairs; the insertions follow them.
 * @throws out_of_memory_error if memory runs out.
 * @throws cuda_error          if the runtime reports an error.
 * @throws not_supported_error if CUDA is not built.
 * @async
 */
template <typename vertex_t>
const std::uint32_t* upload_normalized_batch(const resources& res,
                                             const normalized_batch<vertex_t>& nb);

/**
 * @brief Step 0 of batch_semantics::as_sets on G_t (prepare_batch()): validate the batch, drop
 *        self-loops as the semantics say, sort and deduplicate both lists (the first insertion of
 *        a pair in batch order supplies its weights), drop the deletions of missing edges and the
 *        insertions of existing edges (unless the same edge is deleted too), and count.
 *
 * Nothing is changed; an error leaves `out` unspecified. `out.state_id` is not set (the caller
 * records the state it normalized against).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]  original The out-edge CSR of G_t (host memory; sorted rows without parallel edges).
 * @param[in]  batch    The batch (host memory).
 * @param[in]  props    Properties of the graph (semantics.as_sets set).
 * @param[out] out      The normalized batch.
 * @throws invalid_argument_error on malformed batches or a semantics rule that says error.
 * @throws not_supported_error    if the properties do not allow set semantics, or for vertex
 *         insertions and deletions.
 * @throws capacity_error         if the edge count after the batch does not fit edge_t.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_set_batch(const csr<vertex_t, edge_t, weight_t>& original,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         const graph_properties& props, normalized_batch<vertex_t>& out);

}  // namespace dyng::detail
