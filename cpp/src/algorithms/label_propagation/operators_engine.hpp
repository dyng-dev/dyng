// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file operators_engine.hpp
 * @brief The operators engine of label_propagation on CUDA (Tier A; ADR 0101): the framework's
 *        hooks written with this folder's operators (operators/: binned_frontier,
 *        connected_components, neighbor_reduce). Host-only declarations; operators_engine.cu
 *        holds the work.
 *
 * One update (options::cuda_engine = engine::operators, schedule::damped), on the one stream of
 * the resources, after the fused engine's before_apply on G_t (the marks of the deleted vertices'
 * neighbours: the same pass):
 *   - identify_affected: the seeds' classes, the inserted vertices marked and stamped, G' (the
 *     fused engine's sparsify pass, or the endpoints of a batch without insertions) and its
 *     components with connected_components (Shiloach-Vishkin, one synchronization per round and
 *     one for the count); the labels before the update kept;
 *   - seed: the supernode initialization (the fused engine's label_init pass: 64-bit fixed point,
 *     exact);
 *   - loop: the damped propagation with binned_frontier and neighbor_reduce: the first set from
 *     the affected flags, then per iteration and half-sweep and bin one neighbor_reduce launch
 *     (the bin's mapping: 8 lanes, a warp, a block; the block of 1024 threads while the frontier
 *     holds at most 1024 vertices, the fused engine's persistent block, else 256), one expand
 *     launch, the commit of the new labels, and one synchronization (the next set's size); the
 *     seeds' labels restored; on_limit::fallback_recompute runs IrLP (the fused engine's, the
 *     published baseline);
 *   - finalize: the changed labels counted, G''s size read (one synchronization), the graph's
 *     report settled.
 * The lane mapping, the fused multiply-adds and the division are the fused engine's, in this
 * target's fast-math translation unit, so the labels can be its bits (AD6 asks c x eps with c
 * calibrated in parity/goldens.toml, and identical iterations, components and frontier_visits).
 * schedule::in_place has no operators twin (PLAN 4.5.4): not_supported_error.
 */
#pragma once

#include "algorithms/label_propagation/fused.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace dyng::detail::lp_cuda {

/// The operators engine's own scratch (leased from the pool beside fused.hpp's workspace): the
/// per-position changed flags of the propagation's queues.
struct operators_workspace final : pooled_workspace {
  scratch_buffer<std::uint8_t> changed[num_degree_bins];  ///< per bin, parallel to its queue
  int sized[num_degree_bins] = {0, 0, 0};                 ///< the capacities reserved

  /**
   * @brief Size the flags for the queue capacities (a reservation when they grow).
   * @param[in] res      Resources of the CUDA backend.
   * @param[in] capacity The queue capacities.
   * @throws out_of_memory_error, cuda_error.
   */
  void reserve(const resources& res, const int (&capacity)[num_degree_bins]);

  /// Bytes held.
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/// What the operators engine carries between the hooks of one update, and what it did.
struct operators_batch {
  int components = 0;              ///< components of G'
  int hook_rounds = 0;             ///< Shiloach-Vishkin rounds
  int sparsified = 0;              ///< edges of G' (read in finalize)
  propagation_report propagation;  ///< the propagation (persistent_iterations 0)
  bool capped = false;             ///< the cap was reached with a non-empty frontier
  bool recomputed = false;         ///< on_limit::fallback_recompute ran IrLP
  bool irlp_capped = false;        ///< and IrLP reached the cap too
  int irlp_iterations = 0;         ///< its iterations
  std::int64_t affected = 0;       ///< vertices whose label bits changed
  std::int64_t host_syncs = 0;     ///< the synchronizations of the algorithm phase
  bool settled = false;            ///< the graph's report was settled
  std::uint32_t graph_errors = 0;  ///< the device apply's error bits (settled)
  std::int64_t appended = 0;       ///< half-edges the device apply appended
  std::int64_t dropped = 0;        ///< half-edges it dropped
  std::string error_detail;        ///< the description of graph_errors
};

/**
 * @brief identify_affected on G_{t+1} (see the file comment).
 * @tparam edge_t Edge offset type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in]     g   G_{t+1} (its batch on the device).
 * @param[in,out] st  The result's device state (the seeds' classes, the batch stamps).
 * @param[in,out] ws  The workspace (fused.hpp's, sized by before_apply).
 * @param[in]     a   The update's inputs (tau, the batch stamp).
 * @param[out]    out The components, the rounds.
 * @throws internal_error (the rounds' guard), cuda_error.
 */
template <typename edge_t>
void ops_identify_affected(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                           state& st, workspace& ws, const batch_args& a, operators_batch& out);

/**
 * @brief seed: the supernode initialization of the inserted vertices.
 * @tparam edge_t Edge offset type.
 * @throws internal_error if there are more components than accumulators; cuda_error.
 */
template <typename edge_t>
void ops_seed(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
              workspace& ws, const operators_batch& b);

/**
 * @brief loop: the damped propagation (and IrLP under on_limit::fallback_recompute at the cap).
 * @tparam edge_t Edge offset type.
 * @throws out_of_memory_error, cuda_error.
 */
template <typename edge_t>
void ops_propagate(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
                   workspace& ws, operators_workspace& ows, const batch_args& a,
                   operators_batch& out);

/**
 * @brief The damped propagation of the operators engine on any rows (the loop's core; the
 *        fused engine's launch_iterative_propagation for the operators): the first set from
 *        `affected` (active vertices without a seed label), then the iterations until the set is
 *        empty or `max_iterations`; the seeds' labels restored. One synchronization for the first
 *        set and one per iteration; `after_first_sync` runs after the first.
 * @tparam edge_t Edge offset type.
 * @return What it did (persistent_iterations 0).
 * @throws out_of_memory_error, cuda_error.
 */
template <typename edge_t>
propagation_report ops_iterate(const resources& res, const rows<edge_t>& g, float* labels,
                               const std::int32_t* gt_class, const std::int32_t* affected,
                               float delta, int max_iterations, state& st, workspace& ws,
                               operators_workspace& ows, void (*after_first_sync)(void*),
                               void* arg);

/**
 * @brief finalize: the changed labels counted, G''s size read, the graph's report settled.
 * @tparam edge_t Edge offset type.
 * @throws cuda_error.
 */
template <typename edge_t>
void ops_finalize(const resources& res, const graph<std::int32_t, edge_t, float>& g, state& st,
                  workspace& ws, operators_batch& out);

}  // namespace dyng::detail::lp_cuda
