// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file transaction.hpp
 * @brief The two-phase update of a hypergraph (PLAN Section 4.5.1, "the container's internal
 *        transaction"; ADR 0043): begin -> before() -> commit() -> after() + applied().
 *
 * hypergraph::apply() is a transaction without participants. dyng::update() over a hypergraph
 * (run_update(), hyper_update.cpp) opens one, lets every participant read G_t through before()
 * (the old views stay valid until the commit), commits once, and hands every participant G_{t+1}
 * through after() with the applied record:
 *
 *   - **begin** (the constructor): the placement and poisoning checks, the batch staged to the
 *     host under the copy policy of the resources, then Step 0 once: the inserted hyperedges
 *     validated, sorted and deduplicated, the deletions checked against G_t (dead, out-of-range
 *     and repeated ids skipped and counted). Everything that can be rejected is rejected here,
 *     before anything changes (strong guarantee).
 *   - **before()**: an old_view of G_t, valid until commit(). In Debug builds (NDEBUG undefined)
 *     its use after the commit throws internal_error (invariant I1).
 *   - **commit()**: the store phase in the order of id_reuse, the line graph, one read of the
 *     device error words, the new row views, then the no-throw commit (version, state identity,
 *     weights, the applied record). An error inside the store phase poisons the hypergraph (basic
 *     guarantee). Exactly once per transaction.
 *   - **after()** and **applied()**: a new_view of G_{t+1} and the record of what the commit did
 *     (inserted ids on the host and, on CUDA, the device; deleted ids with their members; touched
 *     hyperedges and vertices; skipped counts; the incidence-delta hook of M11).
 */
#pragma once

#include "core/staging.hpp"
#include "framework/views.hpp"
#include "hypergraph/hypergraph_impl.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>

namespace dyng::detail {

/**
 * @brief One hypergraph transaction (see the file comment). Not copyable or movable: the views it
 *        hands out point into it.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
class hyper_transaction {
 public:
  using container_type = hypergraph<vertex_t, weight_t>;             ///< the container
  using batch_view_type = hyperedge_batch_view<vertex_t, weight_t>;  ///< the batch view

  /**
   * @brief begin: check the call and run Step 0 (see the file comment).
   * @param[in]     res      Execution resources of the hypergraph's backend (copied: the handle
   *                          is shared).
   * @param[in,out] h        The hypergraph (must outlive the transaction).
   * @param[in]     batch    The batch (any memory space; staged to the host once).
   * @param[in]     function The calling function, for messages (e.g. "hypergraph::apply").
   * @throws invalid_argument_error for a placement mismatch, a moved-from or poisoned hypergraph,
   *         an invalid batch, or a copy the copy policy refuses.
   * @throws not_supported_error    for incidence edits (0.4), or a device batch without CUDA.
   * @throws capacity_error         if the batch exceeds the store's int32 payload.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @guarantee Strong: nothing is changed.
   */
  hyper_transaction(const resources& res, container_type& h, const batch_view_type& batch,
                    const char* function);

  hyper_transaction(const hyper_transaction&) = delete;             ///< not copyable
  hyper_transaction& operator=(const hyper_transaction&) = delete;  ///< not copyable
  hyper_transaction(hyper_transaction&&) = delete;                  ///< not movable
  hyper_transaction& operator=(hyper_transaction&&) = delete;       ///< not movable
  ~hyper_transaction();                                             ///< frees the staged batch

  /**
   * @brief The batch as requested, readable on the host (the staged copy of device arrays).
   * @return The host view, valid for the transaction's life.
   */
  [[nodiscard]] const batch_view_type& requested() const noexcept {
    return host_;
  }

  /**
   * @brief Step 0: the batch validated and normalized once against G_t.
   * @return The staged batch, valid for the transaction's life.
   */
  [[nodiscard]] const hyper_staged_batch<weight_t>& staged() const noexcept {
    return staged_;
  }

  /**
   * @brief G_t, before the commit.
   * @return An old_view of the hypergraph; using it after commit() throws internal_error in
   *         Debug builds.
   * @throws internal_error after commit().
   */
  [[nodiscard]] framework::old_view<container_type> before() const;

  /**
   * @brief Apply the staged batch (the store phase and the commit), once.
   * @param[in] record_incidence_delta Fill the applied record's incidence delta (M11's hook).
   * @return What the batch did.
   * @throws internal_error if the transaction was committed already, or the device error word
   *         reports a broken invariant (poisoned).
   * @throws invalid_argument_error if the hypergraph changed since begin (another apply).
   * @throws capacity_error, out_of_memory_error, cuda_error from the store (poisoned).
   * @guarantee Basic: an error inside the store phase poisons the hypergraph; nothing throws
   *            after the store phase.
   */
  hyper_apply_summary commit(bool record_incidence_delta = false);

  /**
   * @brief Whether commit() has run.
   * @return true after a successful commit().
   */
  [[nodiscard]] bool committed() const noexcept {
    return committed_;
  }

  /**
   * @brief G_{t+1}, after the commit.
   * @return A new_view of the hypergraph.
   * @throws internal_error before commit().
   */
  [[nodiscard]] framework::new_view<container_type> after() const;

  /**
   * @brief The applied record of the commit.
   * @return The record (the hypergraph's, valid until its next apply()).
   * @throws internal_error before commit().
   */
  [[nodiscard]] const hyper_applied& applied() const;

 private:
  resources res_;  ///< a copy: the handle is shared, and the caller's may be a temporary
  container_type& h_;
  const char* function_;
  host_input<std::int64_t> deletions_;
  host_input<std::int64_t> offsets_;
  host_input<vertex_t> members_;
  host_input<weight_t> weights_;
  batch_view_type host_;
  hyper_staged_batch<weight_t> staged_;
  bool committed_ = false;
};

}  // namespace dyng::detail
