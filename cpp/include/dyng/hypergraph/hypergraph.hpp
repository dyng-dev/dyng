// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph.hpp
 * @brief hypergraph<V,W>: the dynamic hypergraph container (pimpl, move-only).
 * @ingroup hypergraph
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/hypergraph/hypergraph_view.hpp>

#include <cstdint>
#include <memory>

namespace dyng {

namespace detail {
template <typename vertex_t, typename weight_t>
class hypergraph_impl;
struct hypergraph_access;
}  // namespace detail

/**
 * @brief A dynamic hypergraph: hyperedges (sets of vertices) identified by 0-based int64 ids,
 *        changed by batches of hyperedge deletions and insertions.
 *
 * The hypergraph owns its storage (hypergraph_storage::escher: the ESCHER store, ESCHER
 * IPDPS'26 with the best-fit reuse of ESCHER+, TKDE'26) and a version counter that increases by
 * one with every applied batch, plus a process-wide unique state identity (renewed by every
 * construction and every batch, copied by clone()), so a result computed on it is detected as
 * stale when it is used with another hypergraph or state.
 *
 * **Ids.** Hyperedge i of from_hyperedges() gets id i. A batch's inserted hyperedges get the ids
 * that the store's best-fit reuse chooses (hyper_apply_summary::inserted_ids): deleted slots
 * whose capacity fits are reused, the others are dropped and new hyperedges are numbered after
 * the last live id. Ids must be read from inserted_ids, never predicted: the order of a batch
 * (hypergraph_properties::reuse) changes them, and a dead id can be given out again. Live ids lie
 * in [0, hyperedge_id_bound()); a dead id below the bound has an empty row.
 *
 * **Placement** (PLAN Section 4.6 rule 5). The storage belongs to the backend of the resources
 * that built it: the host backends (sequential, openmp) keep a host store, the cuda backend the
 * device store (authoritative; host copies are made only when something reads them on the host:
 * to_hyperedge_list(), check_integrity(), a clone for a host backend). Calls with resources of
 * the other kind throw instead of copying the hypergraph; clone(res) and to_backend(res) make
 * the copy.
 *
 * **Thread safety.** As graph: read-only calls (the views, the accessors, to_hyperedge_list())
 * may run concurrently with each other; apply() must not overlap any other call on the same
 * hypergraph.
 *
 * Instantiated for (vertex_t, weight_t) = (int32, unweighted) and (int32, int32) on every
 * backend.
 * @tparam vertex_t Vertex id type (signed).
 * @tparam weight_t Hyperedge weight type (unweighted, or std::int32_t in [1, 2^31 - 1]).
 * @ingroup hypergraph
 */
template <typename vertex_t = std::int32_t, typename weight_t = unweighted>
class hypergraph {
 public:
  using vertex_type = vertex_t;                                      ///< vertex id type
  using weight_type = weight_t;                                      ///< hyperedge weight type
  using h2v_view_type = incidence_view<vertex_t>;                    ///< the h2v view type
  using v2h_view_type = incidence_view<std::int64_t>;                ///< the v2h view type
  using line_graph_type = line_graph_view<weight_t>;                 ///< the h2h view type
  using batch_view_type = hyperedge_batch_view<vertex_t, weight_t>;  ///< the batch view type
  using list_type = hyperedge_list<vertex_t, weight_t>;              ///< the owning list type

  /**
   * @brief Build a hypergraph from a list of hyperedges (hyperedge i gets id i).
   *
   * Each hyperedge's members are sorted and repeated members dropped. The list may be empty (no
   * hyperedge; vertices may still exist).
   * @param[in] res        Execution resources (the hypergraph belongs to their backend).
   * @param[in] hyperedges The hyperedges (any memory space: the list is read on the host, so
   *                       arrays in device memory are copied once under res.get_copy_policy()).
   * @param[in] props      The properties of the new hypergraph.
   * @return The hypergraph at version 0.
   * @throws invalid_argument_error if a member is out of [0, num_vertices), a hyperedge is empty,
   *         the offsets are malformed, a weight is outside [1, 2^31 - 1] (or weights are given
   *         for an unweighted list, or missing for a weighted one), the line-graph policy needs
   *         v2h and store_v2h is false, or an array must be copied and the copy policy is
   *         copy_policy::error.
   * @throws not_supported_error    for hypergraph_storage::slack_csr or line_graph_policy::
   *         slack_csr (planned for 0.4), line_graph_policy::sorted_merge (not yet available), a
   *         grow_factor other than 1.5 or 0, or the cuda backend in a build without CUDA.
   * @throws capacity_error         if the storage exceeds the policy's max_payload (by default
   *         the int32 offsets of the store).
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] static hypergraph from_hyperedges(
      const resources& res, hyperedge_list_view<vertex_t, weight_t> hyperedges,
      const hypergraph_properties& props = {});

  /**
   * @brief Move constructor.
   * @param[in,out] other The hypergraph to move from; it may only be assigned to or destroyed.
   */
  hypergraph(hypergraph&& other) noexcept;

  /**
   * @brief Move assignment.
   * @param[in,out] other The hypergraph to move from; it may only be assigned to or destroyed.
   * @return *this.
   */
  hypergraph& operator=(hypergraph&& other) noexcept;

  hypergraph(const hypergraph&) = delete;             ///< not copyable: use clone()
  hypergraph& operator=(const hypergraph&) = delete;  ///< not copyable: use clone()
  ~hypergraph();                                      ///< releases the storage

  /**
   * @brief A deep copy for the resources `res` (same properties, ids, rows, version and state
   *        identity; the store's deleted slots and payload accounting too, so later batches give
   *        the copy the same ids).
   *
   * The copy belongs to the backend of `res`: this is how a hypergraph moves between the host
   * backends and cuda.
   * @param[in] res Execution resources of the copy.
   * @return The copy.
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph.
   * @throws not_supported_error    for the cuda backend in a build without CUDA.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] hypergraph clone(const resources& res) const;

  /**
   * @brief The hypergraph for the backend of `res`: the explicit move between the host backends
   *        and cuda that the placement errors name. The same as clone(res).
   * @param[in] res Execution resources of the new hypergraph.
   * @return The hypergraph for `res`.
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph.
   * @throws not_supported_error    for the cuda backend in a build without CUDA.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] hypergraph to_backend(const resources& res) const;

  /**
   * @brief Number of vertices.
   * @return n; vertex ids lie in [0, n).
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept;

  /**
   * @brief Number of live hyperedges.
   * @return The live count.
   */
  [[nodiscard]] std::int64_t num_hyperedges() const noexcept;

  /**
   * @brief The bound of the hyperedge ids: live ids lie in [0, bound); dead ids may exist below
   *        it.
   *
   * The bound can go down: an insert that appends new rows drops the deleted slots it did not
   * reuse and numbers the new rows after the last live id.
   * @return The bound.
   */
  [[nodiscard]] std::int64_t hyperedge_id_bound() const noexcept;

  /**
   * @brief The properties of the hypergraph.
   * @return The properties given at construction.
   */
  [[nodiscard]] const hypergraph_properties& properties() const noexcept;

  /**
   * @brief The memory space of the storage and of the views.
   * @return memory_space::device for a hypergraph built with CUDA resources, memory_space::host
   *         otherwise.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief The version counter.
   * @return 0 after construction; +1 per applied batch.
   */
  [[nodiscard]] std::uint64_t version() const noexcept;

  /**
   * @brief Hyperedge to members: one sorted row per id in [0, hyperedge_id_bound()), empty for
   *        a dead id.
   *
   * Built from the store on first use per version (in the hypergraph's memory space) and kept
   * until the next apply().
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph.
   * @throws out_of_memory_error    if the view cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] h2v_view_type h2v() const;

  /**
   * @brief Vertex to hyperedges: one row per vertex, the ids of its live hyperedges in
   *        increasing order.
   *
   * Built from the store on first use per version (at most once per version, never by apply())
   * and kept until the next apply().
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph, or one built with
   *         store_v2h = false.
   * @throws out_of_memory_error    if the view cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] v2h_view_type v2h() const;

  /**
   * @brief The line graph: every live hyperedge to the hyperedges it overlaps.
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph, or one built with
   *         line_graph_policy::none.
   * @throws not_supported_error    while the line graph is not yet available (line_graph_policy::
   *         automatic in this build).
   * @sync
   */
  [[nodiscard]] line_graph_type h2h() const;

  /**
   * @brief The hyperedge weights by id (host memory on every backend): one per id in
   *        [0, hyperedge_id_bound()), 0 for a dead id; empty for an unweighted hypergraph.
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from hypergraph.
   */
  [[nodiscard]] array_view<const weight_t> weights() const;

  /**
   * @brief Apply a batch: delete hyperedges by id and insert new ones, in the order of
   *        properties().reuse.
   *
   * Deletions of dead, out-of-range or repeated ids are skipped and counted
   * (hyper_apply_summary::ignored_deletions), as are repeated members of an inserted hyperedge
   * (dropped_members). The inserted hyperedges get the ids the store's best-fit reuse chooses,
   * reported in hyper_apply_summary::inserted_ids.
   * @param[in] res   Execution resources of the hypergraph's backend (and device).
   * @param[in] batch The batch (any memory space: it is validated and grouped on the host, so
   *                  arrays in device memory are copied once under res.get_copy_policy()).
   * @return What the batch did.
   * @throws invalid_argument_error if a member is out of range, an inserted hyperedge is empty,
   *         the offsets are malformed, a weight is outside [1, 2^31 - 1] (or the weights do not
   *         match the insertions), the resources belong to another backend or device, the
   *         hypergraph was moved from or is poisoned, or an array must be copied and the copy
   *         policy is copy_policy::error.
   * @throws not_supported_error    if the batch has incidence edits (planned for 0.4).
   * @throws capacity_error         if the storage exceeds its policy (the hypergraph is then
   *         poisoned).
   * @throws out_of_memory_error    if host or device memory cannot be allocated (poisoned).
   * @throws cuda_error             if the CUDA runtime reports an error (poisoned).
   * @throws internal_error         if the store's device error word reports a broken invariant
   *         (poisoned).
   * @sync
   * @guarantee Strong for every error found before the store changes (the invalid_argument_error
   *            and not_supported_error cases above): version, state identity and contents are
   *            unchanged. Basic otherwise: an error inside a store operation poisons the
   *            hypergraph, and every later call but destruction and assignment throws.
   */
  hyper_apply_summary apply(const resources& res, const batch_view_type& batch);

  /**
   * @brief A host copy of the live hyperedges in increasing id order (ids are not kept: the
   *        list's hyperedge i is the i-th live id).
   * @param[in] res Execution resources (the hypergraph's backend).
   * @return The list, with the weights of a weighted hypergraph.
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph, or resources of
   *         another backend.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] list_type to_hyperedge_list(const resources& res) const;

  /**
   * @brief The storage counters (device bytes in ESCHER-GPU's accounting scope, payload,
   *        growths, compactions, kernel paths).
   * @return The counters of the last completed operation.
   */
  [[nodiscard]] storage_stats store_stats() const noexcept;

  /**
   * @brief Check the storage invariants: the rows of every tree by search from the root, their
   *        reachability, segments and tail metadata (ESCHER's row check), the deleted-slot
   *        counters (subtree_avail, MOSP_ESCHER's store check) against the live ids, and that
   *        v2h is the transpose of h2v.
   * @param[in] res Execution resources (the hypergraph's backend).
   * @throws internal_error         if an invariant is broken.
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph, or resources of
   *         another backend.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  void check_integrity(const resources& res) const;

 private:
  friend struct detail::hypergraph_access;
  using impl_type = detail::hypergraph_impl<vertex_t, weight_t>;
  explicit hypergraph(std::unique_ptr<impl_type> impl) noexcept;
  [[nodiscard]] impl_type& impl();
  [[nodiscard]] const impl_type& impl() const;

  std::unique_ptr<impl_type> impl_;
};

}  // namespace dyng
