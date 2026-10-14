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
#include <dyng/update.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

namespace dyng {

namespace detail {
template <typename vertex_t, typename weight_t>
class hypergraph_impl;
struct hypergraph_access;
template <typename weight_t>
struct hyper_staged_batch;
struct hyper_applied;
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
 * **Results.** dyng::update(res, h, batch, results...) applies a batch once and brings every
 * result of a hypergraph algorithm up to date (detail::participant_of<hypergraph>, below): one
 * validation and normalization of the batch, every result's work on the hypergraph before the
 * batch, one commit, every result's work after it.
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
   *         for an unweighted list, or missing for a weighted one), line_graph_policy::
   *         sorted_merge is asked for without v2h (store_v2h is false), or an array must be copied
   *         and the copy policy is copy_policy::error.
   * @throws not_supported_error    for hypergraph_storage::slack_csr or line_graph_policy::
   *         slack_csr (planned for 0.4), a grow_factor other than 1.5 or 0, or the cuda backend in
   *         a build without CUDA.
   * @throws capacity_error         if the storage (the line graph of sorted_merge included)
   *         exceeds the policy's max_payload (by default the int32 offsets of the store).
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
   * @brief The line graph: every live hyperedge to the hyperedges it overlaps (one sorted row per
   *        id in [0, hyperedge_id_bound()), empty for a dead id; `out` the first neighbour above
   *        the row's id).
   *
   * Under line_graph_policy::sorted_merge the line graph is built with the hypergraph and kept
   * by every apply() (ESCHER-GPU's sorted delta merge); under automatic the first call builds it,
   * and apply() keeps it from then on. The view itself (decoded: 0-based ids, int64 offsets, in
   * the hypergraph's memory space; the weights by id in host memory) is built from the line graph
   * on first use per version and kept until the next apply().
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph, one built with
   *         line_graph_policy::none, or one built with store_v2h = false.
   * @throws capacity_error         if the line graph exceeds the capacity policy's max_payload
   *         (by default the int32 offsets of the store).
   * @throws out_of_memory_error    if the line graph or the view cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] line_graph_type h2h() const;

  /**
   * @brief The hyperedge weights by id (host memory on every backend): one per id in
   *        [0, hyperedge_id_bound()), 0 for a dead id; empty for an unweighted hypergraph.
   * @return The view, invalidated by the next apply().
   * @throws invalid_argument_error for a moved-from or poisoned hypergraph.
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
   *
   * On CUDA the storage works on the stream of the resources the hypergraph was built with;
   * apply() orders that work after the work queued so far on the stream of `res`, and the work
   * later queued on the stream of `res` after it, so either stream may hold readers of the views.
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
   * @throws capacity_error         if the batch inserts more than INT_MAX / 3 (715,827,882)
   *         members in all (found before the store changes: nothing changed), or the storage
   *         exceeds its policy (the hypergraph is then poisoned).
   * @throws out_of_memory_error    if host or device memory cannot be allocated: while the batch
   *         is staged and checked nothing changed, in a store operation the hypergraph is
   *         poisoned.
   * @throws cuda_error             if the CUDA runtime reports an error (poisoned).
   * @throws internal_error         if the store's device error word reports a broken invariant
   *         (poisoned).
   * @sync
   * @guarantee Strong for every error found before the store changes (the invalid_argument_error
   *            and not_supported_error cases above, the batch-size capacity_error, an allocation
   *            failure while the batch is staged and checked): version, state identity and
   *            contents are unchanged. Basic otherwise: an error inside a store operation poisons the
   *            hypergraph; every later call that reads or changes its storage (the views,
   *            weights(), apply(), to_hyperedge_list(), clone(), to_backend(), check_integrity())
   *            throws, and the noexcept accessors return unspecified values, until it is
   *            destroyed or assigned.
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

namespace detail {

/**
 * @brief Whether a type is a hypergraph<V,W> (for the plain-English static_asserts of a
 *        hypergraph algorithm's update_traits: its result takes part in dyng::update() over a
 *        hypergraph only).
 * @tparam type_t The type.
 * @ingroup hypergraph
 */
template <typename type_t>
struct is_hypergraph : std::false_type {};

/**
 * @brief hypergraph<V,W> is a hypergraph.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
struct is_hypergraph<hypergraph<vertex_t, weight_t>> : std::true_type {};

/**
 * @brief is_hypergraph<type_t>::value.
 * @tparam type_t The type.
 * @ingroup hypergraph
 */
template <typename type_t>
inline constexpr bool is_hypergraph_v = is_hypergraph<type_t>::value;

/**
 * @brief One result taking part in an update of a hypergraph (type-erased; each hypergraph
 *        algorithm implements it): the hypergraph's participant interface of dyng::update().
 *
 * run_update() opens the hypergraph's transaction (Step 0 once: the batch staged to the host,
 * validated and normalized), hands the staged batch to every participant (use_normalized()),
 * calls before_apply() of every participant on G_t, commits the batch once, then calls
 * after_apply() of every participant on G_{t+1} with the applied record (invariant I1: subtract
 * on G_t, add on G_{t+1}). The participant of an algorithm is the framework's adapter of its
 * problem (problem_participant in the library's cpp/src/framework/composition.hpp).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
class hyper_update_participant {
 public:
  using graph_type = hypergraph<vertex_t, weight_t>;                   ///< the container
  using batch_type = hyperedge_batch_view<vertex_t, weight_t>;         ///< the batch view
  hyper_update_participant() = default;                                ///< default
  hyper_update_participant(const hyper_update_participant&) = delete;  ///< not copyable
  hyper_update_participant& operator=(const hyper_update_participant&) = delete;  ///< not copyable
  hyper_update_participant(hyper_update_participant&&) = delete;                  ///< not movable
  hyper_update_participant& operator=(hyper_update_participant&&) = delete;       ///< not movable
  virtual ~hyper_update_participant() = default;                                  ///< destructor

  /**
   * @brief The result object this participant updates (to reject duplicates).
   * @return Its address.
   */
  [[nodiscard]] virtual const void* target() const noexcept = 0;

  /**
   * @brief Receive Step 0 (the staged batch: inserted hyperedges sorted and deduplicated, the
   *        live deletions of G_t, the skipped counts), computed once by run_update() for every
   *        participant and the commit; called before before_apply(). The object lives until
   *        run_update() returns.
   * @param[in] staged The staged batch.
   */
  virtual void use_normalized(const hyper_staged_batch<weight_t>* staged) noexcept {
    (void)staged;
  }

  /**
   * @brief Steps 0 and 1a on G_t: validate (version, batch) and read G_t; mutates nothing
   *        visible. The views of G_t stay valid until the commit.
   * @param[in] res   Execution resources.
   * @param[in] h     The hypergraph before the batch.
   * @param[in] batch The batch as requested (host memory).
   */
  virtual void before_apply(const resources& res, const graph_type& h, const batch_type& batch) = 0;

  /**
   * @brief Steps 1b and 2 on G_{t+1}: bring the result up to date.
   * @param[in] res     Execution resources.
   * @param[in] h       The hypergraph after the batch.
   * @param[in] summary What the commit did (the inserted ids among it).
   * @param[in] applied The applied record (inserted ids on the host and, on CUDA, the device;
   *                    deleted ids with their members; touched hyperedges and vertices).
   */
  virtual void after_apply(const resources& res, const graph_type& h,
                           const hyper_apply_summary& summary, const hyper_applied& applied) = 0;

  /**
   * @brief Mark the result unusable after after_apply() threw (later updates throw).
   */
  virtual void poison() noexcept = 0;

  /**
   * @brief Whether before_apply() or after_apply() read the line graph (h2h): run_update() then
   *        makes sure it is maintained before before_apply() (line_graph_policy::automatic builds
   *        it on G_t, and the commit keeps it).
   * @return false (the default).
   */
  [[nodiscard]] virtual bool reads_prepared_graph() const noexcept {
    return false;
  }
};

/**
 * @brief Run one update of a hypergraph over several participants: the transaction's begin
 *        (Step 0 once), every before_apply() on G_t, one commit, then every after_apply() on
 *        G_{t+1}.
 *
 * If the begin, a before_apply() or the commit's validation throws, nothing was changed. If an
 * after_apply() throws, that result is poisoned, the remaining participants still run, and the
 * first exception is rethrown.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in]     res          Execution resources.
 * @param[in,out] h            The hypergraph (version + 1).
 * @param[in]     batch        The batch.
 * @param[in]     participants The participants (distinct targets).
 * @param[in]     count        Number of participants.
 * @param[in]     commit_stage Profiler stage name of the commit (e.g. "triad_count.commit").
 * @return What the commit did.
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
hyper_apply_summary run_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                               const hyperedge_batch_view<vertex_t, weight_t>& batch,
                               hyper_update_participant<vertex_t, weight_t>* const* participants,
                               std::size_t count, std::string_view commit_stage);

/**
 * @brief The participant interface, batch view and run_update() of hypergraph<V,W>: what makes
 *        dyng::update() and update_each() work for hypergraphs.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
struct participant_of<hypergraph<vertex_t, weight_t>> {
  static constexpr bool supported = true;                       ///< a container of update()
  using type = hyper_update_participant<vertex_t, weight_t>;    ///< the interface
  using batch_type = hyperedge_batch_view<vertex_t, weight_t>;  ///< the batch view it takes

  /**
   * @brief Run one update over the participants (run_update() of hypergraphs).
   * @param[in]     res          Execution resources.
   * @param[in,out] h            The hypergraph.
   * @param[in]     batch        The batch.
   * @param[in]     participants The participants.
   * @param[in]     count        Number of participants.
   * @param[in]     commit_stage Profiler stage name of the commit.
   * @return What the commit did.
   */
  static hyper_apply_summary run(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                 const batch_type& batch, type* const* participants,
                                 std::size_t count, std::string_view commit_stage) {
    return run_update(res, h, batch, participants, count, commit_stage);
  }
};

}  // namespace detail

}  // namespace dyng
