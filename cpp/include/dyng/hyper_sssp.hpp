// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp.hpp
 * @brief Dynamic single-source shortest hyperpaths (H-SOSP): compute() and update() on a weighted
 *        hypergraph, kept exact under hyperedge and incidence batches.
 * @ingroup hyper_sssp
 *
 * Draft for 0.4 (M11a): tracked by the API snapshot, frozen at the end of M11b, stable at 0.4.0
 * only on the author's confirmation (HD17; ADR 0090).
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

/**
 * @defgroup hyper_sssp hyper_sssp
 * @brief Single-source shortest hyperpaths on a hypergraph with hyperedge weights, kept up to
 *        date under batches of hyperedge and incidence changes (H-SOSP, IA3 2026).
 *
 * **The problem.** The line graph of a hypergraph has one node per live hyperedge, and two nodes
 * are adjacent when their hyperedges share a vertex. Entering hyperedge h costs its weight w(h).
 * From a source vertex s, the distance D(h) of a live hyperedge is the cheapest walk
 * h_1, ..., h_k = h in the line graph with s in h_1: D(h) = w(h_1) + ... + w(h_k), so D(h) = w(h)
 * for every h that contains s. Its canonical parent is -1 when D(h) = w(h) and s is in h, and
 * otherwise the lowest-id neighbour p with D(p) + w(h) = D(h) (MOSP_ESCHER's tie rule). The
 * distance of a vertex v is the minimum of D(h) over the live hyperedges that contain it, and 0
 * for v = s (distance_to(); D(h) already includes w(h), so nothing is added: PLAN 5.3's
 * "+ w(h)" would count it twice, HD11). Dead and unreachable hyperedges have the distance
 * infinite_distance<std::int64_t>() and the parent -1.
 *
 * **The update.** update() applies the batch to the hypergraph once (its commit: Step 0, the ids,
 * the incidence edits and the order-free line-graph delta inside it, ADR 0086), then brings the
 * result up to date from that delta (MOSP_ESCHER's hsospUpdate):
 *   - identify_affected: the roots are the endpoint b of every deleted line-graph pair (a, b)
 *     whose tree edge it was, every new, recreated or dead id, and (source vertex) every
 *     hyperedge that lost s while its parent was -1; their subtrees of the tree before the batch
 *     are invalidated (distance infinite);
 *   - seed: every invalidated hyperedge pulls its best (distance, parent) over its neighbours
 *     (and its own w(h) if it holds s), and both directions of every inserted pair are relaxed;
 *   - loop: the improved hyperedges push to their neighbours until no distance decreases;
 *   - finalize: the counters.
 * Distances only decrease after the invalidation and ties go to the lowest parent id, so the
 * update of a canonical result is the canonical result of the new hypergraph (update() equals
 * compute() bit for bit).
 *
 * **The work budget** (options::work_budget, max_iterations, on_budget; MOSP_ESCHER's
 * UpdateConfig). The CUDA engines fall back to a recompute from scratch (stats::fallback_used,
 * stats::fallback_reason) when, before a push round, their relaxations exceed work_budget times
 * the live line-graph entries, or they have run max_iterations push rounds; and always when a
 * distance does not fit their packed words (engine::fused: 32 bits; engine::operators: the
 * hyperedge id bound times the largest weight next to the parent bits), or the invalidation's
 * pointer jumping does not end (engine::fused: 64 rounds; engine::operators: ceil(log2 n) + 1).
 * The host engine counts the relaxations the same way, checked before each expansion, and has no
 * iteration, width or depth limit. The fallback gives the same result as
 * the update would have (it is exact); under budget_action::error an exceeded work or iteration
 * budget throws convergence_error instead, after the commit.
 *
 * **Hypergraph requirements** (checked by compute() and before the commit of update()): integer
 * weights (hypergraph<int32_t, int32_t>) in [1, 2^28] (a stricter limit than the container's
 * [1, 2^31 - 1]; MOSP_ESCHER's), hypergraph_storage::slack_csr with its line graph
 * (line_graph_policy::slack_csr, or automatic, which builds it on first use), and the
 * vertex-to-hyperedge rows (store_v2h). Any id_reuse order and capacity policy.
 *
 * **Backends** (HD14): sequential (MOSP_ESCHER's host emulation, emulateSospRecompute and
 * emulateSospUpdate) and cuda, with two engines that return the same bytes: MOSP_ESCHER's kernels
 * as they are (engine::fused, the default) and the shared sssp engine (engine::operators: sssp's
 * operators over the line graph with node weights). There is no openmp backend in 0.4: openmp
 * resources raise not_supported_error. On cuda the hypergraph must be built with (or cloned for)
 * the CUDA resources and the result's arrays are device memory (copy them with to_vector()); the
 * engines synchronize the stream once per push round (engine::fused also once per
 * pointer-jumping round).
 *
 * Types: int32_t vertex ids and int32_t weights (other hypergraph types fail to compile with a
 * static_assert naming these).
 */

namespace dyng::hyper_sssp {

/**
 * @brief What update() does when its work or iteration budget is spent (options::on_budget).
 * @ingroup hyper_sssp
 */
enum class budget_action : std::uint8_t {
  recompute,  ///< recompute from scratch (stats::fallback_used = true): the result is exact
  error,      ///< throw convergence_error; the budget is counted after the commit, so the
              ///< hypergraph keeps the batch and the result is poisoned (the basic guarantee)
};

/**
 * @brief Why an update fell back to a recompute (stats::fallback_reason).
 * @ingroup hyper_sssp
 */
enum class fallback_reason : std::uint8_t {
  none,                ///< no fallback
  work_budget,         ///< the relaxations exceeded options::work_budget x the live entries
  iterations,          ///< the push rounds reached options::max_iterations (cuda)
  distance_width,      ///< a distance did not fit the engine's 32-bit packed word (cuda)
  invalidation_depth,  ///< the pointer jumping of the invalidation needed more than 64 rounds
                       ///< (cuda)
};

/**
 * @brief The lower-case name of a budget action ("recompute", "error").
 * @param[in] a The action.
 * @return A static string naming `a`.
 * @ingroup hyper_sssp
 */
[[nodiscard]] std::string_view to_string(budget_action a) noexcept;

/**
 * @brief The lower-case name of a fallback reason ("none", "work_budget", "iterations",
 *        "distance_width", "invalidation_depth").
 * @param[in] r The reason.
 * @return A static string naming `r`.
 * @ingroup hyper_sssp
 */
[[nodiscard]] std::string_view to_string(fallback_reason r) noexcept;

/**
 * @brief Options of compute() and update() (an aggregate; fields are only ever appended).
 *
 * Every field is a tunable: it changes how an update runs, never its result, and
 * result::set_options() changes it.
 * @ingroup hyper_sssp
 */
struct options {
  /// The work budget of an update, as a multiple of the live line-graph entries (MOSP_ESCHER's
  /// UpdateConfig::workBudget): before each push round (cuda) or expansion (sequential), an update
  /// whose relaxations exceed work_budget x max(1, live entries) takes the on_budget action. 0
  /// acts after any work; +infinity never. Finite, or +infinity, and >= 0.
  double work_budget = 1.0;
  /// The push rounds an update may run before the on_budget action (MOSP_ESCHER's
  /// maxIterations; cuda only: the host engine has no rounds). Also the floor of the recompute's
  /// round cap, max(max_iterations, hyperedge id bound + 1). At least 1.
  std::int64_t max_iterations = 4096;
  /// What an update does when it exceeds work_budget or max_iterations (the 32-bit width and the
  /// invalidation depth always recompute).
  budget_action on_budget = budget_action::recompute;
  /// The CUDA engine: automatic and fused run MOSP_ESCHER's kernels (engine::fused);
  /// engine::operators runs the shared sssp engine (the same results). Ignored by the sequential
  /// backend.
  engine cuda_engine = engine::automatic;
  /// Check the weights against [1, 2^28] (compute(): every live hyperedge; update(): the batch's
  /// insertions, before the commit). Without the check heavier weights still give exact results
  /// (they fall back more often: fallback_reason::distance_width).
  bool validate_inputs = true;
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 *
 * The inherited fields mean:
 *   - `affected`: the hyperedges whose subtree was invalidated (= invalidated; deterministic);
 *   - `iterations`: the push rounds (cuda) or the expansions (sequential) of Step 2;
 *     schedule-dependent on cuda, logged only;
 *   - `frontier_visits`: the largest frontier (cuda) or the expansions (sequential);
 *     schedule-dependent on cuda;
 *   - `fallback_used`: the update fell back to a recompute. **Deterministic on sequential;
 *     schedule-dependent on cuda near the budget**: the relaxations of a push round depend on the
 *     thread schedule, so a batch within a few percent of its budget may fall back in one run and
 *     not in another, with identical results (HD7; the weaker level core/stats.hpp allows);
 *   - `converged`: always true;
 *   - `engine_used`: the CUDA engine that ran (options::cuda_engine; automatic is fused), and
 *     engine::operators on sequential.
 * @ingroup hyper_sssp
 */
struct stats : update_stats {
  /// Deterministic: what applying the batch did to the hypergraph (the inserted ids among it).
  hyper_apply_summary batch;
  /// Deterministic: the line-graph pairs the batch added.
  std::int64_t line_graph_insertions = 0;
  /// Deterministic: the line-graph pairs the batch removed.
  std::int64_t line_graph_deletions = 0;
  /// Deterministic: the hyperedges in invalidated subtrees (the source excluded).
  std::int64_t invalidated = 0;
  /// The relaxations the budget counted (the neighbours read by the pulls and pushes).
  /// Schedule-dependent on cuda; deterministic on sequential.
  std::int64_t relaxations = 0;
  /// Why the update fell back (fallback_reason::none if it did not); schedule-dependent on cuda
  /// near the budget, as fallback_used.
  hyper_sssp::fallback_reason fallback_reason = hyper_sssp::fallback_reason::none;
};

}  // namespace dyng::hyper_sssp

namespace dyng::detail {
struct hyper_sssp_state;
struct hyper_sssp_access;

/**
 * @brief Whether hyper_sssp is instantiated for hypergraph<vertex_t, weight_t>: int32_t vertex
 *        ids and int32_t weights.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 */
template <typename vertex_t, typename weight_t>
inline constexpr bool hyper_sssp_supported_v =
    std::is_same_v<vertex_t, std::int32_t> && std::is_same_v<weight_t, std::int32_t>;

/**
 * @brief The message of hyper_sssp's static_assert for an unsupported hypergraph type.
 * @ingroup hyper_sssp
 */
#define DYNG_HYPER_SSSP_TYPES_MESSAGE                                                          \
  "dyng::hyper_sssp supports hypergraph<int32_t, int32_t> only (weighted hyperedges, weights " \
  "in [1, 2^28]); an unweighted hypergraph has no hyperedge lengths; see "                     \
  "docs/algorithms/hyper_sssp.md"

/**
 * @brief The message of hyper_sssp's static_assert for a container that is not a hypergraph.
 * @ingroup hyper_sssp
 */
#define DYNG_HYPER_SSSP_CONTAINER_MESSAGE                               \
  "dyng::hyper_sssp is a hypergraph algorithm; dyng::update() takes a " \
  "dyng::hypergraph<int32_t, int32_t> for its result"
}  // namespace dyng::detail

namespace dyng::hyper_sssp {

/**
 * @brief Shortest hyperpaths from one source vertex, kept up to date by update() (opaque,
 *        move-only).
 *
 * The result owns its distance and parent arrays by hyperedge id (host memory, or device memory
 * for the cuda backend), its options, the hypergraph version it matches, and on cuda the engine's
 * device state (MOSP_ESCHER's HsospState: the packed words, the invalidation and frontier arrays).
 *
 * **Thread safety.** A result is not thread-safe: its accessors may run concurrently with each
 * other, but not with an update() of it (or a dyng::update() that includes it), which must also
 * not overlap another call on its hypergraph.
 *
 * @tparam vertex_t   Vertex id type (int32_t).
 * @tparam distance_t Distance type: std::int64_t, the only width (another value fails to compile).
 * @ingroup hyper_sssp
 */
template <typename vertex_t, typename distance_t = std::int64_t>
class result {
  static_assert(std::is_same_v<distance_t, std::int64_t>,
                "dyng::hyper_sssp::result: distance_t is std::int64_t");
  static_assert(std::is_same_v<vertex_t, std::int32_t>,
                "dyng::hyper_sssp::result: vertex_t is std::int32_t");

 public:
  /**
   * @brief Move constructor.
   * @param[in,out] other The result to move from; it may only be assigned to or destroyed after.
   */
  result(result&& other) noexcept;

  /**
   * @brief Move assignment.
   * @param[in,out] other The result to move from; it may only be assigned to or destroyed after.
   * @return *this.
   */
  result& operator=(result&& other) noexcept;

  result(const result&) = delete;             ///< not copyable: use clone()
  result& operator=(const result&) = delete;  ///< not copyable: use clone()
  ~result();                                  ///< releases the arrays

  /**
   * @brief The source vertex.
   * @return The source given to compute() (invalid_id() for a moved-from result).
   */
  [[nodiscard]] vertex_t source() const noexcept;

  /**
   * @brief The distances D(h) by hyperedge id.
   * @return hyperedge_id_bound() values of the hypergraph the result matches (host memory; device
   *         memory for a result of the cuda backend); infinite_distance<distance_t>() for dead or
   *         unreachable ids. Valid until the next update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const distance_t> hyperedge_distances() const;

  /**
   * @brief The canonical parents by hyperedge id (the shortest-hyperpath tree).
   * @return As many values as hyperedge_distances() (in its memory space): 0-based hyperedge ids;
   *         -1 for a hyperedge reached directly from the source vertex, and for dead or
   *         unreachable ids. Hyperedge ids of slack_csr storage are below 2^31, so they are
   *         32-bit. Valid until the next update().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const std::int32_t> hyperedge_parents() const;

  /**
   * @brief The distance of a vertex: the minimum of D(h) over the live hyperedges that contain
   *        it, 0 for the source (HD11).
   * @param[in] target A vertex in [0, num_vertices()).
   * @return The distance; infinite_distance<distance_t>() if no live hyperedge of `target` is
   *         reachable (or it is in none).
   * @throws invalid_argument_error if `target` is out of range, or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned), or the
   *         hypergraph the result matches no longer exists or has changed since.
   * @sync Reads the vertex's hyperedges from the hypergraph the result matches and, on cuda, the
   *       distances from the device (the stream is synchronized once per hypergraph version).
   */
  [[nodiscard]] distance_t distance_to(vertex_t target) const;

  /**
   * @brief The distances of every vertex (distance_to() of each), computed on request (HD11).
   * @return num_vertices() values (host memory on every backend). Valid until the next update()
   *         of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     as distance_to().
   * @throws out_of_memory_error    if the array cannot be allocated.
   * @sync As distance_to(); one pass over the incidences per hypergraph version.
   */
  [[nodiscard]] array_view<const distance_t> vertex_distances() const;

  /**
   * @brief The options.
   * @return The options given at compute(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the tunables (every field of options); the next update() uses them.
   * @param[in] opt The new options.
   * @throws invalid_argument_error if a field holds an invalid value (work_budget negative or
   *         NaN, max_iterations below 1, an enumerator out of range), or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @guarantee Strong: every check runs before the options change.
   */
  void set_options(const options& opt);

  /**
   * @brief The hypergraph version this result matches.
   * @return The version of the hypergraph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of hyperedge_distances() and hyperedge_parents().
   * @return memory_space::device for a result of the cuda backend, memory_space::host otherwise.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy (arrays, options, version, the engine's state) for the resources `res`.
   *
   * The copy belongs to the backend of `res` (its arrays are copied between host and device memory
   * as needed): this is how a result moves between sequential and cuda, with its hypergraph
   * (hypergraph::to_backend()).
   * @param[in] res Execution resources of the copy (sequential or cuda).
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is openmp or not built.
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   */
  [[nodiscard]] result clone(const resources& res) const;

 private:
  friend struct detail::hyper_sssp_access;
  explicit result(std::unique_ptr<detail::hyper_sssp_state> state) noexcept;

  std::unique_ptr<detail::hyper_sssp_state> impl_;
};

}  // namespace dyng::hyper_sssp

namespace dyng::detail {

/**
 * @brief hyper_sssp::compute() for a supported hypergraph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] res    Execution resources.
 * @param[in] h      The hypergraph.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The result.
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyper_sssp::result<vertex_t> hyper_sssp_compute(
    const resources& res, const hypergraph<vertex_t, weight_t>& h, vertex_t source,
    const hyper_sssp::options& opt);

/**
 * @brief hyper_sssp::update() for a supported hypergraph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in]     res   Execution resources.
 * @param[in,out] h     The hypergraph.
 * @param[in]     batch The batch.
 * @param[in,out] r     The result.
 * @return Counters of the update.
 */
template <typename vertex_t, typename weight_t>
hyper_sssp::stats hyper_sssp_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                    const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                    hyper_sssp::result<vertex_t>& r);

}  // namespace dyng::detail

namespace dyng::hyper_sssp {

/**
 * @brief Compute the shortest hyperpaths of `h` from the source vertex `source` (the static solve,
 *        the oracle of update()).
 *
 * Every hyperedge that contains `source` starts at its own weight with parent -1; then rounds of
 * pulls (every candidate takes its best (distance, lowest id) neighbour) and collections (the
 * neighbours of the improved hyperedges become the next candidates) run until nothing improves:
 * MOSP_ESCHER's hsospRecompute on cuda (one warp per candidate) and emulateSospRecompute on
 * sequential. Both give the canonical tree.
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam weight_t Hyperedge weight type (int32_t).
 * @param[in] res    Execution resources (sequential or cuda).
 * @param[in] h      The hypergraph (see "Hypergraph requirements" of @ref hyper_sssp; built with,
 *                   or cloned for, the backend of `res`); it is not modified (under
 *                   line_graph_policy::automatic its line graph is built here, once).
 * @param[in] source The source vertex, in [0, num_vertices()).
 * @param[in] opt    Options.
 * @return The result, matching `h.version()`.
 * @throws invalid_argument_error if `source` is out of range, an option is invalid, a live weight
 *         is outside [1, 2^28] (validate_inputs), `h` keeps no vertex-to-hyperedge rows
 *         (store_v2h = false), `h` belongs to another backend than `res`, or `h` was poisoned by a
 *         failed apply().
 * @throws not_supported_error    if the backend of `res` is openmp (hyper_sssp runs on
 *         sequential and cuda in 0.4; use dyng::resources::sequential()) or not built, `h` is
 *         not slack_csr storage with its line graph (line_graph_policy::none or sorted_merge).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @sync On cuda the stream is synchronized once per round of the recompute.
 * @backends sequential, cuda
 * @determinism Bit-exact across backends and runs: the canonical tree (lowest-id ties).
 * @guarantee Strong: `h` is not modified (its line graph may be built under
 *            line_graph_policy::automatic), and nothing is kept if the call throws.
 * @paper H-SOSP (IA3 2026): `dyng::citation("hyper_sssp")`, key hsosp2026 in
 *        docs/references.bib.
 * @ingroup hyper_sssp
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res,
                                       const hypergraph<vertex_t, weight_t>& h, vertex_t source,
                                       const options& opt = {}) {
  static_assert(detail::hyper_sssp_supported_v<vertex_t, weight_t>, DYNG_HYPER_SSSP_TYPES_MESSAGE);
  return detail::hyper_sssp_compute(res, h, source, opt);
}

/**
 * @brief Apply a batch of hyperedge and incidence changes to `h` and update the shortest
 *        hyperpaths `r`.
 *
 * The batch is applied once (hypergraph::apply, with the line-graph delta inside its commit);
 * then the result is updated from the delta (see @ref hyper_sssp). Postcondition: `r` equals
 * compute(res, h, r.source(), r.get_options()) exactly, distances and parents.
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam weight_t Hyperedge weight type (int32_t).
 * @param[in]     res   Execution resources (sequential or cuda; `h` and `r` must belong to their
 *                      backend).
 * @param[in,out] h     The hypergraph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Hyperedge deletions (by id) and insertions, incidence removals and
 *                      additions (host memory, or device memory under the copy policy of `res`).
 * @param[in,out] r     Result of compute() or of a previous update() on `h`.
 * @return Counters of this update (see stats for which are deterministic).
 * @throws stale_result_error     if r.graph_version() != h.version(), `r` was computed on another
 *         hypergraph, or `r` was left unusable by a failed update.
 * @throws invalid_argument_error if a batch id, member or weight is invalid (an inserted weight
 *         outside [1, 2^28] with validate_inputs), `h` belongs to another backend than `res`, or
 *         `h` was poisoned (nothing is changed).
 * @throws not_supported_error    as compute() (nothing is changed).
 * @throws convergence_error      under budget_action::error, if the update exceeds its work or
 *         iteration budget (after the commit: `h` holds the new version and `r` is poisoned).
 * @throws capacity_error         if the hypergraph's storage cannot hold the batch (as
 *         hypergraph::apply).
 * @throws out_of_memory_error    if host or device memory cannot be allocated (before the commit:
 *         nothing is changed; inside it: `h` is poisoned; after it: `r` is poisoned).
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @sync On cuda the stream is synchronized once per pointer-jumping round and once per push round
 *       (the engine reads its counters), plus the commit's own synchronizations.
 * @backends sequential, cuda
 * @determinism Bit-exact across backends and runs (distances, parents, `invalidated`, the line
 *              graph counts); fallback_used, fallback_reason, relaxations and iterations are
 *              schedule-dependent on cuda (see stats).
 * @guarantee Strong for every error found before the batch is applied: `h` and `r` are unchanged.
 *            Basic for an error inside the commit (`h` is poisoned, as hypergraph::apply; `r` is
 *            unchanged and must be recomputed once `h` is rebuilt) and after it (a spent budget
 *            under budget_action::error, a CUDA or allocation failure: `h` holds the new version
 *            and `r` is poisoned, so every later use of it throws stale_result_error until it is
 *            recomputed).
 * @paper H-SOSP (IA3 2026): `dyng::citation("hyper_sssp")`.
 * @ingroup hyper_sssp
 */
template <typename vertex_t, typename weight_t>
stats update(const resources& res, hypergraph<vertex_t, weight_t>& h,
             const hyperedge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r) {
  static_assert(detail::hyper_sssp_supported_v<vertex_t, weight_t>, DYNG_HYPER_SSSP_TYPES_MESSAGE);
  return detail::hyper_sssp_update(res, h, batch, r);
}

}  // namespace dyng::hyper_sssp

namespace dyng::detail {

/**
 * @brief The update participant of a hyper_sssp result (used by dyng::update()).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_hyper_sssp_participant(
    hyper_sssp::result<vertex_t>& r, hyper_sssp::stats& out);

/**
 * @brief update_traits of hyper_sssp::result (dyng::update() support).
 * @tparam vertex_t   Vertex id type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename distance_t>
struct update_traits<hyper_sssp::result<vertex_t, distance_t>> {
  using stats_type = hyper_sssp::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type (a hypergraph).
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      hyper_sssp::result<vertex_t, distance_t>& r, hyper_sssp::stats& out) {
    static_assert(is_hypergraph_v<container_t>, DYNG_HYPER_SSSP_CONTAINER_MESSAGE);
    if constexpr (is_hypergraph_v<container_t>) {
      static_assert(hyper_sssp_supported_v<typename container_t::vertex_type,
                                           typename container_t::weight_type>,
                    DYNG_HYPER_SSSP_TYPES_MESSAGE);
      return make_hyper_sssp_participant<typename container_t::vertex_type,
                                         typename container_t::weight_type>(r, out);
    } else {
      (void)r;
      (void)out;
      return nullptr;
    }
  }
};

}  // namespace dyng::detail
