// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation.hpp
 * @brief label_propagation: binary harmonic label propagation on a weighted similarity graph,
 *        kept up to date under vertex batches (DynLP, ICS 2026; LabelPropagation-CUDA at a276a3a).
 * @ingroup label_propagation
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/update.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

/**
 * @defgroup label_propagation label_propagation
 * @brief Binary harmonic label propagation under vertex batches (DynLP).
 *
 * Every vertex v gets a label F(v) in [0, 1]: the seeds (vertices with a seed label, the ground
 * truth of class 0 or 1) keep their class, and every other vertex the weighted average of its
 * neighbours' labels (the harmonic solution of the graph, approached by iteration until no label
 * changes by more than options::tolerance). The graph is undirected with float weights (a
 * similarity graph); vertices arrive (with their similarity edges, some with a seed label) and
 * leave in batches.
 *
 * **The published setup** (PLAN 6.4.7; ADR 0071): the full graph is known up front (a
 * generators::vertex_arrival_stream), so the graph is graph::with_capacity(stream.row_capacity)
 * with row_layout::slotted and batch_semantics::net_effect(); compute() on the empty graph gives
 * the starting result (every label 0.5), and every batch b0, b1, ... is one update() with the
 * published threshold (options::similarity_threshold = stream.similarity_threshold). b0 (the
 * first ground truth and the edges among it) is an ordinary update.
 *
 * **update()** is DynLP (LabelPropagation-CUDA's runDynLP, one batch): the vertices affected by
 * the batch (the neighbours of the deleted ones on the rows before the batch, the inserted ones,
 * both endpoints of every new edge), the components of the inserted vertices over their strong
 * mutual edges (weight > tau, the sparsified subgraph G'), the initial label of each component
 * from its supernode weights to the two classes (64-bit fixed point), then propagation from the
 * affected vertices until no label changes by more than the tolerance (or max_iterations), and
 * the seeds' labels restored. **compute()** is IrLP, the published static baseline: Jacobi
 * iterations over every unlabelled active vertex from 0.5.
 *
 * Neither is compared with the other: both are compared with the converged harmonic solution of
 * the current graph (dyng::testing::harmonic_solution), the original checker's rule mean |F - F*|
 * <= 2000 tolerance (PLAN 5.1, the oracle kind "reference").
 */

namespace dyng::label_propagation {

/**
 * @brief The update rule of the propagation (update(); compute() is IrLP's Jacobi).
 * @ingroup label_propagation
 */
enum class schedule : std::uint8_t {
  /// The published rule: every iteration updates its frontier in two hashed half-sweeps, each
  /// vertex reading its neighbours' current labels (the second half sees the first's). Fast; not
  /// bitwise reproducible on CUDA (a tolerance, calibrated against the original's runs).
  in_place,
  /// 1/2-damped Jacobi: every new label F' = (F + average) / 2 is computed from the labels of the
  /// previous iteration and committed after both half-sweeps. Bitwise reproducible per backend
  /// and build; the same fixed point.
  damped,
};

/**
 * @brief What update() does when the propagation reaches options::max_iterations with vertices
 *        still changing.
 * @ingroup label_propagation
 */
enum class on_limit : std::uint8_t {
  error,               ///< throw convergence_error (the result is left unusable)
  report,              ///< keep the labels and report stats::converged = false (the default)
  fallback_recompute,  ///< recompute every label with compute()'s IrLP; stats::fallback_used
};

/**
 * @brief Options of compute() and update() (an aggregate; fields are only ever appended). Every
 *        field is a tunable: set_options() may change it between updates.
 * @ingroup label_propagation
 */
struct options {
  /**
   * @brief tau, the sparsification threshold: an edge between two vertices inserted in the same
   *        batch joins G' when its weight is > tau.
   *
   * A value: the published setup passes generators::vertex_arrival_stream::similarity_threshold
   * (the full graph's average edge weight, the original's computeAvgEdgeWeight). std::nullopt:
   * the average weight of the current graph's live directed entries, computed in every update as
   * a double sum in vertex order and slot order, rounded to float (0.5 for a graph without
   * edges), bitwise the same on every backend; on cuda each such update first downloads the
   * graph's rows to the host (a copy of the slot storage and an O(m) pass per update, not part of
   * the published setup). It must not be NaN.
   */
  std::optional<float> similarity_threshold;
  float tolerance = 1e-4F;  ///< delta: a label changes "by more than delta" (> 0)
  /** @brief The cap of the propagation per update() and of compute()'s iterations (> 0; the
   *         original's default). */
  std::int64_t max_iterations = 100000;
  on_limit on_iteration_limit = on_limit::report;  ///< see on_limit
  schedule update_schedule = schedule::in_place;   ///< the published rule by default
  /** @brief The CUDA engine: engine::automatic or engine::fused (the verbatim port of the
   *         original's pipeline, bitwise the original's damped labels; ADR 0075); engine::operators
   *         arrives in M10b (not_supported_error until then). Ignored on the host backends. */
  engine cuda_engine = engine::automatic;
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 *
 * The determinism of the inherited fields (core/stats.hpp's levels; AD20): `affected` (the
 * vertices whose label bits changed), `converged` and `fallback_used` are deterministic per backend
 * under schedule::damped and schedule-dependent under schedule::in_place (where the labels depend
 * on the order of the updates); `iterations` and `frontier_visits` are schedule-dependent (under
 * schedule::damped they repeat from run to run on one backend); `engine_used` is deterministic per
 * backend.
 * @ingroup label_propagation
 */
struct stats : update_stats {
  /// Deterministic: what applying the batch did to the graph.
  apply_summary batch;
  /// Deterministic: the components of the inserted vertices in G' (0 when none was inserted).
  std::int64_t components = 0;
  /// Deterministic: the edges of G' (kept by the tau filter between two inserted vertices).
  std::int64_t sparsified_edges = 0;
  /// Deterministic: the initial frontier of the propagation, |V_aff| (the affected vertices that
  /// are active and have no seed label).
  std::int64_t initial_frontier = 0;
};

}  // namespace dyng::label_propagation

namespace dyng::detail {
template <typename vertex_t>
struct label_propagation_state;
struct label_propagation_access;
}  // namespace dyng::detail

namespace dyng::label_propagation {

/**
 * @brief The labels kept up to date by update() (opaque, move-only).
 *
 * The result owns the label of every vertex, the seed labels, its options, the graph version it
 * matches and the propagation's run-wide state (the original's tags: every update consumes at
 * least one).
 * @tparam vertex_t Vertex id type (std::int32_t).
 * @ingroup label_propagation
 */
template <typename vertex_t>
class result {
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
  ~result();                                  ///< releases the labels

  /**
   * @brief The labels.
   * @return One label per vertex in [0, 1] (memory space(): host on the host backends, the
   *         device of the result's resources on cuda): a seed's
   *         class exactly (0 or 1); a deleted vertex keeps its last label; a vertex that never
   *         entered the graph 0.5 (after compute(): every vertex that is not active 0.5). Valid
   *         until the next update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const float> labels() const;

  /**
   * @brief The seed labels.
   * @return One per vertex: -1 (none), or the class 0 / 1 of a seed (given to compute() or
   *         inserted with a label by a batch; kept when the vertex is deleted), host memory.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const std::int8_t> seed_labels() const;

  /**
   * @brief The options.
   * @return The options given to compute(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the options (every field is a tunable).
   * @param[in] opt The new options.
   * @throws invalid_argument_error if an option is out of range (tolerance <= 0 or NaN,
   *         max_iterations <= 0, a NaN similarity_threshold), or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  void set_options(const options& opt);

  /**
   * @brief The graph version this result matches.
   * @return The version of the graph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of the labels.
   * @return memory_space::host on the host backends, memory_space::device on cuda.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy for the resources `res` (labels, seeds, options, the run-wide state).
   * @param[in] res Execution resources of the copy.
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available for
   *         label_propagation (OpenMP: use dyng.Resources("sequential") or cuda).
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @sync
   */
  [[nodiscard]] result clone(const resources& res) const;

  /**
   * @brief Size the per-batch workspaces of `res` for batches of up to `max_batch_vertices`
   *        inserted vertices and `max_batch_edges` edges (the original's setup), so that the
   *        updates of such batches allocate nothing for the result; a larger batch grows them
   *        later, with headroom.
   *
   * The graph's own per-batch storage is the graph's: on cuda, graph::reserve(res,
   * max_batch_edges) sizes the device scratch of a batch's edges, and the graph's device arrays
   * of a batch's vertex ids and the page-locked staging of its pageable arrays grow on demand
   * (the first vertex batch's size exactly, a later larger batch with headroom; ADR 0074), so an
   * update of the published stream allocates in its first two vertex batches at most.
   * @param[in] res                Execution resources the updates will use.
   * @param[in] max_batch_vertices The largest number of vertex insertions of a batch
   *                               (generators::vertex_arrival_stream::max_batch_vertices).
   * @param[in] max_batch_edges    The largest number of edges of a batch
   *                               (generators::vertex_arrival_stream::max_batch_edges).
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available for
   *         label_propagation.
   * @throws out_of_memory_error    if the workspaces cannot be allocated.
   * @sync
   */
  void reserve(const resources& res, std::size_t max_batch_vertices, std::size_t max_batch_edges);

 private:
  friend struct detail::label_propagation_access;
  explicit result(std::unique_ptr<detail::label_propagation_state<vertex_t>> state) noexcept;

  std::unique_ptr<detail::label_propagation_state<vertex_t>> impl_;
};

}  // namespace dyng::label_propagation

namespace dyng::detail {

/**
 * @brief Whether label_propagation is instantiated for graph<vertex_t, edge_t, weight_t>: float
 *        weights, 32-bit vertex ids, 32- or 64-bit offsets.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
inline constexpr bool label_propagation_supported_v =
    std::is_same_v<weight_t, float> && std::is_same_v<vertex_t, std::int32_t> &&
    (std::is_same_v<edge_t, std::int32_t> || std::is_same_v<edge_t, std::int64_t>);

/**
 * @brief The message of label_propagation's static_assert for an unsupported graph type.
 * @ingroup label_propagation
 */
#define DYNG_LABEL_PROPAGATION_TYPES_MESSAGE                                             \
  "dyng::label_propagation supports graph<int32_t, int32_t or int64_t, float> only (an " \
  "undirected similarity graph with float weights); see docs/algorithms/label_propagation.md"

/**
 * @brief label_propagation::compute() for a supported graph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in] res            Execution resources.
 * @param[in] g              The graph.
 * @param[in] seed_vertices  The seeds.
 * @param[in] seed_labels    Their classes.
 * @param[in] opt            Options.
 * @return The result.
 */
template <typename vertex_t, typename edge_t>
[[nodiscard]] label_propagation::result<vertex_t> label_propagation_compute(
    const resources& res, const graph<vertex_t, edge_t, float>& g,
    array_view<const vertex_t> seed_vertices, array_view<const std::int8_t> seed_labels,
    const label_propagation::options& opt);

/**
 * @brief label_propagation::update() for a supported graph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     res   Execution resources.
 * @param[in,out] g     The graph.
 * @param[in]     batch The batch.
 * @param[in,out] r     The result.
 * @return The update's counters.
 */
template <typename vertex_t, typename edge_t>
label_propagation::stats label_propagation_update(const resources& res,
                                                  graph<vertex_t, edge_t, float>& g,
                                                  const edge_batch_view<vertex_t, float>& batch,
                                                  label_propagation::result<vertex_t>& r);

}  // namespace dyng::detail

namespace dyng::label_propagation {

/**
 * @brief The static propagation from the seeds: IrLP, the published baseline (the original's
 *        runIrLPSnapshot): Jacobi iterations over every active vertex without a seed label, from
 *        0.5, until no label changes by more than options::tolerance or options::max_iterations.
 *
 * The neighbours are read from the graph's current edges with every row sorted by column (the
 * snapshot the original builds from its full graph); a vertex that is not active keeps 0.5. The
 * result's propagation state starts fresh (no tag consumed: the first update() is the original's
 * first batch).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in] res           Execution resources (sequential or cuda; OpenMP is not
 *                          a backend of label_propagation in 0.4).
 * @param[in] g             An undirected graph with float weights: row_layout::slotted (the
 *                          published setup; compute() on graph::with_capacity()'s empty graph gives
 *                          every label 0.5) or compact. It is not modified.
 * @param[in] seed_vertices The seeds: active vertices of `g`, each once (any memory space: read on
 *                          the host).
 * @param[in] seed_labels   One class (0 or 1) per seed.
 * @param[in] opt           Options.
 * @return The result, matching `g.version()`.
 * @throws invalid_argument_error if `g` is directed, an option is out of range (tolerance <= 0 or
 *         NaN, max_iterations <= 0, a NaN similarity_threshold), the seed arrays differ in size, a
 *         seed is out of range, listed twice, not active, or its label is not 0 or 1, or `g` is
 *         not host storage for host resources.
 * @throws not_supported_error    if the backend of `res` is not available for label_propagation
 *         (OpenMP in 0.4: the message names sequential and cuda; use dyng.Resources("sequential")).
 * @throws convergence_error      if the cap is reached and options::on_iteration_limit is
 *         on_limit::error or on_limit::fallback_recompute (compute() is the recomputation).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @backends sequential, cuda (IrLP's kernel verbatim: per snapshot bitwise the original's)
 * @determinism Bitwise per backend and build (Jacobi from fixed starting labels).
 * @paper DynLP (ICS 2026): `dyng::citation("label_propagation")`, key dynlp2026 in
 *        docs/references.bib; IrLP is the paper's baseline.
 * @guarantee Strong: `g` is not modified, and nothing is kept if the call throws.
 * @ingroup label_propagation
 */
template <typename vertex_t, typename edge_t>
[[nodiscard]] result<vertex_t> compute(const resources& res,
                                       const graph<vertex_t, edge_t, float>& g,
                                       array_view<const vertex_t> seed_vertices,
                                       array_view<const std::int8_t> seed_labels,
                                       const options& opt = {}) {
  static_assert(detail::label_propagation_supported_v<vertex_t, edge_t, float>,
                DYNG_LABEL_PROPAGATION_TYPES_MESSAGE);
  return detail::label_propagation_compute(res, g, seed_vertices, seed_labels, opt);
}

/**
 * @brief Apply a vertex batch to `g` and bring the labels `r` up to date (DynLP).
 *
 * The batch holds the arriving vertices (edge_batch_view::insert_vertices, with
 * insert_vertex_labels: -1, or the class 0 / 1 of a new seed), the edges they bring
 * (insert_src / insert_dst / insert_weights) and the departing vertices (delete_vertices),
 * applied with their net effect (batch_semantics::net_effect(), ADR 0069). Then, in the
 * original's order: the affected vertices (the neighbours of the deleted vertices on the rows
 * before the batch; the inserted vertices; both endpoints of every new edge, also in a batch of
 * edges between present vertices), G' (the new edges of weight > tau between two inserted
 * vertices) and its components over the inserted vertices (numbered by their smallest vertex in
 * insertion order), the initial labels (each inserted seed its class; every other inserted vertex
 * W1 / (W0 + W1), its component's summed weights to seeds of class 1 and 0, in 64-bit fixed point,
 * or 0.5 when the sum is at most 1e-10), then the propagation from the affected vertices that are
 * active and have no seed label, with options::update_schedule, until no label changes by more
 * than options::tolerance; finally every seed's label is its class. Deleted vertices keep their
 * last label; a vertex that never entered the graph (or arrived and left in one batch) stays at
 * 0.5.
 *
 * After a seed is deleted, labels far from it can stay stale: the propagation starts at the
 * deleted seed's neighbours and stops where the changes fall below options::tolerance (under
 * schedule::damped each update halves the change), so it may not reach every vertex whose
 * harmonic label moved; those labels catch up when a later batch's affected vertices reach them.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @param[in]     res   Execution resources (sequential or cuda, the result's backend: a result
 *                      moves between them with result::clone()).
 * @param[in,out] g     The graph: undirected, float weights, row_layout::slotted with
 *                      batch_semantics::net_effect() (graph::with_capacity()). The batch is
 *                      applied to it and its version increases by one.
 * @param[in]     batch The vertex batch.
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update.
 * @throws stale_result_error     if r.graph_version() != g.version() or `r` belongs to another
 *         graph.
 * @throws invalid_argument_error if the batch breaks the input contract (graph::apply(): an id or
 *         endpoint out of range, an id listed twice or inserted a second time, a labelled vertex
 *         deleted in the same batch, labels that do not match the insertions), or `g` is not
 *         undirected, or `r` belongs to another backend or device than `res` (move it with
 *         result::clone()); nothing is changed.
 * @throws capacity_error         if an edge does not fit into its row (an edge delivered twice, or
 *         not of the graph the capacities were made for); nothing is changed on the host
 *         backends; on cuda the overflow is found on the device after the batch was applied (the
 *         graph is poisoned and `r` unusable; graph::apply()).
 * @throws not_supported_error    if `g` does not have row_layout::slotted, the batch deletes edges,
 *         or the backend of `res` is not available for label_propagation (OpenMP in 0.4: use
 *         dyng.Resources("sequential")).
 * @throws convergence_error      if the cap is reached under on_limit::error (after the batch was
 *         applied: `r` is left unusable).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @backends sequential, cuda (the fused engine: the original's pipeline on the resources' one
 *           stream, the graph's rows resident on the device; ADR 0075)
 * @determinism schedule::damped: bitwise per backend and build. schedule::in_place: the sequential
 *              backend is deterministic (it updates each half-sweep in vertex order); on CUDA a
 *              tolerance (the published rule is not reproducible). The deterministic stages
 *              (affected set, G', components, initial labels, the rows) are identical everywhere.
 * @paper DynLP (ICS 2026).
 * @guarantee Strong for every error found before the batch is applied (`g` and `r` unchanged);
 *            basic after the commit (`g` updated, `r` poisoned until it is recomputed).
 * @ingroup label_propagation
 */
template <typename vertex_t, typename edge_t>
stats update(const resources& res, graph<vertex_t, edge_t, float>& g,
             const edge_batch_view<vertex_t, float>& batch, result<vertex_t>& r) {
  static_assert(detail::label_propagation_supported_v<vertex_t, edge_t, float>,
                DYNG_LABEL_PROPAGATION_TYPES_MESSAGE);
  return detail::label_propagation_update(res, g, batch, r);
}

}  // namespace dyng::label_propagation

namespace dyng::detail {

/**
 * @brief The update participant of label_propagation::result (used by dyng::update()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (float).
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_label_propagation_participant(
    label_propagation::result<vertex_t>& r, label_propagation::stats& out);

/**
 * @brief update_traits of label_propagation::result (dyng::update() support).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct update_traits<label_propagation::result<vertex_t>> {
  using stats_type = label_propagation::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type.
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      label_propagation::result<vertex_t>& r, label_propagation::stats& out) {
    static_assert(label_propagation_supported_v<typename container_t::vertex_type,
                                                typename container_t::edge_type,
                                                typename container_t::weight_type>,
                  DYNG_LABEL_PROPAGATION_TYPES_MESSAGE);
    return make_label_propagation_participant<vertex_t, typename container_t::edge_type,
                                              typename container_t::weight_type>(r, out);
  }
};

}  // namespace dyng::detail
