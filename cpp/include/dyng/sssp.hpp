// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp.hpp
 * @brief Dynamic single-source shortest paths: compute() and update().
 * @ingroup sssp
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

#include <cstdint>
#include <memory>

/**
 * @defgroup sssp sssp
 * @brief Dynamic single-source shortest paths (the SOSP update of DynaMOSP).
 *
 * compute() builds the canonical shortest-path tree of a graph (ties go to the lowest parent id);
 * update() applies a batch of edge insertions, deletions and weight changes to the graph and
 * brings the tree up to date incrementally. Step 1 invalidates the subtrees below deleted or
 * weight-increased tree edges and pulls each invalidated vertex's best in-neighbour; Step 2
 * relaxes outward until no distance decreases. Every backend returns the same tree, bit for bit.
 *
 * Tie rule (the rule of MOSP-OpenMP's sospUpdateCpu and MOSP-CUDA's sospUpdateGpu): a vertex's
 * parent changes only when the vertex is re-evaluated. update() re-evaluates the vertices of the
 * invalidated subtrees and the insertion heads, which take their best (distance, lowest parent
 * id) pair over all in-neighbours, and the out-neighbours x of every vertex a whose distance
 * decreased, which adopt (d(a) + w(a,x), a) if that pair is smaller than their own. Other
 * vertices keep their parents. Hence:
 *   - from a canonical tree (from compute(), or from_arrays() with canonicalize = true), update()
 *     equals compute() on the new graph exactly;
 *   - from a valid tree whose tie parents are not the lowest ids (from_arrays() with
 *     canonicalize = false, like MOSP's dataset trees), the distances equal compute()'s and the
 *     tree is a valid shortest-path tree, but kept tie parents can differ from compute()'s;
 *   - in the distance-only mode (stats::packed_parents == false: (n - 1) * max weight does not
 *     fit next to the parent ids in 64 bits) every parent is recovered with the lowest-id rule
 *     after the search, so update() equals compute() on every input.
 *
 * Backends: sequential (the reference), openmp (MOSP-OpenMP's sospUpdateCpu) and cuda (MOSP-CUDA's
 * persistent cooperative kernel, the fused engine; options::cuda_engine). On cuda the graph must
 * be built with (or cloned for) the CUDA resources, the result arrays live in device memory
 * (copy them with to_vector()), and compute() and update() synchronize the stream once. Near the
 * packing limit the two parallel engines choose the word format slightly differently (MOSP-CUDA
 * packs when (n - 1) * max weight fits, MOSP-OpenMP when one more edge fits too), so
 * stats::packed_parents may differ between cuda and the host backends there; the trees are
 * identical for canonical input trees.
 */

namespace dyng::sssp {

/**
 * @brief Options of compute() and update() (an aggregate; fields are only ever appended).
 * @ingroup sssp
 */
struct options {
  /// Near-far bucket width; 0 = automatic: max(1, 32 * average weight / average out-degree) of
  /// the graph before the batch (MOSP's defaultDelta). A tunable: it changes the schedule, never
  /// the result.
  std::int64_t delta = 0;
  /// Which weight column of a multi-weight graph is the edge length. Fixed at compute().
  int objective = 0;
  /// Engine of the CUDA backend; ignored by the host backends. engine::automatic and
  /// engine::fused run the fused persistent cooperative kernel (MOSP-CUDA's sospUpdateGpu) and throw
  /// not_supported_error on a device without cooperative launch (the operators engine that would be
  /// the fallback arrives in 0.2); engine::operators throws not_supported_error in this release.
  engine cuda_engine = engine::automatic;
  /// O(n) checks on imported trees in result::from_arrays() (rooted at the source, no parent
  /// cycle, distances in range).
  bool validate_inputs = true;
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 * @ingroup sssp
 */
struct stats : update_stats {
  /// Deterministic: what applying the batch did to the graph.
  apply_summary batch;
  /// Deterministic: vertices in invalidated subtrees (below deleted or weight-increased tree
  /// edges).
  std::int64_t invalidated = 0;
  /// Schedule-dependent: raises of the near-far threshold (0 on the sequential backend).
  std::int64_t epochs = 0;
  /// Schedule-dependent: vertex expansions of the Step 2 loop.
  std::int64_t pushes = 0;
  /// Deterministic per backend: false if distances do not fit next to the parent ids in 64-bit
  /// words, so the engine kept distances only and recovered every parent with the lowest-id rule
  /// after the search. The sequential engine reports the OpenMP engine's value and applies the same
  /// recovery, so both return the same tree; the CUDA engine uses MOSP-CUDA's slightly larger
  /// packing limit (see @ref sssp), so its value can differ right at the limit.
  bool packed_parents = true;
};

}  // namespace dyng::sssp

namespace dyng::detail {
template <typename vertex_t, typename distance_t>
struct sssp_state;
struct sssp_access;
}  // namespace dyng::detail

namespace dyng::sssp {

/**
 * @brief A shortest-path tree kept up to date by update() (opaque, move-only).
 *
 * The result owns its distance and parent arrays (host memory, or device memory for the cuda
 * backend), its options and the graph version it matches.
 * The scratch memory of the engines is not part of it: compute() and update() lease the
 * workspace of the resources handle they run with (sized once, then reused), so results computed
 * and updated through one handle, such as the K objectives of dyng::update_each(), share one
 * workspace as MOSP's objectives share one, and a steady-state update allocates no scratch memory
 * (ADR 0015). The results stay independent: nothing in the workspace carries over from one run to
 * the next.
 *
 * @tparam vertex_t   Vertex id type (int32_t or int64_t).
 * @tparam distance_t Distance type (int64_t).
 * @ingroup sssp
 */
template <typename vertex_t, typename distance_t = std::int64_t>
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
  ~result();                                  ///< releases the arrays

  /**
   * @brief The source vertex.
   * @return The source (invalid_id() for a moved-from result).
   */
  [[nodiscard]] vertex_t source() const noexcept;

  /**
   * @brief The distances.
   * @return One distance per vertex (host memory; device memory for a result of the cuda backend,
   *         see space()); infinite_distance<distance_t>() for vertices that cannot be reached.
   *         Valid until the next update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const distance_t> distances() const;

  /**
   * @brief The parents (the tree).
   * @return One parent per vertex (host memory; device memory for a result of the cuda backend):
   *         an in-neighbour on a shortest path, the lowest-id one unless kept from a non-canonical
   *         input tree (see the tie rule of @ref sssp); -1 for the source and unreachable vertices.
   *         Valid until the next update().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const vertex_t> parents() const;

  /**
   * @brief The options.
   * @return The options given at compute() or from_arrays(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the tunables (delta, cuda_engine, validate_inputs).
   * @param[in] opt The new options; `objective` must stay the same.
   * @throws invalid_argument_error if `opt.objective` differs or `opt.delta` is negative, or for a
   *         moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  void set_options(const options& opt);

  /**
   * @brief The graph version this result matches.
   * @return The version of the graph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of the arrays.
   * @return memory_space::device for a result of the cuda backend, memory_space::host otherwise.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy (arrays, options, version) for the resources `res`.
   *
   * The copy belongs to the backend of `res` (its arrays are copied between host and device memory
   * as needed): this is how a result moves between the host backends and cuda. Also sizes the
   * pooled workspace of `res` for the graph (a no-op if it is large enough), so the first update of
   * the copy through `res` allocates no scratch memory.
   * @param[in] res Execution resources of the copy.
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available for sssp.
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @sync
   */
  [[nodiscard]] result clone(const resources& res) const;

  /**
   * @brief Adopt an existing tree of `g` (e.g. read from MOSP's distance and tree files).
   *
   * With `canonicalize`, every parent is replaced by the lowest-id in-neighbour u with
   * dist[u] + w(u,v) == dist[v] (MOSP's canonicalizeTree()), so that later updates produce the
   * canonical tree (equal to compute()). Without it the parents are kept as given, as MOSP's
   * `mosp` driver keeps them; later updates then keep the tie parents of the vertices they do not
   * re-evaluate (the tie rule of @ref sssp), identically on every backend. With
   * `opt.validate_inputs` the tree is checked in O(n): array sizes, source at distance 0
   * without a parent, distances in [0, (n - 1) * max weight] or unreachable (>=
   * infinite_distance() / 2, stored as infinite_distance()), unreachable vertices without a
   * parent, every other vertex with a reachable parent, and no parent cycle (every chain ends at
   * the source); on the OpenMP backend the checks run in parallel and report the same first
   * problem as the sequential ones. The caller guarantees that the tree is a shortest-path tree
   * of `g`. Also sizes the pooled workspace of `res` for the graph (once for all results built
   * through `res`; ADR 0015). On the cuda backend the tree is imported and checked in host memory
   * and then uploaded (profiler stage sssp.upload); the arrays may be in host or device memory.
   *
   * @tparam edge_t   Edge offset type of the graph.
   * @tparam weight_t Weight type of the graph.
   * @param[in] res          Execution resources (the graph must belong to their backend).
   * @param[in] g            The graph the tree belongs to (with in-edges stored).
   * @param[in] source       The source vertex.
   * @param[in] distances    One distance per vertex (host memory; any memory for cuda).
   * @param[in] parents      One parent per vertex, -1 for none (host memory; any memory for cuda).
   * @param[in] canonicalize Apply the lowest-id tie rule to the parents (default true).
   * @param[in] opt          Options (objective = the weight column the tree belongs to).
   * @return The result, matching `g.version()`.
   * @throws invalid_argument_error if a check fails, the options are invalid, or `g` belongs to
   *         another backend than `res`.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @sync
   */
  template <typename edge_t, typename weight_t>
  [[nodiscard]] static result from_arrays(const resources& res,
                                          const graph<vertex_t, edge_t, weight_t>& g,
                                          vertex_t source, array_view<const distance_t> distances,
                                          array_view<const vertex_t> parents,
                                          bool canonicalize = true, const options& opt = {});

 private:
  friend struct detail::sssp_access;
  using state_type = detail::sssp_state<vertex_t, distance_t>;
  explicit result(std::unique_ptr<state_type> state) noexcept;

  std::unique_ptr<state_type> impl_;
};

/**
 * @brief Compute the canonical shortest-path tree of `g` from `source` (the static solve).
 *
 * The sequential backend runs the Step 2 loop of the sequential engine from the source; the
 * OpenMP backend runs the near-far search of MOSP-OpenMP's sospFromScratchCpu(), the cuda backend
 * MOSP-CUDA's sospFromScratchGpu() (the persistent kernel from the source). All return the
 * Dijkstra tree with lowest-id ties.
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type; the objective's weights must lie in [1, 2^31 - 1].
 * @param[in] res    Execution resources (sequential, openmp or cuda).
 * @param[in] g      The graph (with in-edges stored; built with, or cloned for, the backend of
 *                   `res`); it is not modified.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The result, matching `g.version()`.
 * @throws invalid_argument_error if the source or the objective is out of range, a weight of the
 *         objective is below 1, the graph stores no in-edges, distances could exceed 62 bits, or
 *         `g` belongs to another backend than `res`.
 * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
 *         of options::cuda_engine cannot run (engine::operators; no cooperative launch).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @sync On cuda the stream is synchronized once (the control block of the kernel is read).
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs: the Dijkstra tree with lowest-id ties.
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("sssp")`, keys dynamosp2025 and
 *        dynamosptpds2025 in docs/references.bib.
 * @ingroup sssp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res,
                                       const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                                       const options& opt = {});

/**
 * @brief Apply a batch of edge changes to `g` and update the shortest-path tree `r`.
 *
 * The batch is applied under `g.properties().semantics` (for graph_properties::mosp_compatible()
 * exactly as MOSP's applyChangeBatch()). Step 1 invalidates the subtrees below deleted or
 * weight-increased tree edges and pulls each invalidated vertex's and each insertion head's best
 * in-neighbour; Step 2 relaxes outward until no distance decreases. Postcondition: the distances
 * of `r` equal those of compute(res, g, r.source(), r.get_options()), and `r` is a shortest-path
 * tree of the new graph; if `r` was canonical before the call (it came from compute(), from
 * from_arrays() with canonicalize = true, or from updates of such a result) it equals that
 * compute() exactly (see the tie rule of @ref sssp for trees adopted with canonicalize = false).
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type; weights must lie in [1, 2^31 - 1].
 * @param[in]     res   Execution resources (sequential, openmp or cuda; `g` and `r` must belong to
 *                      their backend).
 * @param[in,out] g     The graph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Insertions (upserts), deletions and weight changes (host memory).
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update; `invalidated` and `affected` are deterministic, `iterations`,
 *         `epochs` and `pushes` are not.
 * @throws stale_result_error     if r.graph_version() != g.version(), `r` was computed on another
 *         graph (or on an earlier state of a graph variable that was reassigned since), or `r` was
 *         left unusable by a failed update.
 * @throws invalid_argument_error if a batch id or weight is invalid, or `g` or `r` belongs to
 *         another backend than `res` (nothing is changed); or, only for a tree imported without
 *         validation, if the tree has a parent cycle or (cuda) a distance outside the packing bound
 *         (then the graph was updated and `r` is left unusable).
 * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
 *         of the result's options::cuda_engine cannot run (nothing is changed).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @sync On cuda the stream is synchronized once per result (the kernel's control block).
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs (distances, parents, `invalidated`,
 *              `affected`), for canonical and non-canonical input trees alike.
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("sssp")`, keys dynamosp2025 and
 *        dynamosptpds2025 in docs/references.bib.
 * @ingroup sssp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r);

}  // namespace dyng::sssp

namespace dyng::detail {

/**
 * @brief The update participant of an sssp result (used by dyng::update()).
 * @tparam vertex_t   Vertex id type.
 * @tparam edge_t     Edge offset type.
 * @tparam weight_t   Weight type.
 * @tparam distance_t Distance type.
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_sssp_participant(
    sssp::result<vertex_t, distance_t>& r, sssp::stats& out);

/**
 * @brief update_traits of sssp::result (dyng::update() support).
 * @tparam vertex_t   Vertex id type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename distance_t>
struct update_traits<sssp::result<vertex_t, distance_t>> {
  using stats_type = sssp::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type.
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      sssp::result<vertex_t, distance_t>& r, sssp::stats& out) {
    return make_sssp_participant<vertex_t, typename container_t::edge_type,
                                 typename container_t::weight_type, distance_t>(r, out);
  }
};

}  // namespace dyng::detail
