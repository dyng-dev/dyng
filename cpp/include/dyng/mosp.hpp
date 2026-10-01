// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp.hpp
 * @brief Dynamic multi-objective shortest paths: compute() and update().
 * @ingroup mosp
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
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

/**
 * @defgroup mosp mosp
 * @brief Dynamic multi-objective shortest paths (the MOSP update of DynaMOSP).
 *
 * A graph with K weight columns (objectives) and a source. The result holds, for every objective
 * k, the canonical shortest-path tree T_k (an sssp result), and the MOSP tree: the shortest-path
 * tree of the *combined graph*, whose edges are the edges of the K trees, an edge (p, v) weighted
 * L * (K + 1) - sum over the trees T_i containing it of L / Pref_i (thesis Chapter 4, Algorithm
 * MOSP_Update, Step 2; Pref_i >= 1 is the preference of objective i, a lower value a higher
 * priority, and L = lcm(Pref) keeps the weights integral, so combined distances are in units of
 * 1/L; with the default Pref = (1, ..., 1) an edge in m trees weighs K + 1 - m). Along the MOSP
 * tree, the path costs give the K objective values of the path to every vertex (Step 3).
 *
 * update() applies the batch once and updates the K trees incrementally (K sssp updates over the
 * objective views of the graph, one after the other on one shared workspace: dyng::update_each()
 * semantics), then rebuilds the combined graph from the K trees (count, scan, fill), solves it
 * from scratch with sssp's engine and recomputes the path costs. Every tree is canonical (lowest
 * parent id among equal distances), so every backend returns the same bytes.
 *
 * Backends: sequential, openmp (MOSP-OpenMP's mospUpdate and combinedGraphSospCpu) and cuda
 * (MOSP-CUDA's mospUpdate and combinedGraphSospGpu; options::cuda_engine chooses sssp's fused or
 * operators engine for the K updates and the combined solve). On cuda the trees and the combined
 * arrays live in device memory; the path costs are computed on the host on every backend in this
 * release, so path_costs() is host memory everywhere.
 */

namespace dyng::mosp {

/**
 * @brief The largest number of objectives (weight columns used); MOSP's originals allow 32.
 * @ingroup mosp
 */
inline constexpr int max_objectives = 64;

/**
 * @brief The largest preference scale L = lcm(Pref) (MOSP's preferenceScale limit, 2^20).
 * @ingroup mosp
 */
inline constexpr std::int64_t max_preference_scale = std::int64_t{1} << 20;

/**
 * @brief Options of compute() and update() (an aggregate; fields are only ever appended).
 * @ingroup mosp
 */
struct options {
  /// One preference per objective, each >= 1 (a lower value is a higher priority); empty = all
  /// ones. lcm(preferences) must not exceed max_preference_scale. Fixed at compute().
  std::vector<std::int32_t> preferences;
  /// Near-far bucket width of the K sssp updates; 0 = automatic per objective (sssp's rule). The
  /// combined solve always uses the automatic width of the combined graph, as the originals do. A
  /// tunable: it changes the schedule, never the result.
  std::int64_t delta = 0;
  /// Engine of the CUDA backend for the K sssp updates and the combined solve (see
  /// sssp::options::cuda_engine); ignored by the host backends. A tunable.
  engine cuda_engine = engine::automatic;
  /// Compute the path costs (Step 3's last line) in compute() and update(). A tunable; while it is
  /// false, path_costs() throws.
  bool compute_path_costs = true;
  /// O(K n) checks on imported trees in result::from_arrays() (sssp's checks, per objective).
  bool validate_inputs = true;
  /// The objectives: the first num_objectives weight columns of the graph; 0 = every column
  /// (MOSP's MospOptions::numberOfObjectives). Fixed at compute().
  int num_objectives = 0;
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 * @ingroup mosp
 */
struct stats : update_stats {
  /// Deterministic: what applying the batch did to the graph.
  apply_summary batch;
  /// The K sssp updates, in objective order (their deterministic counters as in sssp).
  std::vector<sssp::stats> objectives;
  /// Deterministic: edges of the combined graph.
  std::int64_t combined_edges = 0;
  /// Deterministic: L = lcm(preferences), the unit of the combined distances is 1/L.
  std::int64_t preference_scale = 0;
};

}  // namespace dyng::mosp

namespace dyng::detail {
template <typename vertex_t, typename distance_t>
struct mosp_state;
struct mosp_access;
}  // namespace dyng::detail

namespace dyng::mosp {

/**
 * @brief The K shortest-path trees, the MOSP tree and the path costs kept up to date by update()
 *        (opaque, move-only).
 *
 * The result owns its arrays (host memory, or device memory for the cuda backend, except the path
 * costs, which are host memory on every backend), its options and the graph version it matches.
 * Like sssp results, it leases the engines' scratch memory from its resources handle (ADR 0015).
 *
 * **Thread safety.** As sssp::result: accessors may run concurrently with each other, not with an
 * update() of the result.
 *
 * @tparam vertex_t   Vertex id type (int32_t or int64_t).
 * @tparam distance_t Distance type: std::int64_t (as sssp::result).
 * @ingroup mosp
 */
template <typename vertex_t, typename distance_t = std::int64_t>
class result {
  static_assert(std::is_same_v<distance_t, std::int64_t>,
                "dyng::mosp::result: distance_t is std::int64_t (as sssp::result)");
  static_assert(std::is_same_v<vertex_t, std::int32_t> || std::is_same_v<vertex_t, std::int64_t>,
                "dyng::mosp::result: vertex_t is std::int32_t or std::int64_t");

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
   * @brief The number of objectives K.
   * @return K (0 for a moved-from result).
   */
  [[nodiscard]] int num_objectives() const noexcept;

  /**
   * @brief The distances of one objective's tree.
   * @param[in] objective The objective, in [0, num_objectives()).
   * @return One distance per vertex (infinite_distance<distance_t>() if unreachable); device
   *         memory for a result of the cuda backend. Valid until the next update().
   * @throws invalid_argument_error if `objective` is out of range, or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const distance_t> distances(int objective) const;

  /**
   * @brief The parents of one objective's tree.
   * @param[in] objective The objective, in [0, num_objectives()).
   * @return One parent per vertex, -1 for the source and unreachable vertices; device memory for
   *         a result of the cuda backend. Valid until the next update().
   * @throws invalid_argument_error if `objective` is out of range, or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const vertex_t> parents(int objective) const;

  /**
   * @brief The distances in the combined graph.
   * @return One distance per vertex, in units of 1/L (L = preference_scale());
   *         infinite_distance<distance_t>() if unreachable; device memory for the cuda backend.
   *         Valid until the next update().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const distance_t> combined_distances() const;

  /**
   * @brief The MOSP tree: the canonical shortest-path tree of the combined graph.
   * @return One parent per vertex, -1 for the source and unreachable vertices; device memory for
   *         the cuda backend. Valid until the next update().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const vertex_t> combined_parents() const;

  /**
   * @brief The path costs: the K objective values of the MOSP path to every vertex (MOSP's
   *        mospPathCosts: the weights of the tree edge (p, v) are those of the first edge from p
   *        to v in the graph's row order).
   * @return n * K values, vertex-major (costs[v * K + k]); infinite_distance<distance_t>() for
   *         unreachable vertices. Host memory on every backend. Valid until the next update().
   * @throws invalid_argument_error if options::compute_path_costs was false at the last
   *         compute() or update() (the costs were not computed), or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const distance_t> path_costs() const;

  /**
   * @brief The preference scale L = lcm(preferences) (the unit of the combined distances is 1/L).
   * @return L (0 for a moved-from result).
   */
  [[nodiscard]] std::int64_t preference_scale() const noexcept;

  /**
   * @brief The options.
   * @return The options given at compute() or from_arrays(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the tunables (delta, cuda_engine, compute_path_costs, validate_inputs).
   * @param[in] opt The new options; preferences and num_objectives must stay the same.
   * @throws invalid_argument_error if a fixed option differs or `opt.delta` is negative, or for a
   *         moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @guarantee Strong: every check runs before the options change.
   */
  void set_options(const options& opt);

  /**
   * @brief The graph version this result matches.
   * @return The version of the graph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of the trees and the combined arrays.
   * @return memory_space::device for a result of the cuda backend, memory_space::host otherwise
   *         (the path costs are host memory on every backend).
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy (every array, the options, the version) for the resources `res`.
   * @param[in] res Execution resources of the copy (any backend of mosp: the arrays are copied
   *                between host and device memory as needed).
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not built.
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @sync
   */
  [[nodiscard]] result clone(const resources& res) const;

  /**
   * @brief Adopt K existing trees of `g` (e.g. MOSP's obj<k>/distancesOriginal.txt and
   *        SSSPTreeOriginal.txt) and build the combined graph, the MOSP tree and the path costs.
   *
   * Every tree is imported as sssp::result::from_arrays() imports it (with `canonicalize` and
   * `opt.validate_inputs` applied to each); then the combined graph is built from the K trees and
   * solved, as at the end of an update().
   *
   * @tparam edge_t   Edge offset type of the graph.
   * @tparam weight_t Weight type of the graph.
   * @param[in] res          Execution resources (the graph must belong to their backend).
   * @param[in] g            The graph the trees belong to (with in-edges stored).
   * @param[in] source       The source vertex.
   * @param[in] distances    K views of n distances (host memory: the list; any memory space: the
   *                         arrays).
   * @param[in] parents      K views of n parents, -1 for none.
   * @param[in] canonicalize Apply the lowest-id tie rule to every tree (default true).
   * @param[in] opt          Options; K = the number of views, which must equal the objectives the
   *                         options select (opt.num_objectives, or every weight column).
   * @return The result, matching `g.version()`.
   * @throws invalid_argument_error if a check fails (the sssp checks of a tree, K, the options), or
   *         `g` belongs to another backend than `res`.
   * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
   *         of opt.cuda_engine cannot run.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   * @guarantee Strong: `g` and the arrays are not modified, and nothing is kept if the call
   *            throws.
   */
  template <typename edge_t, typename weight_t>
  [[nodiscard]] static result from_arrays(const resources& res,
                                          const graph<vertex_t, edge_t, weight_t>& g,
                                          vertex_t source,
                                          array_view<const array_view<const distance_t>> distances,
                                          array_view<const array_view<const vertex_t>> parents,
                                          bool canonicalize = true, const options& opt = {});

 private:
  friend struct detail::mosp_access;
  using state_type = detail::mosp_state<vertex_t, distance_t>;
  explicit result(std::unique_ptr<state_type> state) noexcept;

  std::unique_ptr<state_type> impl_;
};

}  // namespace dyng::mosp

namespace dyng::detail {

/**
 * @brief Whether mosp is instantiated for graph<vertex_t, edge_t, weight_t> (sssp's types).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
inline constexpr bool mosp_supported_v = sssp_supported_v<vertex_t, edge_t, weight_t>;

/**
 * @brief The message of mosp's static_assert for an unsupported graph type.
 * @ingroup mosp
 */
#define DYNG_MOSP_TYPES_MESSAGE                                                                  \
  "dyng::mosp supports graph<int32_t, int32_t or int64_t, int32_t> and graph<int64_t, int64_t, " \
  "int32_t> only (integer weights in [1, 2^31 - 1]); see 'Graph requirements' in "               \
  "docs/algorithms/mosp.md"

/**
 * @brief mosp::compute() for a supported graph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] res    Execution resources.
 * @param[in] g      The graph.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The result.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] mosp::result<vertex_t> mosp_compute(const resources& res,
                                                  const graph<vertex_t, edge_t, weight_t>& g,
                                                  vertex_t source, const mosp::options& opt);

/**
 * @brief mosp::update() for a supported graph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res   Execution resources.
 * @param[in,out] g     The graph.
 * @param[in]     batch The batch.
 * @param[in,out] r     The result.
 * @return The update's counters.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
mosp::stats mosp_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                        const edge_batch_view<vertex_t, weight_t>& batch,
                        mosp::result<vertex_t>& r);

}  // namespace dyng::detail

namespace dyng::mosp {

/**
 * @brief Compute the K canonical shortest-path trees of `g` from `source`, the combined graph,
 *        its shortest-path tree (the MOSP tree) and the path costs (the static solve).
 *
 * The K trees are sssp::compute() of the objectives; the combined graph is built from their
 * parents and solved with the same engine (the original's Steps 2-3).
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type; every objective's weights must lie in [1, 2^31 - 1].
 * @param[in] res    Execution resources (sequential, openmp or cuda).
 * @param[in] g      The graph (with in-edges stored, at least one weight column; built with, or
 *                   cloned for, the backend of `res`); it is not modified.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The result, matching `g.version()`.
 * @throws invalid_argument_error if the source is out of range, the options are invalid (a
 *         preference below 1, as many preferences as objectives, lcm above
 *         max_preference_scale, num_objectives out of [0, num_weights], more than max_objectives
 *         objectives, a negative delta), a weight is below 1, the graph stores no in-edges or has
 *         no weight column, or `g` belongs to another backend than `res`.
 * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
 *         of options::cuda_engine cannot run.
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error (cuda backend).
 * @sync
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs (every tree canonical, lowest-id ties).
 * @guarantee Strong: `g` is not modified, and nothing is kept if the call throws.
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("mosp")`, keys dynamosp2025 and
 *        dynamosptpds2025 in docs/references.bib.
 * @ingroup mosp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res,
                                       const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                                       const options& opt = {}) {
  static_assert(detail::mosp_supported_v<vertex_t, edge_t, weight_t>, DYNG_MOSP_TYPES_MESSAGE);
  return detail::mosp_compute(res, g, source, opt);
}

/**
 * @brief Apply a batch of edge changes to `g` and update the K trees, the MOSP tree and the path
 *        costs of `r`.
 *
 * The batch is applied once (under `g.properties().semantics`; graph_properties::
 * mosp_compatible() reproduces MOSP's applyChangeBatch()). The K objectives are updated by sssp
 * one after the other (stage mosp.objective around each, its sssp.* stages inside), then the
 * combined graph is rebuilt (mosp.combine), solved (mosp.combined_sssp) and the path costs are
 * recomputed (mosp.path_costs). Postcondition: `r` equals compute(res, g, r.source(),
 * r.get_options()) exactly if its trees were canonical (from compute(), or from_arrays() with
 * canonicalize = true); otherwise as sssp's tie rule describes for each tree, and the MOSP tree is
 * the canonical tree of the combined graph of the K updated trees.
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type.
 * @param[in]     res   Execution resources (`g` and `r` must belong to their backend).
 * @param[in,out] g     The graph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Insertions (with the graph's number of weights each), deletions and weight
 *                      changes (any memory space; read on the host in this release).
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update; `affected` counts the vertices whose combined distance or MOSP
 *         parent changed (deterministic).
 * @throws stale_result_error     if r.graph_version() != g.version(), `r` was computed on another
 *         graph, or `r` was left unusable by a failed update.
 * @throws invalid_argument_error if a batch id or weight is invalid or the batch has another
 *         number of weights than the graph (nothing is changed); or, for trees imported without
 *         validation, as sssp::update() (then the graph was updated and `r` is left unusable).
 * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
 *         cannot run (nothing is changed).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error (after the batch was
 *         applied, `r` is left unusable).
 * @sync
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs (trees, combined arrays, path costs,
 *              `affected`, `combined_edges` and the objectives' deterministic counters).
 * @guarantee Strong for every error found before the batch is applied; basic after the commit
 *            (`g` holds the new version and `r` is poisoned until it is recomputed).
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("mosp")`.
 * @ingroup mosp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r) {
  static_assert(detail::mosp_supported_v<vertex_t, edge_t, weight_t>, DYNG_MOSP_TYPES_MESSAGE);
  return detail::mosp_update(res, g, batch, r);
}

}  // namespace dyng::mosp

namespace dyng::detail {

/**
 * @brief The update participant of an mosp result (used by dyng::update()).
 * @tparam vertex_t   Vertex id type.
 * @tparam edge_t     Edge offset type.
 * @tparam weight_t   Weight type.
 * @tparam distance_t Distance type.
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_mosp_participant(
    mosp::result<vertex_t, distance_t>& r, mosp::stats& out);

/**
 * @brief update_traits of mosp::result (dyng::update() support).
 * @tparam vertex_t   Vertex id type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename distance_t>
struct update_traits<mosp::result<vertex_t, distance_t>> {
  using stats_type = mosp::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type.
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      mosp::result<vertex_t, distance_t>& r, mosp::stats& out) {
    static_assert(
        mosp_supported_v<typename container_t::vertex_type, typename container_t::edge_type,
                         typename container_t::weight_type>,
        DYNG_MOSP_TYPES_MESSAGE);
    return make_mosp_participant<vertex_t, typename container_t::edge_type,
                                 typename container_t::weight_type, distance_t>(r, out);
  }
};

}  // namespace dyng::detail
