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
 * relaxes outward until no distance decreases. The result always equals compute() on the new
 * graph, bit for bit, on every backend.
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
  /// Engine of the CUDA backend (fused, operators or automatic); ignored by the CPU backends.
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
  /// Deterministic: false if distances did not fit next to the parent ids in 64-bit words and the
  /// distance-only fallback recovered the parents after the search (OpenMP backend; always true on
  /// the sequential backend, which needs no packing).
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
 * The result owns its distance and parent arrays, its options, the graph version it matches and
 * the reusable workspace of the incremental engines (reserved once).
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
  ~result();                                  ///< releases the arrays and the workspace

  /**
   * @brief The source vertex.
   * @return The source (invalid_id() for a moved-from result).
   */
  [[nodiscard]] vertex_t source() const noexcept;

  /**
   * @brief The distances.
   * @return One distance per vertex (host memory); infinite_distance<distance_t>() for vertices
   *         that cannot be reached. Valid until the next update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   */
  [[nodiscard]] array_view<const distance_t> distances() const;

  /**
   * @brief The parents (the tree).
   * @return One parent per vertex (host memory): the lowest-id in-neighbour on a shortest path;
   *         -1 for the source and unreachable vertices. Valid until the next update().
   * @throws invalid_argument_error for a moved-from result.
   */
  [[nodiscard]] array_view<const vertex_t> parents() const;

  /**
   * @brief The options.
   * @return The options given at compute() or from_arrays(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the tunables (delta, cuda_engine, validate_inputs).
   * @param[in] opt The new options; `objective` must stay the same.
   * @throws invalid_argument_error if `opt.objective` differs or `opt.delta` is negative.
   */
  void set_options(const options& opt);

  /**
   * @brief The graph version this result matches.
   * @return The version of the graph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of the arrays.
   * @return memory_space::host in this release.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy (arrays, options, version; a fresh workspace).
   * @param[in] res Execution resources of the copy.
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @sync
   */
  [[nodiscard]] result clone(const resources& res) const;

  /**
   * @brief Adopt an existing tree of `g` (e.g. read from MOSP's distance and tree files).
   *
   * With `canonicalize`, every parent is replaced by the lowest-id in-neighbour u with
   * dist[u] + w(u,v) == dist[v] (MOSP's canonicalizeTree()), so that later updates produce the
   * canonical tree. With `opt.validate_inputs` the tree is checked in O(n): array sizes, source
   * at distance 0 without a parent, distances in [0, (n - 1) * max weight] or unreachable
   * (>= infinite_distance() / 2, stored as infinite_distance()), unreachable vertices without a
   * parent, every other vertex with a reachable parent, and no parent cycle (every chain ends at
   * the source). The caller guarantees that the tree is a shortest-path tree of `g`.
   *
   * @tparam edge_t   Edge offset type of the graph.
   * @tparam weight_t Weight type of the graph.
   * @param[in] res          Execution resources (a host backend).
   * @param[in] g            The graph the tree belongs to (with in-edges stored).
   * @param[in] source       The source vertex.
   * @param[in] distances    One distance per vertex (host memory).
   * @param[in] parents      One parent per vertex, -1 for none (host memory).
   * @param[in] canonicalize Apply the lowest-id tie rule to the parents.
   * @param[in] opt          Options (objective = the weight column the tree belongs to).
   * @return The result, matching `g.version()`.
   * @throws invalid_argument_error if a check fails or the options are invalid.
   * @throws not_supported_error    for a device backend.
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
 * OpenMP backend runs the near-far search of MOSP-OpenMP's sospFromScratchCpu(). Both return the
 * Dijkstra tree with lowest-id ties.
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type; the objective's weights must lie in [1, 2^31 - 1].
 * @param[in] res    Execution resources (sequential or openmp).
 * @param[in] g      The graph (with in-edges stored); it is not modified.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The result, matching `g.version()`.
 * @throws invalid_argument_error if the source or the objective is out of range, a weight of the
 *         objective is below 1, the graph stores no in-edges, or distances could exceed 62 bits.
 * @throws not_supported_error    if the backend of `res` is not available for sssp (cuda before
 *         M1b).
 * @sync
 * @backends sequential, openmp
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
 * in-neighbour; Step 2 relaxes outward until no distance decreases. Postcondition: `r` equals
 * compute(res, g, r.source(), r.get_options()) exactly.
 *
 * @tparam vertex_t Vertex id type (int32_t or int64_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Integer weight type; weights must lie in [1, 2^31 - 1].
 * @param[in]     res   Execution resources (sequential or openmp).
 * @param[in,out] g     The graph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Insertions (upserts), deletions and weight changes (host memory).
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update; `invalidated` and `affected` are deterministic, `iterations`,
 *         `epochs` and `pushes` are not.
 * @throws stale_result_error     if r.graph_version() != g.version(), `r` was computed on another
 *         graph (or on an earlier state of a graph variable that was reassigned since), or `r` was
 *         left unusable by a failed update.
 * @throws invalid_argument_error if a batch id or weight is invalid (nothing is changed); or, only
 *         for a tree imported without validation, if the tree has a parent cycle (then the graph
 *         was updated and `r` is left unusable).
 * @throws not_supported_error    if the backend of `res` is not available for sssp.
 * @sync
 * @backends sequential, openmp
 * @determinism Bit-exact across backends: equals compute(res, g, r.source()) with lowest-id ties.
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
