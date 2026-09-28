// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count.hpp
 * @brief Exact k-bounded directed simple-cycle histograms: compute() and update().
 * @ingroup cycle_count
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>

/**
 * @defgroup cycle_count cycle_count
 * @brief Directed simple cycles by length, counted exactly and kept up to date under edge
 *        batches (TruCy / DynTruCy; the kappa-truncated search of the paper is not implemented).
 *
 * compute() counts every directed simple cycle of length 2..max_length of a graph (each cycle
 * once, from its smallest vertex; a 2-cycle is a pair of opposite edges; self-loops lie on no
 * cycle). update() applies a batch of edge insertions and deletions to the graph and changes the
 * histogram by the cycles the batch destroys and creates, without recounting the rest:
 *
 *   - Step 0 reduces the batch to its net structural change: two lists of edges, the deletions of
 *     existing edges and the insertions of new ones, each sorted by (source, destination) without
 *     repeats (CycleEnumeration-GPU's prepare_batch; under batch_semantics::set() exactly the
 *     normalized batch of graph::apply()). The position of an edge in its list is its id.
 *   - before_apply (stage cycle_count.count_minus, on G_t): every cycle of length <= max_length
 *     through a deleted edge is subtracted once, by the deleted edge with the smallest id on it.
 *   - commit: the batch is applied once (graph.apply).
 *   - identify_affected (cycle_count.identify_affected): the inserted edges and their ownership
 *     index.
 *   - count_plus (cycle_count.count_plus, on G_{t+1}): every cycle through an inserted edge is
 *     added once, by the inserted edge with the smallest id on it.
 *   - finalize (cycle_count.finalize): the signed delta is applied to the histogram; a bucket that
 *     would become negative throws internal_error.
 *
 * Graph requirements: rows sorted by neighbour id and no parallel edges (row_order::sorted and
 * multi_edges::forbid, e.g. graph_properties::cycle_enum_compatible() or the default properties);
 * the in-edges are not needed. Weights are ignored. Every batch_semantics is accepted (Step 0
 * reduces a batch to the change of the edge set, so weight-only upserts are no-ops); parity with
 * CycleEnumeration-GPU is defined under batch_semantics::set().
 *
 * Backends: sequential (CycleEnumeration-GPU's sequential Johnson and update_static_histogram()),
 * openmp (its root-parallel OpenMP counter and update_static_histogram_openmp()). The histograms
 * are host arrays on every backend.
 *
 * Types: int32_t vertex ids, int32_t or int64_t edge offsets, unweighted or int32_t weights (other
 * graph types fail to compile with a static_assert naming these).
 */

namespace dyng::cycle_count {

/**
 * @brief The search of compute().
 * @ingroup cycle_count
 */
enum class search_method : std::uint8_t {
  johnson,  ///< Johnson's circuit search from each root (the smallest vertex of its cycles)
};

/**
 * @brief Which cycles are counted.
 * @ingroup cycle_count
 */
enum class cycle_mode : std::uint8_t {
  simple,  ///< static directed simple cycles (time-window and temporal cycles follow in 0.4)
};

/**
 * @brief Options of compute() (an aggregate; fields are only ever appended). Fixed at compute().
 * @ingroup cycle_count
 */
struct options {
  /// Longest counted cycle, >= 2; or -1 for no bound (every simple cycle, however long; the
  /// sequential backend then runs Johnson's algorithm with blocked lists, the OpenMP backend a
  /// plain path search, as the original). update() keeps the bound of compute().
  int max_length = -1;
  search_method method = search_method::johnson;  ///< the search (Johnson)
  cycle_mode mode = cycle_mode::simple;           ///< the cycles counted (simple)
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 *
 * Every counter is deterministic for cycle_count (identical on every backend, thread count and
 * run), including the two that update_stats calls schedule-dependent, because the update has no
 * iterative loop. The inherited fields mean:
 *   - `affected`: the number of lengths whose count changed (deterministic);
 *   - `iterations`: always 0 (no Step 2 loop; the searches are one pass per change edge);
 *   - `frontier_visits`: the change edges searched, `deletions` + `insertions` (deterministic);
 *   - `fallback_used`: always false (the update never recomputes);
 *   - `converged`: always true (no iteration cap);
 *   - `engine_used`: always engine::operators (the hooks of the template, one per stage).
 * @ingroup cycle_count
 */
struct stats : update_stats {
  /// What applying the batch did to the graph.
  apply_summary batch;
  /// Edges of the net structural change that were deleted (the change edges of count_minus).
  std::int64_t deletions = 0;
  /// Edges of the net structural change that were inserted (the change edges of count_plus).
  std::int64_t insertions = 0;
  /// Cycles of length <= the bound that the batch destroyed (subtracted on G_t).
  std::uint64_t cycles_removed = 0;
  /// Cycles of length <= the bound that the batch created (added on G_{t+1}).
  std::uint64_t cycles_added = 0;
};

}  // namespace dyng::cycle_count

namespace dyng::detail {
struct cycle_count_state;
struct cycle_count_access;

/**
 * @brief Whether cycle_count is instantiated for graph<vertex_t, edge_t, weight_t>: int32_t
 *        vertex ids (the ownership table keys two 32-bit ids), int32_t or int64_t offsets,
 *        unweighted or int32_t weights.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
inline constexpr bool cycle_count_supported_v =
    std::is_same_v<vertex_t, std::int32_t> &&
    (std::is_same_v<edge_t, std::int32_t> || std::is_same_v<edge_t, std::int64_t>) &&
    (is_unweighted_v<weight_t> || std::is_same_v<weight_t, std::int32_t>);

/// The message of the static_assert of an unsupported graph type.
#define DYNG_CYCLE_COUNT_TYPES_MESSAGE                                                          \
  "dyng::cycle_count supports graph<int32_t, int32_t or int64_t, unweighted or int32_t> only "  \
  "(int32_t vertex ids: the ownership table keys two 32-bit ids); see 'Graph requirements' in " \
  "docs/algorithms/cycle_count.md"
}  // namespace dyng::detail

namespace dyng::cycle_count {

/**
 * @brief A cycle histogram kept up to date by update() (opaque, move-only).
 *
 * counts()[len] is the number of directed simple cycles of length `len`; entries 0 and 1 are
 * always 0. With a bound (options::max_length = k) the array has k + 1 entries; without one it has
 * max(num_vertices, 2) + 1 entries (no simple cycle is longer than the vertex count), and it grows
 * with the graph. Counts are 64-bit; a count that would exceed 2^64 - 1 throws capacity_error.
 * The scratch memory of the engines is leased from the resources handle (ADR 0015), not owned.
 * @ingroup cycle_count
 */
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
  ~result();                                  ///< releases the histogram

  /**
   * @brief The histogram.
   * @return counts[len] = cycles of length len (host memory), size bound() + 1. Valid until the
   *         next update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const std::uint64_t> counts() const;

  /**
   * @brief The number of cycles of one length.
   * @param[in] length A cycle length (any value; lengths outside [2, bound()] give 0).
   * @return The count.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] std::uint64_t count(std::int64_t length) const;

  /**
   * @brief The number of cycles of every counted length.
   * @return The sum of counts().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws capacity_error         if the sum exceeds 2^64 - 1.
   */
  [[nodiscard]] std::uint64_t total() const;

  /**
   * @brief The longest length the histogram covers.
   * @return options::max_length, or without a bound max(num_vertices, 2) of the graph the result
   *         matches.
   * @throws invalid_argument_error for a moved-from result.
   */
  [[nodiscard]] std::int64_t bound() const;

  /**
   * @brief The options.
   * @return The options given at compute().
   * @throws invalid_argument_error for a moved-from result.
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief The graph version this result matches.
   * @return The version of the graph after the last compute() / update() of this result.
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept;

  /**
   * @brief The memory space of the histogram.
   * @return memory_space::host (on every backend).
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy for the resources `res` (the histogram is a host array on every backend).
   * @param[in] res Execution resources of the copy (sequential or openmp).
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available for cycle_count.
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @sync The copy is complete on return (host memory on every backend).
   */
  [[nodiscard]] result clone(const resources& res) const;

 private:
  friend struct detail::cycle_count_access;
  explicit result(std::unique_ptr<detail::cycle_count_state> state) noexcept;

  std::unique_ptr<detail::cycle_count_state> impl_;
};

}  // namespace dyng::cycle_count

namespace dyng::detail {

/**
 * @brief The implementation of cycle_count::compute() (instantiated for the supported types).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] res Execution resources.
 * @param[in] g   The graph.
 * @param[in] opt Options.
 * @return The histogram.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] cycle_count::result cycle_count_compute(const resources& res,
                                                      const graph<vertex_t, edge_t, weight_t>& g,
                                                      const cycle_count::options& opt);

/**
 * @brief The implementation of cycle_count::update() (instantiated for the supported types).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res   Execution resources.
 * @param[in,out] g     The graph.
 * @param[in]     batch The batch.
 * @param[in,out] r     The result.
 * @return Counters of the update.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
cycle_count::stats cycle_count_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                                      const edge_batch_view<vertex_t, weight_t>& batch,
                                      cycle_count::result& r);

}  // namespace dyng::detail

namespace dyng::cycle_count {

/**
 * @brief Count the directed simple cycles of `g` by length (the static solve).
 *
 * Every cycle is counted once, from its smallest vertex as the root: the search from a root
 * extends paths through larger vertices only and counts a cycle whenever an edge closes back to
 * the root. The sequential backend is CycleEnumeration-GPU's sequential Johnson (path-membership
 * blocking with a bound, Johnson's blocked lists without one), the OpenMP backend its OpenMP
 * counter (roots in parallel, dynamic schedule, one histogram per thread).
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Weight type (unweighted or int32_t; weights are ignored).
 * @param[in] res Execution resources (sequential or openmp).
 * @param[in] g   The graph (sorted rows, no parallel edges); it is not modified.
 * @param[in] opt Options.
 * @return The histogram, matching `g.version()`.
 * @throws invalid_argument_error if `g` does not have sorted rows (row_order::sorted) or allows
 *         parallel edges (multi_edges::allow), options::max_length is neither -1 nor >= 2, or `g`
 *         belongs to another backend than `res`.
 * @throws not_supported_error    if the backend of `res` is not available for cycle_count (cuda
 *         arrives with M2b).
 * @throws capacity_error         if a count exceeds 2^64 - 1.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync The histogram is complete on return (host backends only in this release).
 * @backends sequential, openmp
 * @determinism Exact values: identical histograms on every backend and thread count.
 * @paper TruCy / DynTruCy (submitted to IEEE Transactions on Computers):
 *        `dyng::citation("cycle_count")`, key trucy2026 in docs/references.bib. Exact counts; the
 *        kappa-truncated search of the paper is not implemented.
 * @ingroup cycle_count
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] result compute(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                             const options& opt = {}) {
  static_assert(detail::cycle_count_supported_v<vertex_t, edge_t, weight_t>,
                DYNG_CYCLE_COUNT_TYPES_MESSAGE);
  return detail::cycle_count_compute(res, g, opt);
}

/**
 * @brief Apply a batch of edge changes to `g` and update the histogram `r`.
 *
 * The batch is reduced to its net structural change (see @ref cycle_count), the cycles through the
 * deleted edges are subtracted on the graph before the batch, the batch is applied once, and the
 * cycles through the inserted edges are added on the graph after it; each affected cycle is
 * attributed to exactly one change edge (the smallest id on it among its phase's changes).
 * Postcondition: `r` equals compute(res, g, r.get_options()) exactly. Without a bound the update
 * enumerates every cycle through the change edges, which can take exponential time.
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam edge_t   Edge offset type (int32_t or int64_t).
 * @tparam weight_t Weight type (unweighted or int32_t).
 * @param[in]     res   Execution resources (sequential or openmp).
 * @param[in,out] g     The graph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Insertions and deletions (host memory).
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update (all deterministic).
 * @throws stale_result_error     if r.graph_version() != g.version(), `r` was computed on another
 *         graph, or `r` was left unusable by a failed update.
 * @throws invalid_argument_error if the batch is invalid for the graph's batch_semantics, the
 *         graph does not have sorted rows or allows parallel edges, or `g` belongs to another
 *         backend than `res` (nothing is changed).
 * @throws not_supported_error    if the backend of `res` is not available for cycle_count
 *         (nothing is changed).
 * @throws internal_error         if a bucket would become negative (then the graph was updated and
 *         `r` is left unusable).
 * @throws capacity_error         if a count exceeds 2^64 - 1 (raised after the batch was applied,
 *         it leaves `r` unusable).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync The graph and the histogram are updated on return (host backends only in this release).
 * @backends sequential, openmp
 * @determinism Exact values: identical histograms and counters on every backend and thread count.
 * @paper TruCy / DynTruCy (submitted to IEEE Transactions on Computers):
 *        `dyng::citation("cycle_count")`, key trucy2026 in docs/references.bib.
 * @ingroup cycle_count
 */
template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result& r) {
  static_assert(detail::cycle_count_supported_v<vertex_t, edge_t, weight_t>,
                DYNG_CYCLE_COUNT_TYPES_MESSAGE);
  return detail::cycle_count_update(res, g, batch, r);
}

}  // namespace dyng::cycle_count

namespace dyng::detail {

/**
 * @brief The update participant of a cycle_count result (used by dyng::update()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_cycle_count_participant(
    cycle_count::result& r, cycle_count::stats& out);

/**
 * @brief update_traits of cycle_count::result (dyng::update() support).
 */
template <>
struct update_traits<cycle_count::result> {
  using stats_type = cycle_count::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type.
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      cycle_count::result& r, cycle_count::stats& out) {
    static_assert(
        cycle_count_supported_v<typename container_t::vertex_type, typename container_t::edge_type,
                                typename container_t::weight_type>,
        DYNG_CYCLE_COUNT_TYPES_MESSAGE);
    return make_cycle_count_participant<typename container_t::vertex_type,
                                        typename container_t::edge_type,
                                        typename container_t::weight_type>(r, out);
  }
};

}  // namespace dyng::detail
