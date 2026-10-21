// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count.hpp
 * @brief Hyperedge triad counts (the 30 h-motifs and StatHyper's triad types 1-3) of a hypergraph,
 *        kept exact under hyperedge batches: compute() and update().
 * @ingroup triad_count
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

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

/**
 * @defgroup triad_count triad_count
 * @brief Triads of hyperedges by how they overlap (h-motifs, StatHyper's types), counted exactly
 *        and kept up to date under hyperedge batches (ESCHER, ESCHER+).
 *
 * A **triad** is a set of three live hyperedges {a, b, c} that form a triangle of the line graph:
 * every two of them share a vertex. Its **h-motif** is the pattern of the seven regions of the
 * Venn diagram of a, b and c (the vertices only in a, only in b, only in c, only in a and b, only
 * in b and c, only in c and a, in all three) that are non-empty; up to a permutation of the three
 * hyperedges there are 30 such patterns with at least two pairwise overlaps (Lee et al., "Hypergraph
 * Motifs", VLDB 2020). Bins 0-19 are the triads whose three hyperedges share a vertex, bins 26-29
 * those whose every pair overlaps but no vertex lies in all three; bins 20-25, the open triads
 * (one pair does not overlap), are always 0, because a triad is a triangle of the line graph. The
 * bins are numbered 0-based: ESCHER-GPU's h-motif ids are 1-30, so its open triads 21-26 are bins
 * 20-25.
 * The patterns of options::kind:
 *
 *   - pattern::h_motif: 30 bins, the triads by h-motif;
 *   - pattern::stathyper_type1: one bin, the sum over the live hyperedges of C(|e|, 3) (the
 *     vertex triples inside one hyperedge);
 *   - pattern::stathyper_type2: one bin, the sum over the pairs of adjacent hyperedges (i, j) that
 *     share a >= 2 vertices of C(a, 2) * b * c, with b = |e_i| - a and c = |e_j| - a;
 *   - pattern::stathyper_type3: one bin, the sum over the triads (i, j, k) of x * y * z, the sizes
 *     of the three pairwise-only regions.
 *
 * update() applies a batch of hyperedge deletions and insertions to the hypergraph and changes the
 * counts by the triads the batch destroys and creates, without recounting the rest (the
 * aggregate-delta template of the framework):
 *
 *   - Step 0 (the hypergraph's own, stage triad_count.normalize): the deletions of live ids,
 *     distinct, in batch order; the inserted hyperedges with their members sorted;
 *   - count_minus (triad_count.count_minus, on the hypergraph before the batch): every triad (pair,
 *     hyperedge) that contains a deleted hyperedge is subtracted once, by the deleted hyperedge with
 *     the smallest id in it;
 *   - commit: the batch is applied once (hypergraph.apply);
 *   - identify_affected (triad_count.identify_affected): the ids the inserted hyperedges received
 *     (never the ids a caller may expect: the store's best fit decides them);
 *   - count_plus (triad_count.count_plus, on the hypergraph after the batch): every triad that
 *     contains an inserted hyperedge is added once, by the inserted hyperedge with the smallest id in
 *     it;
 *   - finalize (triad_count.finalize): the counts of both phases are applied to the result; a bin
 *     that would become negative throws internal_error.
 *
 * Hypergraph requirements (checked by compute(), which names the property): h_motif and types 2
 * and 3 read the line graph and the vertex-to-hyperedge rows, so the hypergraph must keep them
 * (hypergraph_properties::line_graph other than line_graph_policy::none, and store_v2h = true;
 * under line_graph_policy::automatic compute() builds the line graph once); type 1 reads the
 * hyperedges' sizes only and needs neither. Weights are ignored. Any storage, id_reuse order and
 * capacity policy is accepted.
 *
 * Backends: cuda (ESCHER-GPU's wedge engine: one thread per oriented wedge of the line graph, the
 * frontier phases of its exact delta), sequential and openmp (the same engine as loops over the
 * sources: the same full count and the same frontier phases, with the same results on every
 * thread count). The counts are host arrays on every backend.
 *
 * Limits: every count is exact up to 2^64 - 1 on every backend (C(|e|, 3), C(a, 2) * b * c and
 * x * y * z are computed without an intermediate overflow; ESCHER-GPU's signed 64-bit arithmetic
 * goes wrong from |e| = 2,097,152 for type 1); a count past 2^64 - 1 throws capacity_error, as
 * does an update that changes one bin by 2^63 or more (stats::delta is signed 64-bit).
 *
 * Types: int32_t vertex ids, unweighted or int32_t weights (other hypergraph types fail to compile
 * with a static_assert naming these).
 */

namespace dyng::triad_count {

/**
 * @brief What a result counts.
 * @ingroup triad_count
 */
enum class pattern : std::uint8_t {
  h_motif,          ///< the 30 h-motifs (bins 20-25, the open triads, are always 0)
  stathyper_type1,  ///< one bin: the sum of C(|e|, 3) over the live hyperedges
  stathyper_type2,  ///< one bin: the sum of C(a, 2) * b * c over the adjacent pairs
  stathyper_type3,  ///< one bin: the sum of x * y * z over the triads
};

/**
 * @brief The lower-case name of a pattern ("h_motif", "stathyper_type1", "stathyper_type2",
 *        "stathyper_type3").
 * @param[in] p The pattern.
 * @return A static string naming `p` (the enumerator's name).
 * @ingroup triad_count
 */
[[nodiscard]] std::string_view to_string(pattern p) noexcept;

/**
 * @brief The number of bins of pattern::h_motif, the size of stats::delta.
 *
 * The most bins a pattern of triad_count has; a pattern appended later has at most as many (a
 * count of more bins is an Op of experimental::count_local_patterns, whose bins are its own).
 * @ingroup triad_count
 */
inline constexpr int max_bins = 30;

/**
 * @brief Options of compute() and update() (an aggregate; fields are only ever appended).
 *
 * kind is fixed at compute() (the result counts that pattern). cuda_engine is a tunable: it
 * chooses how the cuda backend counts, never what it counts, and result::set_options() changes it.
 * @ingroup triad_count
 */
struct options {
  pattern kind = pattern::h_motif;  ///< what the result counts; fixed at compute()

  /// The CUDA engine: automatic and fused select the wedge engine (ESCHER-GPU's kernels, the Tier
  /// B engine of docs/developer/framework.md); operators throws not_supported_error (triad_count
  /// has no operators engine). Ignored on the host backends. A tunable (result::set_options()).
  engine cuda_engine = engine::automatic;
};

/**
 * @brief Counters of one update() (fields are only ever appended).
 *
 * Every counter is deterministic (identical on every backend, thread count and run). The
 * inherited fields mean:
 *   - `affected`: the number of bins whose count changed;
 *   - `iterations`: always 0 (no Step 2 loop; one signed count per phase);
 *   - `frontier_visits`: deleted_frontier + inserted_frontier;
 *   - `fallback_used`: always false (every backend updates the counts incrementally);
 *   - `converged`: always true;
 *   - `engine_used`: engine::fused on cuda (the wedge engine), engine::operators on the host
 *     backends (deterministic per backend).
 * @ingroup triad_count
 */
struct stats : update_stats {
  /// What applying the batch did to the hypergraph (the inserted ids among it).
  hyper_apply_summary batch;
  /// The hyperedges of the subtract phase: the live ids the batch deleted (distinct).
  std::int64_t deleted_frontier = 0;
  /// The hyperedges of the add phase: the hyperedges the batch inserted.
  std::int64_t inserted_frontier = 0;
  /// The triads the batch destroyed, summed over the bins: the count of the subtract phase. For
  /// the types 1-3, whose one bin is a sum of contributions (C(|e|, 3), C(a, 2) * b * c,
  /// x * y * z; the hyperedge weights are ignored), the contributions the batch removed.
  std::uint64_t triads_removed = 0;
  /// The triads the batch created, summed over the bins: the count of the add phase (types 1-3:
  /// the contributions the batch added).
  std::uint64_t triads_added = 0;
  /// The change of every bin: counts() after the update minus counts() before it (bins at and
  /// above result::num_bins() are 0). A fixed-size array, so the stats allocate nothing.
  std::array<std::int64_t, max_bins> delta{};
};

}  // namespace dyng::triad_count

namespace dyng::detail {
struct triad_count_state;
struct triad_count_access;

/**
 * @brief Whether triad_count is instantiated for hypergraph<vertex_t, weight_t>: int32_t vertex
 *        ids, unweighted or int32_t weights (the weights are ignored).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 */
template <typename vertex_t, typename weight_t>
inline constexpr bool triad_count_supported_v =
    std::is_same_v<vertex_t, std::int32_t> &&
    (std::is_same_v<weight_t, unweighted> || std::is_same_v<weight_t, std::int32_t>);

/**
 * @brief The message of triad_count's static_assert for an unsupported hypergraph type.
 * @ingroup triad_count
 */
#define DYNG_TRIAD_COUNT_TYPES_MESSAGE                                                           \
  "dyng::triad_count supports hypergraph<int32_t, unweighted or int32_t> only (the weights are " \
  "ignored); see docs/algorithms/triad_count.md"

/**
 * @brief The message of triad_count's static_assert for a container that is not a hypergraph.
 * @ingroup triad_count
 */
#define DYNG_TRIAD_COUNT_CONTAINER_MESSAGE                               \
  "dyng::triad_count is a hypergraph algorithm; dyng::update() takes a " \
  "dyng::hypergraph<V, W> for its result"
}  // namespace dyng::detail

namespace dyng::triad_count {

/**
 * @brief Triad counts kept up to date by update() (opaque, move-only).
 *
 * counts()[k] is the count of bin k: num_bins() is 30 for pattern::h_motif and 1 for the types.
 * The counts are 64-bit host arrays on every backend. The scratch memory of the engines is leased
 * from the resources handle (ADR 0015), not owned.
 *
 * **Thread safety.** A result is not thread-safe: its accessors may run concurrently with each
 * other, but not with an update() of it (or a dyng::update() that includes it), which must also
 * not overlap another call on its hypergraph.
 * @ingroup triad_count
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
  ~result();                                  ///< releases the counts

  /**
   * @brief What the result counts.
   * @return options::kind of compute().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] pattern kind() const;

  /**
   * @brief The number of bins.
   * @return 30 for pattern::h_motif, 1 for the types.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] int num_bins() const;

  /**
   * @brief The counts.
   * @return counts[k] = the count of bin k (host memory), num_bins() entries. Valid until the next
   *         update() of this result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const std::uint64_t> counts() const;

  /**
   * @brief The count of one bin.
   * @param[in] bin A bin in [0, num_bins()).
   * @return The count.
   * @throws invalid_argument_error if `bin` is outside [0, num_bins()), or for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] std::uint64_t count(int bin) const;

  /**
   * @brief The count of every bin.
   * @return The sum of counts() (for pattern::h_motif: the number of triads).
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws capacity_error         if the sum exceeds 2^64 - 1.
   */
  [[nodiscard]] std::uint64_t total() const;

  /**
   * @brief The options.
   * @return The options given at compute(), as changed by set_options().
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] const options& get_options() const;

  /**
   * @brief Change the tunables (cuda_engine); the next update() uses them.
   * @param[in] opt The new options; kind must stay the same.
   * @throws invalid_argument_error if `opt.kind` differs from the result's, a field holds an
   *         invalid value, or for a moved-from result.
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
   * @brief The memory space of the counts.
   * @return memory_space::host (on every backend).
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief A deep copy for the resources `res` (the counts are a host array on every backend).
   * @param[in] res Execution resources of the copy (any backend).
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available for triad_count.
   * @throws out_of_memory_error    if the copy cannot be allocated.
   * @sync The copy is complete on return (host memory on every backend).
   */
  [[nodiscard]] result clone(const resources& res) const;

 private:
  friend struct detail::triad_count_access;
  explicit result(std::unique_ptr<detail::triad_count_state> state) noexcept;

  std::unique_ptr<detail::triad_count_state> impl_;
};

}  // namespace dyng::triad_count

namespace dyng::detail {

/**
 * @brief The implementation of triad_count::compute() (instantiated for the supported types).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] res Execution resources.
 * @param[in] h   The hypergraph.
 * @param[in] opt Options.
 * @return The counts.
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] triad_count::result triad_count_compute(const resources& res,
                                                      const hypergraph<vertex_t, weight_t>& h,
                                                      const triad_count::options& opt);

/**
 * @brief The implementation of triad_count::update() (instantiated for the supported types).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in]     res   Execution resources.
 * @param[in,out] h     The hypergraph.
 * @param[in]     batch The batch.
 * @param[in,out] r     The result.
 * @return Counters of the update.
 */
template <typename vertex_t, typename weight_t>
triad_count::stats triad_count_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                      const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                      triad_count::result& r);

}  // namespace dyng::detail

namespace dyng::triad_count {

/**
 * @brief Count the triads of `h` (the static count, the oracle of update()).
 *
 * h_motif and type 3: every triangle (a, b, c) of the line graph with a < b < c is counted once,
 * from its out-wedge (a, b): the rows of a after b and of b after b are intersected for the
 * closing c, and the triad is classified with the three hyperedges' member sets (one merge over
 * the three rows). Type 2 visits every pair of adjacent hyperedges (a, b), a < b, once, and type 1
 * every live hyperedge. The CUDA backend runs one thread per out-wedge for h_motif and types 2
 * and 3, and a grid-stride loop over the hyperedges for type 1 (ESCHER-GPU's full counts); the
 * host backends loop over the same wedges and hyperedges.
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam weight_t Hyperedge weight type (unweighted or int32_t; ignored).
 * @param[in] res Execution resources.
 * @param[in] h   The hypergraph; it is not modified (under line_graph_policy::automatic its line
 *                graph is built here, once, for the patterns that read it).
 * @param[in] opt Options.
 * @return The counts, matching `h.version()`.
 * @throws invalid_argument_error if options::kind is not a pattern, the pattern needs the line
 *         graph and `h` keeps none (line_graph_policy::none) or has no vertex-to-hyperedge rows
 *         (store_v2h = false), the pattern classifies triads by their member sets (h_motif,
 *         stathyper_type3) and `h` has 2,147,483,647 vertices (the largest vertex id the counting
 *         merge supports is 2^31 - 3), `h` belongs to another backend than `res`, or `h` was
 *         poisoned by a failed apply().
 * @throws not_supported_error    if the backend of `res` is not available for triad_count, or
 *         options::cuda_engine is engine::operators on cuda.
 * @throws capacity_error         if a count exceeds 2^64 - 1 (on every backend).
 * @throws internal_error         if the device engine meets a triad it cannot classify (a library
 *         bug).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @sync The counts are complete on return (a host array on every backend).
 * @backends sequential, openmp, cuda
 * @determinism Exact values: identical counts on every backend and thread count.
 * @guarantee Strong: `h` is not modified (its line graph may be built under
 *            line_graph_policy::automatic, which changes no observable state but h2h()'s
 *            availability), and nothing is kept if the call throws.
 * @paper ESCHER (IPDPS 2026) and ESCHER+ (TKDE 2026): `dyng::citation("triad_count")`, keys
 *        escher2026 and escherplus2026 in docs/references.bib.
 * @ingroup triad_count
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] result compute(const resources& res, const hypergraph<vertex_t, weight_t>& h,
                             const options& opt = {}) {
  static_assert(detail::triad_count_supported_v<vertex_t, weight_t>,
                DYNG_TRIAD_COUNT_TYPES_MESSAGE);
  return detail::triad_count_compute(res, h, opt);
}

/**
 * @brief Apply a batch of hyperedge changes to `h` and update the counts `r`.
 *
 * The triads through the deleted hyperedges are subtracted on the hypergraph before the batch, the
 * batch is applied once, and the triads through the inserted hyperedges are added on the
 * hypergraph after it, each attributed to exactly one changed hyperedge (the smallest id among its
 * phase's changes in it; see @ref triad_count). Postcondition: `r` equals
 * compute(res, h, r.get_options()) exactly.
 *
 * @tparam vertex_t Vertex id type (int32_t).
 * @tparam weight_t Hyperedge weight type (unweighted or int32_t; ignored).
 * @param[in]     res   Execution resources.
 * @param[in,out] h     The hypergraph; the batch is applied to it and its version increases by one.
 * @param[in]     batch Hyperedge deletions (by id) and insertions (host memory, or device memory
 *                      under the copy policy of `res`).
 * @param[in,out] r     Result of compute() or of a previous update() on `h`.
 * @return Counters of this update (all deterministic; stats::batch.inserted_ids: the ids of the
 *         inserted hyperedges).
 * @throws stale_result_error     if r.graph_version() != h.version(), `r` was computed on another
 *         hypergraph, or `r` was left unusable by a failed update.
 * @throws invalid_argument_error if a batch member or weight is invalid, `h` belongs to another
 *         backend than `res`, or `h` was poisoned (nothing is changed).
 * @throws not_supported_error    if the backend of `res` is not available for triad_count,
 *         options::cuda_engine is engine::operators on cuda, or the batch has incidence edits
 *         (nothing is changed).
 * @throws internal_error         if a bin would become negative, or the device engine meets a
 *         triad it cannot classify (a library bug; then the hypergraph was updated and `r` is left
 *         unusable).
 * @throws capacity_error         if a count exceeds 2^64 - 1 or the change of a bin is outside the
 *         signed 64-bit range of stats::delta (2^63 or more triads added or removed on balance;
 *         both after the batch was applied: `r` is left unusable), or the hypergraph's storage
 *         cannot hold the batch (as hypergraph::apply).
 * @throws out_of_memory_error    if host or device memory cannot be allocated (before the
 *         commit: nothing is changed; inside it: `h` is poisoned; after it: `r` is poisoned; see
 *         @guarantee).
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @sync The hypergraph and the counts are updated on return.
 * @backends sequential, openmp, cuda
 * @determinism Exact values: identical counts and counters on every backend and thread count.
 * @guarantee Strong for every error found before the batch is applied (the ones marked "nothing is
 *            changed" above, and allocation failures before the commit): `h` and `r` are
 *            unchanged. Basic for an error inside the commit (an allocation, storage-capacity or
 *            CUDA failure of hypergraph::apply's store operations): `h` is poisoned (as
 *            hypergraph::apply), and `r` is not changed and not poisoned: it keeps the counts of
 *            the hypergraph before the batch, and must be recomputed with compute() once `h` is
 *            rebuilt (an update of `r` on the rebuilt hypergraph throws stale_result_error). Basic
 *            for an error after the commit (a negative bin, an unclassifiable triad, a count past
 *            2^64 - 1 or outside stats::delta's range, a CUDA or allocation failure): `h` holds
 *            the new version and `r` is poisoned, so every later use of it throws
 *            stale_result_error until it is recomputed.
 * @paper ESCHER (IPDPS 2026) and ESCHER+ (TKDE 2026): `dyng::citation("triad_count")`.
 * @ingroup triad_count
 */
template <typename vertex_t, typename weight_t>
stats update(const resources& res, hypergraph<vertex_t, weight_t>& h,
             const hyperedge_batch_view<vertex_t, weight_t>& batch, result& r) {
  static_assert(detail::triad_count_supported_v<vertex_t, weight_t>,
                DYNG_TRIAD_COUNT_TYPES_MESSAGE);
  return detail::triad_count_update(res, h, batch, r);
}

}  // namespace dyng::triad_count

namespace dyng::detail {

/**
 * @brief The update participant of a triad_count result (used by dyng::update()).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in,out] r   The result.
 * @param[out]    out Receives the stats of the update.
 * @return The participant.
 */
template <typename vertex_t, typename weight_t>
std::unique_ptr<hyper_update_participant<vertex_t, weight_t>> make_triad_count_participant(
    triad_count::result& r, triad_count::stats& out);

/**
 * @brief update_traits of triad_count::result (dyng::update() support).
 */
template <>
struct update_traits<triad_count::result> {
  using stats_type = triad_count::stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type (a hypergraph).
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      triad_count::result& r, triad_count::stats& out) {
    static_assert(is_hypergraph_v<container_t>, DYNG_TRIAD_COUNT_CONTAINER_MESSAGE);
    if constexpr (is_hypergraph_v<container_t>) {
      static_assert(triad_count_supported_v<typename container_t::vertex_type,
                                            typename container_t::weight_type>,
                    DYNG_TRIAD_COUNT_TYPES_MESSAGE);
      return make_triad_count_participant<typename container_t::vertex_type,
                                          typename container_t::weight_type>(r, out);
    } else {
      (void)r;
      (void)out;
      return nullptr;
    }
  }
};

}  // namespace dyng::detail
