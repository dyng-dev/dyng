// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file local_patterns.cuh
 * @brief EXPERIMENTAL PUBLIC; may change in any minor release. count_local_patterns<op_t>: count
 *        a new hypergraph pattern, given as an Op, with triad_count's wedge engine, and keep the
 *        counts exact under hyperedge batches. Needs nvcc.
 *
 * The extension header of PLAN Section 4.1 (the one public header that needs nvcc by nature): it
 * holds the per-Op counting kernels and host work, which must be compiled for the user's Op. The
 * driver (Step 0, the frontiers, the workspace, the commit, the delta) is the library's, the one
 * triad_count runs (ADR 0058). Include it from a CUDA translation unit (.cu) compiled by nvcc and
 * link dyng::dyng; the program runs the Op on the cuda, sequential and openmp backends (the
 * callbacks are DYNG_HD). A host-only program uses <dyng/experimental/local_patterns.hpp>.
 * @ingroup experimental
 */
#pragma once

#if !defined(__CUDACC__)
#error \
    "<dyng/experimental/local_patterns.cuh> needs nvcc: include it from a .cu file compiled by nvcc (the Op's kernels are compiled for your Op). A host-only program can include <dyng/experimental/local_patterns.hpp>."
#endif

#include <dyng/core/array_view.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/detail/local_patterns_kernels.cuh>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace dyng::experimental {

template <class op_t>
struct count_local_patterns;

/**
 * @brief The counts of an Op kept up to date by count_local_patterns<op_t>::update() (opaque,
 *        move-only; the counterpart of triad_count::result).
 *
 * counts()[k] is the count of bin k, num_bins() = op_t::num_bins values, a host array on every
 * backend. The Op value given to compute() is kept (op()): every update counts with it.
 *
 * **Thread safety.** As triad_count::result: accessors may run concurrently with each other, not
 * with an update of the result.
 * @tparam op_t The Op.
 * @ingroup experimental
 */
template <class op_t>
class local_patterns_result {
 public:
  using op_type = op_t;  ///< the Op

  /**
   * @brief Move constructor.
   * @param[in,out] other The result to move from; it may only be assigned to or destroyed after.
   */
  local_patterns_result(local_patterns_result&& other) noexcept = default;

  /**
   * @brief Move assignment.
   * @param[in,out] other The result to move from; it may only be assigned to or destroyed after.
   * @return *this.
   */
  local_patterns_result& operator=(local_patterns_result&& other) noexcept = default;

  local_patterns_result(const local_patterns_result&) = delete;             ///< use clone()
  local_patterns_result& operator=(const local_patterns_result&) = delete;  ///< use clone()
  ~local_patterns_result() = default;                                       ///< releases the counts

  /**
   * @brief The number of bins.
   * @return op_t::num_bins.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] int num_bins() const {
    return detail::local_patterns_num_bins(st_.get());
  }

  /**
   * @brief The counts.
   * @return num_bins() counts (host memory), valid until the next update of the result.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] array_view<const std::uint64_t> counts() const {
    return detail::local_patterns_counts(st_.get());
  }

  /**
   * @brief The count of one bin.
   * @param[in] bin The bin, in [0, num_bins()).
   * @return Its count.
   * @throws invalid_argument_error for a bin outside [0, num_bins()) or a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] std::uint64_t count(int bin) const {
    return detail::local_patterns_count(st_.get(), bin);
  }

  /**
   * @brief The sum of the bins.
   * @return The total.
   * @throws capacity_error         if the sum exceeds 2^64 - 1.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] std::uint64_t total() const {
    return detail::local_patterns_total(st_.get());
  }

  /**
   * @brief The Op the result counts with (the value given to compute()).
   * @return A copy of it.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   */
  [[nodiscard]] op_t op() const {
    op_t value{};
    detail::local_patterns_op(st_.get(), &value, sizeof(op_t));
    return value;
  }

  /**
   * @brief The hypergraph version this result matches.
   * @return The version after the last compute() / update() of this result (0 if moved from).
   */
  [[nodiscard]] std::uint64_t graph_version() const noexcept {
    return detail::local_patterns_graph_version(st_.get());
  }

  /**
   * @brief The memory space of the counts.
   * @return memory_space::host (on every backend).
   */
  [[nodiscard]] memory_space space() const noexcept {
    return memory_space::host;
  }

  /**
   * @brief A deep copy for the resources `res`.
   * @param[in] res Execution resources of the copy (any available backend).
   * @return The copy.
   * @throws invalid_argument_error for a moved-from result.
   * @throws stale_result_error     if a failed update left the result unusable (poisoned).
   * @throws not_supported_error    if the backend of `res` is not available.
   * @sync The copy is complete on return (host memory on every backend).
   */
  [[nodiscard]] local_patterns_result clone(const resources& res) const {
    return local_patterns_result(detail::local_patterns_clone(res, st_.get()));
  }

 private:
  friend struct count_local_patterns<op_t>;
  friend struct detail::update_traits<local_patterns_result>;
  explicit local_patterns_result(detail::local_patterns_state_ptr st) noexcept
      : st_(std::move(st)) {}

  detail::local_patterns_state_ptr st_;
};

/**
 * @brief Count a hypergraph pattern given as an Op, and keep the counts exact under hyperedge
 *        batches: experimental::count_local_patterns<op_t>::compute() and update() (PLAN
 *        Appendix B: ESCHER-GPU's countTriangles<Op>).
 *
 * **The Op concept** (each requirement has its own static_assert message):
 *
 *   - `static constexpr int num_bins`, in [1, max_pattern_bins] (the bins live in a block's
 *     shared memory);
 *   - `static constexpr dyng::experimental::shape shape`: what the engine enumerates;
 *   - the callbacks of the shape, each DYNG_HD (they run on the GPU and on the host) and each
 *     taking the rows handle (`const hyperedge_rows&`) and 0-based hyperedge ids:
 *       - shape::triangles: `void begin_wedge(const hyperedge_rows& rows, int a, int b)` (once
 *         per wedge (a, b) of the line graph) and `template <class bins_t> void triangle(const
 *         hyperedge_rows& rows, int c, bins_t& bins)` (once per triangle {a, b, c} closing it);
 *       - shape::pairs: `template <class bins_t> void pair(const hyperedge_rows& rows, int a, int
 *         b, bins_t& bins)` (once per pair of hyperedges that share a vertex);
 *       - shape::keys: `template <class bins_t> void key(const hyperedge_rows& rows, int a,
 *         bins_t& bins)` (once per live hyperedge);
 *     and `template <class bins_t> void flush(bins_t& bins)`, called once at the end of a
 *     worker's run, which adds what the Op kept in its own members;
 *   - trivially copyable and default-constructible: the Op value given to compute() (op_t{} by
 *     default) is the initial state of every worker (a GPU thread, a host source), which may call
 *     the callbacks any number of times before flush().
 *
 * The bins handle has `add(int bin, std::uint64_t value)` (a bin outside [0, num_bins) is
 * refused and reported) and `overflow()` (report a count of the Op's own past 2^64 - 1). Every
 * count is an unsigned 64-bit sum; a count of a pattern must not depend on which worker or in
 * which order it is counted (the engine's guarantee of exact, deterministic counts rests on it).
 *
 * The Op never sees the frontier, the sign of a phase or the ownership rule: the engine counts
 * each triangle (pair, hyperedge) through the changed hyperedges once, at its smallest changed
 * member (ownership::min_member), subtracts the ones through the deleted hyperedges on the
 * hypergraph before the batch and adds the ones through the inserted hyperedges on the
 * hypergraph after it. The Op never holds a view or a pointer of its own into the hypergraph, so
 * the subtraction cannot read the new state (invariant I1 by construction).
 *
 * The rows: a member code (member_code) is opaque and ascending in the vertex ids;
 * hyperedge_rows::decode() gives the vertex id. intersect_size(), venn3() and intersects3() merge
 * the codes directly (no decode pass on the hot path).
 *
 * Requirements: the shapes triangles and pairs read the line graph and the vertex-to-hyperedge
 * rows (hypergraph_properties::line_graph other than none, store_v2h); keys reads neither. Every
 * Op is refused on a hypergraph with 2,147,483,647 vertices (the set helpers end a row at
 * INT_MAX; the vertex ids up to 2^31 - 3 are supported).
 *
 * Types: hypergraph<int32_t, unweighted or int32_t> (the weights are ignored).
 * @tparam op_t The Op.
 * @ingroup experimental
 */
template <class op_t>
struct count_local_patterns {
  /// Whether op_t is an Op (otherwise a static_assert of the Op concept has fired).
  static constexpr bool valid = detail::local_patterns::check_op<op_t>();

  using result = local_patterns_result<op_t>;  ///< the counts of the Op
  using stats = local_patterns_stats;          ///< the counters of an update

  /**
   * @brief Count the Op's pattern on `h` (the full count, the oracle of update()).
   * @tparam vertex_t Vertex id type (int32_t).
   * @tparam weight_t Hyperedge weight type (unweighted or int32_t; ignored).
   * @param[in] res Execution resources (cuda, sequential or openmp).
   * @param[in] h   The hypergraph (under line_graph_policy::automatic its line graph is built
   *                here, once, for the shapes triangles and pairs).
   * @param[in] op  The Op's initial value (kept by the result for every update; the overload
   *                without it uses op_t{}).
   * @return The counts, matching `h.version()`.
   * @throws invalid_argument_error if the shape needs the line graph or the vertex-to-hyperedge
   *         rows and `h` keeps none, `h` has 2,147,483,647 vertices, `h` belongs to another
   *         backend than `res`, or `h` was poisoned; or an Op named a bin outside [0, num_bins).
   * @throws not_supported_error    if the backend of `res` is not available, or the header and
   *         the library differ.
   * @throws capacity_error         if a count exceeds 2^64 - 1.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync The counts are complete on return.
   * @backends sequential, openmp, cuda (the wedge engine of triad_count)
   * @determinism Exact values: identical counts on every backend and thread count (for an Op
   *              whose counts do not depend on the order of its calls).
   * @guarantee Strong: `h` is not modified (its line graph may be built under automatic).
   * @paper The engine of ESCHER (IPDPS 2026) and ESCHER+ (TKDE 2026):
   *        `dyng::citation("triad_count")`.
   */
  template <typename vertex_t, typename weight_t>
  [[nodiscard]] static result compute(const resources& res, const hypergraph<vertex_t, weight_t>& h,
                                      const op_t& op) {
    static_assert(detail::local_patterns_supported_v<vertex_t, weight_t>,
                  DYNG_LOCAL_PATTERNS_TYPES_MESSAGE);
    if constexpr (valid) {
      const detail::local_patterns::kernels k = detail::local_patterns::make_kernels<op_t>();
      return result(detail::local_patterns_compute(res, h, k, &op));
    } else {
      (void)res;
      (void)h;
      (void)op;
      return result(detail::local_patterns_state_ptr{});
    }
  }

  /**
   * @brief compute() with the value-initialized Op, op_t{}.
   * @tparam vertex_t Vertex id type (int32_t).
   * @tparam weight_t Hyperedge weight type (unweighted or int32_t; ignored).
   * @param[in] res Execution resources.
   * @param[in] h   The hypergraph.
   * @return The counts, matching `h.version()`.
   * @throws invalid_argument_error as compute(res, h, op).
   * @throws not_supported_error    as compute(res, h, op).
   * @throws capacity_error         if a count exceeds 2^64 - 1.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync The counts are complete on return.
   * @backends sequential, openmp, cuda
   * @determinism As compute(res, h, op).
   * @guarantee Strong.
   * @paper The engine of ESCHER (IPDPS 2026) and ESCHER+ (TKDE 2026).
   */
  template <typename vertex_t, typename weight_t>
  [[nodiscard]] static result compute(const resources& res,
                                      const hypergraph<vertex_t, weight_t>& h) {
    if constexpr (valid) {
      return compute(res, h, op_t{});
    } else {
      (void)res;
      (void)h;
      return result(detail::local_patterns_state_ptr{});
    }
  }

  /**
   * @brief Apply a batch of hyperedge changes to `h` and update the counts `r` (count(-) on the
   *        hypergraph before the batch, the commit, count(+) after it).
   *
   * Postcondition: `r` equals compute(res, h, r.op()) exactly.
   * @tparam vertex_t Vertex id type (int32_t).
   * @tparam weight_t Hyperedge weight type (unweighted or int32_t; ignored).
   * @param[in]     res   Execution resources.
   * @param[in,out] h     The hypergraph; the batch is applied to it and its version increases.
   * @param[in]     batch Hyperedge deletions (by id) and insertions.
   * @param[in,out] r     Result of compute() or of a previous update() on `h`.
   * @return Counters of this update (deterministic).
   * @throws stale_result_error     if r.graph_version() != h.version(), `r` was computed on
   *         another hypergraph, or `r` was left unusable by a failed update.
   * @throws invalid_argument_error if a batch member is invalid, `h` belongs to another backend
   *         than `res` or was poisoned (nothing is changed); or an Op named a bin outside
   *         [0, num_bins) (after the commit: `r` is left unusable).
   * @throws not_supported_error    if the backend of `res` is not available or the batch has
   *         incidence edits (nothing is changed).
   * @throws internal_error         if a bin would become negative (an Op whose counts depend on
   *         the order of its calls, or a library bug; `r` is left unusable).
   * @throws capacity_error         if a count exceeds 2^64 - 1 or the change of a bin is outside
   *         the signed 64-bit range (`r` is left unusable), or the storage cannot hold the batch.
   * @throws out_of_memory_error    if memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync The hypergraph and the counts are updated on return.
   * @backends sequential, openmp, cuda
   * @determinism Exact values: identical counts and counters on every backend and thread count.
   * @guarantee As triad_count::update(): strong before the commit, basic after it (the result is
   *            poisoned).
   * @paper The engine of ESCHER (IPDPS 2026) and ESCHER+ (TKDE 2026).
   */
  template <typename vertex_t, typename weight_t>
  static stats update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                      const hyperedge_batch_view<vertex_t, weight_t>& batch, result& r) {
    static_assert(detail::local_patterns_supported_v<vertex_t, weight_t>,
                  DYNG_LOCAL_PATTERNS_TYPES_MESSAGE);
    return detail::local_patterns_update(res, h, batch, r.st_, &r);
  }
};

}  // namespace dyng::experimental

namespace dyng::detail {

/**
 * @brief update_traits of an experimental result (dyng::update() support: its results compose
 *        with triad_count's and the other hypergraph algorithms' in one commit).
 * @tparam op_t The Op.
 */
template <class op_t>
struct update_traits<experimental::local_patterns_result<op_t>> {
  using stats_type = experimental::local_patterns_stats;  ///< the stats type

  /**
   * @brief Create the participant.
   * @tparam container_t The container type (a hypergraph).
   * @param[in,out] r   The result.
   * @param[out]    out Receives the stats.
   * @return The participant.
   */
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      experimental::local_patterns_result<op_t>& r, experimental::local_patterns_stats& out) {
    static_assert(is_hypergraph_v<container_t>,
                  "dyng::experimental::count_local_patterns counts hypergraph patterns; "
                  "dyng::update() takes a dyng::hypergraph<V, W> for its result");
    if constexpr (is_hypergraph_v<container_t>) {
      static_assert(local_patterns_supported_v<typename container_t::vertex_type,
                                               typename container_t::weight_type>,
                    DYNG_LOCAL_PATTERNS_TYPES_MESSAGE);
      return make_local_patterns_participant<typename container_t::vertex_type,
                                             typename container_t::weight_type>(r.st_, &r, out);
    } else {
      (void)r;
      (void)out;
      return nullptr;
    }
  }
};

}  // namespace dyng::detail
