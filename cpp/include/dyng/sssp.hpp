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
#include <type_traits>

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
 *   - in the distance-only mode (stats::packed_parents == false: the distances do not fit next
 *     to the parent ids in 64 bits) every parent is recovered with the lowest-id rule after the
 *     search, so update() equals compute() on every input.
 *
 * Backends: sequential (the reference), openmp (MOSP-OpenMP's sospUpdateCpu) and cuda, which has
 * two engines (options::cuda_engine): the fused engine (MOSP-CUDA's persistent cooperative kernel)
 * and the operators engine (the same algorithm as a sequence of kernels driven by the host; it
 * needs no cooperative launch and returns the same bytes). On cuda the graph must be built with
 * (or cloned for) the CUDA resources, the result arrays live in device memory (copy them with
 * to_vector()), and with the fused engine compute() synchronizes the stream once and update()
 * once per result, plus once per batch inside the commit (the new graph state's upload); the
 * operators engine synchronizes once per near-far round as well (compute() 2 + iterations +
 * epochs times, update() 3 + iterations + epochs times per result). engine::automatic runs the
 * fused engine where cooperative launch is available and the operators engine otherwise.
 *
 * The packing window (ADR 0029, proposed). The originals choose the word format differently near
 * the packing limit: MOSP-CUDA packs when (n - 1) * max weight fits next to the parent bits,
 * MOSP-OpenMP only when one more edge fits too. The host backends follow MOSP-OpenMP and both
 * CUDA engines MOSP-CUDA, so in the window (n - 1) * max weight <= max_distance < n * max weight
 * (with int32 weights only on graphs of at least 65537 vertices) the host backends run
 * distance-only and cuda runs packed. From canonical input trees every backend returns the same
 * tree (only stats::packed_parents differs); from non-canonical input trees inside the window the
 * host backends return the canonical tree and cuda keeps the tie parents of the vertices it does
 * not re-evaluate, so the parents differ between the host backends and cuda (each equal to its
 * original; the distances are equal everywhere).
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
  /// Engine of the CUDA backend; ignored by the host backends. engine::fused runs the fused
  /// persistent cooperative kernel (MOSP-CUDA's sospUpdateGpu) and throws not_supported_error on a
  /// device without cooperative launch; engine::operators runs the multi-kernel operators engine
  /// (the same bytes, on any device); engine::automatic runs the fused engine where it can and the
  /// operators engine otherwise. A tunable: it changes the schedule, never the result.
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
  /// Deterministic per backend (update_stats): false if distances do not fit next to the parent
  /// ids in 64-bit words, so the engine kept distances only and recovered every parent with the
  /// lowest-id rule after the search. The sequential engine reports the OpenMP engine's value and
  /// applies the same recovery, so both return the same tree; the CUDA engines use MOSP-CUDA's
  /// slightly larger packing limit, so its value can differ inside the packing window (see
  /// @ref sssp, ADR 0029).
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
 * **Thread safety.** A result is not thread-safe: its accessors may run concurrently with each
 * other, but not with an update() of it (or a dyng::update() that includes it), which must also
 * not overlap another call on its graph; see graph, "Thread safety". On CUDA, reading its device
 * arrays on one stream while another stream updates it is undefined unless the streams are
 * ordered.
 *
 * @tparam vertex_t   Vertex id type (int32_t or int64_t).
 * @tparam distance_t Distance type: std::int64_t, the only width in 0.1 (the packed (distance, id)
 *                    words and the engines are 64-bit); the parameter keeps room for a later
 *                    width, and another value fails to compile.
 * @ingroup sssp
 */
template <typename vertex_t, typename distance_t = std::int64_t>
class result {
  static_assert(std::is_same_v<distance_t, std::int64_t>,
                "dyng::sssp::result: distance_t is std::int64_t in 0.1 (the packed (distance, id) "
                "words of the engines are 64-bit)");
  static_assert(std::is_same_v<vertex_t, std::int32_t> || std::is_same_v<vertex_t, std::int64_t>,
                "dyng::sssp::result: vertex_t is std::int32_t or std::int64_t");

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
   * @guarantee Strong: every check runs before the options change.
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
   * re-evaluate (the tie rule of @ref sssp), identically on every backend except inside the
   * packing window, where the host backends recover every parent with the lowest-id rule and
   * cuda keeps them (each backend as its original; @ref sssp, ADR 0029). With
   * `opt.validate_inputs` the tree is checked in O(n): array sizes, source at distance 0
   * without a parent, distances in [0, (n - 1) * max weight] or unreachable (>=
   * infinite_distance() / 2, stored as infinite_distance()), unreachable vertices without a
   * parent, every other vertex with a reachable parent, and no parent cycle (every chain ends at
   * the source); on the OpenMP backend the checks run in parallel and report the same first
   * problem as the sequential ones. The caller guarantees that the tree is a shortest-path tree
   * of `g`. Also sizes the pooled workspace of `res` for the graph (once for all results built
   * through `res`; ADR 0015). The tree is imported and checked in host memory on every backend
   * (on cuda it is then uploaded, profiler stage sssp.upload); the arrays may be in any memory
   * space, and arrays in device memory are copied to the host once under res.get_copy_policy()
   * (host arrays for a cuda result are this function's own import, not an implicit copy).
   *
   * @tparam edge_t   Edge offset type of the graph.
   * @tparam weight_t Weight type of the graph.
   * @param[in] res          Execution resources (the graph must belong to their backend).
   * @param[in] g            The graph the tree belongs to (with in-edges stored).
   * @param[in] source       The source vertex.
   * @param[in] distances    One distance per vertex (any memory space).
   * @param[in] parents      One parent per vertex, -1 for none (any memory space).
   * @param[in] canonicalize Apply the lowest-id tie rule to the parents (default true).
   * @param[in] opt          Options (objective = the weight column the tree belongs to).
   * @return The result, matching `g.version()`.
   * @throws invalid_argument_error if a check fails, the options are invalid, `g` belongs to
   *         another backend than `res`, or an array in device memory must be copied and the copy
   *         policy is copy_policy::error.
   * @throws not_supported_error    if an array is in device memory and CUDA is not built.
   * @throws out_of_memory_error    if host or device memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync
   * @guarantee Strong: `g` and the arrays are not modified, and nothing is kept if the call
   *            throws.
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

}  // namespace dyng::sssp

namespace dyng::detail {

/**
 * @brief Whether sssp is instantiated for graph<vertex_t, edge_t, weight_t>: int32_t weights with
 *        int32_t ids and int32_t or int64_t offsets, or int64_t ids and offsets.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
inline constexpr bool sssp_supported_v =
    std::is_same_v<weight_t, std::int32_t> &&
    ((std::is_same_v<vertex_t, std::int32_t> &&
      (std::is_same_v<edge_t, std::int32_t> || std::is_same_v<edge_t, std::int64_t>)) ||
     (std::is_same_v<vertex_t, std::int64_t> && std::is_same_v<edge_t, std::int64_t>));

/**
 * @brief The message of sssp's static_assert for an unsupported graph type.
 * @ingroup sssp
 */
#define DYNG_SSSP_TYPES_MESSAGE                                                                  \
  "dyng::sssp supports graph<int32_t, int32_t or int64_t, int32_t> and graph<int64_t, int64_t, " \
  "int32_t> only (integer weights in [1, 2^31 - 1]); see 'Graph requirements' in "               \
  "docs/algorithms/sssp.md"

/**
 * @brief sssp::compute() for a supported graph type (instantiated in the library).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in] res    Execution resources.
 * @param[in] g      The graph.
 * @param[in] source The source vertex.
 * @param[in] opt    Options.
 * @return The shortest-path tree.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] sssp::result<vertex_t> sssp_compute(const resources& res,
                                                  const graph<vertex_t, edge_t, weight_t>& g,
                                                  vertex_t source, const sssp::options& opt);

/**
 * @brief sssp::update() for a supported graph type (instantiated in the library).
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
sssp::stats sssp_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                        const edge_batch_view<vertex_t, weight_t>& batch,
                        sssp::result<vertex_t>& r);

}  // namespace dyng::detail

namespace dyng::sssp {

/**
 * @brief Compute the canonical shortest-path tree of `g` from `source` (the static solve).
 *
 * The sequential backend runs the Step 2 loop of the sequential engine from the source; the
 * OpenMP backend runs the near-far search of MOSP-OpenMP's sospFromScratchCpu(), the cuda backend
 * MOSP-CUDA's sospFromScratchGpu() (the persistent kernel from the source; with the operators
 * engine, its kernels one by one). All return the Dijkstra tree with lowest-id ties.
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
 *         of options::cuda_engine cannot run (engine::fused without cooperative launch).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error (cuda backend).
 * @sync On cuda with the fused engine the stream is synchronized once (the control block of the
 *       kernel is read); with the operators engine (engine::operators, or engine::automatic on a
 *       device without cooperative launch) 2 + iterations + epochs times (once after the first
 *       split, once per push iteration and threshold raise, once in the finalize step). Once more
 *       before that if the graph's current state is not resident on the device yet (its upload,
 *       profiler stage graph.upload, completes before the kernel runs).
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs: the Dijkstra tree with lowest-id ties.
 * @guarantee Strong: `g` is not modified (on cuda its device copy may be uploaded, which changes
 *            no observable state), and nothing is kept if the call throws.
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("sssp")`, keys dynamosp2025 and
 *        dynamosptpds2025 in docs/references.bib.
 * @ingroup sssp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] result<vertex_t> compute(const resources& res,
                                       const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                                       const options& opt = {}) {
  static_assert(detail::sssp_supported_v<vertex_t, edge_t, weight_t>, DYNG_SSSP_TYPES_MESSAGE);
  return detail::sssp_compute(res, g, source, opt);
}

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
 * @param[in]     batch Insertions (upserts), deletions and weight changes (any memory space: the
 *                      batch is read on the host in this release, so arrays in device memory are
 *                      copied once under res.get_copy_policy()).
 * @param[in,out] r     Result of compute() or of a previous update() on `g`.
 * @return Counters of this update; `invalidated` and `affected` are deterministic, `iterations`,
 *         `epochs` and `pushes` are not.
 * @throws stale_result_error     if r.graph_version() != g.version(), `r` was computed on another
 *         graph (or on an earlier state of a graph variable that was reassigned since), or `r` was
 *         left unusable by a failed update.
 * @throws invalid_argument_error if a batch id or weight is invalid, `g` or `r` belongs to
 *         another backend than `res`, or a batch array must be copied and the copy policy is
 *         copy_policy::error (nothing is changed); or, only for a tree imported without
 *         validation, if the tree has a parent cycle or (cuda) a distance outside the packing bound
 *         (then the graph was updated and `r` is left unusable).
 * @throws not_supported_error    if the backend of `res` is not built, or on cuda if the engine
 *         of the result's options::cuda_engine cannot run (nothing is changed).
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error (cuda backend; after the
 *         batch was applied, `r` is left unusable).
 * @sync On cuda the stream is synchronized once per result with the fused engine (the kernel's
 *       control block is read), or 3 + iterations + epochs times per result with the operators
 *       engine (engine::operators, or engine::automatic on a device without cooperative launch:
 *       once after identify_affected, once after the first split, once per push iteration and
 *       threshold raise, once in the finalize step; ADR 0026), and once per batch inside the
 *       commit, where the new graph state is uploaded (profiler stage graph.upload; ADR 0017
 *       item 7).
 * @backends sequential, openmp, cuda
 * @determinism Bit-exact across backends and runs (distances, parents, `invalidated`,
 *              `affected`), for canonical and non-canonical input trees alike, except the parents
 *              and `affected` of non-canonical input trees inside the packing window, which differ
 *              between the host backends and cuda (each as its original; @ref sssp, ADR 0029).
 * @guarantee Strong for every error found before the batch is applied (a stale result, an
 *            invalid batch, the wrong backend, an unsupported engine, a copy refused by the copy
 *            policy, memory for the normalization): `g` and `r` are unchanged. Basic for an error
 *            after the commit (only an imported tree that was not validated, a CUDA failure or an
 *            allocation failure can cause one): `g` holds the new version and `r` is poisoned, so
 *            every later use of it throws stale_result_error until it is recomputed.
 * @paper DynaMOSP (IPDPS 2025; IEEE TPDS 2025): `dyng::citation("sssp")`, keys dynamosp2025 and
 *        dynamosptpds2025 in docs/references.bib.
 * @ingroup sssp
 */
template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r) {
  static_assert(detail::sssp_supported_v<vertex_t, edge_t, weight_t>, DYNG_SSSP_TYPES_MESSAGE);
  return detail::sssp_update(res, g, batch, r);
}

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
    static_assert(
        sssp_supported_v<typename container_t::vertex_type, typename container_t::edge_type,
                         typename container_t::weight_type>,
        DYNG_SSSP_TYPES_MESSAGE);
    return make_sssp_participant<vertex_t, typename container_t::edge_type,
                                 typename container_t::weight_type, distance_t>(r, out);
  }
};

}  // namespace dyng::detail
