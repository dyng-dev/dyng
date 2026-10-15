// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file problem.hpp
 * @brief The triad_count problem on the framework (cpp/src/framework), over the hypergraph: its
 *        hooks, its state, and the interfaces of the host and CUDA engines.
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md):
 *
 *     normalize -> prepare -> before_apply -> count(-) on G_t -> commit ->
 *     identify_affected -> count(+) on G_{t+1} -> finalize
 *
 * triad_count is an aggregate-delta problem (family::aggregate_delta, ownership_type =
 * ownership::min_member: a triad is counted by the changed hyperedge with the smallest id in it,
 * invariant I2). Its hooks, one profiler stage each:
 *   - begin_update (no stage): the backend, poisoned and stale-result checks, the placement, the
 *     CUDA engine; Step 0 is the transaction's (stage triad_count.normalize, run_update()): the
 *     live distinct deletions of G_t and the normalized insertions (hyper_staged_batch);
 *   - count(-) on the old view (triad_count.count_minus, on G_t): on cuda the frontier of Step
 *     0's deletions (one upload of the host ids) and the wedge engine's subtract phase on G_t's
 *     internal row views; the host backends count nothing here (they recount in count(+));
 *   - commit (triad_count.commit, run_update()): hypergraph.apply;
 *   - cuda, Tier B (select_engine: engine::fused; triad_count.enact_fused): identify_affected
 *     (triad_count.identify_affected: the frontier of the ASSIGNED inserted ids, from the applied
 *     record's device copy), count_plus (triad_count.count_plus: the add phase on G_{t+1}'s
 *     internal row views) and finalize (triad_count.finalize: one read of both phases' bins, the
 *     signed delta applied, a negative bin an internal_error);
 *   - host backends, Tier A: count(+) (triad_count.count_plus) recounts G_{t+1} and finalize
 *     (triad_count.finalize) fills the stats (stats::fallback_used; M9a step 4 makes them
 *     incremental);
 *   - end_update (no stage): record the hypergraph state the result matches, return the scratch.
 * compute() is the static enactor: reset (triad_count.reset) -> count (triad_count.count, the
 * full count) -> finalize on the host backends; compute_fused (triad_count.enact_fused, with the
 * sub-stages reset, count and finalize) on cuda.
 *
 * The engines read the hypergraph through its internal row views
 * (detail::hypergraph_access::views(), ADR 0042): rows by CBST key (id + 1), h2v members + 1,
 * h2h neighbour keys with `out`. The h2v view covers the keys 1..hyperedge_id_bound(), the h2h
 * view 1..the largest bound so far (its rows above the bound are empty), so every loop over h2v
 * keys is bounded by views.h2v.max_key, and the frontier flags cover views.h2h.max_key + 1 keys.
 * The public views (h2v(), h2h()) are never built on the counting path.
 */
#pragma once

#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace dyng::detail {

/// The counts of one result (num_bins() of them are used).
using triad_bins = std::array<std::uint64_t, triad_count::max_bins>;
/// The signed bins of one phase of an update (two's complement on the device).
using triad_signed_bins = std::array<std::int64_t, triad_count::max_bins>;

/**
 * @brief The launch counters of the CUDA engine (M9a criterion 9), per mode (0: the full count,
 *        1: a frontier phase) and shape (0: triangles, 1: pairs, 2: keys): the counting kernels
 *        launched and the calls that ended at one of the two early outs (no sources: an empty
 *        frontier; no wedges: sources without a wedge). Kept by the result across its calls.
 */
struct triad_launch_counters {
  std::int64_t kernels[2][3] = {};     ///< counting kernels launched
  std::int64_t no_sources[2][3] = {};  ///< calls without sources
  std::int64_t no_wedges[2][3] = {};   ///< calls whose sources have no wedge
};

/// The state behind a triad_count::result.
struct triad_count_state {
  triad_count::options opt;        ///< the options (kind fixed at compute())
  triad_bins counts{};             ///< the counts; bins at and above num_bins are 0
  int num_bins = 30;               ///< 30 (h_motif) or 1 (the types)
  std::uint64_t version = 0;       ///< hypergraph version matched
  std::uint64_t graph_state = 0;   ///< hypergraph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;           ///< a failed update left the result unusable
  triad_launch_counters launches;  ///< the CUDA engine's launches (all calls of the result)
};

/// Access to the state of a result.
struct triad_count_access {
  /// The state of `r`, or nullptr for a moved-from result.
  static const triad_count_state* find(const triad_count::result& r) noexcept {
    return r.impl_.get();
  }
  /// The state of `r`.
  static triad_count_state& state(triad_count::result& r);
  /// The state of `r`.
  static const triad_count_state& state(const triad_count::result& r);
  /// A result owning `st`.
  static triad_count::result make(std::unique_ptr<triad_count_state> st);
};

/**
 * @brief The number of bins of a pattern.
 * @param[in] kind The pattern.
 * @return 30 for pattern::h_motif, 1 for the types.
 */
[[nodiscard]] int triad_num_bins(triad_count::pattern kind) noexcept;

/**
 * @brief Whether a pattern reads the line graph (h2h) and the vertex-to-hyperedge rows.
 * @param[in] kind The pattern.
 * @return false for stathyper_type1 (it reads the hyperedges' sizes only), true otherwise.
 */
[[nodiscard]] bool triad_reads_line_graph(triad_count::pattern kind) noexcept;

// ------------------------------------------------------------------------------------------------
// The host backends (sequential.cpp, openmp.cpp): in this release a recount
// ------------------------------------------------------------------------------------------------

/**
 * @brief The full count of a hypergraph's host row views on the sequential backend: ESCHER-GPU's
 *        full count (every out-wedge (a, b) of the line graph closed into its triangles (a, b, k),
 *        k > b; type 2 every out-wedge; type 1 every h2v key) as a loop.
 * @param[in]  kind  The pattern.
 * @param[in]  views The internal row views (host memory).
 * @param[out] counts The counts (zeroed first).
 * @throws capacity_error if a count exceeds 2^64 - 1.
 */
void triad_count_sequential(triad_count::pattern kind, const hyper::row_views& views,
                            triad_bins& counts);

/**
 * @brief The full count on the OpenMP backend (as triad_count_sequential(); the source keys in
 *        parallel, one set of bins per thread).
 * @param[in]  res    The resources (their thread count).
 * @param[in]  kind   The pattern.
 * @param[in]  views  The internal row views (host memory).
 * @param[out] counts The counts (zeroed first).
 * @throws capacity_error if a count exceeds 2^64 - 1.
 */
void triad_count_openmp(const resources& res, triad_count::pattern kind,
                        const hyper::row_views& views, triad_bins& counts);

// ------------------------------------------------------------------------------------------------
// The CUDA backend (cuda.cu; ESCHER-GPU's wedge engine, engine/)
// ------------------------------------------------------------------------------------------------

/// The CUDA engine of one compute() or update() (defined in cuda.cu).
struct triad_cuda_engine;

/// Destroys a triad_cuda_engine (cuda.cu).
struct triad_cuda_engine_deleter {
  /// @brief Destroy `e`.
  /// @param[in] e The engine.
  void operator()(triad_cuda_engine* e) const noexcept;
};

/// The CUDA engine of one call.
using triad_cuda_handle = std::unique_ptr<triad_cuda_engine, triad_cuda_engine_deleter>;

/**
 * @brief The CUDA engine of one call for `kind`.
 * @param[in] res  Resources of the CUDA backend.
 * @param[in] kind The pattern.
 * @return The engine.
 */
[[nodiscard]] triad_cuda_handle triad_cuda_open(const resources& res, triad_count::pattern kind);

/**
 * @brief compute() on cuda: the full count of a state's row views (ESCHER-GPU's count()).
 * @param[in]     res    Resources of the CUDA backend.
 * @param[in,out] e      The engine.
 * @param[in]     views  The internal row views (device memory).
 * @param[out]    counts The counts.
 */
void triad_cuda_compute(const resources& res, triad_cuda_engine& e, const hyper::row_views& views,
                        triad_bins& counts);

/**
 * @brief update() on cuda, count(-): the subtract phase on G_t (ESCHER-GPU's subtract()), with the
 *        frontier of the deleted ids; sizes the engine for the add phase.
 * @param[in]     res        Resources of the CUDA backend.
 * @param[in,out] e          The engine.
 * @param[in]     views      G_t's internal row views.
 * @param[in]     deletions  Step 0's live distinct deletions (ids, host memory).
 * @param[in]     insertions The number of hyperedges the batch inserts.
 */
void triad_cuda_count_minus(const resources& res, triad_cuda_engine& e,
                            const hyper::row_views& views,
                            const std::vector<std::int64_t>& deletions, std::int64_t insertions);

/**
 * @brief update() on cuda, identify_affected: the frontier of the add phase, the ids the commit
 *        ASSIGNED to the inserted hyperedges.
 * @param[in]     res    Resources of the CUDA backend.
 * @param[in,out] e      The engine.
 * @param[in]     views  G_{t+1}'s internal row views.
 * @param[in]     record The applied record (inserted_ids, inserted_ids_device).
 */
void triad_cuda_identify_affected(const resources& res, triad_cuda_engine& e,
                                  const hyper::row_views& views, const hyper_applied& record);

/**
 * @brief update() on cuda, count(+): the add phase on G_{t+1} (ESCHER-GPU's add()).
 * @param[in]     res   Resources of the CUDA backend.
 * @param[in,out] e     The engine.
 * @param[in]     views G_{t+1}'s internal row views.
 */
void triad_cuda_count_plus(const resources& res, triad_cuda_engine& e,
                           const hyper::row_views& views);

/**
 * @brief update() on cuda, finalize: the bins of both phases on the host.
 * @param[in]     res   Resources of the CUDA backend.
 * @param[in,out] e     The engine.
 * @param[out]    minus The subtract phase's bins (<= 0: minus the triads removed).
 * @param[out]    plus  The add phase's bins (>= 0: the triads added).
 * @throws internal_error if the device engine raised its error word (an unclassifiable triad).
 */
void triad_cuda_finalize(const resources& res, triad_cuda_engine& e, triad_signed_bins& minus,
                         triad_signed_bins& plus);

/**
 * @brief Add the engine's launch counters of this call to `into`.
 * @param[in]     e    The engine.
 * @param[in,out] into The result's counters.
 */
void triad_cuda_take_launches(const triad_cuda_engine& e, triad_launch_counters& into) noexcept;

/**
 * @brief The host synchronizations of one update on cuda (the algorithm budget, I9 / C8).
 * @param[in] kind The pattern.
 * @return At most one per phase that launches wedges (its wedge count) and one for the bins.
 */
[[nodiscard]] std::int64_t triad_cuda_host_syncs(triad_count::pattern kind) noexcept;

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

/**
 * @brief The triad_count problem (family::aggregate_delta) that framework::update_enactor and
 *        framework::static_enactor run: the hooks of the file comment. Its members are defined in
 *        triad_count.cpp, the only translation unit that runs it.
 *
 * One problem serves one call: an update of one result (constructed from the result; the
 * participant adapter of framework/composition.hpp owns it) or a compute() (constructed from the
 * state being built). The CUDA engine of an update lives from count(-) (before the commit) to
 * end_update(); everything it needs after the commit is sized before it (invariant I9).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (ignored).
 */
template <typename vertex_t, typename weight_t>
class triad_count_problem final
    : public framework::problem_base<triad_count_problem<vertex_t, weight_t>,
                                     framework::family::aggregate_delta> {
 public:
  static constexpr std::string_view name = "triad_count";   ///< stages "triad_count.<hook>"
  using container_type = hypergraph<vertex_t, weight_t>;    ///< the container
  using stats_type = triad_count::stats;                    ///< the stats of update()
  using frontier_type = framework::internal_frontier;       ///< frontiers are internal
  using ownership_type = framework::ownership::min_member;  ///< the counting rule (I2)
  using old_graph = framework::old_view<container_type>;    ///< G_t
  using new_graph = framework::new_view<container_type>;    ///< G_{t+1}
  /// The batch before the commit (batch.normalized: Step 0, the live deletions of G_t).
  using requested = framework::hyper_requested_batch<vertex_t, weight_t>;
  /// What the commit did (applied.record: the applied record with the assigned ids).
  using applied = framework::hyper_applied_batch<vertex_t, weight_t>;

  /**
   * @brief The problem of one update of `r`.
   * @param[in,out] r The result (must outlive the problem).
   */
  explicit triad_count_problem(triad_count::result& r);

  /**
   * @brief The problem of one compute() that fills `st`.
   * @param[in,out] st The state of the result being built (must outlive the problem).
   */
  explicit triad_count_problem(triad_count_state& st) noexcept : state_(&st), computing_(true) {}

  // ---- lifecycle (no stage) ------------------------------------------------------------------

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return result_;
  }
  /// Whether the update keeps the line graph: true, except for stathyper_type1 (which reads the
  /// hyperedges' sizes only).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return reads_line_graph_;
  }
  /// The checks before anything changes (backend, poisoned and stale result, placement, the CUDA
  /// engine); keeps Step 0's batch.
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Bind G_{t+1}.
  void resume(framework::context& ctx, new_graph g, const applied& applied);
  /// Record the hypergraph state the result matches; return the engine.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (its algorithm phase failed).
  void poison() noexcept;

  // ---- policies (no stage) -------------------------------------------------------------------

  /// engine::fused (the wedge engine) on cuda, engine::operators (the hooks) on the host backends.
  [[nodiscard]] engine select_engine(framework::context& ctx) const noexcept;
  /// The budget of the update's algorithm work (I9, C8): no allocation once reserved; on cuda at
  /// most triad_cuda_host_syncs() host synchronizations, none on the host backends.
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept;

  // ---- Step 1a, on G_t -------------------------------------------------------------------------

  /// triad_count.count_minus: cuda, the subtract phase on G_t; host, nothing (the recount).
  void count(framework::context& ctx, old_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);

  // ---- Tier A (host backends), on G_{t+1} ------------------------------------------------------

  /// update(): triad_count.count_plus, the recount of G_{t+1}; compute(): triad_count.count.
  void count(framework::context& ctx, new_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);
  /// update(): triad_count.finalize, the stats; compute(): the hypergraph state.
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- Tier B (cuda), on G_{t+1} -----------------------------------------------------------

  /// triad_count.enact_fused: identify_affected, count_plus and finalize (sub-stages).
  void enact_fused(framework::context& ctx, new_graph g, const applied& applied, stats_type& stats);

  // ---- compute() -----------------------------------------------------------------------------

  /// triad_count.reset: the zero counts.
  void reset(framework::context& ctx);
  /// triad_count.enact_fused (cuda): the full count, with the sub-stages reset, count, finalize.
  void compute_fused(framework::context& ctx, new_graph g, stats_type& stats);

 private:
  /// The signed delta of both phases applied to the counts, and the stats.
  void apply_delta(const triad_signed_bins& minus, const triad_signed_bins& plus,
                   stats_type& stats);
  /// The counters of the update in the stats (all but the bins).
  void fill_counts(stats_type& stats) const;

  triad_count_state* state_ = nullptr;
  triad_count::result* result_ = nullptr;
  bool computing_ = false;
  bool reads_line_graph_ = true;
  const hyper_staged_batch<weight_t>* normalized_ = nullptr;  ///< Step 0 of the update
  const container_type* graph_ = nullptr;                     ///< G_{t+1} (compute(): G)
  std::int64_t deleted_ = 0;                                  ///< the subtract frontier's size
  std::int64_t inserted_ = 0;                                 ///< the add frontier's size
  triad_bins before_{};  ///< host backends: the counts before the update
  triad_cuda_handle cuda_;
};

}  // namespace dyng::detail
