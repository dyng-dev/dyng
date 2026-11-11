// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/dynLP.cu (runDynLP: the order of the steps of a
// batch), src/sparsifySubgraph.cu, src/connectedComponents.cu, src/labelInitialization.cu,
// src/iterativePropagation.cu and src/validation.cu (IrLP)
/**
 * @file problem.hpp
 * @brief The label_propagation problem on the framework (cpp/src/framework): its state, its
 *        workspace and its hooks (ADRs 0071, 0072).
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md):
 *
 *     normalize (the framework's Step 0 of net_effect) -> before_apply -> commit ->
 *     identify_affected { .sparsify, .components } -> seed -> loop -> finalize
 *
 * DynLP (the original's runDynLP, one batch) as hooks:
 *   - before_apply (G_t): the neighbours of the deleted vertices (Del ∩ V_t) on the rows before the
 *     batch are affected (Algorithm 2, lines 3-6; markDeletionsKernel);
 *   - identify_affected (G_{t+1}): the inserted vertices (Ins \ Del) and both endpoints of every
 *     new edge of the batch that was appended are affected; the seed labels of the batch join the
 *     result; then, when vertices were inserted, the sub-stages `sparsify` (G': the new edges with
 *     weight > tau whose two endpoints were inserted in this batch) and `components` (G''s
 *     components over the inserted vertices, a component's id the rank of its smallest vertex in
 *     insertion order, the original's numbering);
 *   - seed: the supernode initialization of the inserted vertices (64-bit fixed-point sums of the
 *     weights to the seeds of each class; labelInitialization.cu);
 *   - loop: the propagation from the affected vertices that are active and have no seed label,
 *     with the run-wide tags of the original (two hashed half-sweeps per iteration);
 *   - finalize: every seed's label is its class (enforceGroundTruthKernel), the stats.
 * compute() is IrLP (validation.cu): reset -> seed_static -> loop (Jacobi) -> finalize.
 *
 * Lifecycle members without a stage: begin_update (the checks before anything changes), resume
 * (lease and size the workspace, keep the labels before the update), end_update, poison.
 */
#pragma once

#include "algorithms/label_propagation/fused.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/label_propagation.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::detail {

/// The state behind a label_propagation::result.
template <typename vertex_t>
struct label_propagation_state {
  label_propagation::options opt;  ///< the options
  std::vector<float> labels;       ///< F per vertex
  std::vector<std::int8_t> seeds;  ///< per vertex -1, or the class of a seed (kept when deleted)
  /// Per vertex: the tag of the last propagation queue that held it (the original's d_stamp;
  /// run-wide, so it is never cleared).
  std::vector<std::int64_t> stamp;
  std::int64_t next_tag = 1;      ///< the original's ws.nextTag (the first update's tag is 1)
  std::int64_t updates = 0;       ///< updates since compute() (the stage trace's batch number)
  std::uint64_t version = 0;      ///< graph version matched
  std::uint64_t graph_state = 0;  ///< graph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;          ///< a failed update left the result unusable
  /// The reservation of reserve(): per-batch workspaces sized for these batches.
  std::size_t reserved_vertices = 0;
  std::size_t reserved_edges = 0;  ///< see reserved_vertices
  /// The propagation queues' capacities from the graph's row capacities (computed by compute() on
  /// every backend, so a result can move to CUDA; fused.hpp's state::capacity).
  std::array<int, lp_cuda::num_degree_bins> bin_capacity{};
  /// The run-wide device state of a CUDA result (ADR 0075), nullptr on the host backends: there the
  /// labels, seeds' classes and stamps live in the vectors above; on CUDA in it (`labels` and
  /// `stamp` are empty; `seeds` stays current on the host).
  std::shared_ptr<lp_cuda::state> cuda;
};

/// Access to the state of a result.
struct label_propagation_access {
  /// The state of `r`.
  template <typename vertex_t>
  static label_propagation_state<vertex_t>& state(label_propagation::result<vertex_t>& r);
  /// The state of `r`.
  template <typename vertex_t>
  static const label_propagation_state<vertex_t>& state(
      const label_propagation::result<vertex_t>& r);
  /// A result owning `st`.
  template <typename vertex_t>
  static label_propagation::result<vertex_t> make(
      std::unique_ptr<label_propagation_state<vertex_t>> st);
};

/**
 * @brief Scratch space of one update (or compute), leased from the workspace pool of the
 *        resources handle (ADR 0015). Every per-vertex array is sized for the graph's vertices,
 *        every per-batch array for the reserved batch (grown with headroom when a batch is larger,
 *        a reservation, invariant I9); between updates the flags are clear.
 */
struct label_propagation_workspace final : pooled_workspace {
  std::vector<std::uint8_t> affected;   ///< per vertex: in V_aff (cleared per update)
  std::vector<std::uint8_t> inserted;   ///< per vertex: in Ins_eff of this batch (cleared)
  std::vector<std::int32_t> parent;     ///< per vertex: the union-find of the components
  std::vector<std::int32_t> component;  ///< per vertex: the component of an inserted vertex
  std::vector<float> before;            ///< per vertex: the labels before the update
  std::vector<std::int32_t> frontier;   ///< the current queue (up to every vertex)
  std::vector<std::int32_t> next;       ///< the next queue
  std::vector<std::int32_t> halves;     ///< the current queue split into its two half-sweeps
  std::vector<float> side;              ///< damped: the new labels of the current queue
  std::vector<float> irlp_next;         ///< IrLP: the second label buffer
  std::vector<std::uint64_t> w0;        ///< per component: the fixed-point weight to class 0
  std::vector<std::uint64_t> w1;        ///< per component: the fixed-point weight to class 1
  std::vector<std::int32_t> sub_edges;  ///< G' as (min, max) pairs, flattened
  csr<std::int32_t, std::int64_t, float> snapshot;  ///< IrLP: the column-sorted current graph
  std::vector<std::int64_t> snapshot_count;         ///< IrLP: scratch of the snapshot rows

  /**
   * @brief Size the per-vertex arrays for `n` vertices and the per-batch ones for the given
   *        batch (a reservation when they grow).
   * @param[in] n         Vertices of the graph.
   * @param[in] vertices  Inserted vertices of the batch.
   * @param[in] edges     Edges of the batch.
   */
  void reserve(std::size_t n, std::size_t vertices, std::size_t edges);

  /// Bytes of the arrays' capacities.
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/**
 * @brief The rows of a float graph for the host engines: the slot storage with its row ends and
 *        states (slotted), or the CSR with every vertex active (compact).
 * @tparam edge_t Edge offset type.
 */
template <typename edge_t>
struct lp_rows {
  const edge_t* row_ptr = nullptr;        ///< n + 1 offsets (slot ranges for slotted rows)
  const edge_t* row_end = nullptr;        ///< n row ends (nullptr: row_ptr[v + 1])
  const std::int32_t* col_ind = nullptr;  ///< the entries
  const float* weights = nullptr;         ///< their weights
  const vertex_status* status = nullptr;  ///< n states (nullptr: every vertex active)
  std::int64_t n = 0;                     ///< vertices

  /// The end of row v.
  [[nodiscard]] edge_t end(std::int64_t v) const noexcept {
    return row_end != nullptr ? row_end[v] : row_ptr[v + 1];
  }
  /// Whether v is active.
  [[nodiscard]] bool active(std::int64_t v) const noexcept {
    return status == nullptr || status[v] == vertex_status::active;
  }
};

/**
 * @brief The rows of `g` for the host engines.
 * @tparam edge_t Edge offset type.
 * @param[in] g The graph (host storage).
 * @return Views of its storage, valid until the graph changes.
 */
template <typename edge_t>
lp_rows<edge_t> host_rows(const graph<std::int32_t, edge_t, float>& g);

/// What the propagation of one update did.
struct lp_propagation_report {
  std::int64_t initial_frontier = 0;  ///< |V_aff| (the first queue)
  std::int64_t iterations = 0;        ///< iterations run
  std::int64_t frontier_visits = 0;   ///< summed queue sizes
  bool capped = false;                ///< stopped at max_iterations with a non-empty queue
};

/**
 * @brief The sequential propagation (ADR 0072): the queue of the affected, active, unlabelled
 *        vertices, then iterations of two hashed half-sweeps (the original's halfSweep(v, tag)),
 *        each half in ascending vertex order, until the queue is empty or `max_iterations`.
 *
 * schedule::in_place writes each new label at once (a vertex reads the labels written earlier in
 * the sweep: Gauss-Seidel); schedule::damped computes F' = 0.5f (F + average) of every queued
 * vertex from the iteration's labels and commits them after both halves. A label that changes by
 * more than `delta` queues its active unlabelled neighbours and itself for the next iteration
 * (once per iteration: the run-wide stamps). The tags continue the result's (next_tag).
 * @tparam edge_t Edge offset type.
 * @param[in]     rows       The graph (G_{t+1}).
 * @param[in]     seeds      Per vertex the seed label (-1 none).
 * @param[in,out] labels     The labels.
 * @param[in,out] stamp      The run-wide stamps.
 * @param[in,out] next_tag   The result's next tag.
 * @param[in]     affected   Per vertex the affected flag.
 * @param[in]     damped     Whether the schedule is damped.
 * @param[in]     delta      The tolerance.
 * @param[in]     max_iterations The cap.
 * @param[in,out] ws         The workspace (its queues).
 * @return The report.
 */
template <typename edge_t>
lp_propagation_report propagate_sequential(const lp_rows<edge_t>& rows,
                                           const std::vector<std::int8_t>& seeds,
                                           std::vector<float>& labels,
                                           std::vector<std::int64_t>& stamp, std::int64_t& next_tag,
                                           const std::vector<std::uint8_t>& affected, bool damped,
                                           float delta, std::int64_t max_iterations,
                                           label_propagation_workspace& ws);

/// What IrLP did.
struct lp_irlp_report {
  std::int64_t iterations = 0;  ///< iterations run
  bool capped = false;          ///< stopped at max_iterations while a label still changed
};

/**
 * @brief IrLP on the host (ADR 0072; the original's runIrLPSnapshot / launchIrLP): the
 *        column-sorted snapshot of the current graph (its live entries), every label 0.5 except
 *        the seeds' classes, then double-buffered Jacobi over every active vertex without a seed
 *        label (F' = weighted average, the 1e-10 guard, clamped), the seeds restored after every
 *        iteration, until no label changes by more than `delta` or `max_iterations`.
 * @tparam edge_t Edge offset type.
 * @param[in]  rows           The graph.
 * @param[in]  seeds          Per vertex the seed label (-1 none).
 * @param[out] labels         The labels (resized to n).
 * @param[in]  delta          The tolerance.
 * @param[in]  max_iterations The cap.
 * @param[in,out] ws          The workspace (the snapshot and the second buffer).
 * @return The report.
 */
template <typename edge_t>
lp_irlp_report irlp_sequential(const lp_rows<edge_t>& rows, const std::vector<std::int8_t>& seeds,
                               std::vector<float>& labels, float delta, std::int64_t max_iterations,
                               label_propagation_workspace& ws);

/**
 * @brief The average weight of the live directed entries of `rows` (options::
 *        similarity_threshold = std::nullopt): a double sum in vertex order and slot order,
 *        rounded to float; 0.5 without entries (the original's computeAvgEdgeWeight on the
 *        current graph).
 * @tparam edge_t Edge offset type.
 * @param[in] rows The graph.
 * @return tau.
 */
template <typename edge_t>
float average_live_weight(const lp_rows<edge_t>& rows);

/// The update family of label_propagation.
inline constexpr framework::family label_propagation_family = framework::family::fixed_point;

/**
 * @brief The label_propagation problem (see the file comment).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (float).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class label_propagation_problem final
    : public framework::problem_base<label_propagation_problem<vertex_t, edge_t, weight_t>,
                                     label_propagation_family> {
 public:
  static constexpr std::string_view name = "label_propagation";      ///< stages "<name>.<hook>"
  using container_type = graph<vertex_t, edge_t, weight_t>;          ///< the container
  using stats_type = label_propagation::stats;                       ///< the stats of update()
  using frontier_type = framework::internal_frontier;                ///< frontiers are internal
  using old_graph = framework::old_view<container_type>;             ///< G_t
  using new_graph = framework::new_view<container_type>;             ///< G_{t+1}
  using requested = framework::requested_batch<vertex_t, weight_t>;  ///< the batch before commit
  using applied = framework::applied_batch<vertex_t>;                ///< what the commit did

  /// The problem of one update of `r`.
  explicit label_propagation_problem(label_propagation::result<vertex_t>& r)
      : state_(&label_propagation_access::state(r)), target_(&r) {}
  /// The problem of one compute() filling `st`.
  explicit label_propagation_problem(label_propagation_state<vertex_t>& st) : state_(&st) {}

  // ---- lifecycle (no stage) ----

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return target_;
  }
  /// The checks that read nothing of the batch, before Step 0 (run_update() through
  /// framework::checked_participant; ADR 0070): backend, layout, poisoned and stale result,
  /// options, the result's placement and engine.
  void check_entry(const resources& res, const container_type& g);
  /// The checks before anything changes: check_entry's, then the batch's Step 0 and size.
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Bind G_{t+1}: lease and size the workspace, keep the labels before the update.
  void resume(framework::context& ctx, new_graph g, const applied& applied);
  /// Record the graph state the result matches; return the workspace.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (a failed algorithm phase).
  void poison() noexcept {
    state_->poisoned = true;
  }
  /// The engines read the slot storage on the host (the commit prepares nothing for them).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return false;
  }
  /// The budget of the algorithm phase (invariant I9): nothing allocated once reserved; no host
  /// synchronization on the host backend, and on CUDA the run-dependent ones the fused engine
  /// counted (one per Shiloach-Vishkin round and the components' read, one per head iteration
  /// and per persistent launch of the propagation, the final read of the stats), unchecked while
  /// the stage trace downloads the stages.
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept;

  // ---- the engines ----

  /// The engine of the call: options::cuda_engine on CUDA (the fused engine; there is no
  /// operators engine on CUDA before M10b), the host engine (operators) elsewhere.
  [[nodiscard]] engine select_engine(framework::context& ctx) const noexcept;
  /// Whether the fused engine runs here: on CUDA.
  [[nodiscard]] bool fused_available(framework::context& ctx) const noexcept {
    return ctx.on_cuda();
  }
  /// label_propagation.enact_fused: the rest of the batch on CUDA (fused.hpp's after_commit), the
  /// graph's report, the stats.
  void enact_fused(framework::context& ctx, new_graph g, const applied& applied, stats_type& stats);
  /// label_propagation.enact_fused of compute(): IrLP on CUDA.
  void compute_fused(framework::context& ctx, new_graph g, stats_type& stats);

  // ---- Step 1a ----

  /// label_propagation.before_apply on G_t: the neighbours of the deleted vertices are affected.
  void before_apply(framework::context& ctx, old_graph g, const requested& batch,
                    frontier_type& frontier);

  // ---- Step 1b ----

  /// label_propagation.identify_affected: V_aff on G_{t+1}, the batch's seeds, G' and its
  /// components (sub-stages sparsify and components).
  void identify_affected(framework::context& ctx, new_graph g, const applied& applied,
                         frontier_type& frontier);
  /// label_propagation.seed: the supernode initialization of the inserted vertices.
  void seed(framework::context& ctx, new_graph g, frontier_type& frontier);

  // ---- Step 2 ----

  /// label_propagation.loop: the propagation (update) or IrLP (compute), in one call.
  void loop(framework::context& ctx, new_graph g, frontier_type& in, frontier_type& out);
  /// label_propagation.finalize: the seeds' labels, the stats.
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- compute() ----

  /// label_propagation.reset: start the static run.
  void reset(framework::context& ctx);
  /// label_propagation.seed (static): the seeds given to compute() are already in the state.
  void seed_static(framework::context& ctx, new_graph g, frontier_type& f);

 private:
  /// The workspace, leased and sized for `g` (and the batch of this update).
  label_propagation_workspace& workspace(framework::context& ctx, const container_type& g,
                                         std::size_t vertices, std::size_t edges);

  label_propagation_state<vertex_t>* state_ = nullptr;
  const label_propagation::result<vertex_t>* target_ = nullptr;
  bool computing_ = false;
  std::optional<workspace_pool::lease<label_propagation_workspace>> ws_;
  std::optional<workspace_pool::lease<lp_cuda::workspace>> cuda_ws_;  ///< the fused engine's
  std::int64_t cuda_syncs_ = -1;  ///< the fused engine's host synchronizations (-1: unchecked)
  const edge_batch_view<vertex_t, weight_t>* batch_ = nullptr;  ///< the requested batch
  const applied* applied_ = nullptr;                            ///< what the commit did
  float tau_ = 0.5F;                                            ///< tau of this update
  std::int64_t num_components_ = 0;                             ///< components of G'
  std::int64_t sparsified_ = 0;                                 ///< edges of G'
  lp_propagation_report propagation_{};                         ///< what the loop did
  lp_irlp_report irlp_{};                                       ///< compute(): what IrLP did
  bool fallback_ = false;                                       ///< on_limit::fallback_recompute
};

}  // namespace dyng::detail
