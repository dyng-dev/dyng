// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file problem.hpp
 * @brief The dynamic_bfs problem on the framework (cpp/src/framework): its hooks, its state, its
 *        workspace and the engine interface that each backend implements (TEACHING MATERIAL: the
 *        tutorial docs/tutorials/your_first_dynamic_algorithm.md walks through this file).
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md), family fixed_point, Tier A:
 *
 *     commit -> identify_affected -> seed -> { loop until the frontier is empty } -> finalize
 *
 * The result is the BFS level (hop distance) of every vertex from options::source, -1 for the
 * vertices the source does not reach, and a BFS tree (parents) kept for the next update. After
 * the commit, on G_{t+1}:
 *
 * - **identify_affected** (`classify` + `invalidate_subtree`): a deleted edge u -> v that was v's
 *   tree edge (parent[v] == u, and no parallel edge u -> v is left) makes v a root; every root
 *   and every tree descendant of a root is invalidated (level -1, parent -1). Their levels can
 *   only grow, and the old values are no longer an upper bound.
 * - **seed**: every invalidated vertex pulls the best level from its valid in-neighbours
 *   (level + 1), and every inserted edge u -> v offers level[u] + 1 to v. Each vertex whose level
 *   dropped enters the frontier.
 * - **loop** (`advance`, one round per call): every frontier vertex u offers level[u] + 1 to its
 *   out-neighbours (an atomic minimum); a neighbour whose level dropped enters the next frontier.
 *   The enactor stops when the frontier is empty (the default is_converged of a frontier with
 *   empty()). Levels only drop in Step 2, so the loop ends; the fixed point is the BFS level of
 *   every vertex, whatever the order of the offers.
 * - **finalize**: the parents of the touched vertices are repaired (the lowest-id in-neighbour one
 *   level up, a deterministic rule, so every backend keeps the same tree), the invalidation flags
 *   are cleared and the stats are filled.
 *
 * compute() is the static enactor with the same loop: reset (every level -1), seed_static (the
 * source at level 0), loop, finalize.
 *
 * Where the work runs: the hooks here say WHAT happens WHEN; dynamic_bfs_engine (below) says HOW,
 * once for every backend: engine.hpp writes each pass as a functor run by an executor of the
 * framework operators (cpp/src/operators/execution.hpp), and sequential.cpp, openmp.cpp and
 * cuda.cu instantiate it with the sequential, OpenMP and CUDA executors.
 */
#pragma once

#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/dynamic_bfs.hpp>
#include <dyng/graph/graph.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace dyng::detail {

/// The state behind a dynamic_bfs::result: levels and parents in the memory of the backend that
/// computed it (host memory for sequential and openmp, device memory for cuda).
struct dynamic_bfs_state {
  dynamic_bfs::options opt;                 ///< the options
  buffer<std::int64_t> levels;              ///< the level of every vertex, -1 if unreachable
  buffer<std::int64_t> parents;             ///< the BFS tree: parent of every vertex, -1 for none
  memory_space space = memory_space::host;  ///< where levels and parents live
  int device = -1;                          ///< the CUDA device of device arrays, -1 otherwise
  std::uint64_t version = 0;                ///< graph version matched
  std::uint64_t graph_state = 0;            ///< graph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;                    ///< a failed update left the result unusable
};

/// Access to the state of a result.
struct dynamic_bfs_access {
  /// The state of `r`.
  static dynamic_bfs_state& state(dynamic_bfs::result& r);
  /// The state of `r`.
  static const dynamic_bfs_state& state(const dynamic_bfs::result& r);
  /// A result owning `st`.
  static dynamic_bfs::result make(std::unique_ptr<dynamic_bfs_state> st);
};

/**
 * @brief The scratch arrays of one run, leased from the workspace pool of the resources handle
 *        (ADR 0015), in the memory of the backend (scratch_buffer).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct dynamic_bfs_workspace final : pooled_workspace {
  /// The slots of `counters` (sizes and counts the passes write on the backend).
  enum slot : int { list0 = 0, list1 = 1, invalidated_count = 2, touched_count = 3, changed = 4 };
  static constexpr int num_counters = 5;  ///< length of `counters`

  scratch_buffer<std::int64_t> before;          ///< the levels before the update (stats.affected)
  scratch_buffer<vertex_t> lists[2];            ///< the two frontier lists (a loop's in and out)
  scratch_buffer<vertex_t> invalidated;         ///< every vertex invalidated in this update
  scratch_buffer<vertex_t> touched;             ///< every vertex whose level may have changed
  scratch_buffer<vertex_t> changes;             ///< tails, then heads, of the batch's changes
  scratch_buffer<std::uint32_t> frontier_mark;  ///< round stamps: one push per vertex per round
  scratch_buffer<std::uint32_t> touched_mark;   ///< run stamps: one entry in `touched` per run
  scratch_buffer<std::uint32_t> invalid;        ///< 1 while a vertex is invalidated, else 0
  scratch_buffer<std::uint64_t> counters;       ///< see `slot`
  std::uint32_t round = 0;                      ///< the last round stamp used
  std::uint32_t run = 0;                        ///< the last run stamp used

  /// Bytes of the arrays' capacities.
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return before.bytes() + lists[0].bytes() + lists[1].bytes() + invalidated.bytes() +
           touched.bytes() + changes.bytes() + frontier_mark.bytes() + touched_mark.bytes() +
           invalid.bytes() + counters.bytes();
  }
};

/**
 * @brief The frontier of the loop: which workspace list holds it and how many vertices it has
 *        (known on the host, so is_converged needs no device access). Default-constructible
 *        without allocating (framework/frontier.hpp): the storage is in the workspace.
 */
struct dynamic_bfs_frontier {
  int list = 0;           ///< dynamic_bfs_workspace::lists[list]
  std::int64_t size = 0;  ///< number of vertices

  /// Whether the frontier is empty (the enactor's default convergence test).
  [[nodiscard]] bool empty() const noexcept {
    return size == 0;
  }
};

/**
 * @brief One graph state as the engines read it: raw pointers into the host CSR (sequential,
 *        openmp) or into the resident device copy (cuda).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct dynamic_bfs_graph {
  std::int64_t num_vertices = 0;          ///< n
  const edge_t* out_offsets = nullptr;    ///< n + 1 out-edge offsets
  const vertex_t* out_targets = nullptr;  ///< the out-neighbours
  const edge_t* in_offsets = nullptr;     ///< n + 1 in-edge offsets
  const vertex_t* in_sources = nullptr;   ///< the in-neighbours
};

/**
 * @brief The arrays and host-side counters of one run (compute() or update()).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct dynamic_bfs_run {
  std::int64_t* levels = nullptr;                 ///< the result's levels (backend memory)
  std::int64_t* parents = nullptr;                ///< the result's parents (backend memory)
  dynamic_bfs_workspace<vertex_t>* ws = nullptr;  ///< the leased workspace
  std::int64_t invalidated = 0;                   ///< vertices invalidated
  std::int64_t invalidation_rounds = 0;           ///< passes of the invalidation
  std::int64_t iterations = 0;                    ///< loop rounds
  std::int64_t frontier_visits = 0;               ///< frontier vertices expanded
};

/**
 * @brief How each pass runs on one backend (engine.hpp implements it once, for every executor).
 *
 * Every member reads and writes the arrays of `run` (backend memory). On CUDA the only host
 * synchronizations are the reads of a count: one per invalidation pass, one in seed(), one per
 * advance(), two in finish() (the run's budget, invariant I9).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
class dynamic_bfs_engine {
 public:
  using graph_type = dynamic_bfs_graph<vertex_t, edge_t>;  ///< the graph
  using run_type = dynamic_bfs_run<vertex_t>;              ///< the run

  virtual ~dynamic_bfs_engine() = default;

  /**
   * @brief Start a run: size the workspace for the graph, draw a new run stamp and (update)
   *        keep a copy of the levels for stats.affected.
   * @param[in]     res      The resources.
   * @param[in]     g        The graph.
   * @param[in,out] run      The run.
   * @param[in]     snapshot Whether to copy the levels (update) or not (compute).
   */
  virtual void begin(const resources& res, const graph_type& g, run_type& run,
                     bool snapshot) const = 0;

  /**
   * @brief identify_affected: the roots of the deleted tree edges and their subtrees.
   * @param[in]     res   The resources.
   * @param[in]     g     G_{t+1}.
   * @param[in,out] run   The run (invalidated, invalidation_rounds).
   * @param[in]     tails Tails of the deletions (host memory).
   * @param[in]     heads Heads of the deletions (host memory).
   * @param[in]     count Number of deletions.
   */
  virtual void invalidate(const resources& res, const graph_type& g, run_type& run,
                          const vertex_t* tails, const vertex_t* heads,
                          std::int64_t count) const = 0;

  /**
   * @brief seed: the pull of the invalidated vertices and the offers of the inserted edges.
   * @param[in]     res   The resources.
   * @param[in]     g     G_{t+1}.
   * @param[in,out] run   The run.
   * @param[in]     tails Tails of the insertions (host memory).
   * @param[in]     heads Heads of the insertions (host memory).
   * @param[in]     count Number of insertions.
   * @param[out]    f     The first frontier.
   */
  virtual void seed(const resources& res, const graph_type& g, run_type& run, const vertex_t* tails,
                    const vertex_t* heads, std::int64_t count, dynamic_bfs_frontier& f) const = 0;

  /**
   * @brief seed_static (compute): every level -1, the source at level 0 and on the frontier.
   * @param[in]     res    The resources.
   * @param[in]     g      The graph.
   * @param[in,out] run    The run.
   * @param[in]     source The source.
   * @param[out]    f      The first frontier.
   */
  virtual void seed_source(const resources& res, const graph_type& g, run_type& run,
                           std::int64_t source, dynamic_bfs_frontier& f) const = 0;

  /**
   * @brief loop: one round of offers from the frontier `in` into the frontier `out`.
   * @param[in]     res The resources.
   * @param[in]     g   G_{t+1}.
   * @param[in,out] run The run (iterations, frontier_visits).
   * @param[in]     in  The frontier.
   * @param[out]    out The next frontier.
   */
  virtual void advance(const resources& res, const graph_type& g, run_type& run,
                       const dynamic_bfs_frontier& in, dynamic_bfs_frontier& out) const = 0;

  /**
   * @brief finalize: repair the parents of the touched vertices, clear the invalidation flags
   *        and (update) count the vertices whose level changed.
   * @param[in]     res            The resources.
   * @param[in]     g              G_{t+1}.
   * @param[in,out] run            The run.
   * @param[in]     count_affected Whether to count (update) or not (compute).
   * @return The number of vertices whose level changed (0 without count_affected).
   */
  virtual std::int64_t finish(const resources& res, const graph_type& g, run_type& run,
                              bool count_affected) const = 0;
};

/// The engine of the sequential backend (sequential.cpp).
template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_sequential_engine();
/// The engine of the OpenMP backend (openmp.cpp).
template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_openmp_engine();
/// The engine of the CUDA backend (cuda.cu; only in builds with CUDA).
template <typename vertex_t, typename edge_t>
const dynamic_bfs_engine<vertex_t, edge_t>& dynamic_bfs_cuda_engine();

/**
 * @brief The dynamic_bfs problem (see the file comment).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored: BFS counts hops).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class dynamic_bfs_problem final
    : public framework::problem_base<dynamic_bfs_problem<vertex_t, edge_t, weight_t>,
                                     framework::family::fixed_point> {
 public:
  static constexpr std::string_view name = "dynamic_bfs";    ///< stages "dynamic_bfs.<hook>"
  using container_type = graph<vertex_t, edge_t, weight_t>;  ///< the container
  using stats_type = dynamic_bfs::stats;                     ///< the stats of update()
  using frontier_type = dynamic_bfs_frontier;                ///< the loop's frontier
  using old_graph = framework::old_view<container_type>;     ///< G_t
  using new_graph = framework::new_view<container_type>;     ///< G_{t+1}
  using requested = framework::requested_batch<vertex_t, weight_t>;  ///< the batch before commit
  using applied = framework::applied_batch<vertex_t>;                ///< what the commit did

  /// The problem of one update of `r`.
  explicit dynamic_bfs_problem(dynamic_bfs::result& r)
      : state_(&dynamic_bfs_access::state(r)), target_(&r) {}
  /// The problem of one compute() of `g` filling `st`.
  dynamic_bfs_problem(dynamic_bfs_state& st, const container_type& g) : state_(&st), graph_(&g) {}

  // ---- lifecycle (no stage) ----

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return target_;
  }
  /// The checks before anything changes: backend, placement, graph requirements, stale result.
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Bind G_{t+1}: grow the result for new vertices, lease the workspace, start the run.
  void resume(framework::context& ctx, new_graph g, const applied& applied);
  /// Record the graph state the result matches; return the workspace.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (a failed algorithm phase).
  void poison() noexcept {
    state_->poisoned = true;
  }
  /// The engines read the in-edges (host) or the device copy (cuda) the commit prepares.
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return true;
  }
  /// The budget of the algorithm phase (invariant I9): nothing allocated once reserved; no host
  /// synchronization on the host backends, and on cuda one per count the host reads.
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept;

  // ---- Step 1b: on G_{t+1} ----

  /// dynamic_bfs.identify_affected: invalidate the subtrees under the deleted tree edges.
  void identify_affected(framework::context& ctx, new_graph g, const applied& applied,
                         frontier_type& f);
  /// dynamic_bfs.seed: the first frontier (pulls of the invalidated, offers of the insertions).
  void seed(framework::context& ctx, new_graph g, frontier_type& f);

  // ---- Step 2 ----

  /// dynamic_bfs.loop: one round of offers along the out-edges of the frontier.
  void loop(framework::context& ctx, new_graph g, frontier_type& in, frontier_type& out);
  /// dynamic_bfs.finalize: the parents, the flags and the stats.
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- compute() ----

  /// dynamic_bfs.reset: size the result, lease the workspace.
  void reset(framework::context& ctx);
  /// dynamic_bfs.seed (static): the source at level 0.
  void seed_static(framework::context& ctx, new_graph g, frontier_type& f);

 private:
  using engine_type = dynamic_bfs_engine<vertex_t, edge_t>;

  /// Bind the engine of the backend and the graph's arrays (host CSR or device copy).
  void bind(framework::context& ctx, const container_type& g);
  /// The run's arrays (after bind() and the workspace lease).
  void bind_run();

  dynamic_bfs_state* state_ = nullptr;
  const dynamic_bfs::result* target_ = nullptr;
  const container_type* graph_ = nullptr;  // compute(): the graph (reset() has no view)
  const applied* applied_ = nullptr;       // the commit's changes (identify_affected, seed)
  const engine_type* engine_ = nullptr;
  dynamic_bfs_graph<vertex_t, edge_t> view_{};
  dynamic_bfs_run<vertex_t> run_{};
  bool computing_ = false;
  std::optional<workspace_pool::lease<dynamic_bfs_workspace<vertex_t>>> ws_;
};

}  // namespace dyng::detail
