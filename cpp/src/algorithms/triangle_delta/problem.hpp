// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file problem.hpp
 * @brief The triangle_delta problem on the framework (cpp/src/framework): its hooks, its state,
 *        its workspace and the engine interface that each backend implements (TEACHING
 *        MATERIAL; the aggregate-delta companion of dynamic_bfs).
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md), family aggregate_delta, Tier A:
 *
 *     normalize -> count(-) on G_t -> commit -> count(+) on G_{t+1} -> finalize
 *
 * The result is the number of triangles {a, b, c} of an undirected graph. A batch changes it by
 * what it destroys and creates:
 *
 *     T(G_{t+1}) = T(G_t) - #{triangles of G_t through a deleted edge}
 *                         + #{triangles of G_{t+1} through an inserted edge}
 *
 * (a triangle of G_t without a deleted edge is a triangle of G_{t+1} without an inserted one, and
 * the other way round, so nothing else changes).
 *
 * - **normalize** (Step 0, on G_t): the net change of the batch as two lists of undirected edges
 *   (u, v), u < v, sorted, without repeats: the framework's normalized batch under set semantics,
 *   graph/structural_change.hpp otherwise (it reads G_t, so it runs before the commit). The
 *   position of an edge in its list is its ownership id.
 * - **count(-)** (Step 1a, on G_t, before the commit): for every deleted edge (u, v), the common
 *   neighbours w of u and v close a triangle that the batch destroys. Ownership
 *   (ownership::min_member, invariant I2): a triangle through several deleted edges is counted
 *   by the smallest one only, so the deleted edge with id i skips w when (u, w) or (v, w) is a
 *   deleted edge with an id below i.
 * - **count(+)** (Step 2, on G_{t+1}, after the commit): the same for the inserted edges.
 * - **finalize**: count = count - removed + added, and the stats.
 *
 * compute() is the static enactor: reset, then count on the graph (every triangle once, from its
 * smallest vertex), then finalize.
 *
 * As in dynamic_bfs, the hooks say WHAT happens WHEN; triangle_delta_engine (below) says HOW, once
 * for every backend (engine.hpp, with the executors of cpp/src/operators/execution.hpp).
 */
#pragma once

#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "graph/structural_change.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/triangle_delta.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::detail {

/// The state behind a triangle_delta::result (host memory on every backend).
struct triangle_delta_state {
  triangle_delta::options opt;    ///< the options
  std::uint64_t count = 0;        ///< the number of triangles
  std::uint64_t version = 0;      ///< graph version matched
  std::uint64_t graph_state = 0;  ///< graph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;          ///< a failed update left the result unusable
};

/// Access to the state of a result.
struct triangle_delta_access {
  /// The state of `r`.
  static triangle_delta_state& state(triangle_delta::result& r);
  /// The state of `r`.
  static const triangle_delta_state& state(const triangle_delta::result& r);
  /// A result owning `st`.
  static triangle_delta::result make(std::unique_ptr<triangle_delta_state> st);
};

/**
 * @brief The scratch of one update, leased from the workspace pool of the resources handle
 *        (ADR 0015): the change lists on the host, and their copy and a counter in the memory of
 *        the backend.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct triangle_delta_workspace final : pooled_workspace {
  using edge = std::pair<vertex_t, vertex_t>;  ///< an undirected edge (u, v), u < v

  structural_change<vertex_t> change;     ///< Step 0 of the other batch semantics
  std::vector<edge> deletions;            ///< the deleted edges, sorted (their ids)
  std::vector<edge> insertions;           ///< the inserted edges, sorted (their ids)
  std::vector<vertex_t> staging;          ///< tails, then heads, of one list (host)
  scratch_buffer<vertex_t> changes;       ///< the same in the backend's memory
  scratch_buffer<std::uint64_t> counter;  ///< the count of one pass (backend memory)

  /// Bytes of the arrays' capacities.
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return (change.deletions.capacity() + change.insertions.capacity() +
            change.requested.capacity()) *
               sizeof(edge_change<vertex_t>) +
           (deletions.capacity() + insertions.capacity()) * sizeof(edge) +
           staging.capacity() * sizeof(vertex_t) + changes.bytes() + counter.bytes();
  }
};

/**
 * @brief One graph state as the engines read it: raw pointers into the host CSR (sequential,
 *        openmp) or into the resident device copy (cuda). Rows are sorted; an undirected graph
 *        stores both directions, so row u is the neighbourhood of u.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct triangle_delta_graph {
  std::int64_t num_vertices = 0;        ///< n
  const edge_t* offsets = nullptr;      ///< n + 1 row offsets
  const vertex_t* neighbors = nullptr;  ///< the sorted neighbours of each vertex
};

/**
 * @brief How the counts run on one backend (engine.hpp implements it once, for every executor).
 *
 * On CUDA each count reads one counter back (one host synchronization).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
class triangle_delta_engine {
 public:
  using graph_type = triangle_delta_graph<vertex_t, edge_t>;  ///< the graph
  using workspace_type = triangle_delta_workspace<vertex_t>;  ///< the workspace

  virtual ~triangle_delta_engine() = default;

  /**
   * @brief The static count: every triangle once.
   * @param[in]     res The resources.
   * @param[in]     g   The graph.
   * @param[in,out] ws  The workspace.
   * @return The number of triangles.
   */
  virtual std::uint64_t count_all(const resources& res, const graph_type& g,
                                  workspace_type& ws) const = 0;

  /**
   * @brief The triangles of `g` through the edges of `changes`, each counted by its smallest
   *        changed edge (ownership::min_member).
   * @param[in]     res     The resources.
   * @param[in]     g       G_t (deletions) or G_{t+1} (insertions).
   * @param[in]     changes The changed edges (u, v), u < v, sorted (host memory).
   * @param[in,out] ws      The workspace.
   * @return The number of triangles.
   */
  virtual std::uint64_t count_owned(const resources& res, const graph_type& g,
                                    const std::vector<std::pair<vertex_t, vertex_t>>& changes,
                                    workspace_type& ws) const = 0;
};

/// The engine of the sequential backend (sequential.cpp).
template <typename vertex_t, typename edge_t>
const triangle_delta_engine<vertex_t, edge_t>& triangle_delta_sequential_engine();
/// The engine of the OpenMP backend (openmp.cpp).
template <typename vertex_t, typename edge_t>
const triangle_delta_engine<vertex_t, edge_t>& triangle_delta_openmp_engine();
/// The engine of the CUDA backend (cuda.cu; only in builds with CUDA).
template <typename vertex_t, typename edge_t>
const triangle_delta_engine<vertex_t, edge_t>& triangle_delta_cuda_engine();

/**
 * @brief The triangle_delta problem (see the file comment).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (ignored).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class triangle_delta_problem final
    : public framework::problem_base<triangle_delta_problem<vertex_t, edge_t, weight_t>,
                                     framework::family::aggregate_delta> {
 public:
  static constexpr std::string_view name = "triangle_delta";         ///< stages "<name>.<hook>"
  using container_type = graph<vertex_t, edge_t, weight_t>;          ///< the container
  using stats_type = triangle_delta::stats;                          ///< the stats of update()
  using frontier_type = framework::internal_frontier;                ///< the lists are internal
  using ownership_type = framework::ownership::min_member;           ///< the counting rule (I2)
  using old_graph = framework::old_view<container_type>;             ///< G_t
  using new_graph = framework::new_view<container_type>;             ///< G_{t+1}
  using requested = framework::requested_batch<vertex_t, weight_t>;  ///< the batch before commit
  using applied = framework::applied_batch<vertex_t>;                ///< what the commit did

  /// The problem of one update of `r`.
  explicit triangle_delta_problem(triangle_delta::result& r)
      : state_(&triangle_delta_access::state(r)), target_(&r) {}
  /// The problem of one compute() filling `st`.
  explicit triangle_delta_problem(triangle_delta_state& st) : state_(&st) {}

  // ---- lifecycle (no stage) ----

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return target_;
  }
  /// The checks before anything changes; leases the workspace.
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Record the graph state the result matches; return the workspace.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (a failed algorithm phase).
  void poison() noexcept {
    state_->poisoned = true;
  }
  /// The counts read the out-edges only (the commit need not build the in-edges).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return false;
  }
  /// The budget of the algorithm phase (invariant I9): nothing allocated once reserved; on cuda
  /// at most one host synchronization per count (two).
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept {
    return framework::budget::steady_state(ctx.on_cuda() ? 2 : 0);
  }

  // ---- Step 0 and Step 1a: on G_t ----

  /// triangle_delta.normalize: the net change as two sorted lists of undirected edges.
  void normalize(framework::context& ctx, old_graph g, const requested& batch);
  /// triangle_delta.count_minus: the triangles of G_t through the deleted edges.
  void count(framework::context& ctx, old_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);

  // ---- Step 2: on G_{t+1} ----

  /// triangle_delta.count_plus (and triangle_delta.count in compute()): the triangles of G_{t+1}
  /// through the inserted edges (compute(): every triangle).
  void count(framework::context& ctx, new_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);
  /// triangle_delta.finalize: the new count and the stats.
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- compute() ----

  /// triangle_delta.reset: start the static count.
  void reset(framework::context& ctx);

 private:
  using engine_type = triangle_delta_engine<vertex_t, edge_t>;
  using workspace_type = triangle_delta_workspace<vertex_t>;

  /// The engine of the backend and the arrays of `g` (host CSR or device copy).
  std::pair<const engine_type*, triangle_delta_graph<vertex_t, edge_t>> bind(
      framework::context& ctx, const container_type& g) const;

  triangle_delta_state* state_ = nullptr;
  const triangle_delta::result* target_ = nullptr;
  bool computing_ = false;
  std::uint64_t removed_ = 0;  ///< count(-)
  std::uint64_t added_ = 0;    ///< count(+) (compute(): the count)
  std::optional<workspace_pool::lease<workspace_type>> ws_;
};

}  // namespace dyng::detail
