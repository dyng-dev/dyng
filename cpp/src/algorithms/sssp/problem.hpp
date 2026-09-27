// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:headers/sospUpdateCpu.h (SospWorkspace, HostCsr, HostChanges,
// SospStats, defaultDelta, DISTANCE_INF)
/**
 * @file problem.hpp
 * @brief The sssp problem: state, workspace, per-objective inputs and the hook enactors shared by
 *        the sequential and OpenMP backends.
 *
 * Template card (PLAN Section 4.5.2):
 *
 *     normalize -> translate -> prepare -> [before_apply] -> commit ->
 *     identify_affected -> seed -> { FP: loop until is_converged } -> finalize
 *
 * sssp is a fixed-point problem. Its hooks:
 *   - prepare (sssp.prepare, on G_t): largest weight and weight sum of the objective (MOSP
 *     computes them on the graph before the batch), the default near-far width, input checks;
 *   - commit (sssp.commit): graph::apply under the graph's batch_semantics, with the per-edge
 *     classification (apply_delta: deletions and per-objective weight increases);
 *   - identify_affected (sssp.identify_affected): roots = heads of deleted or weight-increased
 *     tree edges (judged against the old parents); their subtrees are invalidated;
 *   - seed (sssp.seed): invalidated vertices and insertion heads pull their best (distance,
 *     lowest id) over their in-neighbours;
 *   - loop (sssp.loop): propagate decreases until no distance changes;
 *   - finalize (sssp.finalize): write distances and parents, count the affected vertices.
 * compute() is the static enactor: reset -> seed_static (the source) -> loop -> finalize.
 *
 * The code stays organized by these hooks so that M3 can extract the framework
 * (cpp/src/framework) without rewriting the engines.
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/sssp.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace dyng::detail {

/**
 * @brief The out- and in-edges of the graph as seen by one objective (MOSP's HostCsr pair).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct sssp_graph {
  vertex_t num_vertices = 0;              ///< n
  const edge_t* out_row_ptr = nullptr;    ///< n + 1 out-edge offsets
  const vertex_t* out_col_ind = nullptr;  ///< out-neighbours
  const weight_t* out_weights = nullptr;  ///< objective column, out-edge order
  const edge_t* in_row_ptr = nullptr;     ///< n + 1 in-edge offsets
  const vertex_t* in_col_ind = nullptr;   ///< in-neighbours
  const weight_t* in_weights = nullptr;   ///< objective column, in-edge order
};

/**
 * @brief The batch as seen by one objective (MOSP's HostChanges).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct sssp_changes {
  /// Edges (from[i], to[i]) that were deleted or whose weight increased; the head of such an edge
  /// is a root if the edge is its tree edge.
  std::vector<vertex_t> changed_from;
  std::vector<vertex_t> changed_to;        ///< heads of the changed edges
  const vertex_t* insert_heads = nullptr;  ///< heads of the insertions (distance may decrease)
  std::size_t num_insert_heads = 0;        ///< number of insertion heads
};

/**
 * @brief Counters of one engine run (MOSP's SospStats plus `affected`).
 */
struct sssp_counters {
  std::int64_t invalidated = 0;  ///< vertices in invalidated subtrees
  std::int64_t iterations = 0;   ///< Step 2 rounds
  std::int64_t epochs = 0;       ///< near-far threshold raises
  std::int64_t pushes = 0;       ///< vertex expansions
  std::int64_t affected = 0;     ///< vertices whose distance or parent changed
  bool packed_parents = true;    ///< false: distance-only words were used
};

/**
 * @brief Scratch space of the engines, reserved once per result and reused by every update.
 *
 * The OpenMP fields are MOSP-OpenMP's SospWorkspace; the sequential fields replace the per-call
 * vectors of sequentialSOSPUpdate().
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct sssp_workspace {
  std::int64_t capacity = 0;  ///< vertices the arrays are sized for
  int generation = 0;         ///< stamp generation (stamps deduplicate list insertions)

  // --- OpenMP engine (sospUpdateCpu) ---
  std::vector<std::uint64_t> packed;    ///< (distance << b | parent)
  std::vector<int> stamp;               ///< last generation listed
  std::vector<char> in_far;             ///< vertex is in the far pile
  std::vector<char> state;              ///< invalidation walk states
  std::vector<vertex_t> near_a;         ///< near frontier
  std::vector<vertex_t> near_b;         ///< next near frontier
  std::vector<vertex_t> far;            ///< far pile
  std::vector<vertex_t> far2;           ///< far pile being rebuilt
  std::vector<vertex_t> candidates;     ///< invalidated vertices and insertion heads
  std::vector<vertex_t> frontier;       ///< vertices improved by the pull pass
  std::vector<vertex_t> saved_parents;  ///< old parents (distance-only fallback only)

  // --- sequential engine (sequentialSOSPUpdate) ---
  std::vector<vertex_t> child_start;       ///< children CSR offsets of the tree (n + 1)
  std::vector<vertex_t> children;          ///< children CSR of the tree
  std::vector<vertex_t> invalid;           ///< invalidated vertices, roots first
  std::vector<vertex_t> affected;          ///< affected vertices of the current round
  std::vector<vertex_t> candidate_list;    ///< candidates of the current round
  std::vector<vertex_t> touched;           ///< vertices whose values were changed
  std::vector<std::int64_t> old_distance;  ///< saved distance of each touched vertex
  std::vector<vertex_t> old_parent;        ///< saved parent of each touched vertex

  /**
   * @brief Size the arrays for `requested` vertices (no-op if large enough; MOSP semantics).
   * @param[in] requested Number of vertices.
   */
  void reserve(std::int64_t requested);

  /**
   * @brief A fresh stamp generation.
   * @return The new generation (> 0).
   */
  int next_generation();
};

/**
 * @brief The state behind sssp::result.
 * @tparam vertex_t   Vertex id type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename distance_t>
struct sssp_state {
  vertex_t source = 0;                ///< source vertex
  sssp::options opt;                  ///< options
  std::uint64_t version = 0;          ///< graph version matched
  std::uint64_t graph_state = 0;      ///< state identifier of the graph matched (graph_impl)
  bool poisoned = false;              ///< a failed update left the arrays inconsistent
  std::vector<distance_t> distances;  ///< distances, infinite_distance() if unreachable
  std::vector<vertex_t> parents;      ///< parents, -1 for none
  sssp_workspace<vertex_t> ws;        ///< engine scratch
};

/**
 * @brief Internal access to sssp::result (for the implementation and the tests).
 */
struct sssp_access {
  /**
   * @brief The state of a result.
   * @tparam vertex_t   Vertex id type.
   * @tparam distance_t Distance type.
   * @param[in] r The result.
   * @return Its state (throws for a moved-from result).
   */
  template <typename vertex_t, typename distance_t>
  static sssp_state<vertex_t, distance_t>& state(sssp::result<vertex_t, distance_t>& r);

  /**
   * @brief Wrap a state into a result.
   * @tparam vertex_t   Vertex id type.
   * @tparam distance_t Distance type.
   * @param[in] state The state.
   * @return The result.
   */
  template <typename vertex_t, typename distance_t>
  static sssp::result<vertex_t, distance_t> make(
      std::unique_ptr<sssp_state<vertex_t, distance_t>> state);
};

/// MOSP's DISTANCE_INF (0x7fffffffffffffff / 4); values >= half of it are unreachable.
constexpr std::int64_t sssp_infinity = infinite_distance<std::int64_t>();

/**
 * @brief MOSP's defaultDelta: 32 * average weight / average out-degree, at least 1.
 * @param[in] num_edges    Number of edges (> 0 for a meaningful value).
 * @param[in] num_vertices Number of vertices.
 * @param[in] weight_sum   Sum of the objective's weights.
 * @return The near-far bucket width.
 */
std::int64_t sssp_default_delta(std::int64_t num_edges, std::int64_t num_vertices,
                                std::int64_t weight_sum);

/**
 * @brief Whether (n - 1) * max_weight plus one more edge fits the engines' 62-bit distances
 *        (MOSP's choosePacking() precondition).
 * @param[in] num_vertices Number of vertices.
 * @param[in] max_weight   Largest weight (>= 1).
 * @return true if distances cannot overflow.
 */
bool sssp_distances_fit(std::int64_t num_vertices, std::int64_t max_weight);

/**
 * @brief Run one hook inside its profiler stage `sssp.<hook>`.
 * @tparam fn_t Callable.
 * @param[in] res  Resources (profiler).
 * @param[in] name Stage name.
 * @param[in] fn   The hook.
 */
template <typename fn_t>
void sssp_hook(const resources& res, const char* name, fn_t&& fn) {
  scoped_stage stage(res, name);
  fn();
}

/**
 * @brief The update enactor: identify_affected -> seed -> loop -> finalize, one stage each.
 * @tparam problem_t A backend problem with those hooks.
 * @param[in]     res     Resources.
 * @param[in,out] problem The problem.
 */
template <typename problem_t>
void sssp_enact_update(const resources& res, problem_t& problem) {
  sssp_hook(res, "sssp.identify_affected", [&] { problem.identify_affected(); });
  sssp_hook(res, "sssp.seed", [&] { problem.seed(); });
  sssp_hook(res, "sssp.loop", [&] { problem.loop(); });
  sssp_hook(res, "sssp.finalize", [&] { problem.finalize(); });
}

/**
 * @brief The static enactor of compute(): reset -> seed_static -> loop -> finalize.
 * @tparam problem_t A backend problem with those hooks.
 * @param[in]     res     Resources.
 * @param[in,out] problem The problem.
 */
template <typename problem_t>
void sssp_enact_compute(const resources& res, problem_t& problem) {
  sssp_hook(res, "sssp.reset", [&] { problem.reset(); });
  sssp_hook(res, "sssp.seed", [&] { problem.seed_static(); });
  sssp_hook(res, "sssp.loop", [&] { problem.loop(); });
  sssp_hook(res, "sssp.finalize", [&] { problem.finalize(); });
}

/**
 * @brief Everything one engine run needs.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct sssp_run {
  sssp_graph<vertex_t, edge_t, weight_t> graph;     ///< the graph after the batch, one objective
  const sssp_changes<vertex_t>* changes = nullptr;  ///< nullptr for compute()
  vertex_t source = 0;                              ///< source
  std::int64_t delta = 1;                           ///< near-far width (> 0)
  std::int64_t max_weight = 1;                      ///< largest weight before or after the batch
  std::int64_t* distances = nullptr;                ///< in: old tree; out: new tree
  vertex_t* parents = nullptr;                      ///< in: old tree; out: new tree
  sssp_workspace<vertex_t>* ws = nullptr;           ///< scratch
  sssp_counters counters;                           ///< out
};

/**
 * @brief Sequential backend: the update (adapted from sequentialSOSPUpdate()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources.
 * @param[in,out] run The run.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_sequential_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

/**
 * @brief Sequential backend: compute() (the same loop from the source).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources.
 * @param[in,out] run The run.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_sequential_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

/**
 * @brief OpenMP backend: the update (MOSP-OpenMP's sospUpdateCpu()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources (thread count).
 * @param[in,out] run The run.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

/**
 * @brief OpenMP backend: compute() (MOSP-OpenMP's sospFromScratchCpu()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources (thread count).
 * @param[in,out] run The run.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

}  // namespace dyng::detail
