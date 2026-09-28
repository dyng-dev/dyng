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

#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"
#include "util/thread_list.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/memory.hpp>
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
  const vertex_t* changed_from = nullptr;
  const vertex_t* changed_to = nullptr;    ///< heads of the changed edges
  std::size_t num_changed = 0;             ///< number of changed edges
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
 * @brief Scratch space of the engines: leased from the workspace pool of the resources handle for
 *        one run and shared by every result run through that handle (ADR 0015).
 *
 * The OpenMP fields are MOSP-OpenMP's SospWorkspace, which mospUpdate() reserves once and shares
 * across the K objectives; the sequential fields replace the per-call vectors of
 * sequentialSOSPUpdate(). Nothing in it carries information from one run to the next, except the
 * generation counter of the stamps, which lives here with the stamps it describes; `in_far` is all
 * zero between runs (a failed run's workspace is discarded by its lease).
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct sssp_workspace final : pooled_workspace {
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
  /// Two lists per thread for the parallel regions (slot 2t and 2t + 1 of thread t), each on its
  /// own cache lines and kept between rounds and runs (util/thread_list.hpp).
  std::vector<padded_thread_list<vertex_t>> thread_lists;

  // --- sequential engine (sequentialSOSPUpdate) ---
  std::vector<vertex_t> child_start;       ///< children CSR offsets of the tree (n + 1)
  std::vector<vertex_t> children;          ///< children CSR of the tree
  std::vector<vertex_t> invalid;           ///< invalidated vertices, roots first
  std::vector<vertex_t> affected;          ///< affected vertices of the current round
  std::vector<vertex_t> candidate_list;    ///< candidates of the current round
  std::vector<vertex_t> touched;           ///< vertices whose values were changed
  std::vector<std::int64_t> old_distance;  ///< saved distance of each touched vertex
  std::vector<vertex_t> old_parent;        ///< saved parent of each touched vertex

  // --- the change lists of one objective (update) ---
  std::vector<vertex_t> changed_from;  ///< tails of the deleted and weight-increased edges
  std::vector<vertex_t> changed_to;    ///< heads of the deleted and weight-increased edges

  /**
   * @brief Size the arrays for `requested` vertices (no-op if large enough; MOSP semantics:
   *        the per-vertex arrays are assigned, the frontier lists only reserved).
   * @param[in] requested Number of vertices.
   */
  void reserve(std::int64_t requested);

  /**
   * @brief A fresh stamp generation.
   * @return The new generation (> 0).
   */
  int next_generation();

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of every array's capacity.
   */
  [[nodiscard]] std::size_t bytes() const noexcept override;

  /**
   * @brief The part of bytes() held by the per-thread lists of the OpenMP engine. Their capacity
   *        follows the largest share of a round each thread has taken so far, which depends on the
   *        dynamic schedule; everything else depends only on the graph and the batches.
   * @return Bytes of the per-thread lists (their headers and their cache-line blocks).
   */
  [[nodiscard]] std::size_t thread_list_bytes() const noexcept;
};

/**
 * @brief Device scratch of the fused CUDA engine (MOSP-CUDA's SospWorkspace), leased from the
 *        workspace pool of a CUDA resources handle and shared by every result run through it, as
 *        mospUpdate() shares one SospWorkspace across the K objectives (ADR 0015).
 *
 * The arrays are sized once (reserve(), a no-op once large enough); stamp, in_far and flag are
 * zero between runs, as SospWorkspace::reserve leaves them and every run leaves them (a failed
 * run's workspace is discarded by its lease). The change lists of one objective are built on the
 * host and uploaded into the device lists here, whose capacity is reused from batch to batch.
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct sssp_cuda_workspace final : pooled_workspace {
  std::int64_t capacity = 0;          ///< vertices the arrays are sized for
  int generation = 0;                 ///< stamp generation (stamps deduplicate list insertions)
  const void* grid_kernel = nullptr;  ///< the kernel grid_blocks was computed for
  int grid_blocks = 0;                ///< co-resident blocks of that kernel (cooperative launch)

  scratch_buffer<unsigned long long> packed;  ///< (distance << b | parent) words
  scratch_buffer<int> stamp;                  ///< last generation a vertex was listed
  scratch_buffer<int> in_far;                 ///< vertex is in the far pile
  scratch_buffer<int> flag;                   ///< invalidation marks
  scratch_buffer<vertex_t> ancestor;          ///< pointer-jumping ancestors
  scratch_buffer<vertex_t> list_a;            ///< near frontier
  scratch_buffer<vertex_t> list_b;            ///< next near frontier
  scratch_buffer<vertex_t> far_a;             ///< far pile
  scratch_buffer<vertex_t> far_b;             ///< far pile being rebuilt
  scratch_buffer<vertex_t> candidates;        ///< invalidated vertices and insertion heads
  scratch_buffer<vertex_t> frontier;          ///< vertices improved by the pull pass
  scratch_buffer<unsigned char> control;      ///< device control block
  buffer<unsigned char> host_control;         ///< pinned copy of the control block

  std::vector<vertex_t> changed_from;            ///< host: tails of the changed edges
  std::vector<vertex_t> changed_to;              ///< host: heads of the changed edges
  scratch_buffer<vertex_t> device_changed_from;  ///< device copy of changed_from
  scratch_buffer<vertex_t> device_changed_to;    ///< device copy of changed_to
  scratch_buffer<vertex_t> device_insert_heads;  ///< device copy of the insertion heads

  /**
   * @brief Size the arrays for `requested` vertices (no-op if large enough) and clear stamp,
   *        in_far and flag, ordered on the stream of `res` (SospWorkspace::reserve).
   * @param[in] res       Resources of the CUDA backend.
   * @param[in] requested Number of vertices.
   * @throws out_of_memory_error if device memory runs out.
   */
  void reserve(const resources& res, std::int64_t requested);

  /**
   * @brief A fresh stamp generation; clears the stamps (on the stream of `res`) well before the
   *        counter could wrap (SospWorkspace::nextGeneration).
   * @param[in] res Resources of the CUDA backend.
   * @return The new generation (> 0).
   */
  int next_generation(const resources& res);

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of every array (device, pinned and host).
   */
  [[nodiscard]] std::size_t bytes() const noexcept override;
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
  std::vector<distance_t> distances;  ///< host backends: distances, infinite_distance() if none
  std::vector<vertex_t> parents;      ///< host backends: parents, -1 for none
  memory_space space = memory_space::host;  ///< where the arrays live (device for cuda)
  int device = -1;                          ///< CUDA device of device arrays, -1 otherwise
  buffer<distance_t> device_distances;      ///< cuda: distances (device memory)
  buffer<vertex_t> device_parents;          ///< cuda: parents (device memory)

  /**
   * @brief The number of vertices of either placement.
   * @return The array length.
   */
  [[nodiscard]] std::size_t num_vertices() const noexcept {
    return space == memory_space::host ? distances.size() : device_distances.size();
  }

  /**
   * @brief The distance array of either placement.
   * @return A mutable pointer (device memory for cuda).
   */
  [[nodiscard]] distance_t* distance_data() noexcept {
    return space == memory_space::host ? distances.data() : device_distances.data();
  }

  /**
   * @brief The parent array of either placement.
   * @return A mutable pointer (device memory for cuda).
   */
  [[nodiscard]] vertex_t* parent_data() noexcept {
    return space == memory_space::host ? parents.data() : device_parents.data();
  }
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
 * @brief Whether the OpenMP engine packs (distance, parent) into one 64-bit word (MOSP's
 *        choosePacking()): the parent needs ceil(log2(n + 1)) bits and the largest candidate
 *        distance, (n - 1) * max_weight + max_weight, the rest. If not, that engine keeps
 *        distances only and recovers every parent with the lowest-id rule after the search; the
 *        sequential engine then does the same, so both return the same tree.
 * @param[in] num_vertices Number of vertices.
 * @param[in] max_weight   Largest weight (values below 1 count as 1).
 * @return true if the parents are packed next to the distances.
 */
bool sssp_packs_parents(std::int64_t num_vertices, std::int64_t max_weight);

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
  sssp_graph<vertex_t, edge_t, weight_t> graph;      ///< the graph after the batch, one objective
  const sssp_changes<vertex_t>* changes = nullptr;   ///< nullptr for compute()
  vertex_t source = 0;                               ///< source
  std::int64_t delta = 1;                            ///< near-far width (> 0)
  std::int64_t max_weight = 1;                       ///< largest weight before or after the batch
  std::int64_t* distances = nullptr;                 ///< in: old tree; out: new tree
  vertex_t* parents = nullptr;                       ///< in: old tree; out: new tree
  sssp_workspace<vertex_t>* ws = nullptr;            ///< host scratch (leased from the pool)
  sssp_cuda_workspace<vertex_t>* cuda_ws = nullptr;  ///< device scratch (cuda backend)
  sssp_counters counters;                            ///< out
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

/**
 * @brief CUDA backend, fused engine: the update (MOSP-CUDA's sospUpdateGpu(), the persistent
 *        cooperative kernel), timed as the profiler stage sssp.enact_fused.
 *
 * `run.graph` and `run.changes` hold device pointers, `run.distances` / `run.parents` the device
 * arrays of the result, `run.cuda_ws` the leased device workspace. Synchronizes the stream once
 * (the control block is read back).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in,out] run The run.
 * @throws invalid_argument_error if an input distance does not fit the packing (the tree does not
 *         belong to the graph) or the input tree has a parent cycle.
 * @throws not_supported_error    if the kernel cannot be launched cooperatively.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

/**
 * @brief CUDA backend, fused engine: compute() (MOSP-CUDA's sospFromScratchGpu()).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in,out] run The run (device pointers; no changes).
 * @throws not_supported_error if the kernel cannot be launched cooperatively.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run);

}  // namespace dyng::detail
