// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:headers/combinedGraphGpu.cuh (CombineWorkspace, CombineStats),
// headers/mospUpdate.cuh (MospOptions, MospResult) and MOSP-OpenMP@c352151:headers/
// combinedGraphCpu.h (CombineWorkspace of the host)
/**
 * @file problem.hpp
 * @brief The mosp problem: K sssp problems composed through the framework, and the finalize step
 *        (the combined graph, its static solve, the path costs); its state and workspaces.
 *
 * Template card (PLAN Section 4.5.2), mosp as a composition (PLAN 4.5.2, "Composition"):
 *
 *     normalize -> translate -> prepare -> [before_apply] -> commit ->
 *     identify_affected -> seed -> { FP: loop until is_converged } -> finalize
 *
 *   - normalize, commit: run_update() (stages mosp.normalize under set semantics, mosp.commit),
 *     once for the K objectives, as MOSP's mospUpdate() applies the batch once;
 *   - translate: the objective projection. Objective k is an sssp problem on the k-th weight
 *     column (sssp::options::objective), and the commit classifies every insertion per objective
 *     (apply_delta::weight_increased, MOSP's weightIncreaseMask) for sssp's prepare;
 *   - prepare .. finalize of each objective: the K sssp problems, each through its own
 *     update_enactor (framework::problem_participant<sssp_problem>), one after the other on the
 *     handle's one sssp workspace (ADR 0015; MOSP's shared SospWorkspace), each inside the stage
 *     mosp.objective;
 *   - finalize (mosp's own): Steps 2-3 of MOSP_Update. The combined graph of the K trees (count,
 *     scan, fill: stage mosp.combine; MOSP's combinedGraphSospGpu / combinedGraphSospCpu), its
 *     static solve through sssp's engine (mosp.combined_sssp; MOSP's sospFromScratch*), the
 *     `affected` count against the previous MOSP tree (mosp.finalize), and the path costs on the
 *     host (mosp.path_costs; MOSP's mospPathCosts).
 *
 * compute() runs K sssp::compute() (each in mosp.objective) and the same finalize.
 * mosp_problem is the participant of one mosp result in run_update(): it owns the K sssp
 * participants and runs the finalize after them; dyng::update(res, g, b, mosp_result, ...)
 * composes it with any other result through the same interface.
 */
#pragma once

#include "algorithms/sssp/problem.hpp"
#include "framework/budgets.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail {

/**
 * @brief The state behind mosp::result.
 * @tparam vertex_t   Vertex id type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename distance_t>
struct mosp_state {
  vertex_t source = 0;            ///< source vertex
  mosp::options opt;              ///< options
  int num_objectives = 0;         ///< K
  std::int64_t scale = 1;         ///< L = lcm(preferences)
  std::uint64_t version = 0;      ///< graph version matched
  std::uint64_t graph_state = 0;  ///< state identifier of the graph matched (graph_impl)
  bool poisoned = false;          ///< a failed update left the arrays inconsistent
  /// The K trees (objective k: sssp::options::objective == k).
  std::vector<sssp::result<vertex_t, distance_t>> objectives;
  memory_space space = memory_space::host;  ///< where the combined arrays live (device for cuda)
  int device = -1;                          ///< CUDA device of device arrays, -1 otherwise
  /// The MOSP tree and its distances (host backends), and the arrays the next solve writes (the
  /// previous tree stays for the `affected` count; the two pairs are swapped after it).
  std::vector<distance_t> combined_distances, spare_distances;
  std::vector<vertex_t> combined_parents, spare_parents;  ///< see combined_distances
  buffer<distance_t> device_combined_distances;           ///< cuda: combined distances (device)
  buffer<distance_t> device_spare_distances;              ///< cuda: the next solve's distances
  buffer<vertex_t> device_combined_parents;               ///< cuda: the MOSP tree (device)
  buffer<vertex_t> device_spare_parents;                  ///< cuda: the next solve's parents
  std::vector<distance_t> path_costs;  ///< n * K, vertex-major (host, every backend)
  bool has_path_costs = false;         ///< path_costs belongs to the current tree

  /**
   * @brief The number of vertices of the combined arrays.
   * @return n.
   */
  [[nodiscard]] std::size_t num_vertices() const noexcept {
    return space == memory_space::host ? combined_distances.size()
                                       : device_combined_distances.size();
  }
};

/**
 * @brief Internal access to mosp::result (for the implementation and the tests).
 */
struct mosp_access {
  /**
   * @brief The state of a result.
   * @tparam vertex_t   Vertex id type.
   * @tparam distance_t Distance type.
   * @param[in] r The result.
   * @return Its state (throws for a moved-from result).
   */
  template <typename vertex_t, typename distance_t>
  static mosp_state<vertex_t, distance_t>& state(mosp::result<vertex_t, distance_t>& r);

  /**
   * @brief Wrap a state into a result.
   * @tparam vertex_t   Vertex id type.
   * @tparam distance_t Distance type.
   * @param[in] state The state.
   * @return The result.
   */
  template <typename vertex_t, typename distance_t>
  static mosp::result<vertex_t, distance_t> make(
      std::unique_ptr<mosp_state<vertex_t, distance_t>> state);
};

/**
 * @brief What the combine step reads: the K parent arrays and the preference terms (the
 *        parameters of MOSP's countEdgesKernel / fillEdgesKernel, passed by value to the kernels).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct mosp_combine_input {
  vertex_t num_vertices = 0;                           ///< n
  vertex_t source = 0;                                 ///< the source (no in-edge)
  int num_objectives = 0;                              ///< K
  std::int32_t base = 0;                               ///< L * (K + 1)
  const vertex_t* parents[mosp::max_objectives] = {};  ///< K parent arrays (n each)
  std::int32_t terms[mosp::max_objectives] = {};       ///< L / Pref_k
};

/**
 * @brief The combined graph built by a combine step.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct mosp_combined {
  sssp_graph<vertex_t, edge_t, weight_t> view;  ///< out-edges (in-edges when built)
  std::int64_t edges = 0;                       ///< number of edges
  std::int64_t weight_sum = 0;                  ///< sum of the weights (the default delta)
};

/**
 * @brief Host scratch of the finalize step (MOSP-OpenMP's CombineWorkspace and the arrays of
 *        mospPathCosts), leased from the handle's pool and shared by every mosp result run
 *        through it.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct mosp_workspace final : pooled_workspace {
  std::vector<edge_t> row_ptr;        ///< n + 1 (children CSR of the combined graph)
  std::vector<edge_t> cursor;         ///< n + 1 (degrees, then fill positions)
  std::vector<vertex_t> col_ind;      ///< the edges' heads
  std::vector<weight_t> weights;      ///< the edges' weights
  std::vector<edge_t> in_row_ptr;     ///< n + 1 (in-edges; sequential distance-only mode only)
  std::vector<vertex_t> in_col_ind;   ///< the in-edges' tails
  std::vector<weight_t> in_weights;   ///< the in-edges' weights
  std::vector<vertex_t> child_start;  ///< n + 1 (path costs: children lists of the MOSP tree)
  std::vector<vertex_t> children;     ///< n
  std::vector<vertex_t> queue;        ///< n (path costs: the traversal order)

  /**
   * @brief Size the arrays for `n` vertices and `k` trees (no-op if large enough): capacities
   *        only, as MOSP-OpenMP's vectors grow on their first use.
   * @param[in] n Number of vertices.
   * @param[in] k Number of objectives.
   */
  void reserve(std::size_t n, int k);

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of every array's capacity.
   */
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/**
 * @brief Device scratch of the combine step (MOSP-CUDA's CombineWorkspace), leased from the pool
 *        of a CUDA handle and shared by every mosp result run through it.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct mosp_cuda_workspace final : pooled_workspace {
  std::int64_t capacity = 0;                ///< vertices the arrays are sized for
  int trees = 0;                            ///< objectives the arrays are sized for
  scratch_buffer<edge_t> row_ptr;           ///< n + 1 (children CSR of the combined graph)
  scratch_buffer<edge_t> cursor;            ///< n + 1 (degrees, then fill positions)
  scratch_buffer<vertex_t> col_ind;         ///< K * n
  scratch_buffer<weight_t> weights;         ///< K * n
  scratch_buffer<unsigned long long> sums;  ///< [edge count, weight sum, affected]
  scratch_buffer<unsigned char> scan_temp;  ///< CUB scan temporary storage
  buffer<unsigned long long> host_sums;     ///< pinned copy of sums
  buffer<vertex_t> host_parents;            ///< pinned copy of the MOSP tree (path costs)

  /**
   * @brief Size the arrays for `n` vertices and `k` trees (no-op if large enough), ordered on the
   *        stream of `res` (CombineWorkspace::reserve).
   * @param[in] res Resources of the CUDA backend.
   * @param[in] n   Number of vertices.
   * @param[in] k   Number of objectives.
   * @throws out_of_memory_error if device or pinned memory runs out.
   */
  void reserve(const resources& res, std::int64_t n, int k);

  /**
   * @brief The memory the workspace holds.
   * @return Bytes of every array.
   */
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/**
 * @brief Steps 2 of the MOSP update on the sequential backend: the combined graph of the K trees
 *        (count, prefix sum, fill), and its in-edges if `with_in_edges`.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     in            The parent arrays (host memory) and the preference terms.
 * @param[in,out] ws            The leased host workspace (sized).
 * @param[in]     with_in_edges Also build the in-edge CSR (the sequential engine's distance-only
 *                              parent recovery reads it).
 * @return The combined graph (pointers into `ws`).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_sequential(
    const mosp_combine_input<vertex_t>& in, mosp_workspace<vertex_t, edge_t, weight_t>& ws,
    bool with_in_edges);

/**
 * @brief Step 2 on the OpenMP backend (MOSP-OpenMP's combinedGraphSospCpu, Step 2).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources (thread count).
 * @param[in]     in  The parent arrays (host memory) and the preference terms.
 * @param[in,out] ws  The leased host workspace (sized).
 * @return The combined graph (out-edges only; pointers into `ws`).
 * @throws not_supported_error if the OpenMP backend is not built.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_openmp(
    const resources& res, const mosp_combine_input<vertex_t>& in,
    mosp_workspace<vertex_t, edge_t, weight_t>& ws);

/**
 * @brief Step 2 on the CUDA backend (MOSP-CUDA's combinedGraphSospGpu, Step 2: countEdgesKernel,
 *        CUB exclusive scan, fillEdgesKernel); synchronizes the stream once (the edge count and
 *        the weight sum are read back).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res Resources of the CUDA backend.
 * @param[in]     in  The parent arrays (device memory) and the preference terms.
 * @param[in,out] ws  The leased device workspace (sized).
 * @return The combined graph (device pointers into `ws`; no in-edges).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_cuda(
    const resources& res, const mosp_combine_input<vertex_t>& in,
    mosp_cuda_workspace<vertex_t, edge_t, weight_t>& ws);

/**
 * @brief The end of the finalize step on CUDA: count the vertices whose combined distance or
 *        parent differs between the previous and the new MOSP tree (when `count`), and copy the
 *        new tree to the workspace's pinned host array (when `download`); one synchronization.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res          Resources of the CUDA backend.
 * @param[in]     n            Number of vertices.
 * @param[in]     old_distance The previous combined distances (device).
 * @param[in]     old_parent   The previous MOSP tree (device).
 * @param[in]     new_distance The new combined distances (device).
 * @param[in]     new_parent   The new MOSP tree (device).
 * @param[in]     count        Count the changed vertices.
 * @param[in]     download     Copy new_parent to ws.host_parents.
 * @param[in,out] ws           The leased device workspace.
 * @return The number of changed vertices (0 without `count`).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::int64_t mosp_finish_cuda(const resources& res, std::int64_t n,
                              const std::int64_t* old_distance, const vertex_t* old_parent,
                              const std::int64_t* new_distance, const vertex_t* new_parent,
                              bool count, bool download,
                              mosp_cuda_workspace<vertex_t, edge_t, weight_t>& ws);

/**
 * @brief Step 3, last line: the K objective values of the MOSP path to every vertex (MOSP's
 *        mospPathCosts, on the host): the children lists of the tree, then a traversal from the
 *        source; the weights of a tree edge (p, v) are those of the first edge from p to v in p's
 *        row.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     out    The graph's out-edges (host memory; weight columns 0..K-1 are used).
 * @param[in]     parent The MOSP tree (host memory, n entries).
 * @param[in]     source The source.
 * @param[in]     k      Number of objectives K.
 * @param[out]    costs  n * K values, vertex-major.
 * @param[in,out] ws     The leased host workspace (sized).
 * @return The first vertex whose tree edge (parent, v) is not an edge of the graph, or -1.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t mosp_path_costs(const csr_view<vertex_t, edge_t, weight_t>& out, const vertex_t* parent,
                         vertex_t source, int k, std::int64_t* costs,
                         mosp_workspace<vertex_t, edge_t, weight_t>& ws);

}  // namespace dyng::detail
