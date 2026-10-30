// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:headers/iterativePropagation.h (DegreeBin,
// degreeBin, SMALL_FRONTIER_LIMIT, COUNTERS_PER_SET, PropagationWorkspace),
// headers/connectedComponents.h (CCWorkspace), headers/labelInitialization.h
// (LabelInitWorkspace), headers/sparsifySubgraph.h, headers/validation.h and src/dynLP.cu
// (runDynLP's device arrays and batch buffers)
/**
 * @file fused.hpp
 * @brief The fused CUDA engine of label_propagation (Tier B; ADR 0075): the verbatim transplant
 *        of LabelPropagation-CUDA's pipeline. Host-only declarations (buffers, no CUDA headers);
 *        the kernels and their launches live in the CUDA TUs of this directory, one per TU of the
 *        original: sparsify.cu (sparsifySubgraph.cu), components.cu (connectedComponents.cu),
 *        label_init.cu (labelInitialization.cu), propagation.cu (iterativePropagation.cu),
 *        irlp.cu (validation.cu) and cuda.cu (dynLP.cu's batch, on the framework's hooks).
 *
 * The pipeline of one update (runDynLP, one batch), on the one stream of the resources (AD21: the
 * library never creates a stream; the original's copy stream and its two events become the
 * stream order: the ids, then the edges, are copied by Step 0, before the kernels that wait for
 * them in the original):
 *   - before_apply (G_t): the affected flags cleared, launchMarkDeletions on the rows before the
 *     batch;
 *   - the commit: the graph's device apply (graph/device_slotted.hpp);
 *   - enact_fused: the seeds' classes (setGroundTruthKernel's gtClass), launchMarkInserted, then,
 *     when vertices were inserted, launchBuildSparsifiedSubgraph, launchConnectedComponents and
 *     launchLabelInitialization; launchIterativePropagation (with enforceGroundTruthKernel); the
 *     graph's report settled after the first synchronization; the stats.
 * compute() is IrLP (runIrLPSnapshot: the column-sorted snapshot of the current graph, built on
 * the device, then launchIrLP).
 */
#pragma once

#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail::lp_cuda {

/// The degree bins of the frontier (iterativePropagation.h: DegreeBin).
enum degree_bin : int {
  bin_small = 0,   ///< deg <= small_bin_max_degree: 8 lanes per vertex
  bin_medium = 1,  ///< deg <= medium_bin_max_degree: one warp per vertex
  bin_large = 2,   ///< larger degrees: one block per vertex
  num_degree_bins = 3,
};

inline constexpr int small_bin_max_degree = 32;               ///< SMALL_BIN_MAX_DEGREE
inline constexpr int medium_bin_max_degree = 1024;            ///< MEDIUM_BIN_MAX_DEGREE
inline constexpr int small_frontier_limit = 1024;             ///< SMALL_FRONTIER_LIMIT
inline constexpr int counters_per_set = 2 * num_degree_bins;  ///< COUNTERS_PER_SET

/**
 * @brief degreeBin on the host (the queue capacities use the kernels' boundaries).
 * @param[in] degree A degree.
 * @return Its bin.
 */
constexpr int degree_bin_of(std::int64_t degree) noexcept {
  if (degree <= small_bin_max_degree) {
    return bin_small;
  }
  if (degree <= medium_bin_max_degree) {
    return bin_medium;
  }
  return bin_large;
}

/**
 * @brief The run-wide device state of a CUDA result (the original's d_labels, d_gtClass,
 *        d_newStamp and the propagation's d_stamp and nextTag): it lives as long as the result.
 */
struct state {
  int device = -1;                 ///< the device of the arrays
  std::int64_t n = 0;              ///< vertices
  buffer<float> labels;            ///< F per vertex
  buffer<std::int32_t> gt_class;   ///< the seed class per vertex, -1 none (d_gtClass)
  buffer<std::int32_t> new_stamp;  ///< the batch stamp of the vertices inserted (d_newStamp)
  buffer<std::int32_t> stamp;      ///< the propagation's queue stamps (ws.d_stamp)
  int next_tag = 1;                ///< ws.nextTag
  int capacity[num_degree_bins] = {0, 0, 0};  ///< queue capacities from the row capacities

  /**
   * @brief The device memory held.
   * @return Bytes.
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return labels.size() * sizeof(float) +
           (gt_class.size() + new_stamp.size() + stamp.size()) * sizeof(std::int32_t);
  }
};

/**
 * @brief Allocate the run-wide state for `n` vertices: every label 0.5 (or the seeds' classes),
 *        the stamps 0, the queue capacities from the row capacities (createPropagationWorkspace).
 * @param[in]  res       Resources of the CUDA backend.
 * @param[in]  n         Vertices.
 * @param[in]  row_ptr   The host slot offsets (n + 1; a row's capacity is its maximum degree).
 * @param[in]  seeds     Per vertex -1 or the class of a seed (host, n values).
 * @param[in]  labels    The starting labels (host, n values).
 * @param[out] out       The state.
 * @tparam edge_t Edge offset type.
 * @throws out_of_memory_error, cuda_error.
 */
template <typename edge_t>
void create_state(const resources& res, std::int64_t n, const edge_t* row_ptr,
                  const std::int8_t* seeds, const float* labels, state& out);

/**
 * @brief The run-wide state from a host result (result::clone to CUDA).
 * @param[in]  res      Resources of the CUDA backend.
 * @param[in]  n        Vertices.
 * @param[in]  capacity The queue capacities.
 * @param[in]  seeds    Per vertex -1 or a seed's class (host, n).
 * @param[in]  labels   The labels (host, n).
 * @param[in]  stamp    The queue stamps (host, n).
 * @param[in]  next_tag The next tag.
 * @param[out] out      The state.
 * @throws capacity_error if a tag does not fit the device's 32-bit stamps; out_of_memory_error,
 *         cuda_error.
 */
void upload_state(const resources& res, std::int64_t n, const int (&capacity)[num_degree_bins],
                  const std::int8_t* seeds, const float* labels, const std::int64_t* stamp,
                  std::int64_t next_tag, state& out);

/**
 * @brief The labels and stamps of a CUDA result on the host (result::clone to a host backend).
 * @param[in]  in     The state.
 * @param[out] labels The labels (resized to n).
 * @param[out] stamp  The queue stamps (resized to n).
 * @throws cuda_error.
 */
void download_state(const state& in, std::vector<float>& labels, std::vector<std::int64_t>& stamp);

/**
 * @brief A deep copy of the run-wide state (result::clone).
 * @param[in]  res Resources of the CUDA backend.
 * @param[in]  in  The state.
 * @param[out] out The copy.
 * @throws out_of_memory_error, cuda_error.
 */
void copy_state(const resources& res, const state& in, state& out);

/**
 * @brief The per-update scratch of the fused engine, leased from the workspace pool of the
 *        resources (ADR 0015): the propagation's queues and counters (PropagationWorkspace without
 *        its stamps), the CC workspace (CCWorkspace), the supernode accumulators
 *        (LabelInitWorkspace), G' (the batch buffers d_subEdgeSrc / d_subEdgeDst /
 *        d_numSubEdges), the affected flags and component ids, the pinned mirrors. Sized by
 *        reserve() for the vertices, the queue capacities and the largest batch; nothing is
 *        allocated once it is reserved.
 */
struct workspace final : pooled_workspace {
  // ---- per vertex ----
  scratch_buffer<std::int32_t> affected;      ///< d_isAffected
  scratch_buffer<std::int32_t> component_id;  ///< d_componentId
  scratch_buffer<std::int32_t> par;           ///< CCWorkspace::d_par
  scratch_buffer<std::int32_t> root_id;       ///< CCWorkspace::d_rootId (also the labels before)
  // ---- per inserted vertex / component ----
  scratch_buffer<std::int32_t> root_flag;        ///< CCWorkspace::d_rootFlag
  scratch_buffer<std::int32_t> root_scan;        ///< CCWorkspace::d_rootScan
  scratch_buffer<unsigned char> scan_temp;       ///< CCWorkspace::d_scanTemp
  scratch_buffer<unsigned long long> weight_l0;  ///< LabelInitWorkspace::d_weightL0
  scratch_buffer<unsigned long long> weight_l1;  ///< LabelInitWorkspace::d_weightL1
  // ---- per batch edge ----
  scratch_buffer<std::int32_t> sub_src;  ///< G' sources
  scratch_buffer<std::int32_t> sub_dst;  ///< G' destinations
  // ---- the propagation ----
  scratch_buffer<std::int32_t> queue[2][num_degree_bins];  ///< PropagationWorkspace::d_queue
  scratch_buffer<float> new_labels[num_degree_bins];       ///< d_newLabels (damped)
  // ---- control words (device) and their pinned mirrors ----
  scratch_buffer<std::int32_t> counts;  ///< 2 x counters_per_set queue sizes (d_counts)
  scratch_buffer<long long> control;    ///< the persistent kernel's results (d_control)
  scratch_buffer<std::int32_t> words;   ///< [0] changed, [1] numComponents, [2] numSubEdges,
                                        ///< [3] labels changed (the stats)
  buffer<std::int32_t> host_counts;     ///< h_counts (pinned)
  buffer<long long> host_control;       ///< h_control (pinned)
  buffer<std::int32_t> host_words;      ///< h_result and the stats (pinned)
  std::size_t scan_temp_bytes = 0;      ///< the CC scan's temporary storage
  std::int64_t sized_vertices = 0;      ///< reserve()'s n
  std::int64_t sized_inserted = 0;      ///< reserve()'s inserted vertices
  std::int64_t sized_edges = 0;         ///< reserve()'s batch edges
  int sized_capacity[num_degree_bins] = {0, 0, 0};  ///< reserve()'s queue capacities
  bool sized_damped = false;                        ///< reserve() sized the damped side arrays

  /**
   * @brief Size the scratch (a reservation when it grows; a no-op once it suffices).
   * @param[in] res      Resources of the CUDA backend.
   * @param[in] n        Vertices.
   * @param[in] capacity The queue capacities.
   * @param[in] inserted The largest number of inserted vertices of a batch.
   * @param[in] edges    The largest number of edges of a batch.
   * @param[in] damped   Whether the damped side arrays are needed.
   * @param[in] exact    Size the per-batch arrays exactly (result::reserve), not with the
   *                     headroom of a growth during an update.
   * @throws out_of_memory_error, cuda_error.
   */
  void reserve(const resources& res, std::int64_t n, const int (&capacity)[num_degree_bins],
               std::int64_t inserted, std::int64_t edges, bool damped, bool exact = false);

  /// Bytes held (device and pinned).
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/// The device rows the kernels read (the original's d_rowPtr / d_rowEnd / d_colInd / d_values and
/// d_vertexStatus).
template <typename edge_t>
struct rows {
  const edge_t* row_start = nullptr;      ///< first entry of each row
  const edge_t* row_end = nullptr;        ///< end of each row's entries
  const std::int32_t* col_ind = nullptr;  ///< the entries
  const float* values = nullptr;          ///< their weights
  const std::int32_t* status = nullptr;   ///< vertex states (vertex_status's values)
  std::int64_t n = 0;                     ///< vertices
};

/// What one propagation did (launchIterativePropagation's outputs and prof:: counters).
struct propagation_report {
  int iterations = 0;                   ///< iterationsUsed
  long long frontier_visits = 0;        ///< prop.frontierVisits
  long long persistent_iterations = 0;  ///< prop.persistentIterations
  long long initial_frontier = 0;       ///< the first queue's size
  int tag_before = 0;                   ///< ws.nextTag at the start (the trace's next_tag)
  bool capped = false;                  ///< stopped at the cap with a non-empty frontier
  std::int64_t host_syncs = 0;          ///< its host synchronizations
};

// ---- cuda.cu (dynLP.cu: one batch on the framework's hooks) ----

/// The inputs of one update on CUDA.
struct batch_args {
  float tau = 0.5F;                         ///< the threshold of G'
  float delta = 1e-4F;                      ///< the tolerance
  int max_iterations = 100000;              ///< the cap
  bool damped = false;                      ///< the schedule
  bool fallback = false;                    ///< on_limit::fallback_recompute
  int batch_stamp = 1;                      ///< the original's batchStamp (b + 1)
  int batch_no = 0;                         ///< the stage trace's batch number
  const std::int32_t* inserted = nullptr;   ///< Ins_eff on the host (the trace only)
  std::size_t num_inserted = 0;             ///< |Ins_eff|
  const std::int8_t* host_seeds = nullptr;  ///< the result's seeds on the host (the trace only)
  std::int32_t num_deleted = 0;             ///< |Del| (the trace only)
};

/// What one update on CUDA did.
struct batch_result {
  propagation_report propagation;  ///< the propagation
  int components = 0;              ///< components of G'
  int sparsified = 0;              ///< edges of G'
  int hook_rounds = 0;             ///< Shiloach-Vishkin rounds
  std::int64_t affected = 0;       ///< vertices whose label bits changed
  bool capped = false;             ///< the cap was reached with a non-empty frontier
  bool recomputed = false;         ///< on_limit::fallback_recompute ran IrLP
  int irlp_iterations = 0;         ///< its iterations
  std::int64_t host_syncs = 0;     ///< the host synchronizations of the algorithm phase
  std::uint32_t graph_errors = 0;  ///< the device apply's error bits (settled)
  std::int64_t appended = 0;       ///< half-edges the device apply appended
  std::int64_t dropped = 0;        ///< half-edges it dropped
  const char* error_detail = "";   ///< the description of graph_errors (static storage)
};

/**
 * @brief before_apply on G_t (runDynLP, b2): size the workspace, clear the affected flags, mark
 *        the neighbours of Del_eff on the rows before the batch (launchMarkDeletions).
 * @tparam edge_t Edge offset type.
 * @param[in]     res      Resources of the CUDA backend.
 * @param[in]     g        G_t (its batch uploaded by Step 0).
 * @param[in]     st       The result's device state.
 * @param[in,out] ws       The workspace (leased by the caller).
 * @param[in]     inserted The reserved (or this batch's) inserted vertices.
 * @param[in]     edges    The reserved (or this batch's) edges.
 * @param[in]     damped   The schedule.
 * @throws out_of_memory_error, cuda_error.
 */
template <typename edge_t>
void before_apply(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                  const state& st, workspace& ws, std::int64_t inserted, std::int64_t edges,
                  bool damped);

/**
 * @brief The rest of the batch on G_{t+1} (runDynLP, b3-b7; see the file comment).
 * @tparam edge_t Edge offset type.
 * @throws internal_error (the CC cap), out_of_memory_error, cuda_error.
 */
template <typename edge_t>
batch_result after_commit(const resources& res, const graph<std::int32_t, edge_t, float>& g,
                          state& st, workspace& ws, const batch_args& a);

/**
 * @brief compute() on CUDA: the run-wide state of a new result for `g` and IrLP from the seeds.
 * @tparam edge_t Edge offset type.
 * @param[in]  res            Resources of the CUDA backend (the graph's device).
 * @param[in]  g              The graph (slotted on the device, or compact: its device copy).
 * @param[in]  seeds          Per vertex -1 or a seed's class (host).
 * @param[in]  delta          The tolerance.
 * @param[in]  max_iterations The cap.
 * @param[out] st             The new state (its labels IrLP's).
 * @param[out] capped         Whether IrLP reached the cap.
 * @return IrLP's iterations.
 */
template <typename edge_t>
int compute(const resources& res, const graph<std::int32_t, edge_t, float>& g,
            const std::int8_t* seeds, float delta, int max_iterations, state& st, bool& capped);

// ---- sparsify.cu (sparsifySubgraph.cu) ----

/**
 * @brief launchMarkDeletions: the neighbours of the deleted vertices on the rows before the batch
 *        are affected.
 */
template <typename edge_t>
void launch_mark_deletions(const edge_t* row_start, const edge_t* row_end,
                           const std::int32_t* col_ind, std::int32_t* affected,
                           const std::int32_t* deleted, int num_deleted, int num_nodes,
                           void* stream);

/// launchMarkInserted: the inserted vertices are affected and stamped as new in this batch.
void launch_mark_inserted(const std::int32_t* inserted, int num_inserted, std::int32_t* affected,
                          std::int32_t* new_stamp, int batch_stamp, void* stream);

/// The seeds' classes of the batch (setGroundTruthKernel's d_gtClass, dynamicGraph.cu).
void launch_set_seed_classes(const std::int32_t* inserted, const std::int8_t* labels,
                             int num_inserted, std::int32_t* gt_class, void* stream);

/// launchBuildSparsifiedSubgraph: G' and the endpoints of the batch's edges affected.
void launch_build_sparsified_subgraph(std::int32_t* affected, const std::int32_t* src,
                                      const std::int32_t* dst, const float* weight, int num_edges,
                                      int num_nodes, const std::int32_t* status,
                                      const std::int32_t* new_stamp, int batch_stamp, float tau,
                                      std::int32_t* sub_src, std::int32_t* sub_dst,
                                      std::int32_t* num_sub, void* stream);

/// A batch without insertions: the endpoints of its edges are affected (the documented fix of
/// the original, which skips them; ADR 0072).
void launch_mark_edge_endpoints(std::int32_t* affected, const std::int32_t* src,
                                const std::int32_t* dst, int num_edges, int num_nodes,
                                const std::int32_t* status, void* stream);

// ---- components.cu (connectedComponents.cu) ----

/**
 * @brief launchConnectedComponents (Shiloach-Vishkin with one host synchronization per round; the
 *        64-round cap is an internal_error).
 * @return The hook rounds (cc.hookRounds).
 * @throws internal_error if the cap is reached.
 */
int launch_connected_components(const std::int32_t* sub_src, const std::int32_t* sub_dst,
                                const std::int32_t* num_sub, int max_edges,
                                const std::int32_t* inserted, int num_inserted,
                                std::int32_t* component_id, int& num_components, workspace& ws,
                                void* stream);

/// The CUB temporary storage of the CC scan for `max_inserted` vertices.
std::size_t components_scan_bytes(int max_inserted);

// ---- label_init.cu (labelInitialization.cu) ----

/**
 * @brief launchLabelInitialization (the supernode weights in 64-bit fixed point, then the labels).
 * @throws internal_error if there are more components than accumulators.
 */
template <typename edge_t>
void launch_label_initialization(const rows<edge_t>& g, float* labels,
                                 const std::int32_t* component_id, int num_components,
                                 const std::int32_t* inserted, int num_inserted,
                                 const std::int32_t* gt_class, workspace& ws, void* stream);

// ---- propagation.cu (iterativePropagation.cu) ----

/**
 * @brief launchIterativePropagation (the degree-binned queues, the persistent small-frontier
 *        block, the damped commit, enforceGroundTruthKernel). One host synchronization per
 *        head iteration and per persistent launch; `after_first_sync` runs right after the first
 *        (the initial frontier's), where the graph's report is settled.
 * @return What it did.
 */
template <typename edge_t>
propagation_report launch_iterative_propagation(const rows<edge_t>& g, float* labels,
                                                const std::int32_t* gt_class,
                                                const std::int32_t* affected, float delta,
                                                int max_iterations, bool damped, state& st,
                                                workspace& ws, void* stream,
                                                void (*after_first_sync)(void*), void* arg);

/// Count the vertices whose label bits differ from `before` (the stats' affected).
void launch_count_changed(const float* labels, const std::int32_t* before_bits, int n,
                          std::int32_t* count, void* stream);

/// Copy the labels' bits into `before_bits` (the labels before the update).
void launch_keep_labels(const float* labels, std::int32_t* before_bits, int n, void* stream);

// ---- irlp.cu (validation.cu) ----

/**
 * @brief The scratch of IrLP (compute()): the column-sorted snapshot and the second label buffer,
 *        leased from the pool.
 */
struct irlp_workspace final : pooled_workspace {
  scratch_buffer<unsigned char> row_ptr;    ///< n + 1 snapshot offsets (edge_t)
  scratch_buffer<unsigned char> count;      ///< n + 1 row counts (edge_t; the scan's input)
  scratch_buffer<std::int32_t> col_ind;     ///< snapshot columns (slot order)
  scratch_buffer<float> values;             ///< snapshot weights (slot order)
  scratch_buffer<std::int32_t> col_sorted;  ///< snapshot columns, each row sorted
  scratch_buffer<float> values_sorted;      ///< their weights
  scratch_buffer<unsigned char> temp;       ///< CUB scratch (scan, segmented sort)
  scratch_buffer<float> labels_buffer;      ///< the second label buffer
  scratch_buffer<std::int32_t> gt_class;    ///< the active seeds' classes (the snapshot's GT)
  scratch_buffer<std::int32_t> status;      ///< every vertex active (a compact graph's states)
  scratch_buffer<long long> words;          ///< [0] changed (d_changed), [1] the entries
  buffer<long long> host_words;             ///< pinned mirror
  /// Bytes held.
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/**
 * @brief runIrLPSnapshot on the device: the snapshot of the live entries of `g` (both endpoints
 *        active), each row sorted by column (the original's full graph's order), the labels 0.5
 *        except the active seeds' classes, then launchIrLP; an empty snapshot is skipped (the
 *        original's runIrLPValidation).
 * @tparam edge_t Edge offset type.
 * @param[in]  res            Resources of the CUDA backend.
 * @param[in]  g              The device rows.
 * @param[in]  gt_class       Per vertex the seed class, -1 none (device; deleted seeds included:
 *                            only active ones count).
 * @param[out] labels         The labels (device, n).
 * @param[in]  delta          The tolerance.
 * @param[in]  max_iterations The cap.
 * @param[in,out] ws          The scratch.
 * @param[out] capped         Whether the cap was reached with a label still changing.
 * @return The iterations (0 for an empty snapshot).
 * @throws out_of_memory_error, cuda_error.
 */
template <typename edge_t>
int run_irlp(const resources& res, const rows<edge_t>& g, const std::int32_t* gt_class,
             float* labels, float delta, int max_iterations, irlp_workspace& ws, bool& capped);

}  // namespace dyng::detail::lp_cuda
