// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph_impl.hpp
 * @brief The private state of graph<V,E,W> and the internal access used by the algorithms.
 *
 * Algorithms (cpp/src/algorithms/<algo>) reach the storage through detail::graph_access. The
 * update of an algorithm applies the batch with graph_access::apply(), which also returns the
 * per-edge classification (apply_delta) the incremental engines need.
 */
#pragma once

#include "graph/device_graph.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dyng::detail {

/**
 * @brief The effective changes of one applied batch, in the order they were applied.
 *
 * This is the per-edge classification of MOSP's applyChangeBatch (its weightIncreaseMask),
 * widened to any number of objectives: one byte per (insertion, objective).
 *
 * - insertions: every insertion that reached the rows (after self-loop dropping; for an
 *   undirected graph each direction), in batch order, including upserts and ignored ones;
 * - weight_increased[i * num_weights + k] = 1 if insertion i names an edge that existed in the
 *   graph before the batch and the final objective-k weight of the first (u,v) in row u is larger
 *   than the objective-k weight of the first (u,v) of row u before the batch (MOSP semantics);
 * - deletions: every requested deletion with both ends in range (after self-loop dropping; for an
 *   undirected graph each direction), in batch order, whether or not it matched an edge.
 *
 * Under batch_semantics::as_sets (batch_semantics::set()) the lists are instead the normalized
 * batch, the net structural change: the deletions of existing edges and the insertions of new (or
 * deleted and re-inserted) edges, each list sorted by (source, destination) without repeats
 * (CycleEnumeration-GPU's prepare_batch), and every weight-increase flag is 0.
 *
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct apply_delta {
  int num_weights = 0;                         ///< objectives per insertion
  std::vector<vertex_t> insert_src;            ///< tails of the insertions
  std::vector<vertex_t> insert_dst;            ///< heads of the insertions
  std::vector<std::uint8_t> weight_increased;  ///< num_weights flags per insertion
  std::vector<vertex_t> delete_src;            ///< tails of the deletions
  std::vector<vertex_t> delete_dst;            ///< heads of the deletions
};

/**
 * @brief A process-wide unique identifier for a graph state (never 0).
 *
 * Every graph construction and every applied batch draws a new one; clone() copies it (the
 * content is identical). Results remember the identifier of the state they match, so a result is
 * detected as stale when it is used with another graph or with a graph variable that was
 * reassigned, even when the two per-graph version counters happen to be equal.
 * @return A value never returned before in this process.
 */
std::uint64_t next_graph_state_id() noexcept;

/**
 * @brief The state behind graph<V,E,W>: properties, version, host CSR storage and, for a graph
 *        built with CUDA resources, the resident device copy.
 *
 * Placement (PLAN Section 4.6 rule 5): a graph belongs to the backend class of the resources that
 * built it (`home`): host (sequential, openmp) or cuda (one device). Algorithms refuse resources
 * of the other class (graph_access::expect_placement); clone(res) moves a graph to the class of
 * `res`. A CUDA graph keeps its authoritative CSR in host memory in this release (the batch is
 * applied on the host, as MOSP-CUDA applies it) and a device copy (out- and in-edges,
 * objective-major weights) that is uploaded on first use per graph state (device_edges()) and
 * dropped by every applied batch, as MOSP-CUDA uploads the updated graph once per batch.
 *
 * The in-edges (when `props.store_transposed`) are built on first use and cached until the
 * out-edges change: building a graph does not transpose it, and an applied batch drops the stale
 * in-edges. Many uses never read the in-edges of a state (a graph built only to receive a batch,
 * the graph before a batch), so they never pay the transposition (M1b; see the sssp page).
 * Reading them is thread-safe (concurrent read-only calls may trigger the build; one thread
 * builds, the others wait); dropping them happens only in mutating calls.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_impl {
 public:
  using csr_type = csr<vertex_t, edge_t, weight_t>;  ///< the storage type

  graph_impl() = default;  ///< an empty state (no vertices; in-edges not built)

  /**
   * @brief A deep copy (the in-edges too, if they are built).
   * @param[in] other The state to copy.
   */
  graph_impl(const graph_impl& other);

  graph_impl& operator=(const graph_impl&) = delete;  ///< not assignable
  graph_impl(graph_impl&&) = delete;                  ///< not movable
  graph_impl& operator=(graph_impl&&) = delete;       ///< not movable
  ~graph_impl() = default;                            ///< releases the storage

  graph_properties props;                          ///< properties (num_weights as stored)
  std::uint64_t version = 0;                       ///< +1 per applied batch
  std::uint64_t state_id = next_graph_state_id();  ///< unique per state; copied by clone()
  csr_type out;                                    ///< out-edges
  int build_threads = 1;  ///< host threads for a transposition requested without resources
  backend home = backend::sequential;  ///< backend of the resources that built the graph
  int home_device = -1;                ///< CUDA device of a cuda graph, -1 otherwise

  /// The device copy type.
  using device_type = device_graph<vertex_t, edge_t, weight_t>;

  /**
   * @brief The in-edges, built from the out-edges on first use (thread-safe).
   *
   * Precondition: props.store_transposed. Deterministic: the same arrays for every thread count.
   * @param[in] threads Host threads for the transposition (OpenMP backend; 1 = sequential).
   * @return The in-edge CSR (objective-major weights), valid until the out-edges change.
   */
  const csr_type& in_edges(int threads) const;

  /**
   * @brief Whether the in-edges of the current state are built.
   * @return true if in_edges() returns without work.
   */
  [[nodiscard]] bool has_in_edges() const noexcept {
    return in_built_.load(std::memory_order_acquire);
  }

  /**
   * @brief Drop the in-edges after the out-edges changed (only in mutating calls).
   */
  void drop_in_edges() noexcept;

  /**
   * @brief The in-edge storage as it is (for graph::reserve; not built by this call).
   * @return The storage.
   */
  csr_type& in_storage() noexcept {
    return in_;
  }

  /**
   * @brief The device copy of the current state, uploaded on first use (thread-safe; profiler
   *        stage graph.upload when it is built).
   *
   * Precondition: home == backend::cuda and `res` is a CUDA handle of home_device.
   * @param[in] res Resources of the CUDA backend (stream and memory resource of the upload).
   * @return The device graph, valid until the out-edges change.
   */
  const device_type& device_edges(const resources& res) const;

  /**
   * @brief Whether the device copy of the current state is built.
   * @return true if device_edges() returns without work.
   */
  [[nodiscard]] bool has_device_edges() const noexcept {
    return device_built_.load(std::memory_order_acquire);
  }

  /**
   * @brief Drop the device copy after the out-edges changed (only in mutating calls).
   */
  void drop_device_edges() noexcept;

 private:
  mutable csr_type in_;                            ///< in-edges (valid if in_built_)
  mutable std::mutex in_mutex_;                    ///< serializes the lazy build
  mutable std::atomic<bool> in_built_{false};      ///< in_ matches out
  mutable std::unique_ptr<device_type> device_;    ///< device copy (valid if device_built_)
  mutable std::mutex device_mutex_;                ///< serializes the upload
  mutable std::atomic<bool> device_built_{false};  ///< device_ matches out
};

/**
 * @brief Internal access to a graph's storage (for the algorithms and the tests).
 */
struct graph_access {
  /**
   * @brief The state of a graph.
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] g The graph.
   * @return Its implementation object.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static const graph_impl<vertex_t, edge_t, weight_t>& impl(
      const graph<vertex_t, edge_t, weight_t>& g) {
    return g.impl();
  }

  /**
   * @brief Apply a batch and report the effective changes (graph::apply plus classification).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in]     res   Execution resources.
   * @param[in,out] g     The graph (version + 1).
   * @param[in]     batch The batch.
   * @param[out]    delta The effective changes (may be nullptr).
   * @return What the batch did.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static apply_summary apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                             const edge_batch_view<vertex_t, weight_t>& batch,
                             apply_delta<vertex_t>* delta);

  /**
   * @brief The out-edges alone (never builds the in-edges).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] g The graph.
   * @return A view of the out-edge CSR, valid until the graph changes.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static csr_view<vertex_t, edge_t, weight_t> out_view(const graph<vertex_t, edge_t, weight_t>& g) {
    return g.impl().out.view();
  }

  /**
   * @brief The full view; builds the in-edges with the threads of `res` if they are stored and not
   *        built yet (profiler stage graph.transpose).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] res Execution resources (thread count, profiler).
   * @param[in] g   The graph.
   * @return The view, valid until the graph changes.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static graph_view<vertex_t, edge_t, weight_t> view(const resources& res,
                                                     const graph<vertex_t, edge_t, weight_t>& g);

  /**
   * @brief The device copy of a CUDA graph, uploaded on first use per graph state.
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] res Resources of the CUDA backend (the graph's device).
   * @param[in] g   A graph built with CUDA resources.
   * @return The device graph, valid until the graph changes.
   * @throws invalid_argument_error if the placement of `g` does not match `res`.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static const device_graph<vertex_t, edge_t, weight_t>& device(
      const resources& res, const graph<vertex_t, edge_t, weight_t>& g);

  /**
   * @brief Build what the engines of `res` read after a commit, once for every participant: the
   *        host in-edges (host backends, if stored) or the device copy (cuda).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] res Execution resources.
   * @param[in] g   The graph.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static void prepare(const resources& res, const graph<vertex_t, edge_t, weight_t>& g);

  /**
   * @brief Check that an algorithm may run on `g` with `res` (PLAN Section 4.6 rule 5): a graph
   *        built with CUDA resources needs CUDA resources of the same device, a host graph needs
   *        host resources. Never a silent copy of a whole graph.
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] res  Execution resources.
   * @param[in] g    The graph.
   * @param[in] what The calling function, for the message.
   * @throws invalid_argument_error on a mismatch (the message names g.clone(res)).
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static void expect_placement(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                               const char* what);
};

}  // namespace dyng::detail
