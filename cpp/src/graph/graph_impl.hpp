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
#include "graph/normalized_batch.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <atomic>
#include <cstddef>
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
 * @brief The apply_delta of a batch applied under batch_semantics::as_sets: the normalized lists,
 *        with zero weight-increase flags.
 * @tparam vertex_t Vertex id type.
 * @param[in]  normalized The normalized batch.
 * @param[out] delta      Receives the lists (its vectors are resized).
 * @param[in]  num_weights The graph's weight columns.
 */
template <typename vertex_t>
void fill_set_delta(const normalized_batch<vertex_t>& normalized, apply_delta<vertex_t>& delta,
                    int num_weights = 0) {
  delta.num_weights = num_weights;
  delta.insert_src.resize(normalized.insertions.size());
  delta.insert_dst.resize(normalized.insertions.size());
  for (std::size_t i = 0; i < normalized.insertions.size(); ++i) {
    delta.insert_src[i] = normalized.insertions[i].source;
    delta.insert_dst[i] = normalized.insertions[i].target;
  }
  delta.weight_increased.assign(
      normalized.insertions.size() * static_cast<std::size_t>(num_weights), 0);
  delta.delete_src.resize(normalized.deletions.size());
  delta.delete_dst.resize(normalized.deletions.size());
  for (std::size_t d = 0; d < normalized.deletions.size(); ++d) {
    delta.delete_src[d] = normalized.deletions[d].source;
    delta.delete_dst[d] = normalized.deletions[d].target;
  }
}

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
 * `res`.
 *
 * A CUDA graph has two copies of its out-edges, at least one of which is current:
 *   - the host CSR, the authoritative copy of every graph built from host data; and
 *   - the device copy (device_graph), uploaded on first use per graph state (device_edges()).
 * A batch is applied on the host (the host CSR is replaced and the device copy dropped, as
 * MOSP-CUDA applies a batch and uploads the new graph) unless the graph is resident and the
 * batch can be merged on the device: batch_semantics::as_sets, no weight columns and 32-bit
 * vertex ids (graph_access::apply). Then the device copy becomes the new state
 * (apply_set_batch_device, CycleEnumeration-GPU's build_next_rows_kernel) and the host CSR is
 * stale until something reads it: host_edges() downloads it once (ADR 0020). num_vertices() and
 * num_edges() never download.
 *
 * The in-edges (when `props.store_transposed`) are built on first use and cached until the
 * out-edges change: building a graph does not transpose it, and an applied batch drops the stale
 * in-edges. Many uses never read the in-edges of a state (a graph built only to receive a batch,
 * the graph before a batch), so they never pay the transposition (M1b; see the sssp page). The
 * device copy builds its in-edges on first use as well (device_edges(res, true)).
 * Reading is thread-safe (concurrent read-only calls may trigger the lazy builds and the
 * download; one thread builds, the others wait); dropping and replacing happen only in mutating
 * calls.
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
   * @brief A deep copy of the host state (the host CSR, downloaded first if stale, and the
   *        in-edges if they are built; never the device copy).
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
  int build_threads = 1;  ///< host threads for a transposition requested without resources
  backend home = backend::sequential;  ///< backend of the resources that built the graph
  int home_device = -1;                ///< CUDA device of a cuda graph, -1 otherwise

  /// The device copy type.
  using device_type = device_graph<vertex_t, edge_t, weight_t>;

  /**
   * @brief The host out-edges, downloaded from the device copy first if a device apply left them
   *        stale (thread-safe; a synchronous copy on the graph's device).
   * @return The out-edge CSR, valid until the graph changes.
   */
  const csr_type& host_edges() const;

  /**
   * @brief The host out-edges for a mutating call (current: downloaded first if stale).
   * @return The out-edge CSR.
   */
  csr_type& host_edges_for_write() {
    (void)host_edges();
    return out_;
  }

  /**
   * @brief Whether the host out-edges are current (host_edges() returns without a download).
   * @return true unless a device apply produced the current state and nothing read it since.
   */
  [[nodiscard]] bool host_current() const noexcept {
    return host_current_.load(std::memory_order_acquire);
  }

  /**
   * @brief The vertex count of the current state (never downloads).
   * @return n.
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept;

  /**
   * @brief The edge count of the current state (never downloads).
   * @return m.
   */
  [[nodiscard]] edge_t num_edges() const noexcept;

  /**
   * @brief The number of weight columns.
   * @return K.
   */
  [[nodiscard]] int num_weights() const noexcept {
    return out_.num_weights;
  }

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
   *        stage graph.upload when it is uploaded, graph.transpose_device when its in-edges are
   *        built; the build waits for the stream of `res`, so the copy can be read on any stream
   *        afterwards).
   *
   * Precondition: home == backend::cuda and `res` is a CUDA handle of home_device.
   * @param[in] res      Resources of the CUDA backend (stream and memory resource of the upload).
   * @param[in] in_edges Whether the in-edges are needed too (built on the device on first use).
   * @return The device graph, valid until the out-edges change.
   */
  const device_type& device_edges(const resources& res, bool in_edges = true) const;

  /**
   * @brief Whether the device copy (out-edges) of the current state is resident.
   * @return true if device_edges(res, false) returns without work.
   */
  [[nodiscard]] bool has_device_edges() const noexcept {
    return device_built_.load(std::memory_order_acquire);
  }

  /**
   * @brief Drop the device copy after the out-edges changed on the host (only in mutating calls;
   *        the host CSR must be current).
   */
  void drop_device_edges() noexcept;

  /**
   * @brief Make a device copy built by a device apply the current state (only in mutating calls):
   *        the host CSR becomes stale and the in-edges are dropped.
   * @param[in] next The device copy of the new state (its work complete on its stream).
   */
  void replace_device_edges(std::unique_ptr<device_type> next) noexcept;

 private:
  mutable csr_type out_;                              ///< out-edges (current if host_current_)
  mutable std::mutex host_mutex_;                     ///< serializes the download
  mutable std::atomic<bool> host_current_{true};      ///< out_ matches the current state
  mutable csr_type in_;                               ///< in-edges (valid if in_built_)
  mutable std::mutex in_mutex_;                       ///< serializes the lazy build
  mutable std::atomic<bool> in_built_{false};         ///< in_ matches out
  mutable std::unique_ptr<device_type> device_;       ///< device copy (valid if device_built_)
  mutable std::mutex device_mutex_;                   ///< serializes the upload
  mutable std::atomic<bool> device_built_{false};     ///< device_ matches the current state
  mutable std::atomic<bool> device_in_built_{false};  ///< device_ has its in-edges
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
   * @param[in]     normalized Under batch_semantics::as_sets: Step 0 already computed for this
   *                graph state and batch (normalize_set_batch), or nullptr to compute it here.
   * @return What the batch did.
   *
   * A CUDA graph whose device copy is resident is updated on the device when the batch can be
   * merged there (as_sets, no weight columns, 32-bit vertex ids, `res` a CUDA handle of its
   * device): the host CSR is then stale (graph_impl). Otherwise the batch is applied to the host
   * CSR and the device copy is dropped.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static apply_summary apply(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                             const edge_batch_view<vertex_t, weight_t>& batch,
                             apply_delta<vertex_t>* delta,
                             const normalized_batch<vertex_t>* normalized = nullptr);

  /**
   * @brief Step 0 of batch_semantics::as_sets for the current state of `g` (normalize_set_batch):
   *        against the host CSR when it is current, else against the resident device copy
   *        (normalize_set_batch_device), so an update never downloads a graph that a device apply
   *        left stale on the host (ADR 0020). Both give the same lists.
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in]  res   Execution resources.
   * @param[in]  g     The graph (semantics.as_sets).
   * @param[in]  batch The batch (host memory).
   * @param[out] out   The normalized batch (its state_id is set to the graph's).
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static void normalize(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                        const edge_batch_view<vertex_t, weight_t>& batch,
                        normalized_batch<vertex_t>& out);

  /**
   * @brief The host out-edges alone (never builds the in-edges; downloads a stale host copy of a
   *        CUDA graph first, see graph_impl::host_edges()).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] g The graph.
   * @return A view of the out-edge CSR, valid until the graph changes.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static csr_view<vertex_t, edge_t, weight_t> out_view(const graph<vertex_t, edge_t, weight_t>& g) {
    return g.impl().host_edges().view();
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
   * @brief The device out-edges of a CUDA graph, uploaded on first use per graph state; the
   *        in-edges are not built (cycle_count reads the out-edges only).
   * @tparam vertex_t Vertex id type.
   * @tparam edge_t   Edge offset type.
   * @tparam weight_t Weight type.
   * @param[in] res Resources of the CUDA backend (the graph's device).
   * @param[in] g   A graph built with CUDA resources.
   * @return The device graph (in-edges possibly absent), valid until the graph changes.
   * @throws invalid_argument_error if the placement of `g` does not match `res`.
   */
  template <typename vertex_t, typename edge_t, typename weight_t>
  static const device_graph<vertex_t, edge_t, weight_t>& device_out(
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
