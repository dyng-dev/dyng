// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file graph.hpp
 * @brief graph<V,E,W>: the dynamic graph container (pimpl, move-only).
 * @ingroup graph
 */
#pragma once

#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>

#include <cstdint>
#include <memory>

/**
 * @defgroup graph Graphs
 * @brief The dynamic graph container, its properties, views and CSR / edge-list types.
 */

namespace dyng {

namespace detail {
template <typename vertex_t, typename edge_t, typename weight_t>
class graph_impl;
struct graph_access;
}  // namespace detail

/**
 * @brief A dynamic directed (or symmetric) graph with K weight columns.
 *
 * The graph owns its storage and a version counter that increases by one with every applied
 * batch; results computed on the graph remember the version they match and the identity of the
 * graph state (a process-wide unique value that every construction and every batch renews and
 * clone() copies), so a result is detected as stale when it is used with another graph, or with
 * a graph variable that was reassigned since, even if the version counters are equal. The storage
 * belongs to the backend of the resources that created it (PLAN Section 4.6 rule 5): algorithms
 * refuse resources of the other kind (host backends versus cuda) instead of copying the graph
 * silently, and clone(res) makes a copy for other resources. A graph built with CUDA resources
 * keeps its CSR in host memory in this release (a batch is applied on the host, as MOSP-CUDA
 * applies it) and a resident device copy of the current state (out- and in-edges, one weight
 * column per objective), uploaded on first use and again after each applied batch.
 *
 * Instantiated for (vertex_t, edge_t, weight_t) = (int32, int32, int32), (int32, int64, int32)
 * and (int64, int64, int32), and without weights for (int32, int32, unweighted) and
 * (int32, int64, unweighted) (PLAN Section 4.4.3). A graph with weight_t = unweighted has no
 * weight columns.
 *
 * @tparam vertex_t Vertex id type (signed).
 * @tparam edge_t   Edge offset type (signed).
 * @tparam weight_t Weight type.
 * @ingroup graph
 */
template <typename vertex_t = std::int32_t, typename edge_t = std::int64_t,
          typename weight_t = std::int32_t>
class graph {
 public:
  using vertex_type = vertex_t;                              ///< vertex id type
  using edge_type = edge_t;                                  ///< edge offset type
  using weight_type = weight_t;                              ///< weight type
  using view_type = graph_view<vertex_t, edge_t, weight_t>;  ///< the view type
  using csr_type = csr<vertex_t, edge_t, weight_t>;          ///< the owning CSR type

  /**
   * @brief An empty graph (no vertices) with the given properties.
   * @param[in] props The properties (see graph_properties).
   * @throws not_supported_error if `props.layout` is not row_layout::compact.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   */
  explicit graph(const graph_properties& props = {});

  /**
   * @brief Build a graph from an edge list.
   *
   * Edges are grouped by source; inside a row they keep their input order under
   * row_order::append and are stably sorted by destination under row_order::sorted. Under
   * multi_edges::forbid, duplicate (u,v) are merged: the first position is kept, the last weights
   * win. Self-loops follow `props.semantics.on_self_loop`. For an undirected graph every edge
   * (u,v) is stored in both directions. The number of weight columns is taken from `edges`.
   *
   * @param[in] res   Execution resources (the graph belongs to its backend).
   * @param[in] edges The edges (host memory); ids must lie in [0, edges.num_vertices).
   * @param[in] props The properties of the new graph.
   * @return The graph at version 0.
   * @throws invalid_argument_error if an id is out of range, the arrays disagree in size, or a
   *         self-loop is present under self_loop::error.
   * @throws not_supported_error    for a layout other than compact.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  [[nodiscard]] static graph from_edges(const resources& res,
                                        edge_list_view<vertex_t, weight_t> edges,
                                        const graph_properties& props = {});

  /**
   * @brief Build a graph from a CSR (the out-edges).
   *
   * The CSR is validated (monotone offsets from 0, neighbours in range, weight array size).
   * Under row_order::sorted rows are stably sorted; under multi_edges::forbid duplicates are
   * merged as in from_edges(). Self-loops follow `props.semantics.on_self_loop`. For an
   * undirected graph the CSR must already be symmetric (each direction stored). The number of
   * weight columns is taken from `csr`. The in-edges (`props.store_transposed`) are built on
   * first use, not here (the same holds for from_edges() and after apply()).
   *
   * @param[in] res   Execution resources (the graph belongs to its backend).
   * @param[in] csr   The out-edge CSR (host memory, objective-major weights).
   * @param[in] props The properties of the new graph.
   * @return The graph at version 0.
   * @throws invalid_argument_error if the CSR is malformed.
   * @throws not_supported_error    for a layout other than compact.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  [[nodiscard]] static graph from_csr(const resources& res,
                                      csr_view<vertex_t, edge_t, weight_t> csr,
                                      const graph_properties& props = {});

  /**
   * @brief Build a graph from a CSR it may take over (the out-edges).
   *
   * The same graph as from_csr(res, csr.view(), props). When the CSR already is what `props` ask
   * for (directed, self-loops kept or absent, rows in the requested order and without forbidden
   * parallel edges; for example any CSR under graph_properties::mosp_compatible()), its arrays
   * are moved into the graph instead of copied; `csr` is then left empty. Otherwise it is left
   * unchanged.
   *
   * @param[in]     res   Execution resources (the graph belongs to its backend).
   * @param[in,out] csr   The out-edge CSR (objective-major weights).
   * @param[in]     props The properties of the new graph.
   * @return The graph at version 0.
   * @throws invalid_argument_error if the CSR is malformed.
   * @throws not_supported_error    for a layout other than compact.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  [[nodiscard]] static graph from_csr(const resources& res, csr_type&& csr,
                                      const graph_properties& props = {});

  /**
   * @brief Move constructor.
   * @param[in,out] other The graph to move from; it may only be assigned to or destroyed after.
   */
  graph(graph&& other) noexcept;

  /**
   * @brief Move assignment.
   * @param[in,out] other The graph to move from; it may only be assigned to or destroyed after.
   * @return *this.
   */
  graph& operator=(graph&& other) noexcept;

  graph(const graph&) = delete;             ///< not copyable: use clone()
  graph& operator=(const graph&) = delete;  ///< not copyable: use clone()
  ~graph();                                 ///< releases the storage

  /**
   * @brief A deep copy (same properties, storage, version and state) for the resources `res`.
   *
   * The copy belongs to the backend of `res`: this is how a graph moves between the host backends
   * and cuda (a CUDA copy uploads its device copy on first use).
   * @param[in] res Execution resources of the copy.
   * @return The copy.
   * @throws invalid_argument_error for a moved-from graph.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  [[nodiscard]] graph clone(const resources& res) const;

  /**
   * @brief Pre-size the current storage (out-edges and, if stored, in-edges) for `edge_capacity`
   *        edges.
   *
   * In this release reserve() only sizes the storage the graph holds now: the host apply() is the
   * straight port of MOSP's applyChangeBatch(), which builds a new CSR per batch, so the first
   * apply() after reserve() allocates new arrays and the reserved capacity is released. It gives
   * no guarantee against reallocation until the resident apply replaces the port (invariant I9
   * exempts the port until then; PLAN Section 4.5.5).
   * @param[in] res           Execution resources.
   * @param[in] edge_capacity Expected maximum number of stored edges.
   * @throws invalid_argument_error if `edge_capacity` is negative or the graph was moved from.
   * @throws out_of_memory_error    if the storage cannot be allocated.
   * @sync
   */
  void reserve(const resources& res, edge_t edge_capacity);

  /**
   * @brief Number of vertices.
   * @return The vertex count.
   */
  [[nodiscard]] vertex_t num_vertices() const noexcept;

  /**
   * @brief Number of stored (directed) edges.
   * @return The edge count (an undirected edge counts twice, a self-loop once).
   */
  [[nodiscard]] edge_t num_edges() const noexcept;

  /**
   * @brief Number of weight columns (objectives).
   * @return K >= 0.
   */
  [[nodiscard]] int num_weights() const noexcept;

  /**
   * @brief The properties of the graph.
   * @return The properties given at construction (num_weights as stored).
   */
  [[nodiscard]] const graph_properties& properties() const noexcept;

  /**
   * @brief Whether the graph is directed.
   * @return properties().directed.
   */
  [[nodiscard]] bool is_directed() const noexcept;

  /**
   * @brief Whether the in-edges are stored.
   * @return properties().store_transposed.
   */
  [[nodiscard]] bool has_transposed() const noexcept;

  /**
   * @brief The memory space of the storage.
   * @return memory_space::device for a graph built with CUDA resources (its resident device copy;
   *         the host CSR that view() and to_csr() read is kept as well in this release),
   *         memory_space::host otherwise.
   */
  [[nodiscard]] memory_space space() const noexcept;

  /**
   * @brief The version counter.
   * @return 0 after construction; +1 per applied batch.
   */
  [[nodiscard]] std::uint64_t version() const noexcept;

  /**
   * @brief A read-only description of the storage at the current version (host memory, also for
   *        a graph built with CUDA resources).
   *
   * If the in-edges are stored but not built yet for this version (they are built on first use),
   * this call builds them (with the thread count of the resources that built the graph); it is
   * safe to call concurrently with other read-only calls.
   * @return The view; invalidated by the next apply().
   * @throws out_of_memory_error if the in-edges must be built and memory cannot be allocated.
   */
  [[nodiscard]] view_type view() const;

  /**
   * @brief Apply a batch to the structure (and weights) of the graph.
   *
   * The batch is interpreted under `properties().semantics` (see batch_semantics). The in-edges
   * of the new version are built on first use. Results computed on the previous version become
   * stale (their update() throws stale_result_error);
   * use the algorithms' update() to apply a batch and keep a result current.
   *
   * @param[in] res   Execution resources (host threads; a CUDA graph drops its device copy).
   * @param[in] batch The batch (host memory).
   * @return What the batch did.
   * @throws invalid_argument_error if an id is negative or out of range (without vertex growth),
   *         the weights do not match num_weights(), or a semantics rule says error.
   * @throws not_supported_error    for vertex insertions or deletions (planned for 0.3).
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  apply_summary apply(const resources& res, const edge_batch_view<vertex_t, weight_t>& batch);

  /**
   * @brief A host copy of the out-edges.
   * @param[in] res Execution resources.
   * @return The out-edge CSR with objective-major weights.
   * @throws invalid_argument_error for a moved-from graph.
   * @throws out_of_memory_error    if host memory cannot be allocated.
   * @sync
   */
  [[nodiscard]] csr_type to_csr(const resources& res) const;

  /**
   * @brief Check the storage invariants (offsets, ids, row order, simple graph, in-edges).
   * @param[in] res Execution resources.
   * @throws internal_error if an invariant is broken.
   * @sync
   */
  void check_integrity(const resources& res) const;

 private:
  friend struct detail::graph_access;
  using impl_type = detail::graph_impl<vertex_t, edge_t, weight_t>;
  explicit graph(std::unique_ptr<impl_type> impl) noexcept;
  [[nodiscard]] impl_type& impl();
  [[nodiscard]] const impl_type& impl() const;

  std::unique_ptr<impl_type> impl_;
};

}  // namespace dyng
