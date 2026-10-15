// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file store.hpp
 * @brief The internal store interface of the hypergraph container (ADR 0041): what the device
 *        ESCHER store (CUDA), the host store (incidence_model, sequential and openmp) and the
 *        slack_csr store of 0.4 implement.
 *
 * A store holds the incidence relations h2v (hyperedge id -> sorted members) and v2h (vertex ->
 * hyperedge ids) and gives the ids of inserted hyperedges (the best-fit reuse of the H2V tree).
 * Ids and vertices are 0-based here: the CBST keys, the +1 shift and the sentinels stay inside
 * the ESCHER store. The container validates a batch, then calls the phases of the store in the
 * order of its id_reuse rule; a store operation that throws leaves the store undefined (the
 * container is poisoned).
 */
#pragma once

#include "hypergraph/row_views.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/hypergraph/hypergraph_view.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail::hyper {

/// Rows of member lists (0-based vertices), back to back: row i is
/// members[offsets[i] .. offsets[i + 1]).
struct member_rows {
  std::vector<std::int64_t> offsets{0};  ///< size() + 1 offsets
  std::vector<std::int32_t> members;     ///< the members of every row

  /// @brief Number of rows. @return offsets.size() - 1.
  [[nodiscard]] std::int64_t size() const noexcept {
    return static_cast<std::int64_t>(offsets.size()) - 1;
  }
  /// @brief Remove every row (keeps the capacity).
  void clear() noexcept {
    offsets.assign(1, 0);
    members.clear();
  }
};

/// (vertex, hyperedge id) incidences grouped by vertex: group g is vertex `vertices[g]` with the
/// ids `ids[prefix[g - 1] .. prefix[g])` (prefix inclusive, the store's call format). Vertices are
/// increasing; inside a group the ids keep the order of the batch (ESCHER-GPU's groupByVertex).
struct grouped_incidences {
  std::vector<std::int32_t> vertices;  ///< one per group, increasing
  std::vector<std::int64_t> ids;       ///< the hyperedge ids of every group
  std::vector<std::int32_t> prefix;    ///< inclusive end of every group in `ids`

  /// @brief Whether there is no incidence. @return vertices.empty().
  [[nodiscard]] bool empty() const noexcept {
    return vertices.empty();
  }
};

/// Groups the incidences of `rows` (row i has the id ids[i]) by vertex, in row order inside a
/// vertex: ESCHER-GPU's groupByVertex (structure/dynamic_hypergraph.cu@abcf9b2:81-99) and
/// MOSP_ESCHER's stable grouping (hypergraph/src/DynamicHypergraph.cpp@4b86159:63-75) give the
/// same layout.
/// @param[in]  rows The rows.
/// @param[in]  ids  The id of every row.
/// @param[out] out  The groups.
void group_by_vertex(const member_rows& rows, const std::vector<std::int64_t>& ids,
                     grouped_incidences& out);

/// The best-fit id state of a store's H2V tree, independent of the backend (ADR 0041, section
/// 5): enough to rebuild a store on any backend that gives the same ids to later batches.
struct allocator_state {
  std::vector<std::int32_t> keys;  ///< every key in the tree (live or a deleted slot), increasing
  std::vector<std::uint8_t> dead;  ///< per key: 1 for a deleted slot
  std::vector<std::int32_t> lengths;  ///< per key: the slots of its first segment
  std::vector<std::int32_t> counts;   ///< per key: the values of its row (0 for a deleted slot)
  std::vector<std::int32_t> tail_capacity;  ///< per key: usable slots of the tail segment
  std::vector<std::int32_t> occupancy;      ///< per key: values in the tail segment
  std::int64_t bump = 0;                    ///< used payload entries (the bump pointer)
  std::int64_t capacity = 0;                ///< payload capacity
  std::int64_t growths = 0;                 ///< payload growths so far
  std::int64_t compactions = 0;             ///< payload compactions so far
};

/// The state of a store on the host (snapshot()): its allocator state, the live rows and the
/// vertex count. v2h is the transpose of the rows; the line graph is rebuilt from the rows.
struct store_snapshot {
  std::int32_t num_vertices = 0;  ///< vertices
  bool has_v2h = true;            ///< whether v2h is kept
  allocator_state h2v;            ///< the H2V tree's ids, slots and payload accounting
  member_rows rows;               ///< the row of every key of h2v.keys (empty for a deleted slot)
  std::int64_t max_bound = 0;     ///< the largest id bound so far (the line graph's key count)
  bool line_graph = false;        ///< whether the line graph is maintained
};

/// The line graph of a store in ESCHER-GPU's layout (the H2H tree's construct and rebuild input,
/// graph_maintenance.cu@abcf9b2): keys 1..n; row k (the neighbour keys of key k, sorted) starts at
/// starts[k - 1] and takes ((counts[k - 1] + 4) / 4) * 4 slots of `payload`, the values, zeros
/// and INT_MIN in the last slot (padding rule 1).
struct line_graph_layout {
  std::vector<std::int32_t> starts;   ///< per key: the offset of its row
  std::vector<std::int32_t> counts;   ///< per key: the entries of its row
  std::vector<std::int32_t> payload;  ///< the padded rows, back to back in key order

  /// @brief The keys. @return starts.size().
  [[nodiscard]] std::int64_t num_keys() const noexcept {
    return static_cast<std::int64_t>(starts.size());
  }
};

/// How often a store built its views (the tests of "the public views are built at most once per
/// version and never by apply()", and "the row views are rebuilt once per commit").
struct view_builds {
  std::int64_t h2v = 0;        ///< public h2v views built
  std::int64_t v2h = 0;        ///< public (sorted) v2h views built
  std::int64_t h2h = 0;        ///< public h2h views built
  std::int64_t row_views = 0;  ///< rebuilds of the internal row views (construction and commits)
  std::int64_t line_graph_builds = 0;  ///< line graphs built from scratch (enable_line_graph)
};

/// Counters of the degree-binned paths (as storage_path_counters) and of the payload.
struct store_counters {
  storage_path_counters fill, reuse, unfill;  ///< kernel paths
};

/// The internal store interface (one per hypergraph).
class incidence_store {
 public:
  incidence_store() = default;
  incidence_store(const incidence_store&) = delete;
  incidence_store& operator=(const incidence_store&) = delete;
  incidence_store(incidence_store&&) = delete;
  incidence_store& operator=(incidence_store&&) = delete;
  virtual ~incidence_store() = default;

  /// @brief A deep copy on `res` (same backend kind as this store). @param[in] res Resources.
  /// @return The copy.
  [[nodiscard]] virtual std::unique_ptr<incidence_store> clone(const resources& res) const = 0;
  /// @brief The state on the host (for a copy to another backend). @return The snapshot.
  [[nodiscard]] virtual store_snapshot snapshot() const = 0;

  /// @brief The host threads of the next operations (the openmp backend). @param[in] threads
  /// The thread count.
  virtual void set_host_threads(int threads) noexcept {
    (void)threads;
  }

  /// @brief Order the store's next work after the work queued so far on the stream of `res`, the
  /// resources of an apply or update (the store works on the stream it was built with; ADR 0041
  /// section 4). Nothing on the host stores. @param[in] res The call's resources.
  virtual void wait_for(const resources& res) {
    (void)res;
  }
  /// @brief Order the work later queued on the stream of `res` after the store's work so far.
  /// Nothing on the host stores. @param[in] res The call's resources.
  virtual void signal_to(const resources& res) {
    (void)res;
  }

  /// @brief The id bound (the largest key of the H2V tree). @return The bound.
  [[nodiscard]] virtual std::int64_t id_bound() const noexcept = 0;
  /// @brief Live hyperedges. @return The count.
  [[nodiscard]] virtual std::int64_t num_live() const noexcept = 0;
  /// @brief Vertices. @return n.
  [[nodiscard]] virtual std::int32_t num_vertices() const noexcept = 0;
  /// @brief Whether `id` is live (a host-side query). @param[in] id The id. @return Live.
  [[nodiscard]] virtual bool is_live(std::int64_t id) const noexcept = 0;

  /// @brief The sorted members of live ids. @param[in] ids Live ids. @param[out] out Their rows.
  virtual void read_rows(const std::vector<std::int64_t>& ids, member_rows& out) = 0;
  /// @brief Delete live ids (h2v erase). @param[in] ids Distinct live ids.
  virtual void erase(const std::vector<std::int64_t>& ids) = 0;
  /// @brief Insert rows (h2v insert, best-fit reuse). @param[in] rows Sorted, distinct members,
  /// at least one per row. @param[out] ids The id of every row.
  virtual void insert(const member_rows& rows, std::vector<std::int64_t>& ids) = 0;
  /// @brief Remove ids from v2h rows. @param[in] g The incidences.
  virtual void v2h_remove(const grouped_incidences& g) = 0;
  /// @brief Append ids to v2h rows. @param[in] g The incidences.
  virtual void v2h_append(const grouped_incidences& g) = 0;
  /// @brief The line-graph phase of an apply (after the h2v and v2h phases, before
  /// finish_apply()): ESCHER-GPU's sorted delta merge of the line graph, when it is maintained.
  /// @param[in] deleted       The deleted ids (live before the batch, distinct).
  /// @param[in] inserted      The ids of the inserted rows.
  /// @param[in] inserted_rows The inserted rows (sorted, distinct members).
  virtual void update_line_graph(const std::vector<std::int64_t>& deleted,
                                 const std::vector<std::int64_t>& inserted,
                                 const member_rows& inserted_rows) = 0;
  /// @brief The end of an apply: read the device error words once, rebuild the row views kept
  /// per commit, drop the views built on request.
  virtual void finish_apply() = 0;

  /// @brief Whether the line graph is maintained. @return It is.
  [[nodiscard]] virtual bool line_graph_enabled() const noexcept = 0;
  /// @brief Build the line graph from the current state (ESCHER-GPU's flatHyperedgeAdjacency)
  /// and maintain it from now on; nothing if it is maintained already. Thread-safe with the other
  /// read-only calls. @throws as the construction of a store.
  virtual void enable_line_graph() const = 0;
  /// @brief The internal row views of the current state. @return The views.
  [[nodiscard]] virtual row_views views() const = 0;
  /// @brief h2h as a CSR in the store's memory space, built once per state (thread-safe); the
  /// line graph must be maintained. @param[out] row_ptr Offsets. @param[out] col_ind Neighbour
  /// ids. @param[out] out Per row: the offset of its first neighbour above the row's id.
  virtual void h2h_view(array_view<const std::int64_t>& row_ptr,
                        array_view<const std::int64_t>& col_ind,
                        array_view<const std::int64_t>& out) const = 0;
  /// @brief The line graph in ESCHER-GPU's layout, on the host (parity and tests).
  /// @return The layout (empty when the line graph is not maintained).
  [[nodiscard]] virtual line_graph_layout line_graph() const = 0;
  /// @brief The view build counters. @return The counters.
  [[nodiscard]] virtual view_builds builds() const noexcept = 0;

  /// @brief Check every invariant. @throws internal_error if one is broken.
  virtual void check_integrity() const = 0;
  /// @brief The storage counters. @return The counters.
  [[nodiscard]] virtual storage_stats stats() const = 0;

  /// @brief h2v as a CSR in the store's memory space, built once per state (thread-safe).
  /// @param[out] row_ptr Offsets. @param[out] col_ind Members.
  virtual void h2v_view(array_view<const std::int64_t>& row_ptr,
                        array_view<const std::int32_t>& col_ind) const = 0;
  /// @brief v2h as a sorted CSR in the store's memory space, built once per state (thread-safe).
  /// @param[out] row_ptr Offsets. @param[out] col_ind Ids.
  virtual void v2h_view(array_view<const std::int64_t>& row_ptr,
                        array_view<const std::int64_t>& col_ind) const = 0;
};

}  // namespace dyng::detail::hyper
