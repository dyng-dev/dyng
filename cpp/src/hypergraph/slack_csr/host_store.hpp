// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/include/HostHypergraph.hpp (the host incidence model)
/**
 * @file host_store.hpp
 * @brief The host store of slack_csr storage (sequential and openmp backends; ADR 0086): the host
 *        incidence model of MOSP_ESCHER@4b86159 (sorted rows by id, v2h by vertex, as its
 *        HostHypergraph keeps them), the best-fit id allocator (the ids of its ESCHER H2V tree,
 *        fills and unfills included, ADR 0085), and the slack CSR line graph (its DeviceH2H on the
 *        host) kept from every batch's pair delta (incidence_delta, inside the commit).
 */
#pragma once

#include "hypergraph/id_allocator.hpp"
#include "hypergraph/slack_csr/slack_line_graph.hpp"
#include "hypergraph/slack_csr/slack_rows.hpp"
#include "hypergraph/store.hpp"

#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dyng::detail::hyper {

/// The host store of slack_csr storage.
class slack_host_store final : public incidence_store, public slack_access {
 public:
  /**
   * @brief A store over `rows` (row i is hyperedge id i; sorted, distinct, in range, non-empty).
   * @param[in] res          The resources (host threads).
   * @param[in] num_vertices Vertices.
   * @param[in] rows         The rows.
   * @param[in] capacity     The capacity policy (the allocator's presets).
   * @param[in] knobs        The internal knobs.
   */
  slack_host_store(const resources& res, std::int32_t num_vertices, const member_rows& rows,
                   const capacity_policy& capacity, const slack::knobs& knobs);

  /**
   * @brief A store from a snapshot of any backend's slack_csr store.
   * @param[in] res      The resources.
   * @param[in] snapshot The state.
   * @param[in] capacity The capacity policy.
   * @param[in] knobs    The internal knobs.
   */
  slack_host_store(const resources& res, const store_snapshot& snapshot,
                   const capacity_policy& capacity, const slack::knobs& knobs);

  slack_host_store(const slack_host_store& other);  ///< deep copy (views are rebuilt)

  [[nodiscard]] std::unique_ptr<incidence_store> clone(const resources& res) const override;
  [[nodiscard]] store_snapshot snapshot() const override;
  void set_host_threads(int threads) noexcept override {
    threads_ = threads > 0 ? threads : 1;
  }
  [[nodiscard]] std::int64_t id_bound() const noexcept override;
  [[nodiscard]] std::int64_t num_live() const noexcept override;
  [[nodiscard]] std::int32_t num_vertices() const noexcept override;
  [[nodiscard]] bool is_live(std::int64_t id) const noexcept override;
  void read_rows(const std::vector<std::int64_t>& ids, member_rows& out) override;
  void erase(const std::vector<std::int64_t>& ids) override;
  void insert(const member_rows& rows, std::vector<std::int64_t>& ids) override;
  void v2h_remove(const grouped_incidences& g) override;
  void v2h_append(const grouped_incidences& g) override;
  void update_line_graph(const std::vector<std::int64_t>& deleted,
                         const std::vector<std::int64_t>& inserted,
                         const member_rows& inserted_rows) override;
  void finish_apply() override;
  [[nodiscard]] bool line_graph_enabled() const noexcept override;
  void enable_line_graph() const override;
  [[nodiscard]] row_views views() const override;
  void h2h_view(array_view<const std::int64_t>& row_ptr, array_view<const std::int64_t>& col_ind,
                array_view<const std::int64_t>& out) const override;
  [[nodiscard]] line_graph_layout line_graph() const override;
  [[nodiscard]] view_builds builds() const noexcept override;
  void check_integrity() const override;
  [[nodiscard]] storage_stats stats() const override;
  void h2v_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int32_t>& col_ind) const override;
  void v2h_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int64_t>& col_ind) const override;

  [[nodiscard]] bool supports_incidence_edits() const noexcept override {
    return true;
  }
  void h2v_edit(const grouped_members& removals, const grouped_members& additions,
                const std::vector<std::int64_t>& ids, const member_rows& post) override;
  void v2h_apply(const std::vector<std::uint64_t>& by_vertex,
                 const std::vector<std::size_t>& group_start, const std::int64_t* ids,
                 const std::int8_t* signs, std::vector<std::int32_t>& lengths) override;
  void begin_line_graph_delta(const std::vector<std::int32_t>& vertices,
                              const std::vector<std::int64_t>& ids) override;
  void line_graph_delta_of(const std::vector<std::int32_t>& vertices,
                           const std::vector<std::int64_t>& ids,
                           const std::vector<std::int8_t>& signs,
                           const std::vector<std::int64_t>& inserted_ids,
                           const touched_rows& touched, line_graph_delta& out) override;
  void finish_line_graph(const line_graph_delta& delta, const std::vector<std::int64_t>& new_ids,
                         const std::vector<std::int64_t>& new_weights) override;
  void set_node_weights(const std::vector<std::int64_t>& by_id) override;
  [[nodiscard]] slack_access* slack() noexcept override {
    return this;
  }
  [[nodiscard]] const slack_access* slack() const noexcept override {
    return this;
  }

  // slack_access
  [[nodiscard]] slack_row_view rows() const override;
  [[nodiscard]] slack_counters counters() const noexcept override;
  [[nodiscard]] slack::knobs knobs() const noexcept override {
    return knobs_;
  }
  void set_entry_headroom(double headroom) override;
  void vertex_lengths(const std::vector<std::int32_t>& vertices,
                      std::vector<std::int32_t>& out) const override;
  [[nodiscard]] const std::vector<std::vector<std::int32_t>>& host_rows() const noexcept override {
    return rows_;
  }
  [[nodiscard]] const std::vector<std::vector<std::int64_t>>& host_v2h() const noexcept override {
    return v2h_;
  }

  /// @brief The allocator (tests). @return It.
  [[nodiscard]] const id_allocator& allocator() const noexcept {
    return alloc_;
  }
  /// @brief The rows by id (the device store's host model, ADR 0089). @return They.
  [[nodiscard]] const std::vector<std::vector<std::int32_t>>& rows_by_id() const noexcept {
    return rows_;
  }
  /// @brief The hyperedges of every vertex (unsorted). @return They.
  [[nodiscard]] const std::vector<std::vector<std::int64_t>>& v2h_by_vertex() const noexcept {
    return v2h_;
  }
  /// @brief The weights by id that the line graph keeps. @return They.
  [[nodiscard]] const std::vector<std::int64_t>& node_weights() const noexcept {
    return weights_;
  }
  /// @brief The line graph of the rows (MOSP_ESCHER's HostHypergraph::lineGraph: the sorted
  /// neighbours of every id below the largest bound so far) into `lg`, on the store's host threads
  /// (the device store's build and its rebuild after a tail overflow, ADR 0089).
  /// @param[out] lg The line graph (no headroom).
  void build_line_graph_into(slack_line_graph& lg) const;
  /// @brief The largest id bound so far (the line graph's nodes). @return It.
  [[nodiscard]] std::int64_t max_bound() const noexcept {
    return max_bound_;
  }

 private:
  void build_v2h_from_rows();
  void drop_views() noexcept;
  void build_line_graph() const;  // under view_mutex_ or before publication
  [[nodiscard]] std::int64_t planned_node_capacity(std::int64_t needed) const noexcept;

  int threads_ = 1;
  std::int32_t num_vertices_ = 0;
  capacity_policy capacity_;
  slack::knobs knobs_;
  id_allocator alloc_;
  std::vector<std::vector<std::int32_t>> rows_;  // by id; empty for a dead id
  // By vertex: the ids, unsorted (MOSP_ESCHER's v2h: appended, swap-removed; the public view and
  // the checks sort them).
  std::vector<std::vector<std::int64_t>> v2h_;
  std::vector<std::int64_t> weights_;    // by id: the weights the line graph keeps
  std::int64_t max_bound_ = 0;           // the largest id bound so far (the CSR's nodes)
  mutable std::int64_t pair_count_ = 0;  // the live pairs, maintained per batch
  std::int64_t candidates_ = 0;          // the last batch's candidate entries
  std::int64_t csr_rebuilds_ = 0;

  // The line graph (built at construction under slack_csr, or on first read under automatic).
  mutable std::atomic<bool> line_graph_{false};
  mutable slack_line_graph lg_;
  mutable std::int64_t line_graph_builds_ = 0;

  // The state before the batch that the pair delta reads (begin_line_graph_delta).
  std::vector<std::int32_t> pre_vertices_;
  std::vector<std::int64_t> pre_v2h_offsets_, pre_v2h_ids_;
  std::vector<std::int64_t> pre_ids_;
  member_rows pre_rows_;
  std::vector<std::int32_t> touched_slot_;  // by id: -1 between batches

  // The views built on request (once per state).
  mutable std::mutex view_mutex_;
  mutable std::atomic<bool> h2v_built_{false}, v2h_built_{false}, h2h_built_{false};
  mutable std::atomic<bool> row_views_built_{false};
  mutable std::vector<std::int64_t> h2v_row_ptr_, v2h_row_ptr_;
  mutable std::vector<std::int32_t> h2v_col_ind_;
  mutable std::vector<std::int64_t> v2h_col_ind_;
  mutable std::vector<std::int64_t> h2h_row_ptr_, h2h_col_ind_, h2h_out_offsets_;
  mutable std::vector<int> h2v_flat_, h2v_loc_, h2v_len_;
  mutable std::vector<int> h2h_flat_, h2h_loc_, h2h_len_, h2h_out_;
  mutable view_builds builds_;
};

}  // namespace dyng::detail::hyper
