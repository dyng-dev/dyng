// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file incidence_model.hpp
 * @brief The host store of the hypergraph container (sequential and openmp backends; ADR 0041,
 *        section 4): sorted h2v rows by id, sorted v2h rows by vertex, and the best-fit id
 *        allocator that gives the same ids as the CUDA store.
 */
#pragma once

#include "hypergraph/id_allocator.hpp"
#include "hypergraph/store.hpp"

#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace dyng::detail::hyper {

/// The host store (MOSP_ESCHER's HostHypergraph in spirit; the ids are the allocator's).
class incidence_model final : public incidence_store {
 public:
  /**
   * @brief A store over `rows` (row i is hyperedge id i; sorted, distinct, in range, non-empty).
   * @param[in] res          The resources (host threads).
   * @param[in] num_vertices Vertices.
   * @param[in] rows         The rows.
   * @param[in] capacity     The capacity policy.
   * @param[in] has_v2h      Whether v2h is kept.
   */
  incidence_model(const resources& res, std::int32_t num_vertices, const member_rows& rows,
                  const capacity_policy& capacity, bool has_v2h);

  /**
   * @brief A store from a snapshot of any backend's store (same ids and allocator state).
   * @param[in] res      The resources.
   * @param[in] snapshot The state.
   * @param[in] capacity The capacity policy.
   */
  incidence_model(const resources& res, const store_snapshot& snapshot,
                  const capacity_policy& capacity);

  incidence_model(const incidence_model& other);  ///< deep copy (views are rebuilt)

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
  void finish_apply() override;
  void check_integrity() const override;
  [[nodiscard]] storage_stats stats() const override;
  void h2v_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int32_t>& col_ind) const override;
  void v2h_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int64_t>& col_ind) const override;

  /// @brief The allocator (tests). @return It.
  [[nodiscard]] const id_allocator& allocator() const noexcept {
    return alloc_;
  }

 private:
  void build_v2h_from_rows();
  void drop_views() noexcept;

  int threads_ = 1;
  std::int32_t num_vertices_ = 0;
  bool has_v2h_ = true;
  capacity_policy capacity_;
  id_allocator alloc_;
  std::vector<std::vector<std::int32_t>> rows_;  // by id; empty for a dead id
  std::vector<std::vector<std::int64_t>> v2h_;   // by vertex; sorted ids
  std::array<std::int64_t, 3> fill_paths_{}, unfill_paths_{};

  // The views built on request (once per state).
  mutable std::mutex view_mutex_;
  mutable std::atomic<bool> h2v_built_{false}, v2h_built_{false};
  mutable std::vector<std::int64_t> h2v_row_ptr_, v2h_row_ptr_;
  mutable std::vector<std::int32_t> h2v_col_ind_;
  mutable std::vector<std::int64_t> v2h_col_ind_;
};

}  // namespace dyng::detail::hyper
