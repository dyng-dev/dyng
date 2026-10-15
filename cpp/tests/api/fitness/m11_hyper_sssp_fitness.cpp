// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file m11_hyper_sssp_fitness.cpp
 * @brief Compile-only fitness skeleton of the frozen hypergraph API for M11 (hyper_sssp and the
 *        slack_csr storage; the hypergraph API review, ADR 0047).
 *
 * Nothing here runs; the target dyng_api_fitness only compiles it, so a change that removes what
 * M11 needs fails the build:
 *  1. the four-op batch: hyperedge deletions and insertions and the incidence edits (h, v) in
 *     one hyperedge_batch / hyperedge_batch_view (apply() raises not_supported_error for the
 *     edits until 0.4);
 *  2. id_reuse::insert_first (H-SOSP's order) with capacity_policy::mosp_escher(extra);
 *  3. (int32, int32): weighted hyperedges, weights in [1, 2^28] (hyper_sssp's limit, stricter
 *     than the container's [1, 2^31 - 1]);
 *  4. node weights in line_graph_view::weights (the weights by id) and overlaps (empty under
 *     escher, filled by slack_csr);
 *  5. hypergraph_storage::slack_csr and line_graph_policy::slack_csr as enumerators, and a
 *     second store behind the internal store interface (detail::hyper::incidence_store);
 *  6. the incidence-delta hook: hypergraph_access::apply(..., record_incidence_delta = true)
 *     fills hyper_applied::incidence_delta with (vertex, id, sign) triples.
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/store.hpp"

#include <dyng/dyng.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace dyng_fitness_m11 {

using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
namespace hyper = dyng::detail::hyper;

/// A slack_csr store (0.4) behind the store interface: every operation the container calls.
class slack_csr_store final : public hyper::incidence_store {
 public:
  [[nodiscard]] std::unique_ptr<hyper::incidence_store> clone(
      const dyng::resources& res) const override {
    (void)res;
    return std::make_unique<slack_csr_store>();
  }
  [[nodiscard]] hyper::store_snapshot snapshot() const override {
    return {};
  }
  [[nodiscard]] std::int64_t id_bound() const noexcept override {
    return static_cast<std::int64_t>(rows_.size());
  }
  [[nodiscard]] std::int64_t num_live() const noexcept override {
    return live_;
  }
  [[nodiscard]] std::int32_t num_vertices() const noexcept override {
    return n_;
  }
  [[nodiscard]] bool is_live(std::int64_t id) const noexcept override {
    return id >= 0 && id < id_bound() && !rows_[static_cast<std::size_t>(id)].empty();
  }
  void read_rows(const std::vector<std::int64_t>& ids, hyper::member_rows& out) override {
    out.clear();
    for (const std::int64_t id : ids) {
      const auto& r = rows_[static_cast<std::size_t>(id)];
      out.members.insert(out.members.end(), r.begin(), r.end());
      out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
    }
  }
  void erase(const std::vector<std::int64_t>& ids) override {
    for (const std::int64_t id : ids) rows_[static_cast<std::size_t>(id)].clear();
  }
  void insert(const hyper::member_rows& rows, std::vector<std::int64_t>& ids) override {
    ids.clear();
    for (std::int64_t i = 0; i < rows.size(); ++i) ids.push_back(id_bound() + i);
  }
  void v2h_remove(const hyper::grouped_incidences& g) override {
    (void)g;
  }
  void v2h_append(const hyper::grouped_incidences& g) override {
    (void)g;
  }
  void update_line_graph(const std::vector<std::int64_t>& deleted,
                         const std::vector<std::int64_t>& inserted,
                         const hyper::member_rows& inserted_rows) override {
    (void)deleted;
    (void)inserted;
    (void)inserted_rows;
  }
  void finish_apply() override {}
  [[nodiscard]] bool line_graph_enabled() const noexcept override {
    return true;
  }
  void enable_line_graph() const override {}
  [[nodiscard]] hyper::row_views views() const override {
    return {};
  }
  void h2h_view(dyng::array_view<const std::int64_t>& row_ptr,
                dyng::array_view<const std::int64_t>& col_ind,
                dyng::array_view<const std::int64_t>& out) const override {
    row_ptr = {};
    col_ind = {};
    out = {};
  }
  [[nodiscard]] hyper::line_graph_layout line_graph() const override {
    return {};
  }
  [[nodiscard]] hyper::view_builds builds() const noexcept override {
    return {};
  }
  void check_integrity() const override {}
  [[nodiscard]] dyng::storage_stats stats() const override {
    return {};
  }
  void h2v_view(dyng::array_view<const std::int64_t>& row_ptr,
                dyng::array_view<const std::int32_t>& col_ind) const override {
    row_ptr = {};
    col_ind = {};
  }
  void v2h_view(dyng::array_view<const std::int64_t>& row_ptr,
                dyng::array_view<const std::int64_t>& col_ind) const override {
    row_ptr = {};
    col_ind = {};
  }

 private:
  std::vector<std::vector<std::int32_t>> rows_;
  std::int64_t live_ = 0;
  std::int32_t n_ = 0;
};

[[maybe_unused]] void fitness() {
  const dyng::resources res = dyng::resources::sequential();
  dyng::hyperedge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 8;
  list.add_hyperedge({0, 1, 2}, 1);
  list.add_hyperedge({2, 3}, std::int32_t{1} << 28);  // hyper_sssp's largest weight
  dyng::hypergraph_properties props;
  props.reuse = dyng::id_reuse::insert_first;  // H-SOSP's order
  props.capacity = dyng::capacity_policy::mosp_escher(std::int64_t{1} << 20);
  props.line_graph = dyng::line_graph_policy::sorted_merge;
  whgraph h = whgraph::from_hyperedges(res, list.view(), props);

  // The four-op batch (the incidence edits raise not_supported_error until 0.4).
  dyng::hyperedge_batch<std::int32_t, std::int32_t> b;
  b.delete_hyperedge(0);
  b.insert_hyperedge({4, 5}, 7);
  b.delete_incidence(1, 3);
  b.insert_incidence(1, 6);
  const dyng::hyperedge_batch_view<std::int32_t, std::int32_t> v = b.view();
  (void)v.delete_incidence_hyperedge.size();
  (void)v.insert_incidence_vertex.size();
  (void)b.insert_incidence_hyperedge().size();

  // The incidence-delta hook: (vertex, id, sign) per incidence change of the batch.
  dyng::hyperedge_batch<std::int32_t, std::int32_t> whole;
  whole.delete_hyperedge(1);
  whole.insert_hyperedge({6, 7}, 3);
  const dyng::hyper_apply_summary s =
      dyng::detail::hypergraph_access::apply(res, h, whole.view(), /*record_incidence_delta=*/true);
  const dyng::detail::hyper_applied& a = dyng::detail::hypergraph_access::applied(h);
  if (a.has_incidence_delta) {
    (void)a.incidence_delta.vertices.size();
    (void)a.incidence_delta.ids.size();
    (void)a.incidence_delta.signs.size();
  }
  (void)s.inserted_ids;
  (void)s.inserted_incidences;
  (void)s.deleted_incidences;

  // Node weights for the line graph's shortest paths, and overlaps (slack_csr).
  const dyng::line_graph_view<std::int32_t> lg = h.h2h();
  (void)lg.weights.size();
  (void)lg.overlaps.size();
  (void)lg.out.size();

  // slack_csr: the enumerators exist; a second store implements the store interface.
  dyng::hypergraph_properties slack;
  slack.storage = dyng::hypergraph_storage::slack_csr;
  slack.line_graph = dyng::line_graph_policy::slack_csr;
  std::unique_ptr<hyper::incidence_store> store = std::make_unique<slack_csr_store>();
  (void)store->id_bound();
  (void)slack;
}

}  // namespace dyng_fitness_m11
