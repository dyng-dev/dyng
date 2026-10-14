// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file m9_triad_count_fitness.cpp
 * @brief Compile-only fitness skeleton of the frozen hypergraph API for M9 (triad_count; the
 *        hypergraph API review, ADR 0047): what triad_count reads, written against the frozen
 *        headers and the internal access it is allowed (cpp/src, internal-stable).
 *
 * Nothing here runs; the target dyng_api_fitness only compiles it (in every build), so a change
 * that removes what M9 needs fails the build:
 *  1. count(-) on G_t: before_apply() reads the old h2h / h2v rows (public views: 0-based, int64
 *     offsets, no sentinels, in the hypergraph's memory space; and the internal row views of
 *     ESCHER-GPU's layout for the kernels) with the frontier = the live deleted ids of Step 0;
 *  2. count(+) on G_{t+1}: after_apply() reads the new rows with the frontier = inserted_ids, on
 *     the host and, on CUDA, in device memory (the applied record), plus the deleted member sets;
 *  3. the `out` pointers: the neighbours above a row's id start at out[r] (each triad is counted
 *     once, by its smallest member: ownership::min_member);
 *  4. dyng::update(res, h, batch, result) through update_traits / participant_of<hypergraph>;
 *  5. replay by key (a .dgb file's expected ids against inserted_ids) and by content (its deleted
 *     member sets resolved against h2v()), and the host backends (sequential, openmp).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/dyng.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dyng_fitness_m9 {

using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using batch_view = dyng::hyperedge_batch_view<std::int32_t, dyng::unweighted>;

/// The closed triads through `frontier` on one state (host form): pairs above the row's id.
std::uint64_t count_triads(const dyng::resources& res, const hgraph& h,
                           const std::vector<std::int64_t>& frontier) {
  const dyng::line_graph_view<dyng::unweighted> h2h = h.h2h();
  const dyng::incidence_view<std::int32_t> h2v = h.h2v();
  const std::vector<std::int64_t> ptr = dyng::to_vector(res, h2h.row_ptr);
  const std::vector<std::int64_t> col = dyng::to_vector(res, h2h.col_ind);
  const std::vector<std::int64_t> out = dyng::to_vector(res, h2h.out);  // first neighbour > r
  std::uint64_t triads = 0;
  if (h2v.num_rows() > h2h.num_rows()) throw dyng::internal_error("h2v and h2h disagree");
  for (const std::int64_t r : frontier) {
    if (r < 0 || r >= h2h.num_rows()) continue;
    const auto u = static_cast<std::size_t>(r);
    for (std::int64_t i = out[u]; i < ptr[u + 1]; ++i) {
      for (std::int64_t j = i + 1; j < ptr[u + 1]; ++j) {
        triads += static_cast<std::uint64_t>(col[static_cast<std::size_t>(j)] >
                                             col[static_cast<std::size_t>(i)]);
      }
    }
  }
  return triads;
}

/// What a device kernel of count(+) is handed: the frontier in device memory and the rows of
/// ESCHER-GPU's layout (keys = id + 1), or the decoded public views.
struct device_frontier {
  const std::int64_t* inserted_ids = nullptr;  ///< the applied record's device copy (CUDA)
  std::size_t count = 0;
  dyng::detail::escher::row_view h2h_rows;  ///< sorted rows, out set
  dyng::detail::escher::row_view h2v_rows;
};

class triad_result {
 public:
  std::uint64_t total = 0;
  std::uint64_t graph_version = 0;
};

struct triad_stats {
  dyng::hyper_apply_summary batch;
  std::uint64_t removed = 0, added = 0;
};

class triad_participant final
    : public dyng::detail::hyper_update_participant<std::int32_t, dyng::unweighted> {
 public:
  triad_participant(triad_result& r, triad_stats& s) : r_(r), s_(s) {}
  [[nodiscard]] const void* target() const noexcept override {
    return &r_;
  }
  void use_normalized(
      const dyng::detail::hyper_staged_batch<dyng::unweighted>* staged) noexcept override {
    staged_ = staged;
  }
  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return true;  // run_update() makes sure the line graph exists on G_t
  }
  void before_apply(const dyng::resources& res, const graph_type& h,
                    const batch_type& batch) override {
    if (r_.graph_version != h.version()) throw dyng::stale_result_error("triad_count: stale");
    (void)batch.num_deletions();
    // count(-): the old rows, frontier = the live deleted ids of G_t (Step 0, computed once)
    s_.removed = count_triads(res, h, staged_->deletions);
    dyng::detail::hypergraph_access::require_line_graph(h, "triad_count");
    old_rows_ = dyng::detail::hypergraph_access::views(h);  // valid until the commit
  }
  void after_apply(const dyng::resources& res, const graph_type& h,
                   const dyng::hyper_apply_summary& summary,
                   const dyng::detail::hyper_applied& applied) override {
    // count(+): the new rows, frontier = inserted_ids (host; device on CUDA)
    s_.added = count_triads(res, h, summary.inserted_ids);
    device_frontier f;
    f.inserted_ids = applied.inserted_ids_device.data();
    f.count = applied.inserted_ids.size();
    const dyng::detail::hyper::row_views now = dyng::detail::hypergraph_access::views(h);
    f.h2h_rows = now.h2h;
    f.h2v_rows = now.h2v;
    (void)f;
    // the deleted ids with their member sets (by-content checks), the touched sets
    (void)applied.deleted_members.size();
    (void)applied.deleted_ids.size();
    (void)applied.touched_hyperedges.size();
    (void)applied.touched_vertices.size();
    r_.total = r_.total - s_.removed + s_.added;
    r_.graph_version = h.version();
    s_.batch = summary;
  }
  void poison() noexcept override {
    r_.graph_version = ~std::uint64_t{0};
  }

 private:
  triad_result& r_;
  triad_stats& s_;
  const dyng::detail::hyper_staged_batch<dyng::unweighted>* staged_ = nullptr;
  dyng::detail::hyper::row_views old_rows_;
};

}  // namespace dyng_fitness_m9

namespace dyng::detail {

/// dyng::update() over a hypergraph with a triad result (the shape triad_count's traits take).
template <>
struct update_traits<dyng_fitness_m9::triad_result> {
  using stats_type = dyng_fitness_m9::triad_stats;
  template <typename container_t>
  static std::unique_ptr<typename participant_of<container_t>::type> make_participant(
      dyng_fitness_m9::triad_result& r, dyng_fitness_m9::triad_stats& out) {
    static_assert(is_hypergraph_v<container_t>, "triad_count runs on a hypergraph");
    return std::make_unique<dyng_fitness_m9::triad_participant>(r, out);
  }
};

}  // namespace dyng::detail

namespace dyng_fitness_m9 {

/// One update on every backend; replay of a .dgb file by key and by content.
[[maybe_unused]] void fitness(const std::string& hg_file, const std::string& dgb_file) {
  for (const dyng::resources& res :
       {dyng::resources::sequential(), dyng::resources::openmp(8), dyng::resources::cuda(0)}) {
    const auto list = dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>(
        hg_file, dyng::io::hypergraph_read_options::escher_gpu());
    dyng::hypergraph_properties props;
    props.line_graph = dyng::line_graph_policy::sorted_merge;
    props.capacity = dyng::capacity_policy::escher_gpu();
    hgraph by_key = hgraph::from_hyperedges(res, list.view(), props);
    hgraph by_content = by_key.clone(res);
    triad_result r{0, by_key.version()};
    dyng::io::batch_reader in(res, dgb_file);
    while (in.next()) {
      const batch_view b = in.hyperedge_batch<std::int32_t, dyng::unweighted>();
      const auto notes = in.annotations<std::int32_t>();
      // by key: the generating run's ids
      const auto [st] = dyng::update(res, by_key, b, r);
      const std::vector<std::int64_t> want = dyng::to_vector(res, notes.expected_ids);
      if (st.batch.inserted_ids != want) throw dyng::internal_error("ids differ");
      // by content: each deleted member set resolved to a live id of the same set
      const auto h2v = by_content.h2v();
      const std::vector<std::int64_t> ptr = dyng::to_vector(res, h2v.row_ptr);
      const std::vector<std::int32_t> col = dyng::to_vector(res, h2v.col_ind);
      std::multimap<std::vector<std::int32_t>, std::int64_t> ids;
      for (std::size_t i = 0; i + 1 < ptr.size(); ++i)
        if (ptr[i + 1] > ptr[i])
          ids.emplace(std::vector<std::int32_t>(col.begin() + ptr[i], col.begin() + ptr[i + 1]),
                      static_cast<std::int64_t>(i));
      std::vector<std::int64_t> deletions;
      const std::vector<std::int64_t> doff = dyng::to_vector(res, notes.deleted_offsets);
      const std::vector<std::int32_t> dmem = dyng::to_vector(res, notes.deleted_members);
      for (std::size_t i = 0; i + 1 < doff.size(); ++i) {
        const auto it =
            ids.find(std::vector<std::int32_t>(dmem.begin() + doff[i], dmem.begin() + doff[i + 1]));
        if (it != ids.end()) {
          deletions.push_back(it->second);
          ids.erase(it);
        }
      }
      batch_view c = b;
      c.delete_hyperedges = dyng::host_view(deletions);
      (void)by_content.apply(res, c);
    }
  }
}

}  // namespace dyng_fitness_m9
