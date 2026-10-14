// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file incidence_model.cpp
 * @brief The host store of the hypergraph container (ADR 0041, section 4) and its line graph
 *        (ADR 0042).
 */
#include "hypergraph/incidence_model.hpp"

#include "core/resources_access.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/line_graph/host_line_graph.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace dyng::detail::hyper {

namespace {

std::int64_t padded4(std::int64_t count) noexcept {
  return ((count + 4) / 4) * 4;
}

// One path counter for every non-empty degree bin of an operation's groups (the store's
// degree_bins: thread < 32, warp < 1024, block otherwise).
void record_group_paths(const grouped_incidences& g, std::array<std::int64_t, 3>& paths) {
  bool used[3] = {false, false, false};
  std::int32_t begin = 0;
  for (const std::int32_t end : g.prefix) {
    const std::int32_t size = end - begin;
    used[size < 32 ? 0 : (size < 1024 ? 1 : 2)] = true;
    begin = end;
  }
  for (int b = 0; b < 3; ++b) paths[static_cast<std::size_t>(b)] += used[b] ? 1 : 0;
}

}  // namespace

incidence_model::incidence_model(const resources& res, std::int32_t num_vertices,
                                 const member_rows& rows, const capacity_policy& capacity,
                                 bool has_v2h)
    : threads_(resources_access::host_threads(res)),
      num_vertices_(num_vertices),
      has_v2h_(has_v2h),
      capacity_(capacity),
      alloc_(to_store_policy(capacity), 0) {
  const std::int64_t n = rows.size();
  std::vector<std::int32_t> counts(static_cast<std::size_t>(n));
  std::int64_t size = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    counts[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
        rows.offsets[static_cast<std::size_t>(i) + 1] - rows.offsets[static_cast<std::size_t>(i)]);
    size += padded4(counts[static_cast<std::size_t>(i)]);
  }
  alloc_ = id_allocator(to_store_policy(capacity), initial_payload(capacity, size));
  alloc_.construct(counts);
  rows_.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    rows_[static_cast<std::size_t>(i)].assign(
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i)],
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i) + 1]);
  }
  max_bound_ = alloc_.max_key();
  build_v2h_from_rows();
  build_row_views();
}

incidence_model::incidence_model(const resources& res, const store_snapshot& snapshot,
                                 const capacity_policy& capacity)
    : threads_(resources_access::host_threads(res)),
      num_vertices_(snapshot.num_vertices),
      has_v2h_(snapshot.has_v2h),
      capacity_(capacity),
      alloc_(to_store_policy(capacity), snapshot.h2v.capacity) {
  alloc_.restore(snapshot.h2v);
  rows_.assign(static_cast<std::size_t>(alloc_.max_key()), {});
  for (std::size_t i = 0; i < snapshot.h2v.keys.size(); ++i) {
    if (snapshot.h2v.dead[i] != 0) continue;
    rows_[static_cast<std::size_t>(snapshot.h2v.keys[i]) - 1].assign(
        snapshot.rows.members.begin() + snapshot.rows.offsets[i],
        snapshot.rows.members.begin() + snapshot.rows.offsets[i + 1]);
  }
  max_bound_ = std::max<std::int64_t>(snapshot.max_bound, alloc_.max_key());
  build_v2h_from_rows();
  build_row_views();
  if (snapshot.line_graph) enable_line_graph();
}

incidence_model::incidence_model(const incidence_model& other)
    : incidence_store(),
      threads_(other.threads_),
      num_vertices_(other.num_vertices_),
      has_v2h_(other.has_v2h_),
      capacity_(other.capacity_),
      alloc_(other.alloc_),
      rows_(other.rows_),
      v2h_(other.v2h_),
      fill_paths_(other.fill_paths_),
      unfill_paths_(other.unfill_paths_),
      max_bound_(other.max_bound_),
      line_graph_(other.line_graph_.load()),
      h2h_(other.h2h_) {
  build_row_views();
}

void incidence_model::build_v2h_from_rows() {
  v2h_.clear();
  if (!has_v2h_) return;
  v2h_.resize(static_cast<std::size_t>(num_vertices_));
  for (std::size_t id = 0; id < rows_.size(); ++id) {
    for (const std::int32_t v : rows_[id]) {
      v2h_[static_cast<std::size_t>(v)].push_back(static_cast<std::int64_t>(id));
    }
  }
}

std::unique_ptr<incidence_store> incidence_model::clone(const resources& res) const {
  auto copy = std::make_unique<incidence_model>(*this);
  copy->threads_ = resources_access::host_threads(res);
  return copy;
}

store_snapshot incidence_model::snapshot() const {
  store_snapshot out;
  out.num_vertices = num_vertices_;
  out.has_v2h = has_v2h_;
  out.h2v = alloc_.state();
  out.max_bound = max_bound_;
  out.line_graph = line_graph_.load(std::memory_order_acquire);
  out.rows.clear();
  for (std::size_t i = 0; i < out.h2v.keys.size(); ++i) {
    if (out.h2v.dead[i] == 0) {
      const auto& row = rows_[static_cast<std::size_t>(out.h2v.keys[i]) - 1];
      out.rows.members.insert(out.rows.members.end(), row.begin(), row.end());
    }
    out.rows.offsets.push_back(static_cast<std::int64_t>(out.rows.members.size()));
  }
  return out;
}

std::int64_t incidence_model::id_bound() const noexcept {
  return alloc_.max_key();
}

std::int64_t incidence_model::num_live() const noexcept {
  return alloc_.live();
}

std::int32_t incidence_model::num_vertices() const noexcept {
  return num_vertices_;
}

bool incidence_model::is_live(std::int64_t id) const noexcept {
  return alloc_.state_of(id + 1) == id_allocator::key_state::live;
}

void incidence_model::read_rows(const std::vector<std::int64_t>& ids, member_rows& out) {
  out.clear();
  for (const std::int64_t id : ids) {
    const auto& row = rows_[static_cast<std::size_t>(id)];
    out.members.insert(out.members.end(), row.begin(), row.end());
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
  }
}

void incidence_model::erase(const std::vector<std::int64_t>& ids) {
  std::vector<std::int32_t> keys(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) keys[i] = static_cast<std::int32_t>(ids[i] + 1);
  alloc_.erase(keys);
  for (const std::int64_t id : ids) {
    std::vector<std::int32_t>().swap(rows_[static_cast<std::size_t>(id)]);
  }
}

void incidence_model::insert(const member_rows& rows, std::vector<std::int64_t>& ids) {
  const std::int64_t n = rows.size();
  std::vector<std::int32_t> counts(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    counts[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
        rows.offsets[static_cast<std::size_t>(i) + 1] - rows.offsets[static_cast<std::size_t>(i)]);
  }
  const std::vector<std::int32_t> keys = alloc_.insert(counts);
  // A surplus rebuild drops the deleted slots it did not reuse: the bound may go down (their
  // rows are empty already).
  rows_.resize(static_cast<std::size_t>(alloc_.max_key()));
  max_bound_ = std::max<std::int64_t>(max_bound_, alloc_.max_key());
  ids.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t id = static_cast<std::int64_t>(keys[static_cast<std::size_t>(i)]) - 1;
    ids[static_cast<std::size_t>(i)] = id;
    rows_[static_cast<std::size_t>(id)].assign(
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i)],
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i) + 1]);
  }
}

void incidence_model::v2h_remove(const grouped_incidences& g) {
  if (!has_v2h_ || g.empty()) return;
  record_group_paths(g, unfill_paths_);
  const auto groups = static_cast<std::int64_t>(g.vertices.size());
  const int threads = threads_;
  const bool parallel = threads > 1 && groups > 1024;
  (void)threads;
  (void)parallel;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(dynamic, 64)
#endif
  for (std::int64_t k = 0; k < groups; ++k) {
    auto& row = v2h_[static_cast<std::size_t>(g.vertices[static_cast<std::size_t>(k)])];
    const std::int32_t begin = k == 0 ? 0 : g.prefix[static_cast<std::size_t>(k) - 1];
    for (std::int32_t i = begin; i < g.prefix[static_cast<std::size_t>(k)]; ++i) {
      // ESCHER's unfill removes every occurrence of the value; a v2h row holds an id once.
      const auto it = std::lower_bound(row.begin(), row.end(), g.ids[static_cast<std::size_t>(i)]);
      if (it != row.end() && *it == g.ids[static_cast<std::size_t>(i)]) row.erase(it);
    }
  }
}

void incidence_model::v2h_append(const grouped_incidences& g) {
  if (!has_v2h_ || g.empty()) return;
  record_group_paths(g, fill_paths_);
  const auto groups = static_cast<std::int64_t>(g.vertices.size());
  const int threads = threads_;
  const bool parallel = threads > 1 && groups > 1024;
  (void)threads;
  (void)parallel;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(dynamic, 64)
#endif
  for (std::int64_t k = 0; k < groups; ++k) {
    auto& row = v2h_[static_cast<std::size_t>(g.vertices[static_cast<std::size_t>(k)])];
    const std::int32_t begin = k == 0 ? 0 : g.prefix[static_cast<std::size_t>(k) - 1];
    for (std::int32_t i = begin; i < g.prefix[static_cast<std::size_t>(k)]; ++i) {
      const std::int64_t id = g.ids[static_cast<std::size_t>(i)];
      row.insert(std::lower_bound(row.begin(), row.end(), id), id);
    }
  }
}

void incidence_model::drop_views() noexcept {
  h2v_built_.store(false, std::memory_order_release);
  v2h_built_.store(false, std::memory_order_release);
  h2h_built_.store(false, std::memory_order_release);
  h2v_row_ptr_ = {};
  h2v_col_ind_ = {};
  v2h_row_ptr_ = {};
  v2h_col_ind_ = {};
  h2h_row_ptr_ = {};
  h2h_col_ind_ = {};
  h2h_out_offsets_ = {};
}

void incidence_model::update_line_graph(const std::vector<std::int64_t>& deleted,
                                        const std::vector<std::int64_t>& inserted,
                                        const member_rows& inserted_rows) {
  if (!line_graph_.load(std::memory_order_acquire)) return;
  merge_line_graph(h2h_, max_bound_, deleted, inserted, inserted_rows, v2h_, threads_,
                   payload_limit(capacity_));
}

void incidence_model::finish_apply() {
  drop_views();
  build_row_views();
}

void incidence_model::h2v_csr(std::vector<std::int64_t>& ptr,
                              std::vector<std::int32_t>& cols) const {
  ptr.assign(rows_.size() + 1, 0);
  for (std::size_t id = 0; id < rows_.size(); ++id)
    ptr[id + 1] = ptr[id] + static_cast<std::int64_t>(rows_[id].size());
  cols.resize(static_cast<std::size_t>(ptr.back()));
  for (std::size_t id = 0; id < rows_.size(); ++id)
    std::copy(rows_[id].begin(), rows_[id].end(),
              cols.begin() + static_cast<std::ptrdiff_t>(ptr[id]));
}

void incidence_model::build_row_views() const {
  // h2v by key (A's h2vRows): the members + 1, contiguous; dead and absent keys loc = len = 0.
  const std::size_t bound = rows_.size();
  h2v_loc_.assign(bound + 1, 0);
  h2v_len_.assign(bound + 1, 0);
  std::size_t total = 0;
  for (const auto& r : rows_) total += r.size();
  h2v_flat_.resize(std::max<std::size_t>(total, 1));
  std::size_t at = 0;
  for (std::size_t id = 0; id < bound; ++id) {
    const auto& r = rows_[id];
    if (r.empty()) continue;
    h2v_loc_[id + 1] = static_cast<int>(at);
    h2v_len_[id + 1] = static_cast<int>(r.size());
    for (const std::int32_t v : r) h2v_flat_[at++] = v + 1;
  }
  if (line_graph_.load(std::memory_order_acquire))
    line_graph_locator(h2h_, h2h_loc_, h2h_len_, h2h_out_);
  ++builds_.row_views;
}

bool incidence_model::line_graph_enabled() const noexcept {
  return line_graph_.load(std::memory_order_acquire);
}

void incidence_model::enable_line_graph() const {
  if (line_graph_.load(std::memory_order_acquire)) return;
  const std::lock_guard<std::mutex> lock(view_mutex_);
  if (line_graph_.load(std::memory_order_relaxed)) return;
  DYNG_EXPECTS(has_v2h_,
               "hypergraph: the line graph needs v2h; build the hypergraph with store_v2h = true");
  std::vector<std::int64_t> ptr;
  std::vector<std::int32_t> cols;
  h2v_csr(ptr, cols);
  h2h_ = flat_hyperedge_adjacency(ptr, cols, num_vertices_, max_bound_, threads_,
                                  payload_limit(capacity_));
  line_graph_locator(h2h_, h2h_loc_, h2h_len_, h2h_out_);
  ++builds_.line_graph_builds;
  line_graph_.store(true, std::memory_order_release);
}

row_views incidence_model::views() const {
  const std::lock_guard<std::mutex> lock(view_mutex_);
  row_views v;
  v.space = memory_space::host;
  v.device = -1;
  v.id_bound = alloc_.max_key();
  v.num_vertices = num_vertices_;
  v.h2v.flat = h2v_flat_.data();
  v.h2v.loc = h2v_loc_.data();
  v.h2v.len = h2v_len_.data();
  v.h2v.out = nullptr;
  v.h2v.max_key = static_cast<int>(rows_.size());
  if (line_graph_.load(std::memory_order_acquire)) {
    v.has_h2h = true;
    v.h2h.flat = h2h_.payload.data();
    v.h2h.loc = h2h_loc_.data();
    v.h2h.len = h2h_len_.data();
    v.h2h.out = h2h_out_.data();
    v.h2h.max_key = static_cast<int>(h2h_.num_keys());
  }
  return v;
}

void incidence_model::h2h_view(array_view<const std::int64_t>& row_ptr,
                               array_view<const std::int64_t>& col_ind,
                               array_view<const std::int64_t>& out) const {
  DYNG_EXPECTS(line_graph_.load(std::memory_order_acquire),
               "hypergraph::h2h: the line graph is not maintained");
  if (!h2h_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2h_built_.load(std::memory_order_relaxed)) {
      // Rows by id 0..bound - 1 (key id + 1), decoded: 0-based ids, out as an offset of col_ind.
      const std::size_t bound = rows_.size();
      std::vector<std::int64_t> ptr(bound + 1, 0), outs(bound, 0);
      for (std::size_t id = 0; id < bound; ++id) ptr[id + 1] = ptr[id] + h2h_len_[id + 1];
      std::vector<std::int64_t> cols(static_cast<std::size_t>(ptr.back()));
      for (std::size_t id = 0; id < bound; ++id) {
        const int base = h2h_loc_[id + 1];
        for (int i = 0; i < h2h_len_[id + 1]; ++i) {
          cols[static_cast<std::size_t>(ptr[id] + i)] =
              static_cast<std::int64_t>(h2h_.payload[static_cast<std::size_t>(base + i)]) - 1;
        }
        outs[id] = ptr[id] + (h2h_out_[id + 1] - base);
      }
      h2h_row_ptr_ = std::move(ptr);
      h2h_col_ind_ = std::move(cols);
      h2h_out_offsets_ = std::move(outs);
      ++builds_.h2h;
      h2h_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(h2h_row_ptr_);
  col_ind = host_view(h2h_col_ind_);
  out = host_view(h2h_out_offsets_);
}

line_graph_layout incidence_model::line_graph() const {
  const std::lock_guard<std::mutex> lock(view_mutex_);
  return line_graph_.load(std::memory_order_acquire) ? h2h_ : line_graph_layout{};
}

view_builds incidence_model::builds() const noexcept {
  return builds_;
}

void incidence_model::check_integrity() const {
  std::vector<std::string> problems = alloc_.check();
  const auto bound = static_cast<std::size_t>(alloc_.max_key());
  if (rows_.size() != bound) problems.push_back("the rows do not cover the id bound");
  std::int64_t live = 0;
  std::vector<std::vector<std::int64_t>> transposed;
  if (has_v2h_) transposed.resize(static_cast<std::size_t>(num_vertices_));
  for (std::size_t id = 0; id < rows_.size() && problems.size() < 20; ++id) {
    const auto& row = rows_[id];
    const bool alive = is_live(static_cast<std::int64_t>(id));
    if (alive) {
      ++live;
      if (row.empty()) problems.push_back("live id " + std::to_string(id) + " has an empty row");
      if (static_cast<std::int64_t>(row.size()) != alloc_.count(static_cast<std::int64_t>(id) + 1))
        problems.push_back("id " + std::to_string(id) + ": the row and its slot disagree");
    } else if (!row.empty()) {
      problems.push_back("dead id " + std::to_string(id) + " has a row");
    }
    for (std::size_t k = 0; k < row.size(); ++k) {
      if (row[k] < 0 || row[k] >= num_vertices_ || (k > 0 && row[k] <= row[k - 1])) {
        problems.push_back("the row of id " + std::to_string(id) +
                           " is not sorted, distinct and in range");
        break;
      }
      if (has_v2h_)
        transposed[static_cast<std::size_t>(row[k])].push_back(static_cast<std::int64_t>(id));
    }
  }
  if (live != alloc_.live()) problems.push_back("the live count is wrong");
  if (has_v2h_ && transposed != v2h_) problems.push_back("v2h is not the transpose of h2v");
  if (max_bound_ < alloc_.max_key())
    problems.push_back("the line graph's key count is below the bound");
  if (line_graph_.load(std::memory_order_acquire)) {
    // The line graph against its rebuild from h2v (ESCHER-GPU's checkIntegrity compares H2H with
    // the reference line graph of its host model), layout included.
    std::vector<std::int64_t> ptr;
    std::vector<std::int32_t> cols;
    h2v_csr(ptr, cols);
    const line_graph_layout want = flat_hyperedge_adjacency(
        ptr, cols, num_vertices_, max_bound_, threads_, std::numeric_limits<std::int64_t>::max());
    if (want.counts != h2h_.counts || want.starts != h2h_.starts || want.payload != h2h_.payload) {
      std::int64_t key = 0;
      while (key < std::min(want.num_keys(), h2h_.num_keys()) &&
             want.counts[static_cast<std::size_t>(key)] ==
                 h2h_.counts[static_cast<std::size_t>(key)])
        ++key;
      problems.push_back("the line graph differs from its rebuild from h2v (first at key " +
                         std::to_string(key + 1) + " of " + std::to_string(h2h_.num_keys()) + ")");
    }
    std::vector<int> loc, len, out;
    line_graph_locator(h2h_, loc, len, out);
    if (loc != h2h_loc_ || len != h2h_len_ || out != h2h_out_)
      problems.push_back("the line graph's row view is stale");
  }
  if (!problems.empty()) {
    std::string all;
    for (const auto& p : problems) all += "\n  " + p;
    DYNG_FAIL("hypergraph integrity (host store):", all);
  }
}

storage_stats incidence_model::stats() const {
  storage_stats st;
  std::int64_t bytes =
      static_cast<std::int64_t>(rows_.size() * sizeof(rows_[0]) + v2h_.size() * sizeof(v2h_[0]));
  for (const auto& r : rows_) bytes += static_cast<std::int64_t>(r.capacity() * sizeof(r[0]));
  for (const auto& r : v2h_) bytes += static_cast<std::int64_t>(r.capacity() * sizeof(r[0]));
  bytes += static_cast<std::int64_t>(
      (h2h_.payload.capacity() + h2h_.starts.capacity() + h2h_.counts.capacity() +
       h2v_flat_.capacity() + h2v_loc_.capacity() + h2v_len_.capacity() + h2h_loc_.capacity() +
       h2h_len_.capacity() + h2h_out_.capacity()) *
      sizeof(int));
  st.host_bytes = bytes;
  if (h2v_built_.load(std::memory_order_acquire)) {
    st.view_bytes += static_cast<std::int64_t>(h2v_row_ptr_.size() * sizeof(std::int64_t) +
                                               h2v_col_ind_.size() * sizeof(std::int32_t));
  }
  if (v2h_built_.load(std::memory_order_acquire)) {
    st.view_bytes += static_cast<std::int64_t>((v2h_row_ptr_.size() + v2h_col_ind_.size()) *
                                               sizeof(std::int64_t));
  }
  if (h2h_built_.load(std::memory_order_acquire)) {
    st.view_bytes += static_cast<std::int64_t>(
        (h2h_row_ptr_.size() + h2h_col_ind_.size() + h2h_out_offsets_.size()) *
        sizeof(std::int64_t));
  }
  st.payload_capacity = alloc_.capacity();
  st.payload_used = alloc_.bump();
  st.growths = alloc_.growths();
  st.compactions = alloc_.compactions();
  const auto& reuse = alloc_.reuse_paths();
  st.reuse = {reuse[0], reuse[1], reuse[2]};
  st.fill = {fill_paths_[0], fill_paths_[1], fill_paths_[2]};
  st.unfill = {unfill_paths_[0], unfill_paths_[1], unfill_paths_[2]};
  return st;
}

void incidence_model::h2v_view(array_view<const std::int64_t>& row_ptr,
                               array_view<const std::int32_t>& col_ind) const {
  if (!h2v_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2v_built_.load(std::memory_order_relaxed)) {
      std::vector<std::int64_t> ptr(rows_.size() + 1, 0);
      for (std::size_t id = 0; id < rows_.size(); ++id) {
        ptr[id + 1] = ptr[id] + static_cast<std::int64_t>(rows_[id].size());
      }
      std::vector<std::int32_t> cols(static_cast<std::size_t>(ptr.back()));
      for (std::size_t id = 0; id < rows_.size(); ++id) {
        std::copy(rows_[id].begin(), rows_[id].end(),
                  cols.begin() + static_cast<std::ptrdiff_t>(ptr[id]));
      }
      h2v_row_ptr_ = std::move(ptr);
      h2v_col_ind_ = std::move(cols);
      ++builds_.h2v;
      h2v_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(h2v_row_ptr_);
  col_ind = host_view(h2v_col_ind_);
}

void incidence_model::v2h_view(array_view<const std::int64_t>& row_ptr,
                               array_view<const std::int64_t>& col_ind) const {
  DYNG_EXPECTS(has_v2h_, "hypergraph::v2h: the hypergraph was built with store_v2h = false");
  if (!v2h_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!v2h_built_.load(std::memory_order_relaxed)) {
      std::vector<std::int64_t> ptr(v2h_.size() + 1, 0);
      for (std::size_t v = 0; v < v2h_.size(); ++v) {
        ptr[v + 1] = ptr[v] + static_cast<std::int64_t>(v2h_[v].size());
      }
      std::vector<std::int64_t> cols(static_cast<std::size_t>(ptr.back()));
      for (std::size_t v = 0; v < v2h_.size(); ++v) {
        std::copy(v2h_[v].begin(), v2h_[v].end(),
                  cols.begin() + static_cast<std::ptrdiff_t>(ptr[v]));
      }
      v2h_row_ptr_ = std::move(ptr);
      v2h_col_ind_ = std::move(cols);
      ++builds_.v2h;
      v2h_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(v2h_row_ptr_);
  col_ind = host_view(v2h_col_ind_);
}

}  // namespace dyng::detail::hyper
