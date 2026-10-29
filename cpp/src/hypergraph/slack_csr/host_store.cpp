// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (the host incidence model)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (finishBatch: the H2V
// unfill and fill; h2hPairCount)
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu (applyBatch: the delta, the CSR apply,
// its rebuild after a tail overflow)
/**
 * @file host_store.cpp
 * @brief The host store of slack_csr storage (ADR 0086).
 */
#include "hypergraph/slack_csr/host_store.hpp"

#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/line_graph/incidence_delta.hpp"
#include "operators/execution.hpp"
#include "util/region_exceptions.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace dyng::detail::hyper {

namespace {

namespace ops = dyng::detail::operators;

std::int64_t padded4(std::int64_t count) noexcept {
  return ((count + 4) / 4) * 4;
}

// The host executor of `threads` threads (the sequential one on one thread or without OpenMP):
// the delta and the CSR apply are one source on both (ADR 0086).
template <typename work_t>
void with_exec(int threads, work_t&& work) {
  (void)threads;  // without OpenMP: the sequential executor only
#if DYNG_HAS_OPENMP
  if (threads > 1) {
    const ops::openmp_exec exec(resources::openmp(threads));
    work(exec);
    return;
  }
#endif
  const ops::sequential_exec exec(resources::sequential());
  work(exec);
}

// Resets the touched slots of a batch, also when the delta throws.
struct slot_reset {
  std::vector<std::int32_t>& slots;
  const std::vector<std::int64_t>& a;
  const std::vector<std::int64_t>& b;
  ~slot_reset() {
    for (const std::int64_t id : a)
      if (id >= 0 && static_cast<std::size_t>(id) < slots.size())
        slots[static_cast<std::size_t>(id)] = -1;
    for (const std::int64_t id : b)
      if (id >= 0 && static_cast<std::size_t>(id) < slots.size())
        slots[static_cast<std::size_t>(id)] = -1;
  }
};

}  // namespace

slack_host_store::slack_host_store(const resources& res, std::int32_t num_vertices,
                                   const member_rows& rows, const capacity_policy& capacity,
                                   const slack::knobs& knobs)
    : threads_(resources_access::host_threads(res)),
      num_vertices_(num_vertices),
      capacity_(capacity),
      knobs_(knobs),
      alloc_(to_store_policy(capacity), 0) {
  const std::int64_t n = rows.size();
  std::vector<std::int32_t> counts(static_cast<std::size_t>(n));
  std::int64_t size = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    counts[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
        rows.offsets[static_cast<std::size_t>(i) + 1] - rows.offsets[static_cast<std::size_t>(i)]);
    size += padded4(counts[static_cast<std::size_t>(i)]);
  }
  // The ids are those of MOSP_ESCHER's H2V tree (constructFromRows: the padded rows, the
  // capacity of the policy).
  alloc_ = id_allocator(to_store_policy(capacity), initial_payload(capacity, size));
  alloc_.construct(counts);
  rows_.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    rows_[static_cast<std::size_t>(i)].assign(
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i)],
        rows.members.begin() + rows.offsets[static_cast<std::size_t>(i) + 1]);
  }
  max_bound_ = alloc_.max_key();
  weights_.assign(static_cast<std::size_t>(max_bound_), 1);
  build_v2h_from_rows();
}

slack_host_store::slack_host_store(const resources& res, const store_snapshot& snapshot,
                                   const capacity_policy& capacity, const slack::knobs& knobs)
    : threads_(resources_access::host_threads(res)),
      num_vertices_(snapshot.num_vertices),
      capacity_(capacity),
      knobs_(knobs),
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
  weights_.assign(static_cast<std::size_t>(max_bound_), 1);
  build_v2h_from_rows();
  if (snapshot.line_graph) enable_line_graph();
}

slack_host_store::slack_host_store(const slack_host_store& other)
    : incidence_store(),
      slack_access(),
      threads_(other.threads_),
      num_vertices_(other.num_vertices_),
      capacity_(other.capacity_),
      knobs_(other.knobs_),
      alloc_(other.alloc_),
      rows_(other.rows_),
      v2h_(other.v2h_),
      weights_(other.weights_),
      max_bound_(other.max_bound_),
      pair_count_(other.pair_count_),
      candidates_(other.candidates_),
      csr_rebuilds_(other.csr_rebuilds_),
      line_graph_(other.line_graph_.load()),
      lg_(other.lg_),
      line_graph_builds_(other.line_graph_builds_) {}

void slack_host_store::build_v2h_from_rows() {
  v2h_.assign(static_cast<std::size_t>(num_vertices_), {});
  for (std::size_t id = 0; id < rows_.size(); ++id)
    for (const std::int32_t v : rows_[id])
      v2h_[static_cast<std::size_t>(v)].push_back(static_cast<std::int64_t>(id));
}

std::unique_ptr<incidence_store> slack_host_store::clone(const resources& res) const {
  auto copy = std::make_unique<slack_host_store>(*this);
  copy->threads_ = resources_access::host_threads(res);
  return copy;
}

store_snapshot slack_host_store::snapshot() const {
  store_snapshot out;
  out.num_vertices = num_vertices_;
  out.has_v2h = true;
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

std::int64_t slack_host_store::id_bound() const noexcept {
  return alloc_.max_key();
}

std::int64_t slack_host_store::num_live() const noexcept {
  return alloc_.live();
}

std::int32_t slack_host_store::num_vertices() const noexcept {
  return num_vertices_;
}

bool slack_host_store::is_live(std::int64_t id) const noexcept {
  return alloc_.state_of(id + 1) == id_allocator::key_state::live;
}

void slack_host_store::read_rows(const std::vector<std::int64_t>& ids, member_rows& out) {
  out.clear();
  for (const std::int64_t id : ids) {
    // An id at or above the bound (a deleted id a surplus insert dropped) has an empty row.
    if (id >= 0 && static_cast<std::size_t>(id) < rows_.size()) {
      const auto& row = rows_[static_cast<std::size_t>(id)];
      out.members.insert(out.members.end(), row.begin(), row.end());
    }
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
  }
}

void slack_host_store::vertex_lengths(const std::vector<std::int32_t>& vertices,
                                      std::vector<std::int32_t>& out) const {
  out.resize(vertices.size());
  for (std::size_t i = 0; i < vertices.size(); ++i)
    out[i] = static_cast<std::int32_t>(v2h_[static_cast<std::size_t>(vertices[i])].size());
}

void slack_host_store::erase(const std::vector<std::int64_t>& ids) {
  std::vector<std::int32_t> keys(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i) keys[i] = static_cast<std::int32_t>(ids[i] + 1);
  alloc_.erase(keys);
  for (const std::int64_t id : ids)
    std::vector<std::int32_t>().swap(rows_[static_cast<std::size_t>(id)]);
}

void slack_host_store::insert(const member_rows& rows, std::vector<std::int64_t>& ids) {
  const std::int64_t n = rows.size();
  std::vector<std::int32_t> counts(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    counts[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
        rows.offsets[static_cast<std::size_t>(i) + 1] - rows.offsets[static_cast<std::size_t>(i)]);
  }
  const std::vector<std::int32_t> keys = alloc_.insert(counts);
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

void slack_host_store::v2h_remove(const grouped_incidences& g) {
  if (g.empty()) return;
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
      // lower_bound and erase of int64 values: nothing allocates or throws.
      const auto it = std::lower_bound(row.begin(), row.end(), g.ids[static_cast<std::size_t>(i)]);
      if (it != row.end() && *it == g.ids[static_cast<std::size_t>(i)]) row.erase(it);
    }
  }
}

void slack_host_store::v2h_append(const grouped_incidences& g) {
  if (g.empty()) return;
  const auto groups = static_cast<std::int64_t>(g.vertices.size());
  const int threads = threads_;
  const bool parallel = threads > 1 && groups > 1024;
  (void)threads;
  (void)parallel;
  // A row's insert allocates: its exception is carried out of the region (ADR 0061).
  region_exceptions errors;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(dynamic, 64)
#endif
  for (std::int64_t k = 0; k < groups; ++k) {
    errors.run([&] {
      auto& row = v2h_[static_cast<std::size_t>(g.vertices[static_cast<std::size_t>(k)])];
      const std::int32_t begin = k == 0 ? 0 : g.prefix[static_cast<std::size_t>(k) - 1];
      for (std::int32_t i = begin; i < g.prefix[static_cast<std::size_t>(k)]; ++i) {
        const std::int64_t id = g.ids[static_cast<std::size_t>(i)];
        row.insert(std::lower_bound(row.begin(), row.end(), id), id);
      }
    });
  }
  errors.rethrow();
}

void slack_host_store::h2v_remove(const grouped_members& g) {
  if (g.empty()) return;
  // MOSP_ESCHER's H2V unfill (its values are the vertices + 1, as its CBST keeps them).
  std::vector<std::int32_t> keys(g.ids.size()), values(g.members.size());
  for (std::size_t i = 0; i < g.ids.size(); ++i) keys[i] = static_cast<std::int32_t>(g.ids[i] + 1);
  for (std::size_t i = 0; i < g.members.size(); ++i) values[i] = g.members[i] + 1;
  alloc_.remove_values(keys, g.prefix, values);
  for (std::size_t k = 0; k < g.ids.size(); ++k) {
    auto& row = rows_[static_cast<std::size_t>(g.ids[k])];
    for (std::int32_t i = k == 0 ? 0 : g.prefix[k - 1]; i < g.prefix[k]; ++i) {
      const auto it =
          std::lower_bound(row.begin(), row.end(), g.members[static_cast<std::size_t>(i)]);
      if (it != row.end() && *it == g.members[static_cast<std::size_t>(i)]) row.erase(it);
    }
  }
}

void slack_host_store::h2v_append(const grouped_members& g) {
  if (g.empty()) return;
  // MOSP_ESCHER's H2V fill: an overflow segment for what does not fit a row's tail; a fill that
  // does not fit the payload compacts first (and, under mosp_escher, then fails).
  std::vector<std::int32_t> keys(g.ids.size()), values(g.members.size());
  for (std::size_t i = 0; i < g.ids.size(); ++i) keys[i] = static_cast<std::int32_t>(g.ids[i] + 1);
  for (std::size_t i = 0; i < g.members.size(); ++i) values[i] = g.members[i] + 1;
  alloc_.append(keys, g.prefix, values);
  for (std::size_t k = 0; k < g.ids.size(); ++k) {
    auto& row = rows_[static_cast<std::size_t>(g.ids[k])];
    for (std::int32_t i = k == 0 ? 0 : g.prefix[k - 1]; i < g.prefix[k]; ++i) {
      const std::int32_t v = g.members[static_cast<std::size_t>(i)];
      row.insert(std::lower_bound(row.begin(), row.end(), v), v);
    }
  }
}

void slack_host_store::update_line_graph(const std::vector<std::int64_t>& /*deleted*/,
                                         const std::vector<std::int64_t>& /*inserted*/,
                                         const member_rows& /*inserted_rows*/) {
  DYNG_FAIL("hypergraph: the sorted merge of escher storage reached a slack_csr store");
}

void slack_host_store::begin_line_graph_delta(const std::vector<std::int32_t>& vertices,
                                              const std::vector<std::int64_t>& ids) {
  pre_vertices_.clear();
  pre_v2h_offsets_.assign(1, 0);
  pre_v2h_ids_.clear();
  pre_ids_.clear();
  pre_rows_.clear();
  if (!line_graph_.load(std::memory_order_acquire)) return;
  // The hyperedges of the touched vertices and the rows of the deleted and edited ids before the
  // batch (MOSP_ESCHER: pass 0 reads the mirror before applyVertexRowsKernel; the shipped
  // pre-batch rows of the touched hyperedges).
  pre_vertices_ = vertices;
  for (const std::int32_t v : vertices) {
    const auto& row = v2h_[static_cast<std::size_t>(v)];
    pre_v2h_ids_.insert(pre_v2h_ids_.end(), row.begin(), row.end());
    pre_v2h_offsets_.push_back(static_cast<std::int64_t>(pre_v2h_ids_.size()));
  }
  pre_ids_ = ids;
  read_rows(ids, pre_rows_);
}

void slack_host_store::line_graph_delta_of(const std::vector<std::int32_t>& vertices,
                                           const std::vector<std::int64_t>& ids,
                                           const std::vector<std::int8_t>& signs,
                                           const std::vector<std::int64_t>& inserted_ids,
                                           const touched_rows& /*touched*/, line_graph_delta& out) {
  out.clear();
  if (!line_graph_.load(std::memory_order_acquire)) return;
  // The rows before the batch of every touched id: the kept rows of the deleted and edited ids,
  // and an empty row for an inserted id that was not live (dead or above the bound before).
  if (touched_slot_.size() < static_cast<std::size_t>(max_bound_))
    touched_slot_.resize(static_cast<std::size_t>(max_bound_), -1);
  const slot_reset reset{touched_slot_, pre_ids_, inserted_ids};
  for (std::size_t i = 0; i < pre_ids_.size(); ++i)
    touched_slot_[static_cast<std::size_t>(pre_ids_[i])] = static_cast<std::int32_t>(i);
  for (const std::int64_t id : inserted_ids) {
    if (touched_slot_[static_cast<std::size_t>(id)] >= 0) continue;  // deleted, then reused
    touched_slot_[static_cast<std::size_t>(id)] = static_cast<std::int32_t>(pre_rows_.size());
    pre_rows_.offsets.push_back(static_cast<std::int64_t>(pre_rows_.members.size()));
  }
  std::vector<std::uint64_t> keys;
  std::vector<std::int8_t> net;
  net_incidence_changes(vertices, ids, signs, keys, net);
  incidence_delta_rows rows;
  rows.post_rows = &rows_;
  rows.touched_slot = &touched_slot_;
  rows.pre_rows = &pre_rows_;
  rows.pre_vertices = &pre_vertices_;
  rows.pre_v2h_offsets = &pre_v2h_offsets_;
  rows.pre_v2h_ids = &pre_v2h_ids_;
  rows.post_v2h = &v2h_;
  with_exec(threads_, [&](const auto& exec) {
    incidence_delta(exec, keys, rows, knobs_.max_candidates, out);
  });
  candidates_ = out.candidates;
}

std::int64_t slack_host_store::planned_node_capacity(std::int64_t needed) const noexcept {
  // MOSP_ESCHER plans maxNodes = the rows + the planned inserts + 1024; the parity harness sets it
  // (knobs.node_capacity). Without it: the bound and room for an eighth more (at least 1024).
  if (knobs_.node_capacity >= needed) return knobs_.node_capacity;
  return needed + std::max<std::int64_t>(1024, needed / 8);
}

void slack_host_store::build_line_graph() const {
  const container_scope container;  // the container's own work (budgets, I9)
  const std::int64_t nodes = std::max(planned_node_capacity(max_bound_), lg_.node_capacity());
  with_exec(threads_, [&](const auto& exec) {
    lg_.build(exec, rows_, v2h_, max_bound_, nodes, knobs_.entry_headroom, weights_);
  });
  ++line_graph_builds_;
}

void slack_host_store::build_line_graph_into(slack_line_graph& lg) const {
  with_exec(threads_, [&](const auto& exec) {
    lg.build(exec, rows_, v2h_, max_bound_, max_bound_, 1.0, weights_);
  });
}

void slack_host_store::finish_line_graph(const line_graph_delta& delta,
                                         const std::vector<std::int64_t>& new_ids,
                                         const std::vector<std::int64_t>& new_weights) {
  weights_.resize(static_cast<std::size_t>(max_bound_), 0);
  for (std::size_t i = 0; i < new_ids.size(); ++i)
    weights_[static_cast<std::size_t>(new_ids[i])] = new_weights[i];
  if (!line_graph_.load(std::memory_order_acquire)) return;
  if (max_bound_ > lg_.node_capacity()) lg_.grow_nodes(max_bound_);
  bool applied = false;
  with_exec(threads_, [&](const auto& exec) {
    applied = lg_.apply(exec, delta, max_bound_, new_ids, new_weights);
  });
  if (!applied) {
    // The tail is exhausted: rebuild the CSR from the incidence lists (applyBatch's buildCsr with
    // its entryHeadroom argument).
    build_line_graph();
    ++csr_rebuilds_;
  }
  const auto inserted = static_cast<std::int64_t>(delta.inserted_a.size());
  const auto deleted = static_cast<std::int64_t>(delta.deleted_a.size());
#if defined(DYNG_MUTATION_HSOSP_PAIR_COUNT_ADDS_DELETIONS)
  // MUTATION (PLAN 8.4; M11a criterion 16 #7): the maintained pair count adds the deletions.
  pair_count_ += inserted + deleted;
#else
  pair_count_ += inserted - deleted;  // MOSP_ESCHER's h2hPairCount += (insKeys - delKeys) / 2
#endif
}

void slack_host_store::set_node_weights(const std::vector<std::int64_t>& by_id) {
  weights_.assign(by_id.begin(), by_id.end());
  weights_.resize(static_cast<std::size_t>(max_bound_), 0);
  if (line_graph_.load(std::memory_order_acquire)) {
    for (std::size_t id = 0;
         id < weights_.size() && static_cast<std::int64_t>(id) < lg_.node_capacity(); ++id)
      lg_.set_weight(static_cast<std::int64_t>(id), weights_[id]);
  }
}

void slack_host_store::set_entry_headroom(double headroom) {
  DYNG_EXPECTS(headroom >= 1.0,
               "hypergraph: the slack_csr line graph's entry headroom must be at "
               "least 1, got ",
               headroom);
  knobs_.entry_headroom = headroom;
}

void slack_host_store::drop_views() noexcept {
  h2v_built_.store(false, std::memory_order_release);
  v2h_built_.store(false, std::memory_order_release);
  h2h_built_.store(false, std::memory_order_release);
  row_views_built_.store(false, std::memory_order_release);
  h2v_row_ptr_ = {};
  h2v_col_ind_ = {};
  v2h_row_ptr_ = {};
  v2h_col_ind_ = {};
  h2h_row_ptr_ = {};
  h2h_col_ind_ = {};
  h2h_out_offsets_ = {};
  h2v_flat_ = {};
  h2v_loc_ = {};
  h2v_len_ = {};
  h2h_flat_ = {};
  h2h_loc_ = {};
  h2h_len_ = {};
  h2h_out_ = {};
}

void slack_host_store::finish_apply() {
  drop_views();
}

bool slack_host_store::line_graph_enabled() const noexcept {
  return line_graph_.load(std::memory_order_acquire);
}

void slack_host_store::enable_line_graph() const {
  if (line_graph_.load(std::memory_order_acquire)) return;
  const std::lock_guard<std::mutex> lock(view_mutex_);
  if (line_graph_.load(std::memory_order_relaxed)) return;
  build_line_graph();
  pair_count_ = lg_.num_entries() / 2;
  line_graph_.store(true, std::memory_order_release);
}

row_views slack_host_store::views() const {
  const std::lock_guard<std::mutex> lock(view_mutex_);
  if (!row_views_built_.load(std::memory_order_relaxed)) {
    const container_scope container;  // the container's own work (budgets, I9)
    // h2v by key (A's h2vRows), built on request: the slack store keeps no ESCHER row view.
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
    if (line_graph_.load(std::memory_order_acquire)) {
      // h2h by key in the same layout (ESCHER-GPU's RowLocator over sorted rows): the readers of
      // the row views (the kit's fakes) work on slack_csr storage too.
      std::vector<std::int64_t> ptr, cols, outs;
      lg_.sorted_rows(max_bound_, ptr, cols, outs);
      h2h_flat_.resize(std::max<std::size_t>(cols.size(), 1));
      for (std::size_t e = 0; e < cols.size(); ++e) h2h_flat_[e] = static_cast<int>(cols[e] + 1);
      h2h_loc_.assign(static_cast<std::size_t>(max_bound_) + 1, 0);
      h2h_len_.assign(static_cast<std::size_t>(max_bound_) + 1, 0);
      h2h_out_.assign(static_cast<std::size_t>(max_bound_) + 1, 0);
      for (std::int64_t r = 0; r < max_bound_; ++r) {
        const auto k = static_cast<std::size_t>(r) + 1;
        h2h_loc_[k] = static_cast<int>(ptr[static_cast<std::size_t>(r)]);
        h2h_len_[k] = static_cast<int>(ptr[static_cast<std::size_t>(r) + 1] -
                                       ptr[static_cast<std::size_t>(r)]);
        h2h_out_[k] = static_cast<int>(outs[static_cast<std::size_t>(r)]);
      }
    }
    ++builds_.row_views;
    row_views_built_.store(true, std::memory_order_release);
  }
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
    // Built on request (the slack line graph itself is read through slack(), slack_rows.hpp).
    v.has_h2h = true;
    v.h2h.flat = h2h_flat_.data();
    v.h2h.loc = h2h_loc_.data();
    v.h2h.len = h2h_len_.data();
    v.h2h.out = h2h_out_.data();
    v.h2h.max_key = static_cast<int>(max_bound_);
  }
  return v;
}

void slack_host_store::h2h_view(array_view<const std::int64_t>& row_ptr,
                                array_view<const std::int64_t>& col_ind,
                                array_view<const std::int64_t>& out) const {
  DYNG_EXPECTS(line_graph_.load(std::memory_order_acquire),
               "hypergraph::h2h: the line graph is not maintained");
  if (!h2h_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2h_built_.load(std::memory_order_relaxed)) {
      const container_scope container;  // the container's own work (budgets, I9)
      // The slack rows sorted into a contiguous CSR (HD9: a sorted view on request).
      lg_.sorted_rows(static_cast<std::int64_t>(rows_.size()), h2h_row_ptr_, h2h_col_ind_,
                      h2h_out_offsets_);
      ++builds_.h2h;
      h2h_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(h2h_row_ptr_);
  col_ind = host_view(h2h_col_ind_);
  out = host_view(h2h_out_offsets_);
}

line_graph_layout slack_host_store::line_graph() const {
  // The rows in ESCHER-GPU's layout (keys 1..max_bound_, the neighbours' keys sorted, padding rule
  // 1): what the tests compare across storages.
  line_graph_layout g;
  if (!line_graph_.load(std::memory_order_acquire)) return g;
  std::vector<std::int64_t> ptr, cols, out;
  lg_.sorted_rows(max_bound_, ptr, cols, out);
  g.starts.resize(static_cast<std::size_t>(max_bound_));
  g.counts.resize(static_cast<std::size_t>(max_bound_));
  std::int64_t at = 0;
  for (std::int64_t k = 0; k < max_bound_; ++k) {
    const auto c = ptr[static_cast<std::size_t>(k) + 1] - ptr[static_cast<std::size_t>(k)];
    g.starts[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(at);
    g.counts[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(c);
    at += padded4(c);
  }
  g.payload.assign(static_cast<std::size_t>(at), 0);
  for (std::int64_t k = 0; k < max_bound_; ++k) {
    const auto base = static_cast<std::size_t>(g.starts[static_cast<std::size_t>(k)]);
    for (std::int64_t e = ptr[static_cast<std::size_t>(k)];
         e < ptr[static_cast<std::size_t>(k) + 1]; ++e)
      g.payload[base + static_cast<std::size_t>(e - ptr[static_cast<std::size_t>(k)])] =
          static_cast<std::int32_t>(cols[static_cast<std::size_t>(e)] + 1);
    g.payload[base + static_cast<std::size_t>(padded4(g.counts[static_cast<std::size_t>(k)])) - 1] =
        INT_MIN;
  }
  return g;
}

view_builds slack_host_store::builds() const noexcept {
  view_builds b = builds_;
  b.line_graph_builds = line_graph_builds_;
  return b;
}

void slack_host_store::check_integrity() const {
  std::vector<std::string> problems = alloc_.check();
  const auto bound = static_cast<std::size_t>(alloc_.max_key());
  if (rows_.size() != bound) problems.push_back("the rows do not cover the id bound");
  std::int64_t live = 0;
  std::vector<std::vector<std::int64_t>> transposed(static_cast<std::size_t>(num_vertices_));
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
      transposed[static_cast<std::size_t>(row[k])].push_back(static_cast<std::int64_t>(id));
    }
  }
  if (live != alloc_.live()) problems.push_back("the live count is wrong");
  if (transposed != v2h_) problems.push_back("v2h is not the transpose of h2v");
  if (max_bound_ < alloc_.max_key()) problems.push_back("the node count is below the bound");
  if (line_graph_.load(std::memory_order_acquire)) {
    // The slack CSR against the incidence lists (MOSP_ESCHER's checks: the device rows against
    // its rebuilt line graph), its layout, the maintained pair count and the node weights.
    for (const auto& p : lg_.check(rows_, v2h_)) problems.push_back(p);
    if (pair_count_ * 2 != lg_.num_entries())
      problems.push_back("the maintained pair count " + std::to_string(pair_count_) +
                         " is not the line graph's " + std::to_string(lg_.num_entries() / 2));
    for (std::size_t id = 0; id < rows_.size(); ++id) {
      if (is_live(static_cast<std::int64_t>(id)) && lg_.node_weight()[id] != weights_[id]) {
        problems.push_back("the line graph's weight of id " + std::to_string(id) + " is stale");
        break;
      }
    }
  }
  if (!problems.empty()) {
    std::string all;
    for (const auto& p : problems) all += "\n  " + p;
    DYNG_FAIL("hypergraph integrity (slack_csr host store):", all);
  }
}

storage_stats slack_host_store::stats() const {
  storage_stats st;
  std::int64_t bytes =
      static_cast<std::int64_t>(rows_.size() * sizeof(rows_[0]) + v2h_.size() * sizeof(v2h_[0]) +
                                weights_.capacity() * sizeof(std::int64_t));
  for (const auto& r : rows_) bytes += static_cast<std::int64_t>(r.capacity() * sizeof(r[0]));
  for (const auto& r : v2h_) bytes += static_cast<std::int64_t>(r.capacity() * sizeof(r[0]));
  bytes += lg_.bytes();
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
  if (row_views_built_.load(std::memory_order_acquire)) {
    st.view_bytes += static_cast<std::int64_t>(
        (h2v_flat_.size() + h2v_loc_.size() + h2v_len_.size() + h2h_flat_.size() + h2h_loc_.size() +
         h2h_len_.size() + h2h_out_.size()) *
        sizeof(int));
  }
  st.payload_capacity = alloc_.capacity();
  st.payload_used = alloc_.bump();
  st.growths = alloc_.growths();
  st.compactions = alloc_.compactions();
  // The kernel paths of the H2V tree's operations (the ids' model; v2h is no tree here).
  const auto& reuse = alloc_.reuse_paths();
  const auto& fill = alloc_.fill_paths();
  const auto& unfill = alloc_.unfill_paths();
  st.reuse = {reuse[0], reuse[1], reuse[2]};
  st.fill = {fill[0], fill[1], fill[2]};
  st.unfill = {unfill[0], unfill[1], unfill[2]};
  return st;
}

void slack_host_store::h2v_view(array_view<const std::int64_t>& row_ptr,
                                array_view<const std::int32_t>& col_ind) const {
  if (!h2v_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2v_built_.load(std::memory_order_relaxed)) {
      const container_scope container;  // the container's own work (budgets, I9)
      std::vector<std::int64_t> ptr(rows_.size() + 1, 0);
      for (std::size_t id = 0; id < rows_.size(); ++id)
        ptr[id + 1] = ptr[id] + static_cast<std::int64_t>(rows_[id].size());
      std::vector<std::int32_t> cols(static_cast<std::size_t>(ptr.back()));
      for (std::size_t id = 0; id < rows_.size(); ++id)
        std::copy(rows_[id].begin(), rows_[id].end(),
                  cols.begin() + static_cast<std::ptrdiff_t>(ptr[id]));
      h2v_row_ptr_ = std::move(ptr);
      h2v_col_ind_ = std::move(cols);
      ++builds_.h2v;
      h2v_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(h2v_row_ptr_);
  col_ind = host_view(h2v_col_ind_);
}

void slack_host_store::v2h_view(array_view<const std::int64_t>& row_ptr,
                                array_view<const std::int64_t>& col_ind) const {
  if (!v2h_built_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!v2h_built_.load(std::memory_order_relaxed)) {
      const container_scope container;  // the container's own work (budgets, I9)
      std::vector<std::int64_t> ptr(v2h_.size() + 1, 0);
      for (std::size_t v = 0; v < v2h_.size(); ++v)
        ptr[v + 1] = ptr[v] + static_cast<std::int64_t>(v2h_[v].size());
      std::vector<std::int64_t> cols(static_cast<std::size_t>(ptr.back()));
      for (std::size_t v = 0; v < v2h_.size(); ++v)
        std::copy(v2h_[v].begin(), v2h_[v].end(),
                  cols.begin() + static_cast<std::ptrdiff_t>(ptr[v]));
      v2h_row_ptr_ = std::move(ptr);
      v2h_col_ind_ = std::move(cols);
      ++builds_.v2h;
      v2h_built_.store(true, std::memory_order_release);
    }
  }
  row_ptr = host_view(v2h_row_ptr_);
  col_ind = host_view(v2h_col_ind_);
}

slack_row_view slack_host_store::rows() const {
  DYNG_EXPECTS(line_graph_.load(std::memory_order_acquire),
               "hypergraph: the slack_csr line graph is not maintained");
  slack_row_view v;
  v.space = memory_space::host;
  v.device = -1;
  v.num_nodes = lg_.num_nodes();
  v.node_capacity = lg_.node_capacity();
  v.num_entries = lg_.num_entries();
  v.used_entries = lg_.used_entries();
  v.cap_entries = lg_.cap_entries();
  v.pair_count = pair_count_;
  v.row_start = lg_.row_start();
  v.deg = lg_.deg();
  v.cap = lg_.cap();
  v.col_ind = lg_.col_ind();
  v.node_weight = lg_.node_weight();
  return v;
}

slack_counters slack_host_store::counters() const noexcept {
  slack_counters c;
  c.csr_rebuilds = csr_rebuilds_;
  c.relocations = lg_.relocations();
  c.tail_entries = lg_.tail_entries();
  c.node_growths = lg_.node_growths();
  c.mirror_reuploads = 0;  // the host store keeps no device mirror
  c.candidates = candidates_;
  c.line_graph_builds = line_graph_builds_;
  return c;
}

}  // namespace dyng::detail::hyper
