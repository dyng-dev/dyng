// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (build, update)
/**
 * @file escher_store.cu
 * @brief The device store of the hypergraph container: the H2V and V2H CBSTs of M8a and the row
 *        view of H2V (ESCHER-GPU's DynamicHypergraph without its host model; ADR 0041).
 *
 * The device trees are authoritative. The host keeps only what validation and the id bound need:
 * one byte per id (absent, live, or a deleted slot still in the tree) and the live count. The
 * members of deleted ids are read back from the device (cbst_store::rows), never mirrored. Keys
 * and stored values are the store's (1-based: key = id + 1, a stored vertex is vertex + 1).
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/escher/integrity.hpp"
#include "hypergraph/escher/row_view.cuh"
#include "hypergraph/escher_store.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"
#include "util/thrust_allocator.cuh"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail::hyper {

// One row length per id (the key's length; keys 1..bound of the row view).
__global__ void h2v_row_lengths_kernel(const int* len, std::int64_t bound, std::int64_t* row_ptr) {
  const std::int64_t id = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (id < bound) row_ptr[id + 1] = len[id + 1];
}

// The members of every row, 0-based (the stored values are vertex + 1).
__global__ void gather_h2v_kernel(const int* flat, const int* loc, const int* len,
                                  std::int64_t bound, const std::int64_t* row_ptr,
                                  std::int32_t* col_ind) {
  const std::int64_t id = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (id >= bound) return;
  const int n = len[id + 1];
  const int base = loc[id + 1];
  const std::int64_t out = row_ptr[id];
  for (int i = 0; i < n; ++i) col_ind[out + i] = flat[base + i] - 1;
}

// One row length per vertex (keys 1..n of the sorted copy of V2H).
__global__ void v2h_row_lengths_kernel(const int* len, std::int64_t n, std::int64_t* row_ptr) {
  const std::int64_t v = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (v < n) row_ptr[v + 1] = len[v + 1];
}

// The ids of every vertex, 0-based (the stored values are id + 1), in the sorted copy's order.
__global__ void gather_v2h_kernel(const int* flat, const int* loc, const int* len, std::int64_t n,
                                  const std::int64_t* row_ptr, std::int64_t* col_ind) {
  const std::int64_t v = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (v >= n) return;
  const int count = len[v + 1];
  const int base = loc[v + 1];
  const std::int64_t out = row_ptr[v];
  for (int i = 0; i < count; ++i) col_ind[out + i] = static_cast<std::int64_t>(flat[base + i]) - 1;
}

DYNG_REGISTER_KERNEL(h2v_row_lengths_kernel);
DYNG_REGISTER_KERNEL(gather_h2v_kernel);
DYNG_REGISTER_KERNEL(v2h_row_lengths_kernel);
DYNG_REGISTER_KERNEL(gather_v2h_kernel);

namespace {

constexpr int block_size = 256;

cudaStream_t stream_of(const resources& res) {
  return static_cast<cudaStream_t>(res.stream().get());
}

int blocks_for(std::int64_t n) {
  return static_cast<int>((n + block_size - 1) / block_size);
}

// A copy of one value from the device, with its (counted) synchronization.
template <typename value_t>
value_t read_value(const resources& res, const value_t* device) {
  value_t out{};
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(&out, device, sizeof(value_t), cudaMemcpyDeviceToHost, stream_of(res)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res)));
  note_host_sync();
  return out;
}

// The construct layout of padding rule 1 (ESCHER-GPU's flatten2DVector): a row of c values takes
// ((c + 4) / 4) * 4 slots, the values (+1), zeros and INT_MIN in the last slot.
struct flat_layout {
  std::vector<int> keys, starts, counts, values;
};

template <typename row_fn>
flat_layout flatten(std::int64_t num_rows, row_fn&& row, const std::string& name,
                    std::int64_t limit) {
  flat_layout f;
  f.keys.resize(static_cast<std::size_t>(num_rows));
  f.starts.resize(static_cast<std::size_t>(num_rows));
  f.counts.resize(static_cast<std::size_t>(num_rows));
  std::int64_t total = 0;
  for (std::int64_t r = 0; r < num_rows; ++r) {
    const auto [begin, end] = row(r);
    const std::int64_t c = end - begin;
    f.keys[static_cast<std::size_t>(r)] = static_cast<int>(r + 1);
    f.starts[static_cast<std::size_t>(r)] =
        static_cast<int>(std::min<std::int64_t>(total, INT_MAX));
    f.counts[static_cast<std::size_t>(r)] = static_cast<int>(c);
    total += ((c + 4) / 4) * 4;
  }
  if (total > limit) {
    throw capacity_error("dyng: hypergraph: the " + name + " tree needs " + std::to_string(total) +
                         " payload entries, more than its capacity policy allows (" +
                         std::to_string(limit) + ")");
  }
  f.values.assign(static_cast<std::size_t>(total), 0);
  for (std::int64_t r = 0; r < num_rows; ++r) {
    const auto [begin, end] = row(r);
    std::size_t at = static_cast<std::size_t>(f.starts[static_cast<std::size_t>(r)]);
    for (auto it = begin; it != end; ++it) f.values[at++] = static_cast<int>(*it) + 1;
    const std::int64_t slots = ((end - begin + 4) / 4) * 4;
    f.values[static_cast<std::size_t>(f.starts[static_cast<std::size_t>(r)] + slots - 1)] = INT_MIN;
  }
  return f;
}

class escher_device_store final : public incidence_store {
 public:
  escher_device_store(const resources& res, const capacity_policy& capacity, std::int32_t n,
                      bool has_v2h)
      : res_(res),
        capacity_(capacity),
        policy_(to_store_policy(capacity)),
        num_vertices_(n),
        has_v2h_(has_v2h) {}

  // The trees over the rows (ESCHER-GPU's DynamicHypergraph::build: H2V keys = ids + 1, V2H keys
  // = vertices + 1, both with the construct layout; the caller's row counts as the tails'
  // occupancy, the scan's result).
  void build(const member_rows& rows) {
    const std::int64_t n = rows.size();
    flat_layout h2v, v2h;
    {
      scoped_stage stage(res_, "hypergraph.build");
      const auto row_of = [&](std::int64_t r) {
        return std::make_pair(rows.members.begin() + rows.offsets[static_cast<std::size_t>(r)],
                              rows.members.begin() + rows.offsets[static_cast<std::size_t>(r) + 1]);
      };
      h2v = flatten(n, row_of, "H2V", payload_limit(capacity_));
      if (has_v2h_) {
        std::vector<std::int64_t> ptr(static_cast<std::size_t>(num_vertices_) + 1, 0);
        for (const std::int32_t v : rows.members) ++ptr[static_cast<std::size_t>(v) + 1];
        for (std::size_t v = 0; v < static_cast<std::size_t>(num_vertices_); ++v)
          ptr[v + 1] += ptr[v];
        std::vector<std::int64_t> ids(rows.members.size());
        std::vector<std::int64_t> fill(ptr.begin(), ptr.end() - 1);
        for (std::int64_t r = 0; r < n; ++r) {
          for (std::int64_t k = rows.offsets[static_cast<std::size_t>(r)];
               k < rows.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
            ids[static_cast<std::size_t>(fill[static_cast<std::size_t>(rows.members[k])]++)] = r;
          }
        }
        const auto vertex_row = [&](std::int64_t v) {
          return std::make_pair(ids.begin() + ptr[static_cast<std::size_t>(v)],
                                ids.begin() + ptr[static_cast<std::size_t>(v) + 1]);
        };
        v2h = flatten(num_vertices_, vertex_row, "V2H", payload_limit(capacity_));
      }
    }
    scoped_stage stage(res_, "hypergraph.upload");
    h2v_ = make_tree("H2V", static_cast<std::int64_t>(h2v.values.size()));
    construct(*h2v_, h2v);
    if (has_v2h_) {
      v2h_ = make_tree("V2H", static_cast<std::int64_t>(v2h.values.size()));
      construct(*v2h_, v2h);
    }
    state_.assign(static_cast<std::size_t>(n), live_slot);
    live_ = n;
    bound_ = n;
    build_row_view();
  }

  // The trees from another backend's state: the H2V layout of the snapshot's keys and slot
  // lengths (dead slots emptied and erased after the construct), the bump pointer carried over;
  // V2H constructed from the transposed rows.
  void restore(const store_snapshot& s) {
    const allocator_state& a = s.h2v;
    flat_layout h2v;
    std::vector<std::int64_t> transposed_ptr, transposed_ids;
    {
      scoped_stage stage(res_, "hypergraph.build");
      std::int64_t total = 0;
      for (const std::int32_t len : a.lengths) total += len;
      h2v.keys = a.keys;
      h2v.starts.resize(a.keys.size());
      h2v.counts.resize(a.keys.size());
      h2v.values.assign(static_cast<std::size_t>(total), 0);
      std::int64_t at = 0;
      for (std::size_t i = 0; i < a.keys.size(); ++i) {
        h2v.starts[i] = static_cast<int>(at);
        const std::int64_t begin = s.rows.offsets[i], end = s.rows.offsets[i + 1];
        h2v.counts[i] = static_cast<int>(end - begin);
        for (std::int64_t k = begin; k < end; ++k)
          h2v.values[static_cast<std::size_t>(at + k - begin)] = s.rows.members[k] + 1;
        at += a.lengths[i];
        h2v.values[static_cast<std::size_t>(at - 1)] = INT_MIN;
      }
    }
    scoped_stage stage(res_, "hypergraph.upload");
    h2v_ = std::make_unique<escher::cbst_store>(res_, "H2V", static_cast<int>(a.capacity),
                                                policy_.alignment, policy_);
    construct(*h2v_, h2v);
    std::vector<int> dead_keys;
    for (std::size_t i = 0; i < a.keys.size(); ++i)
      if (a.dead[i] != 0) dead_keys.push_back(a.keys[i]);
    h2v_->erase(dead_keys);
    h2v_->set_used_payload(a.bump);
    bound_ = a.keys.empty() ? 0 : a.keys.back();
    state_.assign(static_cast<std::size_t>(bound_), absent_slot);
    live_ = 0;
    for (std::size_t i = 0; i < a.keys.size(); ++i) {
      state_[static_cast<std::size_t>(a.keys[i]) - 1] = a.dead[i] != 0 ? dead_slot : live_slot;
      live_ += a.dead[i] != 0 ? 0 : 1;
    }
    if (has_v2h_) {
      member_rows by_id;
      std::vector<std::int64_t> ids;
      for (std::size_t i = 0; i < a.keys.size(); ++i) {
        if (a.dead[i] != 0) continue;
        by_id.members.insert(by_id.members.end(), s.rows.members.begin() + s.rows.offsets[i],
                             s.rows.members.begin() + s.rows.offsets[i + 1]);
        by_id.offsets.push_back(static_cast<std::int64_t>(by_id.members.size()));
        ids.push_back(static_cast<std::int64_t>(a.keys[i]) - 1);
      }
      std::vector<std::int64_t> ptr(static_cast<std::size_t>(num_vertices_) + 1, 0);
      for (const std::int32_t v : by_id.members) ++ptr[static_cast<std::size_t>(v) + 1];
      for (std::size_t v = 0; v < static_cast<std::size_t>(num_vertices_); ++v)
        ptr[v + 1] += ptr[v];
      std::vector<std::int64_t> flat(by_id.members.size());
      std::vector<std::int64_t> fill(ptr.begin(), ptr.end() - 1);
      for (std::int64_t r = 0; r < by_id.size(); ++r) {
        for (std::int64_t k = by_id.offsets[static_cast<std::size_t>(r)];
             k < by_id.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
          flat[static_cast<std::size_t>(fill[static_cast<std::size_t>(by_id.members[k])]++)] =
              ids[static_cast<std::size_t>(r)];
        }
      }
      const auto vertex_row = [&](std::int64_t v) {
        return std::make_pair(flat.begin() + ptr[static_cast<std::size_t>(v)],
                              flat.begin() + ptr[static_cast<std::size_t>(v) + 1]);
      };
      flat_layout v2h = flatten(num_vertices_, vertex_row, "V2H", payload_limit(capacity_));
      v2h_ = make_tree("V2H", static_cast<std::int64_t>(v2h.values.size()));
      construct(*v2h_, v2h);
    }
    build_row_view();
  }

  [[nodiscard]] std::unique_ptr<incidence_store> clone(const resources& res) const override {
    auto copy = std::make_unique<escher_device_store>(res, capacity_, num_vertices_, has_v2h_);
    const scoped_device guard(res.device());
    copy->h2v_ = std::make_unique<escher::cbst_store>(h2v_->clone(res));
    if (v2h_) copy->v2h_ = std::make_unique<escher::cbst_store>(v2h_->clone(res));
    copy->state_ = state_;
    copy->live_ = live_;
    copy->bound_ = bound_;
    copy->build_row_view();
    return copy;
  }

  [[nodiscard]] store_snapshot snapshot() const override {
    const scoped_device guard(res_.device());
    store_snapshot out;
    out.num_vertices = num_vertices_;
    out.has_v2h = has_v2h_;
    const escher::cbst_host_state st = h2v_->download();
    // Nodes by key (the download gives them by heap position, with their avail).
    std::vector<std::size_t> by_key(st.nodes.size());
    for (std::size_t t = 0; t < st.nodes.size(); ++t) by_key[t] = t;
    std::sort(by_key.begin(), by_key.end(),
              [&](std::size_t x, std::size_t y) { return st.nodes[x].index < st.nodes[y].index; });
    std::vector<int> live_keys;
    allocator_state& a = out.h2v;
    for (const std::size_t t : by_key) {
      const auto& node = st.nodes[t];
      a.keys.push_back(node.index);
      a.dead.push_back(st.avail[t] != 0 ? 1 : 0);
      a.lengths.push_back(node.length);
      a.tail_capacity.push_back(node.tail_capacity);
      a.occupancy.push_back(node.occupancy);
      if (st.avail[t] == 0) live_keys.push_back(node.index);
    }
    const std::vector<std::vector<int>> rows = h2v_->rows(live_keys);
    std::size_t next_live = 0;
    for (std::size_t i = 0; i < a.keys.size(); ++i) {
      int count = 0;
      if (a.dead[i] == 0) {
        const auto& row = rows[next_live++];
        for (const int v : row) out.rows.members.push_back(v - 1);
        count = static_cast<int>(row.size());
      }
      a.counts.push_back(count);
      out.rows.offsets.push_back(static_cast<std::int64_t>(out.rows.members.size()));
    }
    a.bump = st.initial_payload_size;
    a.capacity = st.fixed_size;
    const escher::cbst_store_stats stats = h2v_->stats();
    a.growths = stats.growths;
    a.compactions = stats.compactions;
    return out;
  }

  [[nodiscard]] std::int64_t id_bound() const noexcept override {
    return bound_;
  }
  [[nodiscard]] std::int64_t num_live() const noexcept override {
    return live_;
  }
  [[nodiscard]] std::int32_t num_vertices() const noexcept override {
    return num_vertices_;
  }
  [[nodiscard]] bool is_live(std::int64_t id) const noexcept override {
    return id >= 0 && id < bound_ && state_[static_cast<std::size_t>(id)] == live_slot;
  }

  void read_rows(const std::vector<std::int64_t>& ids, member_rows& out) override {
    out.clear();
    if (ids.empty()) return;
    const scoped_device guard(res_.device());
    const std::vector<std::vector<int>> rows = h2v_->rows(to_keys(ids));
    for (const auto& row : rows) {
      for (const int v : row) out.members.push_back(v - 1);
      out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
    }
  }

  void erase(const std::vector<std::int64_t>& ids) override {
    if (ids.empty()) return;
    const scoped_device guard(res_.device());
    h2v_->erase(to_keys(ids));
    for (const std::int64_t id : ids) state_[static_cast<std::size_t>(id)] = dead_slot;
    live_ -= static_cast<std::int64_t>(ids.size());
  }

  void insert(const member_rows& rows, std::vector<std::int64_t>& ids) override {
    const std::int64_t n = rows.size();
    ids.assign(static_cast<std::size_t>(n), 0);
    if (n == 0) return;
    const scoped_device guard(res_.device());
    // The requested keys are ignored by the store (best fit decides); they are the keys after the
    // bound, as ESCHER-GPU requested for its appended rows.
    std::vector<int> requested(static_cast<std::size_t>(n)), payload, prefix;
    payload.reserve(rows.members.size());
    prefix.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
      requested[static_cast<std::size_t>(i)] = static_cast<int>(bound_ + i + 1);
      for (std::int64_t k = rows.offsets[static_cast<std::size_t>(i)];
           k < rows.offsets[static_cast<std::size_t>(i) + 1]; ++k)
        payload.push_back(rows.members[k] + 1);
      prefix.push_back(static_cast<int>(payload.size()));
    }
    std::int64_t last_live = bound_;
    while (last_live > 0 && state_[static_cast<std::size_t>(last_live - 1)] != live_slot)
      --last_live;
    const escher::insert_mapping mapping = h2v_->insert(requested, payload, prefix);
    // The mirror. The tree now holds the live keys, the items' keys and the deleted slots that
    // survived: none when the insert rebuilt the tree (a surplus item: the deleted slots it did
    // not reuse are dropped and the surplus keys follow the last live key, so a dead key can be
    // given out again), D - K when every item reused a slot. The record count (host-side, no
    // synchronization) tells the two apart.
    const std::int64_t surviving_dead =
        static_cast<std::int64_t>(h2v_->context().num_records) - (live_ + n);
    std::int64_t top = last_live;
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t key = mapping.item_to_key[static_cast<std::size_t>(i)];
      DYNG_EXPECTS(key >= 1, "CBST H2V: insert gave item ", i, " no key");
      top = std::max(top, key);
    }
    if (surviving_dead == 0) {
      for (auto& s : state_)
        if (s == dead_slot) s = absent_slot;
      bound_ = top;
      state_.resize(static_cast<std::size_t>(bound_), absent_slot);
    }
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t id = mapping.item_to_key[static_cast<std::size_t>(i)] - 1;
      state_[static_cast<std::size_t>(id)] = live_slot;
      ids[static_cast<std::size_t>(i)] = id;
    }
    live_ += n;
  }

  void v2h_remove(const grouped_incidences& g) override {
    if (!v2h_ || g.empty()) return;
    const scoped_device guard(res_.device());
    std::vector<int> keys, values;
    to_store_groups(g, keys, values);
    v2h_->remove_values(keys, values, g.prefix);
  }

  void v2h_append(const grouped_incidences& g) override {
    if (!v2h_ || g.empty()) return;
    const scoped_device guard(res_.device());
    std::vector<int> keys, values;
    to_store_groups(g, keys, values);
    v2h_->append(keys, values, g.prefix);
  }

  void finish_apply() override {
    const scoped_device guard(res_.device());
    // The read of the device error words (the stores' read points of an apply without one).
    h2v_->check_errors();
    if (v2h_) v2h_->check_errors();
    drop_views();
    scoped_stage stage(res_, "hypergraph.apply.views");
    build_row_view();
  }

  void check_integrity() const override {
    const scoped_device guard(res_.device());
    std::ostringstream log;
    long long errors = 0;
    // The rows of every live id by the row view (one pass over the nodes), as expected rows of
    // the search from the root (ESCHER's checkTreeRows) and as the source of the transpose.
    std::vector<std::vector<int>> h2v_rows(static_cast<std::size_t>(bound_));
    {
      array_view<const std::int64_t> row_ptr;
      array_view<const std::int32_t> col_ind;
      h2v_view(row_ptr, col_ind);
      std::vector<std::int64_t> ptr(row_ptr.size());
      std::vector<std::int32_t> cols(col_ind.size());
      if (!ptr.empty())
        DYNG_CUDA_TRY(cudaMemcpyAsync(ptr.data(), row_ptr.data(), ptr.size() * sizeof(ptr[0]),
                                      cudaMemcpyDeviceToHost, stream_of(res_)));
      if (!cols.empty())
        DYNG_CUDA_TRY(cudaMemcpyAsync(cols.data(), col_ind.data(), cols.size() * sizeof(cols[0]),
                                      cudaMemcpyDeviceToHost, stream_of(res_)));
      DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
      note_host_sync();
      for (std::int64_t id = 0; id < bound_; ++id) {
        auto& row = h2v_rows[static_cast<std::size_t>(id)];
        for (std::int64_t k = ptr[static_cast<std::size_t>(id)];
             k < ptr[static_cast<std::size_t>(id) + 1]; ++k) {
          row.push_back(cols[static_cast<std::size_t>(k)] + 1);
        }
        const bool alive = state_[static_cast<std::size_t>(id)] == live_slot;
        if (alive == row.empty()) {
          log << "[check] H2V: id " << id
              << (alive ? " is live with an empty row\n" : " is not live but has a row\n");
          ++errors;
        }
        for (std::size_t i = 0; i < row.size(); ++i) {
          if (row[i] < 1 || row[i] > num_vertices_ || (i > 0 && row[i] <= row[i - 1])) {
            log << "[check] H2V: the row of id " << id << " is not sorted, distinct and in range\n";
            ++errors;
            break;
          }
        }
      }
    }
    errors += escher::check_tree_rows(h2v_->context(), h2v_rows, false, "H2V", log);
    errors += escher::check_subtree_avail(h2v_->context(), "H2V", log);
    errors += escher::check_tree_index(h2v_->context(), "H2V", log);
    // The deleted slots: avail of every key against the host's states.
    {
      const escher::cbst_host_state st = h2v_->download();
      std::int64_t live = 0, top = 0;
      for (std::size_t t = 0; t < st.nodes.size(); ++t) {
        const std::int64_t key = st.nodes[t].index;
        top = std::max(top, key);
        const std::uint8_t want = st.avail[t] != 0 ? dead_slot : live_slot;
        if (key < 1 || key > bound_ || state_[static_cast<std::size_t>(key - 1)] != want) {
          log << "[check] H2V: key " << key << " is "
              << (st.avail[t] != 0 ? "a deleted slot" : "live")
              << " on the device, not on the host\n";
          ++errors;
        }
        live += st.avail[t] == 0 ? 1 : 0;
      }
      if (live != live_ || top != bound_) {
        log << "[check] H2V: " << live << " live keys up to " << top << " on the device, " << live_
            << " up to " << bound_ << " on the host\n";
        ++errors;
      }
    }
    if (v2h_) {
      // V2H must be the transpose of H2V (order-insensitive rows).
      std::vector<std::vector<int>> v2h_rows(static_cast<std::size_t>(num_vertices_));
      for (std::int64_t id = 0; id < bound_; ++id)
        for (const int v : h2v_rows[static_cast<std::size_t>(id)])
          v2h_rows[static_cast<std::size_t>(v) - 1].push_back(static_cast<int>(id + 1));
      errors += escher::check_tree_rows(v2h_->context(), v2h_rows, true, "V2H", log);
      errors += escher::check_subtree_avail(v2h_->context(), "V2H", log);
      errors += escher::check_tree_index(v2h_->context(), "V2H", log);
    }
    if (errors != 0)
      DYNG_FAIL("hypergraph integrity (device store): ", errors, " violations\n", log.str());
  }

  [[nodiscard]] storage_stats stats() const override {
    storage_stats st;
    const escher::cbst_store_stats h = h2v_->stats();
    st.device_bytes = h.device_bytes + static_cast<std::int64_t>(h2v_rows_.device_bytes());
    st.payload_capacity = h.payload_capacity;
    st.payload_used = h.payload_used;
    st.growths = h.growths;
    st.compactions = h.compactions;
    st.reuse = {h.bin_launches.reuse[0], h.bin_launches.reuse[1], h.bin_launches.reuse[2]};
    st.fill = {h.bin_launches.fill[0], h.bin_launches.fill[1], h.bin_launches.fill[2]};
    st.unfill = {h.bin_launches.unfill[0], h.bin_launches.unfill[1], h.bin_launches.unfill[2]};
    if (v2h_) {
      const escher::cbst_store_stats v = v2h_->stats();
      st.device_bytes += v.device_bytes;
      st.growths += v.growths;
      st.compactions += v.compactions;
      st.fill.thread += v.bin_launches.fill[0];
      st.fill.warp += v.bin_launches.fill[1];
      st.fill.block += v.bin_launches.fill[2];
      st.unfill.thread += v.bin_launches.unfill[0];
      st.unfill.warp += v.bin_launches.unfill[1];
      st.unfill.block += v.bin_launches.unfill[2];
    }
    if (h2v_built_.load(std::memory_order_acquire))
      st.view_bytes += static_cast<std::int64_t>(h2v_row_ptr_.size() * sizeof(std::int64_t) +
                                                 h2v_col_ind_.size() * sizeof(std::int32_t));
    if (v2h_built_.load(std::memory_order_acquire))
      st.view_bytes += static_cast<std::int64_t>((v2h_row_ptr_.size() + v2h_col_ind_.size()) *
                                                 sizeof(std::int64_t));
    return st;
  }

  void h2v_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int32_t>& col_ind) const override {
    if (!h2v_built_.load(std::memory_order_acquire)) {
      const std::lock_guard<std::mutex> lock(view_mutex_);
      if (!h2v_built_.load(std::memory_order_relaxed)) {
        const scoped_device guard(res_.device());
        const container_scope container;
        const escher::row_view& rv = h2v_rows_.view();
        buffer<std::int64_t> ptr(res_, static_cast<std::size_t>(bound_) + 1);
        DYNG_CUDA_TRY(cudaMemsetAsync(ptr.data(), 0, sizeof(std::int64_t), stream_of(res_)));
        std::int64_t total = 0;
        if (bound_ > 0) {
          h2v_row_lengths_kernel<<<blocks_for(bound_), block_size, 0, stream_of(res_)>>>(
              rv.len, bound_, ptr.data());
          DYNG_CHECK_KERNEL(stream_of(res_));
          thrust_allocator alloc(res_);
          thrust::inclusive_scan(thrust_policy(res_, alloc),
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + 1,
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + bound_ + 1,
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + 1);
          total = read_value(res_, ptr.data() + bound_);
        }
        buffer<std::int32_t> cols(res_, static_cast<std::size_t>(std::max<std::int64_t>(total, 0)));
        if (total > 0) {
          gather_h2v_kernel<<<blocks_for(bound_), block_size, 0, stream_of(res_)>>>(
              rv.flat, rv.loc, rv.len, bound_, ptr.data(), cols.data());
          DYNG_CHECK_KERNEL(stream_of(res_));
        }
        // Complete before any reader on another stream (read-only calls may run concurrently).
        DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
        note_host_sync();
        h2v_row_ptr_ = std::move(ptr);
        h2v_col_ind_ = std::move(cols);
        h2v_built_.store(true, std::memory_order_release);
      }
    }
    row_ptr = array_view<const std::int64_t>(h2v_row_ptr_.data(), h2v_row_ptr_.size(),
                                             memory_space::device, res_.device());
    col_ind = array_view<const std::int32_t>(h2v_col_ind_.data(), h2v_col_ind_.size(),
                                             memory_space::device, res_.device());
  }

  void v2h_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int64_t>& col_ind) const override {
    DYNG_EXPECTS(has_v2h_, "hypergraph::v2h: the hypergraph was built with store_v2h = false");
    if (!v2h_built_.load(std::memory_order_acquire)) {
      const std::lock_guard<std::mutex> lock(view_mutex_);
      if (!v2h_built_.load(std::memory_order_relaxed)) {
        const scoped_device guard(res_.device());
        const container_scope container;
        // ESCHER-GPU's buildSortedCopy: the chained, unsorted V2H rows sorted into a copy.
        escher::row_locator sorted;
        sorted.build_sorted_copy(v2h_->context(), num_vertices_);
        const escher::row_view& rv = sorted.view();
        const std::int64_t n = num_vertices_;
        buffer<std::int64_t> ptr(res_, static_cast<std::size_t>(n) + 1);
        DYNG_CUDA_TRY(cudaMemsetAsync(ptr.data(), 0, sizeof(std::int64_t), stream_of(res_)));
        std::int64_t total = 0;
        if (n > 0) {
          v2h_row_lengths_kernel<<<blocks_for(n), block_size, 0, stream_of(res_)>>>(rv.len, n,
                                                                                    ptr.data());
          DYNG_CHECK_KERNEL(stream_of(res_));
          thrust_allocator alloc(res_);
          thrust::inclusive_scan(thrust_policy(res_, alloc),
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + 1,
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + n + 1,
                                 thrust::device_ptr<std::int64_t>(ptr.data()) + 1);
          total = read_value(res_, ptr.data() + n);
        }
        buffer<std::int64_t> cols(res_, static_cast<std::size_t>(std::max<std::int64_t>(total, 0)));
        if (total > 0) {
          gather_v2h_kernel<<<blocks_for(n), block_size, 0, stream_of(res_)>>>(
              rv.flat, rv.loc, rv.len, n, ptr.data(), cols.data());
          DYNG_CHECK_KERNEL(stream_of(res_));
        }
        DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
        note_host_sync();
        v2h_row_ptr_ = std::move(ptr);
        v2h_col_ind_ = std::move(cols);
        v2h_built_.store(true, std::memory_order_release);
      }
    }
    row_ptr = array_view<const std::int64_t>(v2h_row_ptr_.data(), v2h_row_ptr_.size(),
                                             memory_space::device, res_.device());
    col_ind = array_view<const std::int64_t>(v2h_col_ind_.data(), v2h_col_ind_.size(),
                                             memory_space::device, res_.device());
  }

 private:
  static constexpr std::uint8_t absent_slot = 0, live_slot = 1, dead_slot = 2;

  std::unique_ptr<escher::cbst_store> make_tree(const char* name, std::int64_t size) const {
    const std::int64_t capacity = initial_payload(capacity_, size);
    return std::make_unique<escher::cbst_store>(res_, name, static_cast<int>(capacity),
                                                policy_.alignment, policy_);
  }

  static void construct(escher::cbst_store& tree, flat_layout& f) {
    tree.construct(f.keys.data(), f.starts.data(), static_cast<int>(f.keys.size()), f.values.data(),
                   static_cast<int>(f.values.size()), f.counts.data());
  }

  // The row view of H2V for the keys 1..bound (A's locator: one pass over the nodes).
  void build_row_view() {
    h2v_rows_.build(h2v_->context(), static_cast<int>(bound_), false);
  }

  void drop_views() noexcept {
    h2v_built_.store(false, std::memory_order_release);
    v2h_built_.store(false, std::memory_order_release);
    h2v_row_ptr_ = buffer<std::int64_t>();
    h2v_col_ind_ = buffer<std::int32_t>();
    v2h_row_ptr_ = buffer<std::int64_t>();
    v2h_col_ind_ = buffer<std::int64_t>();
  }

  static std::vector<int> to_keys(const std::vector<std::int64_t>& ids) {
    std::vector<int> keys(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) keys[i] = static_cast<int>(ids[i] + 1);
    return keys;
  }

  static void to_store_groups(const grouped_incidences& g, std::vector<int>& keys,
                              std::vector<int>& values) {
    keys.resize(g.vertices.size());
    for (std::size_t i = 0; i < g.vertices.size(); ++i) keys[i] = g.vertices[i] + 1;
    values.resize(g.ids.size());
    for (std::size_t i = 0; i < g.ids.size(); ++i) values[i] = static_cast<int>(g.ids[i] + 1);
  }

  resources res_;
  capacity_policy capacity_;
  escher::store_policy policy_;
  std::int32_t num_vertices_ = 0;
  bool has_v2h_ = true;
  std::unique_ptr<escher::cbst_store> h2v_, v2h_;
  escher::row_locator h2v_rows_;
  std::vector<std::uint8_t> state_;  // by id: absent, live or a deleted slot in the tree
  std::int64_t live_ = 0, bound_ = 0;

  mutable std::mutex view_mutex_;
  mutable std::atomic<bool> h2v_built_{false}, v2h_built_{false};
  mutable buffer<std::int64_t> h2v_row_ptr_, v2h_row_ptr_, v2h_col_ind_;
  mutable buffer<std::int32_t> h2v_col_ind_;
};

}  // namespace

std::unique_ptr<incidence_store> make_escher_store(const resources& res, std::int32_t num_vertices,
                                                   const member_rows& rows,
                                                   const capacity_policy& capacity, bool has_v2h) {
  DYNG_EXPECTS(res.get_backend() == backend::cuda, "the ESCHER device store needs CUDA resources");
  const scoped_device guard(res.device());
  auto store = std::make_unique<escher_device_store>(res, capacity, num_vertices, has_v2h);
  store->build(rows);
  return store;
}

std::unique_ptr<incidence_store> make_escher_store(const resources& res,
                                                   const store_snapshot& snapshot,
                                                   const capacity_policy& capacity) {
  DYNG_EXPECTS(res.get_backend() == backend::cuda, "the ESCHER device store needs CUDA resources");
  const scoped_device guard(res.device());
  auto store =
      std::make_unique<escher_device_store>(res, capacity, snapshot.num_vertices, snapshot.has_v2h);
  store->restore(snapshot);
  return store;
}

}  // namespace dyng::detail::hyper
