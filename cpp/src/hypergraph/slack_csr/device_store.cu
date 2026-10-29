// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu (applyBatch: the steps of a batch on
// the device, the CSR rebuild after a tail overflow)
/**
 * @file device_store.cu
 * @brief The store of slack_csr storage on the CUDA backend (ADR 0089): the host incidence model
 *        and id allocator of the slack_csr host store stay authoritative (MOSP_ESCHER keeps its
 *        HostHypergraph), and the device holds MOSP_ESCHER's incidence mirror, slack CSR line
 *        graph and node weights (slack_csr_store.hpp), kept from every batch by its order-free
 *        pair delta (derive_delta) and its warp-per-row CSR apply (apply_delta_pairs).
 *
 * The container's commit runs MOSP_ESCHER's applyBatch steps through the store interface: the
 * host phases (ids, H2V and v2h rows) on the host model, then line_graph_delta_of() ships the
 * batch's incidence changes, the touched rows before and after it and the touched vertices'
 * lengths (MOSP_ESCHER's IncidenceBatch) and derives the pairs on the device, then
 * finish_line_graph() applies them to the CSR (a tail overflow rebuilds the CSR from the host
 * line graph inside the commit) and scatters the new weights. The pairs stay on the device
 * (hyper_sssp reads them there); the applied record copies them on request.
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/slack_csr/device_store.hpp"
#include "hypergraph/slack_csr/host_store.hpp"
#include "hypergraph/slack_csr/slack_csr_store.hpp"
#include "hypergraph/slack_csr/slack_line_graph.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/config.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::detail::hyper {

// The degree of every dead id is 0 (MOSP_ESCHER's hsospUpdate zeroed it on the engine's side;
// here it is the container's CSR apply, so the engine never writes container state).
__global__ void zero_dead_degrees_kernel(const int* dead, int n, int* deg) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) deg[dead[i]] = 0;
}
DYNG_REGISTER_KERNEL(zero_dead_degrees_kernel);

namespace {

constexpr int block_size = 256;

cudaStream_t stream_of(const resources& res) {
  return static_cast<cudaStream_t>(res.stream().get());
}

// Makes the stream of `waiter` wait for the work queued so far on the stream of `signaller` (an
// event; nothing when both name the same stream on the same device): the store works on the stream
// of the resources it was built with, an apply joins its own resources' stream at the start and the
// end of the store phase (as the escher store, escher_store.cu).
void order_streams(const resources& signaller, const resources& waiter) {
  if (signaller.stream().get() == waiter.stream().get() && signaller.device() == waiter.device())
    return;
  cudaEvent_t event = nullptr;
  {
    const scoped_device guard(signaller.device());
    DYNG_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    const cudaError_t recorded = cudaEventRecord(event, stream_of(signaller));
    if (recorded != cudaSuccess) {
      (void)cudaEventDestroy(event);
      DYNG_CUDA_TRY(recorded);
    }
  }
  const scoped_device guard(waiter.device());
  const cudaError_t waited = cudaStreamWaitEvent(stream_of(waiter), event, 0);
  (void)cudaEventDestroy(event);
  DYNG_CUDA_TRY(waited);
}

class slack_device_store final : public incidence_store, public slack_access {
 public:
  slack_device_store(const resources& res, std::unique_ptr<slack_host_store> host,
                     const slack::knobs& knobs)
      : res_(res),
        host_(std::move(host)),
        knobs_(knobs),
        threads_(resources_access::host_threads(res)) {
    // Every operation of the device graph on the store's stream.
    dev_.stream = stream_of(res_);
    dev_.inc.stream = stream_of(res_);
    // Every array of the device graph from the store's resources (buffers).
    dev_.res = &res_;
    dev_.inc.res = &res_;
    dev_.inc.max_entries = knobs_.max_entries;
  }

  [[nodiscard]] std::unique_ptr<incidence_store> clone(const resources& res) const override {
    const scoped_device guard(res.device());
    auto copy = std::make_unique<slack_device_store>(
        res, std::make_unique<slack_host_store>(*host_), knobs_);
    copy->host_->set_host_threads(resources_access::host_threads(res));
    if (line_graph_.load(std::memory_order_acquire)) copy->enable_line_graph();
    return copy;
  }

  [[nodiscard]] store_snapshot snapshot() const override {
    store_snapshot out = host_->snapshot();
    out.line_graph = line_graph_.load(std::memory_order_acquire);
    return out;
  }

  void set_host_threads(int threads) noexcept override {
    threads_ = threads > 0 ? threads : 1;
    host_->set_host_threads(threads);
  }

  void wait_for(const resources& res) override {
    order_streams(res, res_);
  }

  void signal_to(const resources& res) override {
    order_streams(res_, res);
  }

  [[nodiscard]] std::int64_t id_bound() const noexcept override {
    return host_->id_bound();
  }
  [[nodiscard]] std::int64_t num_live() const noexcept override {
    return host_->num_live();
  }
  [[nodiscard]] std::int32_t num_vertices() const noexcept override {
    return host_->num_vertices();
  }
  [[nodiscard]] bool is_live(std::int64_t id) const noexcept override {
    return host_->is_live(id);
  }
  void read_rows(const std::vector<std::int64_t>& ids, member_rows& out) override {
    host_->read_rows(ids, out);
  }
  void erase(const std::vector<std::int64_t>& ids) override {
    host_->erase(ids);
  }
  void insert(const member_rows& rows, std::vector<std::int64_t>& ids) override {
    host_->insert(rows, ids);
  }
  void v2h_remove(const grouped_incidences& g) override {
    host_->v2h_remove(g);
  }
  void v2h_append(const grouped_incidences& g) override {
    host_->v2h_append(g);
  }
  void h2v_remove(const grouped_members& g) override {
    host_->h2v_remove(g);
  }
  void h2v_append(const grouped_members& g) override {
    host_->h2v_append(g);
  }
  void update_line_graph(const std::vector<std::int64_t>& /*deleted*/,
                         const std::vector<std::int64_t>& /*inserted*/,
                         const member_rows& /*inserted_rows*/) override {
    DYNG_FAIL("hypergraph: the sorted merge of escher storage reached a slack_csr store");
  }

  [[nodiscard]] bool supports_incidence_edits() const noexcept override {
    return true;
  }

  void begin_line_graph_delta(const std::vector<std::int32_t>& /*vertices*/,
                              const std::vector<std::int64_t>& /*ids*/) override {
    // The state before the batch is read on the device (the mirror, pass 0) and from the record
    // (the touched rows before it): nothing to keep here.
  }

  void line_graph_delta_of(const std::vector<std::int32_t>& vertices,
                           const std::vector<std::int64_t>& ids,
                           const std::vector<std::int8_t>& signs,
                           const std::vector<std::int64_t>& inserted_ids,
                           const touched_rows& touched, line_graph_delta& out) override {
    out.clear();
    if (!line_graph_.load(std::memory_order_acquire)) return;
    const scoped_device guard(res_.device());
    const std::int64_t bound = host_->max_bound();
    if (bound > dev_.max_nodes) grow_nodes(bound);  // dynG (MOSP_ESCHER throws)
    // MOSP_ESCHER's IncidenceBatch (hyperedge ids 1-based).
    slack::incidence_batch& inc = inc_;
    const std::size_t n = vertices.size();
    inc.inc_key.resize(n);
    inc.inc_sign.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      inc.inc_key[i] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertices[i])) << 32) |
                       static_cast<std::uint32_t>(ids[i] + 1);
      inc.inc_sign[i] = signs[i];
    }
    const auto& touched_ids = *touched.ids;
    inc.touched.resize(touched_ids.size());
    for (std::size_t t = 0; t < touched_ids.size(); ++t)
      inc.touched[t] = static_cast<int>(touched_ids[t] + 1);
    const auto ship_rows = [](const member_rows& rows, std::vector<int>& off,
                              std::vector<int>& vals) {
      if (rows.members.size() > static_cast<std::size_t>(INT_MAX)) {
        throw capacity_error("dyng: hypergraph: the touched rows of the batch hold " +
                             std::to_string(rows.members.size()) +
                             " members, more than the int32 offsets of the device delta");
      }
      off.assign(rows.offsets.begin(), rows.offsets.end());
      vals.assign(rows.members.begin(), rows.members.end());
    };
    ship_rows(*touched.pre, inc.pre_off, inc.pre_vals);
    ship_rows(*touched.post, inc.post_off, inc.post_vals);
    inc.touched_vertices.assign(touched.vertices->begin(), touched.vertices->end());
    inc.touched_vertex_len.assign(touched.vertex_lengths->begin(), touched.vertex_lengths->end());
    inc.new_he.resize(inserted_ids.size());
    for (std::size_t i = 0; i < inserted_ids.size(); ++i)
      inc.new_he[i] = static_cast<int>(inserted_ids[i] + 1);
    inc.new_w.assign(touched.new_weights->begin(), touched.new_weights->end());
    inc.dead_he.resize(touched.dead_ids->size());
    for (std::size_t i = 0; i < touched.dead_ids->size(); ++i)
      inc.dead_he[i] = static_cast<int>((*touched.dead_ids)[i] + 1);
    int reuploads = 0;
    const long long candidates =
        slack::derive_delta(dev_, host_model(), inc, reuploads, knobs_.max_candidates);
    mirror_reuploads_ += reuploads;
    candidates_ = candidates;
    out.candidates = candidates;
  }

  [[nodiscard]] bool pairs_on_device() const noexcept override {
    return line_graph_.load(std::memory_order_acquire);
  }

  void download_line_graph_delta(line_graph_delta& out) const override {
    const std::int64_t candidates = out.candidates;
    out.clear();
    out.candidates = candidates;
    if (!line_graph_.load(std::memory_order_acquire)) return;
    const scoped_device guard(res_.device());
    const slack::device_delta& dd = dev_.delta;
    const std::size_t total = static_cast<std::size_t>(dd.num_del) + dd.num_ins;
    std::vector<std::int32_t> pairs(2 * total);
    if (total > 0) {
      DYNG_CUDA_TRY(cudaMemcpyAsync(pairs.data(), dd.d_pairs.data(), sizeof(int2) * total,
                                    cudaMemcpyDeviceToHost, stream_of(res_)));
      DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
      note_host_sync();
    }
    for (std::size_t i = 0; i < total; ++i) {
      const bool deleted = i < static_cast<std::size_t>(dd.num_del);
      (deleted ? out.deleted_a : out.inserted_a).push_back(pairs[2 * i]);
      (deleted ? out.deleted_b : out.inserted_b).push_back(pairs[2 * i + 1]);
    }
  }

  void finish_line_graph(const line_graph_delta& delta, const std::vector<std::int64_t>& new_ids,
                         const std::vector<std::int64_t>& new_weights) override {
    // The weights by id of the host model (its line graph is not kept: the device has it).
    host_->finish_line_graph(delta, new_ids, new_weights);
    if (!line_graph_.load(std::memory_order_acquire)) return;
    const scoped_device guard(res_.device());
    slack::device_delta& dd = dev_.delta;
    // applyBatch's step 3: the device CSR, rebuilt from the host incidence if its tail overflows
    // (MOSP_ESCHER's S5; the delta stays on the device).
    dev_.num_nodes = static_cast<int>(host_->max_bound());
    dev_.num_entries += 2LL * (dd.num_ins - dd.num_del);
    const long long used_before = dev_.used_entries;
    if (!slack::apply_delta_pairs(dev_)) {
      const scoped_stage stage(res_, "hypergraph.apply.line_graph.rebuild");
      slack_line_graph lg;
      host_line_graph(lg);
      check_entries(lg, knobs_.entry_headroom);
      slack::build_csr(dev_, host_model(), csr_of(lg), dev_.max_nodes, knobs_.entry_headroom);
      ++csr_rebuilds_;
    } else {
      tail_entries_ += dev_.used_entries - used_before;
    }
    if (dd.num_dead > 0) {
      zero_dead_degrees_kernel<<<slack::grid_for(dd.num_dead, block_size), block_size, 0,
                                 stream_of(res_)>>>(dd.d_ids.data() + dd.num_new, dd.num_dead,
                                                    dev_.d_deg.data());
      DYNG_CUDA_TRY(cudaGetLastError());
    }
#if defined(DYNG_MUTATION_HSOSP_PAIR_COUNT_ADDS_DELETIONS)
    // MUTATION (PLAN 8.4; M11a criterion 16 #7, the CUDA store): the pair count adds deletions.
    pair_count_ += dd.num_ins + dd.num_del;
#else
    pair_count_ += dd.num_ins - dd.num_del;  // MOSP_ESCHER's h2hPairCount
#endif
  }

  void set_node_weights(const std::vector<std::int64_t>& by_id) override {
    host_->set_node_weights(by_id);
    if (!line_graph_.load(std::memory_order_acquire)) return;
    const scoped_device guard(res_.device());
    upload_node_weights();
  }

  void finish_apply() override {
    host_->finish_apply();  // the host model's views of the new state are built on request
    drop_views();
  }

  [[nodiscard]] bool line_graph_enabled() const noexcept override {
    return line_graph_.load(std::memory_order_acquire);
  }

  void enable_line_graph() const override {
    if (line_graph_.load(std::memory_order_acquire)) return;
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (line_graph_.load(std::memory_order_relaxed)) return;
    const container_scope container;  // the container's own work (budgets, I9)
    const scoped_device guard(res_.device());
    slack_line_graph lg;
    {
      const scoped_stage stage(res_, "hypergraph.build");
      host_line_graph(lg);
    }
    const scoped_stage stage(res_, "hypergraph.upload");
    const std::int64_t nodes = planned_node_capacity(host_->max_bound());
    // MOSP_ESCHER's buildDeviceH2H: the CSR and the incidence mirror, both with the entry
    // headroom.
    check_entries(lg, knobs_.entry_headroom);
    slack::build_device_h2h(dev_, host_model(), csr_of(lg), static_cast<int>(nodes),
                            knobs_.entry_headroom);
    pair_count_ = dev_.num_entries / 2;
    ++line_graph_builds_;
    line_graph_.store(true, std::memory_order_release);
  }

  [[nodiscard]] row_views views() const override {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!row_views_built_) {
      const container_scope container;
      const scoped_device guard(res_.device());
      // h2v by key (the host model's layout, A's h2vRows), uploaded.
      const row_views host = host_->views();
      const auto bound = static_cast<std::size_t>(host.h2v.max_key) + 1;
      std::size_t flat = 0;
      for (std::size_t k = 0; k < bound; ++k)
        flat = std::max<std::size_t>(flat, static_cast<std::size_t>(host.h2v.loc[k]) +
                                               static_cast<std::size_t>(host.h2v.len[k]));
      upload_ints(host.h2v.flat, std::max<std::size_t>(flat, 1), rv_h2v_flat_);
      upload_ints(host.h2v.loc, bound, rv_h2v_loc_);
      upload_ints(host.h2v.len, bound, rv_h2v_len_);
      h2v_max_key_ = host.h2v.max_key;
      if (line_graph_.load(std::memory_order_acquire)) {
        // h2h by key from the device CSR, sorted (ESCHER-GPU's layout: neighbour keys).
        const std::int64_t keys = dev_.num_nodes;
        const std::vector<std::vector<int>> sorted =
            slack::download_rows(dev_, static_cast<int>(keys));
        std::vector<int> f, loc(static_cast<std::size_t>(keys) + 1, 0),
            len(static_cast<std::size_t>(keys) + 1, 0), out(static_cast<std::size_t>(keys) + 1, 0);
        for (std::int64_t r = 0; r < keys; ++r) {
          const auto k = static_cast<std::size_t>(r) + 1;
          const auto& row = sorted[static_cast<std::size_t>(r)];  // 1-based ids = keys
          loc[k] = static_cast<int>(f.size());
          len[k] = static_cast<int>(row.size());
          out[k] = loc[k] +
                   static_cast<int>(std::upper_bound(row.begin(), row.end(), static_cast<int>(k)) -
                                    row.begin());
          f.insert(f.end(), row.begin(), row.end());
        }
        if (f.empty()) f.push_back(0);
        upload_ints(f.data(), f.size(), rv_h2h_flat_);
        upload_ints(loc.data(), loc.size(), rv_h2h_loc_);
        upload_ints(len.data(), len.size(), rv_h2h_len_);
        upload_ints(out.data(), out.size(), rv_h2h_out_);
        h2h_max_key_ = static_cast<int>(keys);
      }
      finish_views();
      ++builds_.row_views;
      row_views_built_ = true;
    }
    row_views v;
    v.space = memory_space::device;
    v.device = res_.device();
    v.id_bound = host_->id_bound();
    v.num_vertices = host_->num_vertices();
    v.h2v.flat = rv_h2v_flat_.data();
    v.h2v.loc = rv_h2v_loc_.data();
    v.h2v.len = rv_h2v_len_.data();
    v.h2v.out = nullptr;
    v.h2v.max_key = h2v_max_key_;
    if (line_graph_.load(std::memory_order_acquire)) {
      v.has_h2h = true;
      v.h2h.flat = rv_h2h_flat_.data();
      v.h2h.loc = rv_h2h_loc_.data();
      v.h2h.len = rv_h2h_len_.data();
      v.h2h.out = rv_h2h_out_.data();
      v.h2h.max_key = h2h_max_key_;
    }
    return v;
  }

  void h2h_view(array_view<const std::int64_t>& row_ptr, array_view<const std::int64_t>& col_ind,
                array_view<const std::int64_t>& out) const override {
    DYNG_EXPECTS(line_graph_.load(std::memory_order_acquire),
                 "hypergraph::h2h: the line graph is not maintained");
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2h_built_) {
      const container_scope container;
      const scoped_device guard(res_.device());
      // The slack rows decoded to sorted, contiguous rows (HD9: a sorted view on request).
      const std::int64_t rows = host_->id_bound();
      const std::vector<std::vector<int>> sorted = slack::download_rows(
          dev_, static_cast<int>(std::min<std::int64_t>(rows, dev_.num_nodes)));
      std::vector<std::int64_t> ptr(static_cast<std::size_t>(rows) + 1, 0), cols,
          first(static_cast<std::size_t>(rows), 0);
      for (std::int64_t r = 0; r < rows; ++r) {
        const auto i = static_cast<std::size_t>(r);
        if (i < sorted.size()) {
          for (const int key : sorted[i]) cols.push_back(static_cast<std::int64_t>(key) - 1);
        }
        ptr[i + 1] = static_cast<std::int64_t>(cols.size());
        const auto b = cols.begin() + ptr[i];
        first[i] = ptr[i] + (std::upper_bound(b, cols.end(), r) - b);
      }
      upload(ptr, h2h_row_ptr_);
      upload(cols, h2h_col_ind_);
      upload(first, h2h_out_);
      finish_views();
      ++builds_.h2h;
      h2h_built_ = true;
    }
    row_ptr = h2h_row_ptr_.view();
    col_ind = h2h_col_ind_.view();
    out = h2h_out_.view();
  }

  [[nodiscard]] line_graph_layout line_graph() const override {
    // The rows in ESCHER-GPU's layout (keys 1..nodes, the neighbours' keys sorted, padding rule
    // 1): what the tests compare across storages.
    line_graph_layout g;
    if (!line_graph_.load(std::memory_order_acquire)) return g;
    const scoped_device guard(res_.device());
    const std::int64_t keys = host_->max_bound();
    const std::vector<std::vector<int>> sorted = slack::download_rows(dev_, static_cast<int>(keys));
    g.starts.resize(static_cast<std::size_t>(keys));
    g.counts.resize(static_cast<std::size_t>(keys));
    std::int64_t at = 0;
    for (std::int64_t k = 0; k < keys; ++k) {
      const auto c = static_cast<std::int64_t>(sorted[static_cast<std::size_t>(k)].size());
      g.starts[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(at);
      g.counts[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(c);
      at += ((c + 4) / 4) * 4;
    }
    g.payload.assign(static_cast<std::size_t>(at), 0);
    for (std::int64_t k = 0; k < keys; ++k) {
      const auto base = static_cast<std::size_t>(g.starts[static_cast<std::size_t>(k)]);
      const auto& row = sorted[static_cast<std::size_t>(k)];
      std::copy(row.begin(), row.end(), g.payload.begin() + static_cast<std::ptrdiff_t>(base));
      const auto c = static_cast<std::int64_t>(row.size());
      g.payload[base + static_cast<std::size_t>(((c + 4) / 4) * 4) - 1] = INT_MIN;
    }
    return g;
  }

  [[nodiscard]] view_builds builds() const noexcept override {
    view_builds b = builds_;
    b.line_graph_builds = line_graph_builds_;
    return b;
  }

  void check_integrity() const override {
    host_->check_integrity();
    if (!line_graph_.load(std::memory_order_acquire)) return;
    const scoped_device guard(res_.device());
    std::vector<std::string> problems;
    const auto hg = host_model();
    // The incidence mirror against the host model (MOSP_ESCHER's incidenceMirrorMismatches).
    const long long bad = slack::incidence_mirror_mismatches(dev_.inc, hg);
    if (bad != 0)
      problems.push_back("the incidence mirror differs from the host model in " +
                         std::to_string(bad) + " rows");
    // The slack CSR against the line graph of the incidence lists, as multisets.
    const std::int64_t nodes = host_->max_bound();
    if (dev_.num_nodes != nodes)
      problems.push_back("the device line graph has " + std::to_string(dev_.num_nodes) +
                         " nodes, the host model " + std::to_string(nodes));
    slack_line_graph lg;
    host_line_graph(lg);
    const std::vector<std::vector<int>> got = slack::download_rows(dev_, dev_.num_nodes);
    std::int64_t entries = 0;
    for (std::int64_t r = 0; r < nodes && r < dev_.num_nodes && problems.size() < 20; ++r) {
      const auto& row = got[static_cast<std::size_t>(r)];
      entries += static_cast<std::int64_t>(row.size());
      std::vector<int> want(static_cast<std::size_t>(lg.deg()[r]));
      for (std::size_t e = 0; e < want.size(); ++e)
        want[e] = lg.col_ind()[lg.row_start()[r] + static_cast<std::int64_t>(e)] + 1;
      if (row != want)
        problems.push_back("the device row of node " + std::to_string(r) + " has " +
                           std::to_string(row.size()) + " entries, the incidence lists give " +
                           std::to_string(want.size()));
    }
    if (entries != dev_.num_entries)
      problems.push_back("the device line graph counts " + std::to_string(dev_.num_entries) +
                         " entries, its rows hold " + std::to_string(entries));
    if (pair_count_ * 2 != entries)
      problems.push_back("the maintained pair count " + std::to_string(pair_count_) +
                         " is not the line graph's " + std::to_string(entries / 2));
    // The capacity invariants: every degree within its slot, the slots below the tail cursor,
    // the tail within the entries, no row above the nodes; the mirror's tails.
    std::vector<long long> start(static_cast<std::size_t>(dev_.max_nodes));
    std::vector<int> deg(static_cast<std::size_t>(dev_.max_nodes)),
        cap(static_cast<std::size_t>(dev_.max_nodes));
    std::vector<long long> weight(static_cast<std::size_t>(dev_.max_nodes));
    download(dev_.d_row_start.data(), start);
    download(dev_.d_deg.data(), deg);
    download(dev_.d_cap.data(), cap);
    download(dev_.d_node_w.data(), weight);
    for (std::size_t r = 0; r < deg.size() && problems.size() < 20; ++r) {
      if (deg[r] < 0 || deg[r] > cap[r] || (deg[r] > 0 && start[r] + cap[r] > dev_.used_entries) ||
          (static_cast<std::int64_t>(r) >= dev_.num_nodes && deg[r] != 0))
        problems.push_back("the device row of node " + std::to_string(r) + " overruns its slot");
    }
    if (dev_.used_entries > dev_.cap_entries)
      problems.push_back("the device line graph's tail overruns its entries");
    if (dev_.inc.he_tail > dev_.inc.he_capacity || dev_.inc.v_tail > dev_.inc.v_capacity)
      problems.push_back("a tail of the incidence mirror overruns its capacity");
    // The node weights of the live ids.
    const auto& w = host_->node_weights();
    for (std::size_t id = 0; id < w.size() && id < weight.size(); ++id) {
      if (host_->is_live(static_cast<std::int64_t>(id)) && weight[id] != w[id]) {
        problems.push_back("the device weight of id " + std::to_string(id) + " is stale");
        break;
      }
    }
    if (!problems.empty()) {
      std::string all;
      for (const auto& p : problems) all += "\n  " + p;
      DYNG_FAIL("hypergraph integrity (slack_csr device store):", all);
    }
  }

  [[nodiscard]] storage_stats stats() const override {
    storage_stats st = host_->stats();
    st.view_bytes = 0;
    if (line_graph_.load(std::memory_order_acquire)) {
      const slack::device_delta& dd = dev_.delta;
      // The CSR and the mirror (B's graph_mb scope), the delta's buffers and the delta's
      // scratch (documented in ADR 0089).
      std::size_t scratch = dd.d_pairs.bytes() + dd.d_ids.bytes() + dd.d_new_w.bytes() +
                            dd.d_keys.bytes() + dd.d_keys_alt.bytes() + dd.d_rows.bytes() +
                            dd.d_row_len.bytes() + dd.d_row_off.bytes() + dd.d_temp.bytes() +
                            dd.d_num_rows.size() * sizeof(int);
      if (const auto* sc = dev_.inc.scratch.get()) {
        scratch += sc->inc_keys.bytes() + sc->inc_keys_alt.bytes() + sc->signs.bytes() +
                   sc->signs_alt.bytes() + sc->uniq.bytes() + sc->net.bytes() + sc->cnt0.bytes() +
                   sc->cnt1.bytes() + sc->off0.bytes() + sc->off1.bytes() + sc->seg_len.bytes() +
                   sc->seg_off.bytes() + sc->seg_vert.bytes() + sc->cand.bytes() +
                   sc->cand_alt.bytes() + sc->cls.bytes() + sc->selected.bytes() +
                   sc->touched.bytes() + sc->pre_off.bytes() + sc->post_off.bytes() +
                   sc->he_write_off.bytes() + sc->pre_vals.bytes() + sc->post_vals.bytes() +
                   sc->t_verts.bytes() + sc->t_vert_len.bytes() + sc->reloc_verts.bytes() +
                   sc->reloc_off.bytes() + sc->temp.bytes() +
                   (sc->totals.size() + 0) * sizeof(long long) + sc->counts.size() * sizeof(int);
      }
      st.device_bytes = dev_.device_bytes() + 4 + static_cast<std::int64_t>(scratch);
    }
    const auto add = [&](std::size_t bytes) { st.view_bytes += static_cast<std::int64_t>(bytes); };
    if (h2v_built_) add((h2v_row_ptr_.size() * 8) + (h2v_col_ind_.size() * 4));
    if (v2h_built_) add((v2h_row_ptr_.size() + v2h_col_ind_.size()) * 8);
    if (h2h_built_) add((h2h_row_ptr_.size() + h2h_col_ind_.size() + h2h_out_.size()) * 8);
    if (row_views_built_)
      add((rv_h2v_flat_.size() + rv_h2v_loc_.size() + rv_h2v_len_.size() + rv_h2h_flat_.size() +
           rv_h2h_loc_.size() + rv_h2h_len_.size() + rv_h2h_out_.size()) *
          4);
    return st;
  }

  void h2v_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int32_t>& col_ind) const override {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!h2v_built_) {
      const container_scope container;
      const scoped_device guard(res_.device());
      array_view<const std::int64_t> p;
      array_view<const std::int32_t> c;
      host_->h2v_view(p, c);
      upload(std::vector<std::int64_t>(p.begin(), p.end()), h2v_row_ptr_);
      upload(std::vector<std::int32_t>(c.begin(), c.end()), h2v_col_ind_);
      finish_views();
      ++builds_.h2v;
      h2v_built_ = true;
    }
    row_ptr = h2v_row_ptr_.view();
    col_ind = h2v_col_ind_.view();
  }

  void v2h_view(array_view<const std::int64_t>& row_ptr,
                array_view<const std::int64_t>& col_ind) const override {
    const std::lock_guard<std::mutex> lock(view_mutex_);
    if (!v2h_built_) {
      const container_scope container;
      const scoped_device guard(res_.device());
      array_view<const std::int64_t> p;
      array_view<const std::int64_t> c;
      host_->v2h_view(p, c);
      upload(std::vector<std::int64_t>(p.begin(), p.end()), v2h_row_ptr_);
      upload(std::vector<std::int64_t>(c.begin(), c.end()), v2h_col_ind_);
      finish_views();
      ++builds_.v2h;
      v2h_built_ = true;
    }
    row_ptr = v2h_row_ptr_.view();
    col_ind = v2h_col_ind_.view();
  }

  [[nodiscard]] slack_access* slack() noexcept override {
    return this;
  }
  [[nodiscard]] const slack_access* slack() const noexcept override {
    return this;
  }

  // slack_access
  [[nodiscard]] slack_row_view rows() const override {
    DYNG_EXPECTS(line_graph_.load(std::memory_order_acquire),
                 "hypergraph: the slack_csr line graph is not maintained");
    slack_row_view v;
    v.space = memory_space::device;
    v.device = res_.device();
    v.num_nodes = dev_.num_nodes;
    v.node_capacity = dev_.max_nodes;
    v.num_entries = dev_.num_entries;
    v.used_entries = dev_.used_entries;
    v.cap_entries = dev_.cap_entries;
    v.pair_count = pair_count_;
    v.row_start = reinterpret_cast<const std::int64_t*>(dev_.d_row_start.data());
    v.deg = dev_.d_deg.data();
    v.cap = dev_.d_cap.data();
    v.col_ind = dev_.d_col_ind.data();
    v.node_weight = reinterpret_cast<const std::int64_t*>(dev_.d_node_w.data());
    v.mirror_he_capacity = dev_.inc.he_capacity;
    v.mirror_he_tail = dev_.inc.he_tail;
    v.mirror_v_capacity = dev_.inc.v_capacity;
    v.mirror_v_tail = dev_.inc.v_tail;
    return v;
  }

  [[nodiscard]] slack_delta_view last_delta() const noexcept override {
    slack_delta_view v;
    if (!line_graph_.load(std::memory_order_acquire)) return v;
    const slack::device_delta& dd = dev_.delta;
    v.space = memory_space::device;
    v.device = res_.device();
    v.pairs = reinterpret_cast<const std::int32_t*>(dd.d_pairs.data());
    v.ids = dd.d_ids.data();
    v.new_weights = reinterpret_cast<const std::int64_t*>(dd.d_new_w.data());
    v.num_deleted = dd.num_del;
    v.num_inserted = dd.num_ins;
    v.num_new = dd.num_new;
    v.num_dead = dd.num_dead;
    return v;
  }

  [[nodiscard]] std::int64_t graph_scope_bytes() const noexcept override {
    return line_graph_.load(std::memory_order_acquire) ? dev_.device_bytes() : 0;
  }

  [[nodiscard]] slack_counters counters() const noexcept override {
    slack_counters c;
    c.csr_rebuilds = csr_rebuilds_;
    c.tail_entries = tail_entries_;
    c.node_growths = node_growths_;
    c.mirror_reuploads = mirror_reuploads_;
    c.candidates = candidates_;
    c.line_graph_builds = line_graph_builds_;
    return c;
  }

  [[nodiscard]] slack::knobs knobs() const noexcept override {
    return knobs_;
  }

  void set_entry_headroom(double headroom) override {
    DYNG_EXPECTS(headroom >= 1.0,
                 "hypergraph: the slack_csr line graph's entry headroom must be at least 1, got ",
                 headroom);
    knobs_.entry_headroom = headroom;
  }

  void vertex_lengths(const std::vector<std::int32_t>& vertices,
                      std::vector<std::int32_t>& out) const override {
    host_->vertex_lengths(vertices, out);
  }

 private:
  // MOSP_ESCHER's HostHypergraph over the host model (rows by 0-based id, v2h, weights; the ids
  // up to the largest bound so far).
  [[nodiscard]] slack::host_hypergraph host_model() const {
    slack::host_hypergraph hg;
    hg.num_vertices = host_->num_vertices();
    hg.bound = static_cast<int>(host_->max_bound());
    hg.he_verts.rows = &host_->rows_by_id();
    hg.v2h.rows = &host_->v2h_by_vertex();
    hg.he_w.w = &host_->node_weights();
    hg.alive.rows = hg.he_verts;
    return hg;
  }

  // MOSP_ESCHER's HostHypergraph::lineGraph: the sorted rows of every id below the largest bound.
  void host_line_graph(slack_line_graph& lg) const {
    host_->build_line_graph_into(lg);
  }

  [[nodiscard]] static slack::line_graph_csr csr_of(const slack_line_graph& lg) {
    return slack::line_graph_csr{lg.row_start(), lg.deg(), lg.col_ind()};
  }

  // MOSP_ESCHER plans maxNodes = the rows + the planned inserts + 1024; the parity harness sets it
  // (knobs.node_capacity). Without it: the bound and room for an eighth more (at least 1024).
  [[nodiscard]] std::int64_t planned_node_capacity(std::int64_t needed) const {
    std::int64_t nodes = knobs_.node_capacity >= needed
                             ? knobs_.node_capacity
                             : needed + std::max<std::int64_t>(1024, needed / 8);
    const std::int64_t limit = std::min<std::int64_t>(knobs_.max_ids, INT_MAX);
    if (nodes > limit) nodes = std::max(needed, limit);
    if (nodes > limit) {
      throw capacity_error("dyng: hypergraph: " + std::to_string(needed) + " ids exceed " +
                           std::to_string(limit) +
                           " (the int32 node ids of the slack_csr line graph)");
    }
    return nodes;
  }

  // The CSR's entry capacity of a build or rebuild (MOSP_ESCHER's buildCsr: max(cursor, cursor x
  // headroom) + 4096) within the int32 counts, before anything is allocated.
  void check_entries(const slack_line_graph& lg, double headroom) const {
    const std::int64_t capacity = slack::entry_capacity(lg.used_entries(), headroom);
    if (capacity > knobs_.max_entries) {
      throw capacity_error("dyng: hypergraph: the slack_csr line graph needs " +
                           std::to_string(capacity) + " entries, more than " +
                           std::to_string(knobs_.max_entries) +
                           " (the int32 counts of its entries)");
    }
  }

  // The node arrays grown by half (at least to `needed`; dynG's capacity rule, ADR 0089: MOSP_ESCHER
  // throws "node capacity exceeded"): the CSR's row starts, degrees, slots and weights, the
  // mirror's hyperedge offsets and lengths and its touched index, kept; the new nodes have no row.
  void grow_nodes(std::int64_t needed) {
    const std::int64_t limit = std::min<std::int64_t>(knobs_.max_ids, INT_MAX);
    std::int64_t to = std::max<std::int64_t>(needed, dev_.max_nodes + dev_.max_nodes / 2);
    to = std::min(to, limit);
    if (needed > to) {
      throw capacity_error("dyng: hypergraph: " + std::to_string(needed) + " ids exceed " +
                           std::to_string(limit) +
                           " (the int32 node ids of the slack_csr line graph)");
    }
    const auto old_n = static_cast<std::size_t>(dev_.max_nodes);
    const auto new_n = static_cast<std::size_t>(to);
    const auto grow = [&](auto& b, int fill) {
      using value_t = typename std::remove_reference_t<decltype(b)>::value_type;
      buffer<value_t> next(res_, new_n);
      if (old_n > 0)
        DYNG_CUDA_TRY(cudaMemcpyAsync(next.data(), b.data(), old_n * sizeof(value_t),
                                      cudaMemcpyDeviceToDevice, stream_of(res_)));
      DYNG_CUDA_TRY(cudaMemsetAsync(next.data() + old_n, fill, (new_n - old_n) * sizeof(value_t),
                                    stream_of(res_)));
      b = std::move(next);
    };
    grow(dev_.d_row_start, 0);
    grow(dev_.d_deg, 0);
    grow(dev_.d_cap, 0);
    grow(dev_.d_node_w, 0);
    grow(dev_.inc.d_he_off, 0);
    grow(dev_.inc.d_he_len, 0);
    grow(dev_.inc.d_touched_idx, 0xff);  // -1: no touched slot
    dev_.inc.he_off.resize(new_n, 0);
    dev_.inc.he_cap.resize(new_n, 0);
    dev_.max_nodes = static_cast<int>(to);
    dev_.inc.max_nodes = static_cast<int>(to);
    ++node_growths_;
  }

  void upload_node_weights() {
    std::vector<long long> w(static_cast<std::size_t>(dev_.max_nodes), 0);
    const auto& by_id = host_->node_weights();
    for (std::size_t i = 0; i < by_id.size() && i < w.size(); ++i) w[i] = by_id[i];
    DYNG_CUDA_TRY(cudaMemcpyAsync(dev_.d_node_w.data(), w.data(), sizeof(long long) * w.size(),
                                  cudaMemcpyHostToDevice, stream_of(res_)));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));  // `w` is a local
    note_host_sync();
  }

  template <typename value_t>
  void download(const value_t* device, std::vector<value_t>& out) const {
    if (out.empty()) return;
    DYNG_CUDA_TRY(cudaMemcpyAsync(out.data(), device, sizeof(value_t) * out.size(),
                                  cudaMemcpyDeviceToHost, stream_of(res_)));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
    note_host_sync();
  }

  // A view's array on the device (the copy completes before the view is handed out:
  // finish_views()).
  template <typename value_t>
  void upload(const std::vector<value_t>& host, buffer<value_t>& out) const {
    out = buffer<value_t>(res_, host.size());
    if (!host.empty()) copy<value_t>(res_, host_view(host), out.view());
  }

  void upload_ints(const int* host, std::size_t n, buffer<int>& out) const {
    out = buffer<int>(res_, n);
    if (n > 0) copy<int>(res_, array_view<const int>(host, n), out.view());
  }

  // The views' uploads have completed (their host sources are temporaries).
  void finish_views() const {
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_of(res_)));
    note_host_sync();
  }

  void drop_views() noexcept {
    h2v_built_ = false;
    v2h_built_ = false;
    h2h_built_ = false;
    row_views_built_ = false;
    h2v_row_ptr_ = {};
    h2v_col_ind_ = {};
    v2h_row_ptr_ = {};
    v2h_col_ind_ = {};
    h2h_row_ptr_ = {};
    h2h_col_ind_ = {};
    h2h_out_ = {};
    rv_h2v_flat_ = {};
    rv_h2v_loc_ = {};
    rv_h2v_len_ = {};
    rv_h2h_flat_ = {};
    rv_h2h_loc_ = {};
    rv_h2h_len_ = {};
    rv_h2h_out_ = {};
  }

  resources res_;
  std::unique_ptr<slack_host_store> host_;  // the host model, without a host line graph
  slack::knobs knobs_;
  int threads_ = 1;

  // The device graph (built at construction under slack_csr, on first read under automatic).
  mutable std::atomic<bool> line_graph_{false};
  mutable slack::device_h2h dev_;
  mutable std::int64_t pair_count_ = 0;
  mutable std::int64_t line_graph_builds_ = 0;
  std::int64_t candidates_ = 0;
  std::int64_t csr_rebuilds_ = 0;
  std::int64_t mirror_reuploads_ = 0;
  std::int64_t tail_entries_ = 0;
  std::int64_t node_growths_ = 0;
  slack::incidence_batch inc_;

  // The views built on request (once per state), in device memory.
  mutable std::mutex view_mutex_;
  mutable bool h2v_built_ = false, v2h_built_ = false, h2h_built_ = false;
  mutable bool row_views_built_ = false;
  mutable buffer<std::int64_t> h2v_row_ptr_, v2h_row_ptr_, v2h_col_ind_;
  mutable buffer<std::int32_t> h2v_col_ind_;
  mutable buffer<std::int64_t> h2h_row_ptr_, h2h_col_ind_, h2h_out_;
  mutable buffer<int> rv_h2v_flat_, rv_h2v_loc_, rv_h2v_len_;
  mutable buffer<int> rv_h2h_flat_, rv_h2h_loc_, rv_h2h_len_, rv_h2h_out_;
  mutable int h2v_max_key_ = 0, h2h_max_key_ = 0;
  mutable view_builds builds_;
};

}  // namespace

std::unique_ptr<incidence_store> make_slack_device_store(const resources& res,
                                                         std::int32_t num_vertices,
                                                         const member_rows& rows,
                                                         const capacity_policy& capacity,
                                                         const slack::knobs& knobs) {
  const scoped_device guard(res.device());
  return std::make_unique<slack_device_store>(
      res, std::make_unique<slack_host_store>(res, num_vertices, rows, capacity, knobs), knobs);
}

std::unique_ptr<incidence_store> make_slack_device_store(const resources& res,
                                                         const store_snapshot& snapshot,
                                                         const capacity_policy& capacity,
                                                         const slack::knobs& knobs) {
  const scoped_device guard(res.device());
  store_snapshot host = snapshot;
  host.line_graph = false;  // the host model keeps no line graph: the device has it
  auto store = std::make_unique<slack_device_store>(
      res, std::make_unique<slack_host_store>(res, host, capacity, knobs), knobs);
  if (snapshot.line_graph) store->enable_line_graph();
  return store;
}

}  // namespace dyng::detail::hyper
