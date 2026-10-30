// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda.cu
 * @brief hyper_sssp's CUDA backend (Tier B, engine::fused): the problem's entry points around
 *        MOSP_ESCHER's transplanted engine (engine.cu, kernels.cuh, state.cuh; ADR 0091): the
 *        device state of a result, the device graph read from the slack_csr store (its rows,
 *        weights and last delta: slack_access::rows() and last_delta()), the endpoints of the two
 *        modes, the budget, and the counters.
 */
#include "algorithms/hyper_sssp/problem.hpp"
#include "algorithms/hyper_sssp/state.cuh"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "util/cuda_check.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace dyng::detail {

void hyper_sssp_device_state_deleter::operator()(hsosp::device_state* st) const noexcept {
  delete st;
}

namespace {

/// The engine's int node ids: the slack_csr store's ids are 32-bit (its colInd).
int node_id(std::int64_t id) {
  return static_cast<int>(id);
}

/// MOSP_ESCHER's DeviceH2H (and DeviceDelta) of the slack_csr store, with the endpoints.
hsosp::device_graph device_graph_of(const hyper_sssp_graph& g, const hyper_sssp_batch* b) {
  DYNG_EXPECTS(g.rows.space == memory_space::device,
               "hyper_sssp: the line graph is not on the device");
  DYNG_EXPECTS(g.rows.num_nodes <= std::numeric_limits<int>::max(),
               "hyper_sssp: ", g.rows.num_nodes, " line-graph nodes exceed the engine's int32 ids");
  hsosp::device_graph dev;
  dev.num_nodes = static_cast<int>(g.rows.num_nodes);
  dev.num_entries = g.rows.num_entries;
  dev.d_row_start = reinterpret_cast<const long long*>(g.rows.row_start);
  dev.d_deg = g.rows.deg;
  dev.d_col_ind = g.rows.col_ind;
  dev.d_node_w = reinterpret_cast<const long long*>(g.rows.node_weight);
  dev.zero_a = g.endpoints.zero_weight_a < 0 ? -1 : node_id(g.endpoints.zero_weight_a);
  dev.zero_b = g.endpoints.zero_weight_b < 0 ? -1 : node_id(g.endpoints.zero_weight_b);
  if (b != nullptr) {
    dev.delta.d_pairs = reinterpret_cast<const int2*>(b->device.pairs);
    dev.delta.d_ids = b->device.ids;
    dev.delta.num_del = static_cast<int>(b->device.num_deleted);
    dev.delta.num_ins = static_cast<int>(b->device.num_inserted);
    dev.delta.num_new = static_cast<int>(b->device.num_new);
    dev.delta.num_dead = static_cast<int>(b->device.num_dead);
  }
  return dev;
}

/// MOSP_ESCHER's UpdateConfig of the budget.
hsosp::engine_config config_of(const hyper_sssp_budget& b, double work_budget) {
  hsosp::engine_config cfg;
  cfg.max_iterations = b.max_iterations;
  cfg.work_budget = work_budget;
  cfg.block_size = b.block_size;
  cfg.error_on_budget = b.on_budget == hyper_sssp::budget_action::error;
  return cfg;
}

/// MOSP_ESCHER's sourceId (1-based): the virtual mode's source row, 0 (source0 = -1) in the public
/// mode.
int source_id_of(const hyper_sssp_state& st) {
  return st.endpoints.is_virtual() ? node_id(st.endpoints.source_hyperedge) + 1 : 0;
}

/// The engine's counters as the problem keeps them.
hyper_sssp_engine_counters counters_of(const hsosp::engine_stats& s) {
  hyper_sssp_engine_counters c;
  c.iterations = s.iterations;
  c.fallback = s.fallback_recompute;
  c.fallback_iterations = s.fallback_iterations;
  c.max_frontier = s.max_frontier;
  c.seed_count = s.seed_count;
  c.jump_rounds = s.jump_rounds;
  c.work = s.work;
  c.reason = s.fallback_recompute ? s.reason : hyper_sssp::fallback_reason::none;
  return c;
}

/// The public mode's base: the live hyperedges of the source vertex.
std::vector<int> base_of(const hyper_sssp_graph& g) {
  std::vector<int> out;
  const std::int32_t s = g.endpoints.source_vertex;
  if (s < 0 || static_cast<std::size_t>(s) >= g.host_v2h->size()) return out;
  for (const std::int64_t h : (*g.host_v2h)[static_cast<std::size_t>(s)]) {
    if (g.live(h)) out.push_back(node_id(h));
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// The engine runs on the legacy default stream in this commit (MOSP_ESCHER's): the work of the
/// resources' stream before it (the commit) completes first.
void wait_for_stream(const resources& res) {
  DYNG_CUDA_TRY(cudaStreamSynchronize(static_cast<cudaStream_t>(res.stream().get())));
}

}  // namespace

void hyper_sssp_cuda_reserve(const resources& res, hyper_sssp_state& st, std::int64_t nodes) {
  const scoped_device guard(res.device());
  DYNG_EXPECTS(nodes <= std::numeric_limits<int>::max(), "hyper_sssp: ", nodes,
               " node slots exceed the engine's int32 ids");
  const auto n = static_cast<int>(nodes);
  if (st.dev != nullptr && st.dev->max_nodes >= n) return;
  note_reservation();  // the result's device state grows (invariant I9)
  hyper_sssp_device_ptr bigger(new hsosp::device_state);
  bigger->allocate(n);
  if (!st.endpoints.is_virtual()) bigger->allocate_holds();
  if (st.dev != nullptr) {
    // A growth of the store's node arrays: the distances, parents and flags carry over (the new
    // slots are infinite, -1 and 0); the epoch stamps start again from 0 (any epoch is newer).
    const auto old = static_cast<std::size_t>(st.dev->max_nodes);
    DYNG_CUDA_TRY(cudaMemcpy(bigger->d_dist, st.dev->d_dist, sizeof(long long) * old,
                             cudaMemcpyDeviceToDevice));
    DYNG_CUDA_TRY(cudaMemcpy(bigger->d_parent, st.dev->d_parent, sizeof(int) * old,
                             cudaMemcpyDeviceToDevice));
    if (st.dev->d_holds != nullptr && bigger->d_holds != nullptr) {
      DYNG_CUDA_TRY(cudaMemcpy(bigger->d_holds, st.dev->d_holds, old, cudaMemcpyDeviceToDevice));
    }
    bigger->epoch = st.dev->epoch;
  }
  st.dev = std::move(bigger);
}

hyper_sssp_engine_counters hyper_sssp_cuda_recompute(const resources& res, hyper_sssp_state& st,
                                                     const hyper_sssp_graph& g,
                                                     const hyper_sssp_budget& budget) {
  const scoped_device guard(res.device());
  wait_for_stream(res);
  hsosp::device_graph dev = device_graph_of(g, nullptr);
  std::vector<int> base;
  if (!st.endpoints.is_virtual()) {
    // The source flags on the device (the host copy is current: compute() rebuilt it).
    DYNG_CUDA_TRY(cudaMemcpy(
        st.dev->d_holds, st.holds_source.data(),
        std::min<std::size_t>(st.holds_source.size(), static_cast<std::size_t>(st.dev->max_nodes)),
        cudaMemcpyHostToDevice));
    dev.d_holds = st.dev->d_holds;
    base = base_of(g);
    dev.base = &base;
  }
  const hsosp::engine_stats s =
      hsosp::recompute(dev, *st.dev, source_id_of(st), config_of(budget, st.opt.work_budget));
  return counters_of(s);
}

hyper_sssp_engine_counters hyper_sssp_cuda_update(const resources& res, hyper_sssp_state& st,
                                                  const hyper_sssp_graph& g,
                                                  const hyper_sssp_batch& b,
                                                  const hyper_sssp_budget& budget,
                                                  const framework::hook_stages& /*names*/) {
  const scoped_device guard(res.device());
  wait_for_stream(res);
  hsosp::device_graph dev = device_graph_of(g, &b);
  std::vector<int> base, lost, gained;
  if (!st.endpoints.is_virtual()) {
    dev.d_holds = st.dev->d_holds;
    base = base_of(g);
    dev.base = &base;
    for (const std::int64_t id : *b.lost_source) lost.push_back(node_id(id));
    for (const std::int64_t id : *b.gained_source) gained.push_back(node_id(id));
    dev.lost = &lost;
    dev.gained = &gained;
  }
  const hsosp::engine_stats s =
      hsosp::update(dev, *st.dev, source_id_of(st), config_of(budget, st.opt.work_budget));
  return counters_of(s);
}

void hyper_sssp_cuda_download(const resources& res, const hyper_sssp_state& st,
                              std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent) {
  const scoped_device guard(res.device());
  const auto n = static_cast<std::size_t>(st.n);
  dist.resize(n);
  parent.resize(n);
  if (n == 0) return;
  DYNG_CUDA_TRY(
      cudaMemcpy(dist.data(), st.dev->d_dist, sizeof(long long) * n, cudaMemcpyDeviceToHost));
  DYNG_CUDA_TRY(
      cudaMemcpy(parent.data(), st.dev->d_parent, sizeof(int) * n, cudaMemcpyDeviceToHost));
}

void hyper_sssp_cuda_upload(const resources& res, hyper_sssp_state& st,
                            const std::vector<std::int64_t>& dist,
                            const std::vector<std::int32_t>& parent) {
  const scoped_device guard(res.device());
  const std::size_t n = std::min(dist.size(), static_cast<std::size_t>(st.dev->max_nodes));
  if (n == 0) return;
  DYNG_CUDA_TRY(
      cudaMemcpy(st.dev->d_dist, dist.data(), sizeof(long long) * n, cudaMemcpyHostToDevice));
  DYNG_CUDA_TRY(
      cudaMemcpy(st.dev->d_parent, parent.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
  if (st.dev->d_holds != nullptr && !st.holds_source.empty()) {
    DYNG_CUDA_TRY(
        cudaMemcpy(st.dev->d_holds, st.holds_source.data(),
                   std::min(st.holds_source.size(), static_cast<std::size_t>(st.dev->max_nodes)),
                   cudaMemcpyHostToDevice));
  }
}

const std::int64_t* hyper_sssp_cuda_distances(const hyper_sssp_state& st) noexcept {
  return st.dev == nullptr ? nullptr : reinterpret_cast<const std::int64_t*>(st.dev->d_dist);
}

const std::int32_t* hyper_sssp_cuda_parents(const hyper_sssp_state& st) noexcept {
  return st.dev == nullptr ? nullptr : st.dev->d_parent;
}

void hyper_sssp_cuda_set_holds(const resources& res, hyper_sssp_state& st,
                               const std::vector<std::int64_t>& ids, std::uint8_t value) {
  const scoped_device guard(res.device());
  for (const std::int64_t id : ids) {
    DYNG_CUDA_TRY(cudaMemset(st.dev->d_holds + id, value, 1));
  }
}

std::int64_t hyper_sssp_cuda_bytes(const hyper_sssp_state& st) noexcept {
  return st.dev == nullptr ? 0 : st.dev->device_bytes();
}

}  // namespace dyng::detail
