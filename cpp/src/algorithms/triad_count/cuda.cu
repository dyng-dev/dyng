// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/driver.cu (run(): count(), subtract() on the snapshot before
// graph.update(), add() on the snapshot after it, with the keys the CBST insert assigned) and
// include/frontier.hpp (DeviceFrontier)
/**
 * @file cuda.cu
 * @brief The CUDA backend of triad_count: ESCHER-GPU's counters (engine/) behind the hooks of the
 *        problem (problem.hpp), on a pooled workspace.
 *
 * The counters run as ESCHER-GPU's driver runs them (src/driver.cu): count() for compute(), the
 * subtract phase on G_t before the commit and the add phase on G_{t+1} after it, on the
 * hypergraph's internal row views. Their memory is a workspace leased from the resources handle
 * for one call (ADR 0015) and sized before the commit for both phases (invariant I9): the wedge
 * offsets and CUB's scan storage, the frontier (its keys and a flag per line-graph key: ESCHER-GPU's
 * DeviceFrontier, built here by a kernel from the deleted ids, uploaded once, and from the applied
 * record's device copy of the ASSIGNED inserted ids, which need no upload), and the bins (the
 * subtract and add phases kept apart, and the device error word).
 */
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace dyng::detail {

/// The device memory of one compute() or update(), pooled (ADR 0015).
struct triad_cuda_workspace final : pooled_workspace {
  scratch_buffer<long long> offsets;        ///< the wedge offsets of a phase's sources
  scratch_buffer<unsigned char> scan_temp;  ///< CUB's storage of their scan
  scratch_buffer<std::int64_t> ids;         ///< the deleted ids (uploaded)
  scratch_buffer<int> keys;                 ///< the frontier's keys (id + 1)
  scratch_buffer<unsigned char> flags;      ///< a flag per line-graph key (the frontier)
  scratch_buffer<unsigned long long> bins;  ///< subtract bins, add bins, the error word

  /// The memory held.
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return offsets.bytes() + scan_temp.bytes() + ids.bytes() + keys.bytes() + flags.bytes() +
           bins.bytes();
  }
};

/// The engine of one call: the leased workspace and the sizes it was reserved for.
struct triad_cuda_engine {
  triad_count::pattern kind = triad_count::pattern::h_motif;
  std::optional<workspace_pool::lease<triad_cuda_workspace>> ws;
  std::size_t scan_bytes = 0;  ///< the reserved CUB storage
  int num_keys = 0;            ///< the keys of the current phase's frontier
  int max_keys = 0;            ///< the frontier keys (and wedge sources) reserved
  std::int64_t max_flags = 0;  ///< the flags reserved
};

void triad_cuda_engine_deleter::operator()(triad_cuda_engine* e) const noexcept {
  delete e;
}

namespace {

/// The frontier of one phase (ESCHER-GPU's DeviceFrontier): keys[i] = ids[i] + 1 and flags[key] = 1
/// for every key up to max_key (the flags are cleared before).
__global__ void frontier_kernel(const std::int64_t* ids, int n, int max_key, int* keys,
                                unsigned char* flags) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const int key = static_cast<int>(ids[i] + 1);
  keys[i] = key;
  if (flags != nullptr && key >= 0 && key <= max_key) flags[key] = 1;
}

triad::wedge::scratch scratch_of(triad_cuda_engine& e) {
  triad::wedge::scratch s;
  s.offsets = e.ws->get().offsets.data();
  s.scan_temp = e.ws->get().scan_temp.data();
  s.scan_temp_bytes = e.scan_bytes;
  return s;
}

/// Size the wedge scratch for phases of up to `sources` sources.
void reserve_wedges(const resources& res, triad_cuda_engine& e, int sources) {
  triad_cuda_workspace& ws = e.ws->get();
  if (!triad_reads_line_graph(e.kind)) return;  // type 1 launches no wedges
  ws.offsets.reserve(res, static_cast<std::size_t>(sources) + 1);
  e.scan_bytes = std::max<std::size_t>(triad::wedge::scan_temp_bytes(sources + 1), 1);
  ws.scan_temp.reserve(res, e.scan_bytes);
}

/// The frontier of a phase from device ids (keys and, for the line-graph patterns, the flags).
void build_frontier(triad_cuda_engine& e, const std::int64_t* d_ids, int n,
                    const hyper::row_views& views) {
  triad_cuda_workspace& ws = e.ws->get();
  e.num_keys = n;
  if (n == 0) return;
  unsigned char* flags = nullptr;
  int max_key = 0;
  if (n > e.max_keys) {
    DYNG_FAIL("triad_count: a frontier of ", n, " keys exceeds the ", e.max_keys,
              " the workspace reserved before the commit");
  }
  if (triad_reads_line_graph(e.kind)) {
    max_key = views.h2h.max_key;
    if (static_cast<std::int64_t>(max_key) + 1 > e.max_flags) {
      DYNG_FAIL("triad_count: the frontier flags of ", max_key,
                " line-graph keys exceed the workspace reserved before the commit (", e.max_flags,
                " flags)");
    }
    flags = ws.flags.data();
    DYNG_CUDA_TRY(cudaMemset(flags, 0, static_cast<std::size_t>(max_key) + 1));
  }
  constexpr int block = 256;
  frontier_kernel<<<(n + block - 1) / block, block>>>(d_ids, n, max_key, ws.keys.data(), flags);
  DYNG_CUDA_TRY(cudaGetLastError());
}

}  // namespace

triad_cuda_handle triad_cuda_open(const resources& res, triad_count::pattern kind) {
  const scoped_device guard(res.device());
  triad_cuda_handle e(new triad_cuda_engine);
  e->kind = kind;
  e->ws.emplace(resources_access::workspaces(res).acquire<triad_cuda_workspace>(res));
  e->ws->get().bins.reserve(res, triad::bin_slots);
  return e;
}

void triad_cuda_compute(const resources& res, triad_cuda_engine& e, const hyper::row_views& views,
                        triad_bins& counts) {
  const scoped_device guard(res.device());
  triad_cuda_workspace& ws = e.ws->get();
  std::vector<unsigned long long> values(triad::bin_slots, 0);
  {
    scoped_stage stage(res, "triad_count.count");
    reserve_wedges(res, e, views.has_h2h ? views.h2h.max_key : 0);
    DYNG_CUDA_TRY(cudaMemset(ws.bins.data(), 0, sizeof(unsigned long long) * triad::bin_slots));
    triad::count_full(e.kind, views, ws.bins.data(), ws.bins.data() + triad::error_slot,
                      scratch_of(e));
  }
  scoped_stage stage(res, "triad_count.finalize");
  DYNG_CUDA_TRY(cudaMemcpy(values.data(), ws.bins.data(),
                           sizeof(unsigned long long) * triad::bin_slots, cudaMemcpyDeviceToHost));
  // The offsets of a full count (one per key: 20 MB on coauth-DBLP) are not kept for the updates.
  ws.offsets.release();
  ws.scan_temp.release();
  const auto word = static_cast<std::uint32_t>(values[triad::error_slot]);
  if (word != 0) {
    throw_device_errors(word, "triad_count::compute",
                        "a triangle of the line graph has a Venn pattern of no h-motif class");
  }
  counts.fill(0);
  for (int k = 0; k < triad_num_bins(e.kind); ++k) {
    counts[static_cast<std::size_t>(k)] = values[static_cast<std::size_t>(k)];
  }
}

void triad_cuda_count_minus(const resources& res, triad_cuda_engine& e,
                            const hyper::row_views& views,
                            const std::vector<std::int64_t>& deletions, std::int64_t insertions) {
  const scoped_device guard(res.device());
  triad_cuda_workspace& ws = e.ws->get();
  const auto deleted = static_cast<int>(deletions.size());
  const auto inserted = static_cast<int>(insertions);
  // Everything both phases need, before the commit (I9): the frontier of the larger phase, and
  // flags for every key the line graph can have after the batch (its keys never shrink and grow
  // by at most the insertions).
  const int sources = std::max(deleted, inserted);
  ws.ids.reserve(res, static_cast<std::size_t>(std::max(deleted, 1)));
  ws.keys.reserve(res, static_cast<std::size_t>(std::max(sources, 1)));
  e.max_keys = sources;
  reserve_wedges(res, e, sources);
  if (triad_reads_line_graph(e.kind)) {
    e.max_flags = static_cast<std::int64_t>(views.h2h.max_key) + insertions + 1;
    ws.flags.reserve(res, static_cast<std::size_t>(e.max_flags));
  }
  DYNG_CUDA_TRY(cudaMemset(ws.bins.data(), 0, sizeof(unsigned long long) * triad::bin_slots));
  if (deleted == 0) return;
  DYNG_CUDA_TRY(cudaMemcpy(ws.ids.data(), deletions.data(), sizeof(std::int64_t) * deletions.size(),
                           cudaMemcpyHostToDevice));
  build_frontier(e, ws.ids.data(), deleted, views);
  triad::count_phase(e.kind, views, ws.keys.data(), deleted, ws.flags.data(), -1, ws.bins.data(),
                     ws.bins.data() + triad::error_slot, scratch_of(e));
}

void triad_cuda_identify_affected(const resources& res, triad_cuda_engine& e,
                                  const hyper::row_views& views, const hyper_applied& record) {
  const scoped_device guard(res.device());
  build_frontier(e, record.inserted_ids_device.data(), static_cast<int>(record.inserted_ids.size()),
                 views);
}

void triad_cuda_count_plus(const resources& res, triad_cuda_engine& e,
                           const hyper::row_views& views) {
  const scoped_device guard(res.device());
  triad_cuda_workspace& ws = e.ws->get();
  triad::count_phase(e.kind, views, ws.keys.data(), e.num_keys, ws.flags.data(), 1,
                     ws.bins.data() + triad::phase_bins, ws.bins.data() + triad::error_slot,
                     scratch_of(e));
}

void triad_cuda_finalize(const resources& res, triad_cuda_engine& e, triad_signed_bins& minus,
                         triad_signed_bins& plus) {
  const scoped_device guard(res.device());
  std::array<unsigned long long, triad::bin_slots> values{};
  DYNG_CUDA_TRY(cudaMemcpy(values.data(), e.ws->get().bins.data(),
                           sizeof(unsigned long long) * triad::bin_slots, cudaMemcpyDeviceToHost));
  const auto word = static_cast<std::uint32_t>(values[triad::error_slot]);
  if (word != 0) {
    throw_device_errors(word, "triad_count::update",
                        "a triangle of the line graph has a Venn pattern of no h-motif class");
  }
  // Two's complement: the subtract phase's bins hold -(the triads removed).
  for (std::size_t k = 0; k < minus.size(); ++k) {
    minus[k] = static_cast<std::int64_t>(values[k]);
    plus[k] = static_cast<std::int64_t>(values[triad::phase_bins + k]);
  }
}

std::int64_t triad_cuda_host_syncs(triad_count::pattern kind) noexcept {
  // The wedge count of each phase that launches wedges and the read of the bins.
  return kind == triad_count::pattern::stathyper_type1 ? 1 : 3;
}

}  // namespace dyng::detail
