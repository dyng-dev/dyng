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
 * subtract and add phases kept apart, and the device error word). Everything runs on the stream of
 * the resources; the host waits only at ESCHER-GPU's read points (the wedge count of each phase
 * that has sources, the bins at the end of the call), each counted (triad_cuda_host_syncs()).
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
#include "util/kernel_registry.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace dyng::detail {

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

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// The engine's sign of a phase (-1: subtract, +1: add).
long long sign_of(framework::sign s) noexcept {
  return s == framework::sign::minus ? -1 : 1;
}

/// The pinned slot of the wedge count, after the bins.
constexpr int wedge_slot = triad::bin_slots;

/// The shape of a pattern's kernels (triad_launch_counters: 0 triangles, 1 pairs, 2 keys).
int shape_of(triad_count::pattern kind) noexcept {
  switch (kind) {
    case triad_count::pattern::stathyper_type1:
      return 2;
    case triad_count::pattern::stathyper_type2:
      return 1;
    default:
      return 0;
  }
}

/// Count one call's outcome (mode 0: full count, 1: a frontier phase).
void record(triad_cuda_engine& e, int mode, triad::wedge::launch outcome) noexcept {
  const int shape = shape_of(e.kind);
  switch (outcome) {
    case triad::wedge::launch::kernel:
      ++e.launches.kernels[mode][shape];
      break;
    case triad::wedge::launch::no_sources:
      ++e.launches.no_sources[mode][shape];
      break;
    case triad::wedge::launch::no_wedges:
      ++e.launches.no_wedges[mode][shape];
      break;
  }
}

/// The wedge scratch of an update's phase, or (full) of compute()'s full count.
triad::wedge::scratch scratch_of(const resources& res, triad_cuda_engine& e, bool full = false) {
  triad::wedge::scratch s;
  triad_cuda_workspace& ws = e.ws->get();
  s.offsets = full ? ws.full_offsets.data() : ws.offsets.data();
  s.scan_temp = full ? ws.full_scan_temp.data() : ws.scan_temp.data();
  s.scan_temp_bytes = e.scan_bytes;
  s.stream = native(res);
  s.host_wedges = reinterpret_cast<long long*>(e.ws->get().host.data() + wedge_slot);
  return s;
}

/// Read the bins of the call into the pinned buffer (the host synchronization of its end).
const unsigned long long* read_bins(const resources& res, triad_cuda_engine& e) {
  triad_cuda_workspace& ws = e.ws->get();
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host.data(), ws.bins.data(),
                                sizeof(unsigned long long) * triad::bin_slots,
                                cudaMemcpyDeviceToHost, native(res)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
  note_host_sync();  // the update's budget (I9) counts it
  return ws.host.data();
}

/// Size the wedge scratch for phases of up to `sources` sources.
void reserve_wedges(const resources& res, triad_cuda_engine& e, int sources, bool full = false) {
  triad_cuda_workspace& ws = e.ws->get();
  if (!triad_reads_line_graph(e.kind)) return;  // type 1 launches no wedges
  e.scan_bytes = std::max<std::size_t>(triad::wedge::scan_temp_bytes(sources + 1), 1);
  (full ? ws.full_offsets : ws.offsets).reserve(res, static_cast<std::size_t>(sources) + 1);
  (full ? ws.full_scan_temp : ws.scan_temp).reserve(res, e.scan_bytes);
}

/// The frontier of a phase from device ids (keys and, for the line-graph patterns, the flags).
void build_frontier(const resources& res, triad_cuda_engine& e, const std::int64_t* d_ids, int n,
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
    DYNG_CUDA_TRY(cudaMemsetAsync(flags, 0, static_cast<std::size_t>(max_key) + 1, native(res)));
  }
  constexpr int block = 256;
  frontier_kernel<<<(n + block - 1) / block, block, 0, native(res)>>>(d_ids, n, max_key,
                                                                      ws.keys.data(), flags);
  DYNG_CHECK_KERNEL(native(res));
}

}  // namespace

void triad_cuda_open(const resources& res, triad_count::pattern kind, triad_cuda_engine& e) {
  const scoped_device guard(res.device());
  e.kind = kind;
  e.ws.emplace(resources_access::workspaces(res).acquire<triad_cuda_workspace>(res));
  triad_cuda_workspace& ws = e.ws->get();
  ws.bins.reserve(res, triad::bin_slots);
  if (ws.host.size() < static_cast<std::size_t>(wedge_slot) + 1) {
    note_reservation();  // a deliberate growth of a reusable array (I9)
    ws.host = buffer<unsigned long long>();
    ws.host = buffer<unsigned long long>(static_cast<std::size_t>(wedge_slot) + 1, res.stream(),
                                         resources_access::staging_memory(res), res.device());
  }
}

void triad_cuda_compute(const resources& res, triad_cuda_engine& e, const hyper::row_views& views,
                        triad_bins& counts) {
  const scoped_device guard(res.device());
  triad_cuda_workspace& ws = e.ws->get();
  {
    scoped_stage stage(res, "triad_count.count");
    reserve_wedges(res, e, views.has_h2h ? views.h2h.max_key : 0, true);
    DYNG_CUDA_TRY(cudaMemsetAsync(ws.bins.data(), 0, sizeof(unsigned long long) * triad::bin_slots,
                                  native(res)));
    record(e, 0,
           triad::count_full(e.kind, views, ws.bins.data(), ws.bins.data() + triad::error_slot,
                             scratch_of(res, e, true)));
  }
  scoped_stage stage(res, "triad_count.finalize");
  const unsigned long long* values = read_bins(res, e);
  // The offsets of a full count (one per key: 20 MB on coauth-DBLP) are not kept for the updates.
  ws.full_offsets.release();
  ws.full_scan_temp.release();
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
                            const std::vector<std::int64_t>& deletions, std::int64_t insertions,
                            framework::sign s, framework::ownership::min_member rule) {
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
  DYNG_CUDA_TRY(cudaMemsetAsync(ws.bins.data(), 0, sizeof(unsigned long long) * triad::bin_slots,
                                native(res)));
  if (deleted == 0) {
    record(e, 1, triad::wedge::launch::no_sources);
    return;
  }
  // One upload of Step 0's host ids (pageable: the copy has read them when it returns).
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.ids.data(), deletions.data(),
                                sizeof(std::int64_t) * deletions.size(), cudaMemcpyHostToDevice,
                                native(res)));
  build_frontier(res, e, ws.ids.data(), deleted, views);
  record(
      e, 1,
      triad::count_phase(e.kind, views, ws.keys.data(), deleted, ws.flags.data(), rule, sign_of(s),
                         ws.bins.data(), ws.bins.data() + triad::error_slot, scratch_of(res, e)));
}

void triad_cuda_identify_affected(const resources& res, triad_cuda_engine& e,
                                  const hyper::row_views& views,
                                  const triad_inserted_frontier& ids) {
  const scoped_device guard(res.device());
  const std::int64_t* d_ids = ids.device;
  if (d_ids == nullptr && ids.size > 0) {
    // Ids without a device copy (only the recorded mutation triad_inserted_requested_ids makes
    // them): one upload.
    triad_cuda_workspace& ws = e.ws->get();
    ws.ids.reserve(res, static_cast<std::size_t>(ids.size));
    DYNG_CUDA_TRY(cudaMemcpyAsync(ws.ids.data(), ids.host, sizeof(std::int64_t) * ids.size,
                                  cudaMemcpyHostToDevice, native(res)));
    d_ids = ws.ids.data();
  }
  build_frontier(res, e, d_ids, ids.size, views);
}

void triad_cuda_count_plus(const resources& res, triad_cuda_engine& e,
                           const hyper::row_views& views, framework::sign s,
                           framework::ownership::min_member rule) {
  const scoped_device guard(res.device());
  triad_cuda_workspace& ws = e.ws->get();
  record(e, 1,
         triad::count_phase(e.kind, views, ws.keys.data(), e.num_keys, ws.flags.data(), rule,
                            sign_of(s), ws.bins.data() + triad::phase_bins,
                            ws.bins.data() + triad::error_slot, scratch_of(res, e)));
}

void triad_cuda_finalize(const resources& res, triad_cuda_engine& e, triad_signed_bins& minus,
                         triad_signed_bins& plus) {
  const scoped_device guard(res.device());
  const unsigned long long* values = read_bins(res, e);
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

void triad_cuda_take_launches(const triad_cuda_engine& e, triad_launch_counters& into) noexcept {
  for (int m = 0; m < 2; ++m) {
    for (int k = 0; k < 3; ++k) {
      into.kernels[m][k] += e.launches.kernels[m][k];
      into.no_sources[m][k] += e.launches.no_sources[m][k];
      into.no_wedges[m][k] += e.launches.no_wedges[m][k];
    }
  }
}

std::int64_t triad_cuda_host_syncs(triad_count::pattern kind) noexcept {
  // The wedge count of each phase that launches wedges and the read of the bins.
  return kind == triad_count::pattern::stathyper_type1 ? 1 : 3;
}

// The frontier kernel (resources::warm_up() loads it; the wedge kernels: engine/counters.cu).
DYNG_REGISTER_KERNEL(frontier_kernel);

}  // namespace dyng::detail
