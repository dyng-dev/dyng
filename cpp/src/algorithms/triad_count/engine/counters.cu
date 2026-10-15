// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/HMotifCount.cu
// Derived from ESCHER-GPU@abcf9b2:src/HMotifCountUpdate.cu
/**
 * @file counters.cu
 * @brief The h-motif full count and the frontier phases of the h-motif delta (ESCHER-GPU's
 *        computeMotifCounts, runFrontierPhase and the delta accumulator).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/engine/device_frontier.cuh"
#include "algorithms/triad_count/engine/motif_ops.cuh"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dyng::detail::triad {

// dynG: the 30 bins are followed by the device error word of motif_op (a triangle of no class),
// read with them; a raised word throws internal_error (ESCHER-GPU added such a triangle to
// bins[-1]).
constexpr int motif_slots = 30 + 1;

static void throw_if_unclassified(const std::vector<long long>& slots) {
  const auto word = static_cast<std::uint32_t>(slots[30]);
  if (word != 0)
    throw_device_errors(word, "triad_count",
                        "a triangle of the line graph has a Venn pattern of no h-motif class");
}

// Full count: every triangle i < j < k of the line graph, one thread per
// out-wedge (i, j), j > i (kernel/wedge.cuh), classified into 30 bins.
std::vector<long long> compute_motif_counts(const escher::row_view& h2v,
                                            const escher::row_view& h2h) {
  unsigned long long* d_counts = nullptr;
  DYNG_CUDA_TRY(cudaMalloc(&d_counts, motif_slots * sizeof(unsigned long long)));
  DYNG_CUDA_TRY(cudaMemset(d_counts, 0, motif_slots * sizeof(unsigned long long)));
  wedge::count_triangles(h2h, nullptr, h2h.max_key, nullptr, make_motif_op(h2v, d_counts + 30), 1,
                         d_counts);
  DYNG_CUDA_TRY(cudaDeviceSynchronize());

  std::vector<long long> counts(motif_slots, 0);
  DYNG_CUDA_TRY(
      cudaMemcpy(counts.data(), d_counts, motif_slots * sizeof(long long), cudaMemcpyDeviceToHost));
  DYNG_CUDA_TRY(cudaFree(d_counts));
  throw_if_unclassified(counts);
  counts.resize(30);
  return counts;
}
// Exact frontier-incident delta: the triangles containing at least one
// frontier hyperedge, each counted once at its smallest frontier member
// (include/frontier.hpp). Subtract runs on the old snapshot with the
// deleted hyperedges, add on the new snapshot with the inserted ones; one
// thread per frontier wedge (f, j) (kernel/wedge.cuh).
static void run_frontier_phase(long long sign, const escher::row_view& h2v,
                               const escher::row_view& h2h, const std::vector<int>& frontier,
                               unsigned long long* d_counts) {
  if (frontier.empty()) return;
  device_frontier f(frontier, h2h.max_key);
  wedge::count_triangles(h2h, f.ids(), f.size(), f.flags(), make_motif_op(h2v, d_counts + 30), sign,
                         d_counts);
  DYNG_CUDA_TRY(cudaDeviceSynchronize());
}

// motif_delta_accumulator implementation
void motif_delta_accumulator::init() {
  DYNG_CUDA_TRY(cudaMalloc(&d_counts, motif_slots * sizeof(unsigned long long)));
  DYNG_CUDA_TRY(cudaMemset(d_counts, 0, motif_slots * sizeof(unsigned long long)));
}

// Bins hold signed deltas in two's complement (subtract adds ~0ULL == -1).
void motif_delta_accumulator::read_to(std::vector<long long>& out) {
  out.resize(motif_slots);
  DYNG_CUDA_TRY(
      cudaMemcpy(out.data(), d_counts, motif_slots * sizeof(long long), cudaMemcpyDeviceToHost));
  DYNG_CUDA_TRY(cudaFree(d_counts));
  d_counts = nullptr;
  throw_if_unclassified(out);
  out.resize(30);
}

// Subtract phase: call BEFORE modifying data structures
void compute_motif_subtract(const escher::row_view& h2v, const escher::row_view& h2h,
                            const std::vector<int>& deleted_ids, motif_delta_accumulator& acc) {
  run_frontier_phase(-1, h2v, h2h, deleted_ids, acc.d_counts);
}

// Add phase: call AFTER building new snapshots
void compute_motif_add(const escher::row_view& h2v, const escher::row_view& h2h,
                       const std::vector<int>& inserted_ids, motif_delta_accumulator& acc) {
  run_frontier_phase(1, h2v, h2h, inserted_ids, acc.d_counts);
}

}  // namespace dyng::detail::triad
