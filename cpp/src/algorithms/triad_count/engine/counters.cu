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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace dyng::detail::triad {

static inline void check_cuda_local(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    std::exit(-1);
  }
}

// Full count: every triangle i < j < k of the line graph, one thread per
// out-wedge (i, j), j > i (kernel/wedge.cuh), classified into 30 bins.
std::vector<long long> compute_motif_counts(const escher::row_view& h2v,
                                            const escher::row_view& h2h) {
  unsigned long long* d_counts = nullptr;
  check_cuda_local(cudaMalloc(&d_counts, 30 * sizeof(unsigned long long)));
  check_cuda_local(cudaMemset(d_counts, 0, 30 * sizeof(unsigned long long)));
  wedge::count_triangles(h2h, nullptr, h2h.max_key, nullptr, make_motif_op(h2v), 1, d_counts);
  check_cuda_local(cudaDeviceSynchronize());

  std::vector<long long> counts(30, 0);
  check_cuda_local(
      cudaMemcpy(counts.data(), d_counts, 30 * sizeof(long long), cudaMemcpyDeviceToHost));
  check_cuda_local(cudaFree(d_counts));
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
  wedge::count_triangles(h2h, f.ids(), f.size(), f.flags(), make_motif_op(h2v), sign, d_counts);
  check_cuda_local(cudaDeviceSynchronize());
}

// motif_delta_accumulator implementation
void motif_delta_accumulator::init() {
  check_cuda_local(cudaMalloc(&d_counts, 30 * sizeof(unsigned long long)));
  check_cuda_local(cudaMemset(d_counts, 0, 30 * sizeof(unsigned long long)));
}

// Bins hold signed deltas in two's complement (subtract adds ~0ULL == -1).
void motif_delta_accumulator::read_to(std::vector<long long>& out) {
  out.resize(30);
  check_cuda_local(
      cudaMemcpy(out.data(), d_counts, 30 * sizeof(long long), cudaMemcpyDeviceToHost));
  check_cuda_local(cudaFree(d_counts));
  d_counts = nullptr;
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
