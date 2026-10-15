// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/HMotifCount.cu
// Derived from ESCHER-GPU@abcf9b2:src/HMotifCountUpdate.cu
// Derived from ESCHER-GPU@abcf9b2:src/type1.cu
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu
/**
 * @file counters.cu
 * @brief The full count and the frontier phases of the four counters (ESCHER-GPU's
 *        computeMotifCounts, runFrontierPhase and the counters' count() / phase()); the wedge
 *        kernels are instantiated here, once per operation.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 *
 * dynG (streams, c4): every launch on the call's stream (wedge::scratch); ESCHER-GPU's
 * cudaDeviceSynchronize after the h-motif count and phases is gone: the host waits only where it
 * reads (the wedge count of a phase, the bins at the end of the call).
 */
#include "algorithms/triad_count/engine/count_ops.cuh"
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/engine/motif_ops.cuh"
#include "algorithms/triad_count/engine/stathyper_ops.cuh"

namespace dyng::detail::triad {

// Full count: every triangle i < j < k of the line graph, one thread per
// out-wedge (i, j), j > i (kernel/wedge.cuh), classified into 30 bins.
void count_full(triad_count::pattern kind, const hyper::row_views& views,
                unsigned long long* d_bins, unsigned long long* d_error, const wedge::scratch& s) {
  switch (kind) {
    case triad_count::pattern::h_motif:  // computeMotifCounts
      wedge::count_triangles(views.h2h, nullptr, views.h2h.max_key, nullptr,
                             make_motif_op(views.h2v, d_error), 1, d_bins, s);
      return;
    case triad_count::pattern::stathyper_type1:  // Type1Counter::count
      wedge::count_keys(nullptr, views.h2v.max_key, make_choose3_op(views.h2v), 1, d_bins, s);
      return;
    case triad_count::pattern::stathyper_type2:  // Type2Counter::count
      wedge::count_pairs(views.h2h, nullptr, views.h2h.max_key, nullptr, make_type2_op(views.h2v),
                         1, d_bins, s);
      return;
    case triad_count::pattern::stathyper_type3:  // Type3Counter::count
      wedge::count_triangles(views.h2h, nullptr, views.h2h.max_key, nullptr,
                             make_type3_op(views.h2v), 1, d_bins, s);
      return;
  }
}

// Exact frontier-incident delta: the triangles containing at least one
// frontier hyperedge, each counted once at its smallest frontier member
// (include/frontier.hpp). Subtract runs on the old snapshot with the
// deleted hyperedges, add on the new snapshot with the inserted ones; one
// thread per frontier wedge (f, j) (kernel/wedge.cuh).
void count_phase(triad_count::pattern kind, const hyper::row_views& views, const int* d_keys,
                 int num_keys, const unsigned char* d_flags, long long sign,
                 unsigned long long* d_bins, unsigned long long* d_error, const wedge::scratch& s) {
  if (num_keys == 0) return;
  switch (kind) {
    case triad_count::pattern::h_motif:  // runFrontierPhase
      wedge::count_triangles(views.h2h, d_keys, num_keys, d_flags,
                             make_motif_op(views.h2v, d_error), sign, d_bins, s);
      return;
    case triad_count::pattern::stathyper_type1:  // Type1Counter::phase
      wedge::count_keys(d_keys, num_keys, make_choose3_op(views.h2v), sign, d_bins, s);
      return;
    case triad_count::pattern::stathyper_type2:  // Type2Counter::phase
      wedge::count_pairs(views.h2h, d_keys, num_keys, d_flags, make_type2_op(views.h2v), sign,
                         d_bins, s);
      return;
    case triad_count::pattern::stathyper_type3:  // Type3Counter::phase
      wedge::count_triangles(views.h2h, d_keys, num_keys, d_flags, make_type3_op(views.h2v), sign,
                             d_bins, s);
      return;
  }
}

}  // namespace dyng::detail::triad
