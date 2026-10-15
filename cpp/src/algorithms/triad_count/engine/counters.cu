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
 *
 * dynG (the Op form, c6): the operations are value-initialized and read the h2v rows through the
 * rows handle the engine passes; the subtract and add phases name the exactly-once rule they
 * follow (ownership::min_member).
 */
#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "util/kernel_registry.hpp"

namespace dyng::detail::triad {

namespace {

/// The rows handle of a state's h2v view.
hyperedge_rows rows_of(const hyper::row_views& views) {
  return hyperedge_rows{views.h2v.flat, views.h2v.loc, views.h2v.len};
}

}  // namespace

// Full count: every triangle i < j < k of the line graph, one thread per
// out-wedge (i, j), j > i (kernel/wedge.cuh), classified into 30 bins.
wedge::launch count_full(triad_count::pattern kind, const hyper::row_views& views,
                         unsigned long long* d_bins, unsigned long long* d_error,
                         const wedge::scratch& s) {
  const hyperedge_rows rows = rows_of(views);
  // dynG: the source keys of phases.hpp (shared with the host engine).
  const int keys = full_count_keys(kind, views);
  switch (kind) {
    case triad_count::pattern::h_motif:  // computeMotifCounts
      return wedge::count_triangles(views.h2h, rows, nullptr, keys, nullptr, motif_op{}, 1, d_bins,
                                    d_error, s);
    case triad_count::pattern::stathyper_type1:  // Type1Counter::count
      return wedge::count_keys(rows, nullptr, keys, choose3_op{}, 1, d_bins, d_error, s);
    case triad_count::pattern::stathyper_type2:  // Type2Counter::count
      return wedge::count_pairs(views.h2h, rows, nullptr, keys, nullptr, type2_op{}, 1, d_bins,
                                d_error, s);
    case triad_count::pattern::stathyper_type3:  // Type3Counter::count
      return wedge::count_triangles(views.h2h, rows, nullptr, keys, nullptr, type3_op{}, 1, d_bins,
                                    d_error, s);
  }
  return wedge::launch::no_sources;
}

// Exact frontier-incident delta: the triangles containing at least one
// frontier hyperedge, each counted once at its smallest frontier member
// (include/frontier.hpp). Subtract runs on the old snapshot with the
// deleted hyperedges, add on the new snapshot with the inserted ones; one
// thread per frontier wedge (f, j) (kernel/wedge.cuh).
wedge::launch count_phase(triad_count::pattern kind, const hyper::row_views& views,
                          const int* d_keys, int num_keys, const unsigned char* d_flags,
                          framework::ownership::min_member /*rule*/, long long sign,
                          unsigned long long* d_bins, unsigned long long* d_error,
                          const wedge::scratch& s) {
  if (num_keys == 0) return wedge::launch::no_sources;
  const hyperedge_rows rows = rows_of(views);
  switch (kind) {
    case triad_count::pattern::h_motif:  // runFrontierPhase
      return wedge::count_triangles(views.h2h, rows, d_keys, num_keys, d_flags, motif_op{}, sign,
                                    d_bins, d_error, s);
    case triad_count::pattern::stathyper_type1:  // Type1Counter::phase
      return wedge::count_keys(rows, d_keys, num_keys, choose3_op{}, sign, d_bins, d_error, s);
    case triad_count::pattern::stathyper_type2:  // Type2Counter::phase
      return wedge::count_pairs(views.h2h, rows, d_keys, num_keys, d_flags, type2_op{}, sign,
                                d_bins, d_error, s);
    case triad_count::pattern::stathyper_type3:  // Type3Counter::phase
      return wedge::count_triangles(views.h2h, rows, d_keys, num_keys, d_flags, type3_op{}, sign,
                                    d_bins, d_error, s);
  }
  return wedge::launch::no_sources;
}

// The kernels, once per operation (resources::warm_up() loads them).
DYNG_REGISTER_KERNEL(wedge::count_wedges_kernel);
DYNG_REGISTER_KERNEL(wedge::triangle_kernel<motif_op>);
DYNG_REGISTER_KERNEL(wedge::triangle_kernel<type3_op>);
DYNG_REGISTER_KERNEL(wedge::pair_kernel<type2_op>);
DYNG_REGISTER_KERNEL(wedge::key_kernel<choose3_op>);

}  // namespace dyng::detail::triad
