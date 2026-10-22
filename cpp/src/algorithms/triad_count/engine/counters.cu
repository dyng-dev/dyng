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
 *
 * dynG (M9b, ADR 0058): a count runs any Op through its table (local_patterns::kernels): the
 * kernels of the four built-in operations are instantiated here (triad_device_launch()), those
 * of the Ops of users in the users' translation units (<dyng/experimental/local_patterns.cuh>).
 */
#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/counters.cuh"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "util/kernel_registry.hpp"

namespace dyng::detail::triad {

namespace {

/// The rows handle of a state's h2v view.
hyperedge_rows rows_of(const hyper::row_views& views) {
  return make_rows(views.h2v.flat, views.h2v.loc, views.h2v.len);
}

}  // namespace

// Full count: every triangle i < j < k of the line graph, one thread per
// out-wedge (i, j), j > i (kernel/wedge.cuh), classified into 30 bins (h_motif; the Op's bins).
wedge::launch count_full(const local_patterns::kernels& k, const void* op,
                         const hyper::row_views& views, unsigned long long* d_bins,
                         unsigned long long* d_error, const wedge::scratch& s) {
  // dynG: the source keys of phases.hpp (shared with the host engine).
  const int keys = full_count_keys(k.pattern_shape, views);
  return wedge::count_shape(k, op, views.h2h, rows_of(views), nullptr, keys, nullptr, d_bins,
                            d_error, s);
}

// Exact frontier-incident delta: the triangles containing at least one
// frontier hyperedge, each counted once at its smallest frontier member
// (include/frontier.hpp). Subtract runs on the old snapshot with the
// deleted hyperedges, add on the new snapshot with the inserted ones; one
// thread per frontier wedge (f, j) (kernel/wedge.cuh). dynG (ADR 0057): both phases add unsigned
// magnitudes; the caller keeps the phase's sign.
wedge::launch count_phase(const local_patterns::kernels& k, const void* op,
                          const hyper::row_views& views, const int* d_keys, int num_keys,
                          const unsigned char* d_flags, framework::ownership::min_member /*rule*/,
                          unsigned long long* d_bins, unsigned long long* d_error,
                          const wedge::scratch& s) {
  if (num_keys == 0) return wedge::launch::no_sources;
  // The keys shape reads no flags (type 1: Type1Counter::phase).
  const unsigned char* flags = k.pattern_shape == experimental::shape::keys ? nullptr : d_flags;
  return wedge::count_shape(k, op, views.h2h, rows_of(views), d_keys, num_keys, flags, d_bins,
                            d_error, s);
}

}  // namespace dyng::detail::triad

namespace dyng::detail {

local_patterns::launch_fn triad_device_launch(triad_count::pattern kind) noexcept {
  switch (kind) {
    case triad_count::pattern::h_motif:  // computeMotifCounts, runFrontierPhase
      return &local_patterns::launch<triad::motif_op>;
    case triad_count::pattern::stathyper_type1:  // Type1Counter
      return &local_patterns::launch<triad::choose3_op>;
    case triad_count::pattern::stathyper_type2:  // Type2Counter
      return &local_patterns::launch<triad::type2_op>;
    case triad_count::pattern::stathyper_type3:  // Type3Counter
      return &local_patterns::launch<triad::type3_op>;
  }
  return nullptr;
}

// The kernels, once per operation (resources::warm_up() loads them).
DYNG_REGISTER_KERNEL(triad::wedge::count_wedges_kernel);
DYNG_REGISTER_KERNEL(local_patterns::triangle_kernel<triad::motif_op>);
DYNG_REGISTER_KERNEL(local_patterns::triangle_kernel<triad::type3_op>);
DYNG_REGISTER_KERNEL(local_patterns::pair_kernel<triad::type2_op>);
DYNG_REGISTER_KERNEL(local_patterns::key_kernel<triad::choose3_op>);

}  // namespace dyng::detail
