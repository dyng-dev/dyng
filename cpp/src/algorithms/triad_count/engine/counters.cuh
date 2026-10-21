// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/type1.cu
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu
// Derived from ESCHER-GPU@abcf9b2:include/motif.hpp
// Derived from ESCHER-GPU@abcf9b2:include/motif_update.hpp
// Derived from ESCHER-GPU@abcf9b2:src/main.cu
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu
/**
 * @file counters.cuh
 * @brief The four dynamic counters: the full count and the subtract / add phases of the h-motif
 *        counts (MotifCounter, computeMotifCounts, the delta accumulator) and of StatHyper's types
 *        1-3 (Type1Counter .. Type3Counter), as ESCHER-GPU's executables run them.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 *
 * dynG (memory, c3 of the port): the counters keep no device memory. ESCHER-GPU's counter classes
 * owned their bins (DeviceBins, the delta accumulator) and allocated a DeviceFrontier per phase;
 * here the bins, the frontier and the wedge scratch are the caller's workspace, so a counter is its
 * pattern's choice of shape and operation: count_full() is the classes' count(), count_phase()
 * their phase() (subtract() with the deleted keys, add() with the inserted keys; dynG, ADR 0057:
 * both as unsigned magnitudes, the signs applied on the host). The bins of one call: a stride of
 * max(num_bins, 30) for the subtract phase, the same for the add phase (kept apart), and the
 * device error word (an invalid bin: motif_op's pattern of no class, or a bin an Op of a user
 * named outside its range; a count past 2^64 - 1). Since M9b (ADR 0058) a counter is a table of
 * the local-pattern engine (local_patterns::kernels): the built-in patterns' and the Ops of users
 * run through the same two functions.
 */
#pragma once

#include "algorithms/triad_count/engine/wedge.cuh"
#include "framework/policies.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/experimental/detail/local_patterns_engine.hpp>

namespace dyng::detail::triad {

/// The smallest stride of a call's phase bins (30: h_motif's bins; M9a's layout, which the
/// built-in patterns keep).
constexpr int phase_bins = 30;

/**
 * @brief The stride of a call's phase bins on the device: the subtract phase's bins at 0, the
 *        add phase's at the stride, the device error word at twice the stride.
 * @param[in] num_bins The pattern's bins.
 * @return max(num_bins, phase_bins).
 */
constexpr int bin_stride(int num_bins) noexcept {
  return num_bins > phase_bins ? num_bins : phase_bins;
}

// The full count of a pattern (ESCHER-GPU's count(): computeMotifCounts for the h-motifs, the
// keys of H2V for type 1, the out-pairs and out-triangles of H2H for types 2 and 3; dynG M9b: any
// Op, through its table `k` and its value `op`), added to d_bins[0..num_bins). The source keys are
// the line graph's (views.h2h.max_key; ESCHER-GPU's numHyperedgeKeys) and, for the keys shape, the
// h2v view's (views.h2v.max_key: dynG's id bound can go down and the h2v view covers the keys up
// to the bound only).
wedge::launch count_full(const local_patterns::kernels& k, const void* op,
                         const hyper::row_views& views, unsigned long long* d_bins,
                         unsigned long long* d_error, const wedge::scratch& s);

// One frontier phase of a pattern (ESCHER-GPU's runFrontierPhase and the counters' phase()):
// the triads (pairs, keys) through the frontier keys d_keys[0..num_keys), each counted at its
// smallest frontier member (the rule `rule`, ownership.hpp; flags d_flags over the line graph's
// keys), added to d_bins as unsigned magnitudes (dynG, ADR 0057: the caller keeps the phase's
// sign; a count past 2^64 - 1 sets the capacity bit of d_error).
wedge::launch count_phase(const local_patterns::kernels& k, const void* op,
                          const hyper::row_views& views, const int* d_keys, int num_keys,
                          const unsigned char* d_flags, framework::ownership::min_member rule,
                          unsigned long long* d_bins, unsigned long long* d_error,
                          const wedge::scratch& s);

}  // namespace dyng::detail::triad
