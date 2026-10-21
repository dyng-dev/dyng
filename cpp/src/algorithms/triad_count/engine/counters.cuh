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
 * both as unsigned magnitudes, the signs applied on the host). The bins of one call: 30 of the
 * subtract phase, 30 of the add phase (kept apart), and the device error word (motif_op's pattern
 * of no class; a count past 2^64 - 1).
 */
#pragma once

#include "algorithms/triad_count/engine/wedge.cuh"
#include "framework/policies.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/triad_count.hpp>

namespace dyng::detail::triad {

/// The bins of one phase (30 for h_motif; the types use the first).
constexpr int phase_bins = 30;
/// The device slots of one call: the subtract phase's bins, the add phase's, the error word.
constexpr int bin_slots = 2 * phase_bins + 1;
/// The slot of the device error word (motif_op: a triangle of no class).
constexpr int error_slot = 2 * phase_bins;

// The full count of a pattern (ESCHER-GPU's count(): computeMotifCounts for the h-motifs, the
// keys of H2V for type 1, the out-pairs and out-triangles of H2H for types 2 and 3), added to
// d_bins[0..30). The source keys are the line graph's (views.h2h.max_key; ESCHER-GPU's
// numHyperedgeKeys) and, for type 1, the h2v view's (views.h2v.max_key: dynG's id bound can go
// down and the h2v view covers the keys up to the bound only).
wedge::launch count_full(triad_count::pattern kind, const hyper::row_views& views,
                         unsigned long long* d_bins, unsigned long long* d_error,
                         const wedge::scratch& s);

// One frontier phase of a pattern (ESCHER-GPU's runFrontierPhase and the counters' phase()):
// the triads (pairs, keys) through the frontier keys d_keys[0..num_keys), each counted at its
// smallest frontier member (the rule `rule`, ownership.hpp; flags d_flags over the line graph's
// keys), added to d_bins as unsigned magnitudes (dynG, ADR 0057: the caller keeps the phase's
// sign; a count past 2^64 - 1 sets the capacity bit of d_error).
wedge::launch count_phase(triad_count::pattern kind, const hyper::row_views& views,
                          const int* d_keys, int num_keys, const unsigned char* d_flags,
                          framework::ownership::min_member rule, unsigned long long* d_bins,
                          unsigned long long* d_error, const wedge::scratch& s);

}  // namespace dyng::detail::triad
