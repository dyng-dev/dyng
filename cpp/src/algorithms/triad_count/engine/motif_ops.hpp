// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/motif_utils.cuh
/**
 * @file motif_ops.hpp
 * @brief The 30-bin h-motif classification of a line-graph triangle: the class map on the device,
 *        the class index from the region counts and the wedge operation MotifOp (ESCHER-GPU's
 *        kernel/motif_utils.cuh).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 *
 * dynG (the Op form, c6): MotifOp is a host-and-device struct; it reads its rows through the rows
 * handle and adds to the bins handle, which refuses a pattern of no class (ESCHER-GPU added it to
 * bins[-1]); the class map has a device copy (the kernels') and a host copy (the host engine's).
 */
#pragma once

#include "algorithms/triad_count/engine/motif_table.hpp"
#include "algorithms/triad_count/engine/sets.hpp"

#include <dyng/experimental/detail/local_patterns_engine.hpp>

#include <climits>
#include <type_traits>

namespace dyng::detail::triad {

#if defined(__CUDACC__)
// Header-only device data and helpers to avoid cross-TU device linking
static __device__ int id_to_index[128] = DYNG_TRIAD_MOTIF_ID_TO_INDEX;
#endif
// dynG: the same map on the host (the host engine's operations).
inline constexpr int host_id_to_index[128] = DYNG_TRIAD_MOTIF_ID_TO_INDEX;

// Inline device helper to compute motif index [0..29] from region counts (-1: no class)
static inline DYNG_TRIAD_HD int compute_motif_index(int deg_a, int deg_b, int deg_c, int c_ab,
                                                    int c_bc, int c_ca, int g_abc) {
  int a = deg_a - (c_ab + c_ca) + g_abc;
  int b = deg_b - (c_bc + c_ab) + g_abc;
  int c = deg_c - (c_ca + c_bc) + g_abc;
  int d = c_ab - g_abc;
  int e = c_bc - g_abc;
  int f = c_ca - g_abc;
  int g = g_abc;

  int motif_id = (a > 0) | ((b > 0) << 1) | ((c > 0) << 2) | ((d > 0) << 3) | ((e > 0) << 4) |
                 ((f > 0) << 5) | ((g > 0) << 6);
#if defined(__CUDA_ARCH__)
  int index = id_to_index[motif_id] - 1;  // [0..29]
#else
  int index = host_id_to_index[motif_id] - 1;  // [0..29]
#endif
  return index;
}

// 30-bin classification of a line-graph triangle (a, b, c) for the wedge
// kernels (kernel/wedge.cuh). The rows, degrees and |A ∩ B| of the wedge
// (a, b) are computed once per wedge, and (|B ∩ C|, |C ∩ A|, |A ∩ B ∩ C|)
// come from a single merge over the three H2V rows.
struct motif_op {
  static constexpr int num_bins = 30;
  /// dynG (M9b, ADR 0058): the Op's shape (experimental::count_local_patterns).
  static constexpr experimental::shape shape = experimental::shape::triangles;
  // dynG (ADR 0057): every add is 1, so a block's bins cannot pass 2^64 - 1 and the device's
  // shared bins are added to unchecked on this hot path (wedge.cuh); the global bins are checked.
  static constexpr bool unit_counts = true;
  const int* row_a;
  const int* row_b;
  int da, db, cab;

  template <class rows_t>
  DYNG_TRIAD_HD void begin_wedge(const rows_t& rows, int a, int b) {
    const row_ref ra = rows.row(a);
    const row_ref rb = rows.row(b);
    row_a = ra.begin();
    row_b = rb.begin();
    da = ra.size();
    db = rb.size();
    cab = intersect_size(row_a, da, row_b, db);
  }
  template <class rows_t, class bins_t>
  DYNG_TRIAD_HD void triangle(const rows_t& rows, int c, bins_t& bins) {
    const row_ref rc = rows.row(c);
    int cbc, cca, g;
    venn3(row_a, da, row_b, db, rc.begin(), rc.size(), cbc, cca, g);
    // The class is invariant under permutations of (a, b, c); the bins handle refuses -1 (no
    // class)
    bins.add(compute_motif_index(da, db, rc.size(), cab, cbc, cca, g), 1ULL);
  }
  template <class bins_t>
  DYNG_TRIAD_HD void flush(bins_t&) {}
};

}  // namespace dyng::detail::triad

namespace dyng::detail::local_patterns {

/// motif_op is one of triad_count's built-in patterns (local_patterns_kernels.cuh's bins handle).
template <>
struct builtin_op<triad::motif_op> : std::true_type {};

}  // namespace dyng::detail::local_patterns
