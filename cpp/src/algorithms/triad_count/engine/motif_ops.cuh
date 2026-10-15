// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/motif_utils.cuh
/**
 * @file motif_ops.cuh
 * @brief The 30-bin h-motif classification of a line-graph triangle: the class map on the device,
 *        the class index from the region counts and the wedge operation MotifOp (ESCHER-GPU's
 *        kernel/motif_utils.cuh).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

#include "algorithms/triad_count/engine/motif_table.hpp"
#include "algorithms/triad_count/engine/wedge.cuh"
#include "util/device_error_flags.hpp"

#include <climits>

namespace dyng::detail::triad {

// Header-only device data and helpers to avoid cross-TU device linking
static __device__ int id_to_index[128] = DYNG_TRIAD_MOTIF_ID_TO_INDEX;

// Inline device helper to compute motif index [0..29] from region counts
static inline __device__ int compute_motif_index(int deg_a, int deg_b, int deg_c, int c_ab,
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
  int index = id_to_index[motif_id] - 1;  // [0..29]
  return index;
}

// 30-bin classification of a line-graph triangle (a, b, c) for the wedge
// kernels (kernel/wedge.cuh). The rows, degrees and |A ∩ B| of the wedge
// (a, b) are computed once per wedge, and (|B ∩ C|, |C ∩ A|, |A ∩ B ∩ C|)
// come from a single merge over the three H2V rows.
struct motif_op {
  static constexpr int num_bins = 30;
  /// dynG: the bits a triangle of no class sets in the device error word.
  static constexpr unsigned long long unclassified =
      static_cast<unsigned long long>(bits_of(device_error::internal));
  escher::row_view h2v;
  unsigned long long* error;  // dynG: the device error word (a triangle of no class raises it)
  const int* row_a;
  const int* row_b;
  int da, db, cab;

  __device__ void begin_wedge(int a, int b) {
    row_a = h2v.flat + h2v.loc[a];
    row_b = h2v.flat + h2v.loc[b];
    da = h2v.len[a];
    db = h2v.len[b];
    cab = wedge::intersect_size(row_a, da, row_b, db);
  }
  __device__ void triangle(int c, unsigned long long* bins) {
    const int* row_c = h2v.flat + h2v.loc[c];
    int dc = h2v.len[c];
    int cbc, cca, g;
    wedge::venn3(row_a, da, row_b, db, row_c, dc, cbc, cca, g);
    // The class is invariant under permutations of (a, b, c)
    const int bin = compute_motif_index(da, db, dc, cab, cbc, cca, g);
    if (bin < 0) {  // dynG: a pattern of no class raises the error word (ESCHER-GPU: bins[-1])
      atomicOr(error, unclassified);
      return;
    }
    atomicAdd(&bins[bin], 1ULL);
  }
  __device__ void flush(unsigned long long*) {}
};

inline motif_op make_motif_op(const escher::row_view& h2v, unsigned long long* error) {
  motif_op op{};
  op.h2v = h2v;
  op.error = error;
  return op;
}

}  // namespace dyng::detail::triad
