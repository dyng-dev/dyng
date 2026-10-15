// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu
/**
 * @file stathyper_ops.hpp
 * @brief The wedge operations of StatHyper's types 2 and 3: C(a, 2) * b * c of an adjacent pair, x
 *        * y * z of a triangle's pairwise-only overlaps (ESCHER-GPU's Type2Op and Type3Op).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 *
 * dynG (the Op form, c6): host-and-device structs that read their rows through the rows handle
 * and add to the bins handle.
 */
#pragma once

#include "algorithms/triad_count/engine/sets.hpp"

namespace dyng::detail::triad {

// ---------------------------------------------------------------------------
// Type-2 Motif:  For each pair (i,j) with i<j sharing >=2 vertices,
// count C(a,2)*b*c where a = |intersection|, b = |e_i| - a, c = |e_j| - a
// Uses H2V and H2H. One thread per adjacent pair (wedge of H2H).
// ---------------------------------------------------------------------------
struct type2_op {
  static constexpr int num_bins = 1;
  unsigned long long sum;

  template <class rows_t, class bins_t>
  DYNG_TRIAD_HD void pair(const rows_t& rows, int i, int j, bins_t&) {
    const row_ref ri = rows.row(i);
    const row_ref rj = rows.row(j);
    const int* row_i = ri.data;
    const int* row_j = rj.data;
    int deg_i = ri.size, deg_j = rj.size;
    long long a = intersect_size(row_i, deg_i, row_j, deg_j);
    long long b = deg_i - a, c = deg_j - a;
    if (a >= 2 && b > 0 && c > 0) sum += static_cast<unsigned long long>(a * (a - 1) / 2 * b * c);
  }
  template <class bins_t>
  DYNG_TRIAD_HD void flush(bins_t& bins) {
    if (sum) bins.add(0, sum);
  }
};

// ---------------------------------------------------------------------------
// Type-3 Motif:  For each triple (i,j,k) with i<j<k that form a triangle
// in H2H, compute x*y*z where:
//   A = con(i,j), B = con(j,k), C = con(i,k), G = group(i,j,k)
//   x = A - G, y = B - G, z = C - G
// Uses H2V and H2H. One thread per wedge of H2H (kernel/wedge.cuh).
// ---------------------------------------------------------------------------
struct type3_op {
  static constexpr int num_bins = 1;
  unsigned long long sum;
  const int *row_a, *row_b;
  int da, db, cab;

  template <class rows_t>
  DYNG_TRIAD_HD void begin_wedge(const rows_t& rows, int a, int b) {
    const row_ref ra = rows.row(a);
    const row_ref rb = rows.row(b);
    row_a = ra.data;
    row_b = rb.data;
    da = ra.size;
    db = rb.size;
    cab = intersect_size(row_a, da, row_b, db);
  }
  template <class rows_t, class bins_t>
  DYNG_TRIAD_HD void triangle(const rows_t& rows, int c, bins_t&) {
    const row_ref rc = rows.row(c);
    int cbc, cca, g;
    venn3(row_a, da, row_b, db, rc.data, rc.size, cbc, cca, g);
    long long x = cab - g, y = cbc - g, z = cca - g;
    if (x > 0 && y > 0 && z > 0) sum += static_cast<unsigned long long>(x * y * z);
  }
  template <class bins_t>
  DYNG_TRIAD_HD void flush(bins_t& bins) {
    if (sum) bins.add(0, sum);
  }
};

}  // namespace dyng::detail::triad
