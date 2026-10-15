// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/type2.cu
// Derived from ESCHER-GPU@abcf9b2:src/type3.cu
/**
 * @file stathyper_ops.cuh
 * @brief The wedge operations of StatHyper's types 2 and 3: C(a, 2) * b * c of an adjacent pair, x
 *        * y * z of a triangle's pairwise-only overlaps (ESCHER-GPU's Type2Op and Type3Op).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

#include "algorithms/triad_count/engine/wedge.cuh"

namespace dyng::detail::triad {

// ---------------------------------------------------------------------------
// Type-2 Motif:  For each pair (i,j) with i<j sharing >=2 vertices,
// count C(a,2)*b*c where a = |intersection|, b = |e_i| - a, c = |e_j| - a
// Uses H2V and H2H. One thread per adjacent pair (wedge of H2H).
// ---------------------------------------------------------------------------
struct type2_op {
  static constexpr int num_bins = 1;
  escher::row_view h2v;
  unsigned long long sum;

  __device__ void pair(int i, int j, unsigned long long*) {
    const int* row_i = h2v.flat + h2v.loc[i];
    const int* row_j = h2v.flat + h2v.loc[j];
    int deg_i = h2v.len[i], deg_j = h2v.len[j];
    long long a = wedge::intersect_size(row_i, deg_i, row_j, deg_j);
    long long b = deg_i - a, c = deg_j - a;
    if (a >= 2 && b > 0 && c > 0) sum += static_cast<unsigned long long>(a * (a - 1) / 2 * b * c);
  }
  __device__ void flush(unsigned long long* bins) {
    if (sum) atomicAdd(&bins[0], sum);
  }
};

static type2_op make_type2_op(const escher::row_view& h2v) {
  type2_op op{};
  op.h2v = h2v;
  return op;
}
// ---------------------------------------------------------------------------
// Type-3 Motif:  For each triple (i,j,k) with i<j<k that form a triangle
// in H2H, compute x*y*z where:
//   A = con(i,j), B = con(j,k), C = con(i,k), G = group(i,j,k)
//   x = A - G, y = B - G, z = C - G
// Uses H2V and H2H. One thread per wedge of H2H (kernel/wedge.cuh).
// ---------------------------------------------------------------------------
struct type3_op {
  static constexpr int num_bins = 1;
  escher::row_view h2v;
  unsigned long long sum;
  const int *row_a, *row_b;
  int da, db, cab;

  __device__ void begin_wedge(int a, int b) {
    row_a = h2v.flat + h2v.loc[a];
    row_b = h2v.flat + h2v.loc[b];
    da = h2v.len[a];
    db = h2v.len[b];
    cab = wedge::intersect_size(row_a, da, row_b, db);
  }
  __device__ void triangle(int c, unsigned long long*) {
    int cbc, cca, g;
    wedge::venn3(row_a, da, row_b, db, h2v.flat + h2v.loc[c], h2v.len[c], cbc, cca, g);
    long long x = cab - g, y = cbc - g, z = cca - g;
    if (x > 0 && y > 0 && z > 0) sum += static_cast<unsigned long long>(x * y * z);
  }
  __device__ void flush(unsigned long long* bins) {
    if (sum) atomicAdd(&bins[0], sum);
  }
};

static type3_op make_type3_op(const escher::row_view& h2v) {
  type3_op op{};
  op.h2v = h2v;
  return op;
}

}  // namespace dyng::detail::triad
