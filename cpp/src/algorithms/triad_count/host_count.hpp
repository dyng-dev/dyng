// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (forEachWedge, TriangleBody, intersectSize,
// venn3), kernel/motif_utils.cuh (compute_motif_index, MotifOp), kernel/count_ops.cuh (Choose3Op),
// src/type2.cu (Type2Op), src/type3.cu (Type3Op)
/**
 * @file host_count.hpp
 * @brief The full count of the host backends (sequential.cpp, openmp.cpp): ESCHER-GPU's full count
 *        as a loop over the source keys of the internal row views (one source's out-wedges at a
 *        time), with the operations of its kernels on the host.
 *
 * The same semantics as the device engine's full mode (engine/wedge.cuh): for the triangle
 * patterns (h_motif, stathyper_type3) every out-wedge (a, b), b > a, of the line graph is closed
 * into its triangles (a, b, k) by the merge of the rows of a after b and of b after b (k > b); for
 * stathyper_type2 every out-wedge is a pair; stathyper_type1 sums C(|e|, 3) over the h2v keys
 * 1..views.h2v.max_key. The rows are the store's encoding (h2v members + 1, h2h neighbour keys);
 * the h2h keys above the id bound have empty rows. This is the recount the host backends run in
 * M9a step 3 (stats::fallback_used); step 4 replaces it by the incremental host engine.
 */
#pragma once

#include "algorithms/triad_count/engine/motif_table.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/error.hpp>
#include <dyng/triad_count.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace dyng::detail::triad_host {

/// The h-motif id (1..30, 0: not a triad) of each 7-bit Venn pattern (ESCHER-GPU's table).
inline constexpr int id_to_index[128] = DYNG_TRIAD_MOTIF_ID_TO_INDEX;

/// |A n B| of two sorted rows (intersectSize).
inline int intersect_size(const int* a, int na, const int* b, int nb) noexcept {
  int x = 0;
  int y = 0;
  int c = 0;
  while (x < na && y < nb) {
    const int va = a[x];
    const int vb = b[y];
    c += va == vb ? 1 : 0;
    x += va <= vb ? 1 : 0;
    y += vb <= va ? 1 : 0;
  }
  return c;
}

/// One merge over rows a, b, c: |B n C|, |C n A| and |A n B n C| (venn3). The rows end at
/// INT_MAX: compute() refuses the classifying patterns on hypergraphs whose largest vertex id
/// would be encoded as INT_MAX.
inline void venn3(const int* a, int na, const int* b, int nb, const int* c, int nc, int& cbc,
                  int& cca, int& g) noexcept {
  int x = 0;
  int y = 0;
  int z = 0;
  cbc = 0;
  cca = 0;
  g = 0;
  while (true) {
    const int va = x < na ? a[x] : INT_MAX;
    const int vb = y < nb ? b[y] : INT_MAX;
    const int vc = z < nc ? c[z] : INT_MAX;
    const int m = va < vb ? (va < vc ? va : vc) : (vb < vc ? vb : vc);
    if (m == INT_MAX) {
      break;
    }
    const int ia = va == m ? 1 : 0;
    const int ib = vb == m ? 1 : 0;
    const int ic = vc == m ? 1 : 0;
    cbc += ib & ic;
    cca += ic & ia;
    g += ia & ib & ic;
    x += ia;
    y += ib;
    z += ic;
  }
}

/// The h-motif bin (0..29) of a triad from its region counts (compute_motif_index), or -1.
inline int motif_bin(int deg_a, int deg_b, int deg_c, int c_ab, int c_bc, int c_ca,
                     int g_abc) noexcept {
  const int a = deg_a - (c_ab + c_ca) + g_abc;
  const int b = deg_b - (c_bc + c_ab) + g_abc;
  const int c = deg_c - (c_ca + c_bc) + g_abc;
  const int d = c_ab - g_abc;
  const int e = c_bc - g_abc;
  const int f = c_ca - g_abc;
  const int id = (a > 0 ? 1 : 0) | (b > 0 ? 2 : 0) | (c > 0 ? 4 : 0) | (d > 0 ? 8 : 0) |
                 (e > 0 ? 16 : 0) | (f > 0 ? 32 : 0) | (g_abc > 0 ? 64 : 0);
  return id_to_index[id] - 1;
}

/// bin += v, or capacity_error past 2^64 - 1.
inline void add_checked(std::uint64_t& bin, std::uint64_t v) {
  if (__builtin_add_overflow(bin, v, &bin)) {
    throw capacity_error("dyng: triad_count: a count exceeds 2^64 - 1");
  }
}

/// The number of source keys of the full count: h2v's keys for type 1, h2h's otherwise.
inline int num_sources(triad_count::pattern kind, const hyper::row_views& views) noexcept {
  if (kind == triad_count::pattern::stathyper_type1) {
    return views.h2v.max_key;
  }
  return views.has_h2h ? views.h2h.max_key : 0;
}

/**
 * @brief The full count's contribution of one source key `a` (1-based) to `bins`.
 * @param[in]     kind  The pattern.
 * @param[in]     views The row views (host memory).
 * @param[in]     a     The source key.
 * @param[in,out] bins  The bins.
 * @throws internal_error for a triad of no class; capacity_error past 2^64 - 1.
 */
inline void count_source(triad_count::pattern kind, const hyper::row_views& views, int a,
                         triad_bins& bins) {
  const escher::row_view& h2v = views.h2v;
  if (kind == triad_count::pattern::stathyper_type1) {
    const auto n = static_cast<std::int64_t>(h2v.len[a]);
    if (n >= 3) {
      add_checked(bins[0], static_cast<std::uint64_t>(n * (n - 1) * (n - 2) / 6));
    }
    return;
  }
  const escher::row_view& adj = views.h2h;
  const int end_a = adj.loc[a] + adj.len[a];
  for (int p = adj.out[a]; p < end_a; ++p) {
    const int b = adj.flat[p];
    const int* row_a = h2v.flat + h2v.loc[a];
    const int* row_b = h2v.flat + h2v.loc[b];
    const int da = h2v.len[a];
    const int db = h2v.len[b];
    const int cab = intersect_size(row_a, da, row_b, db);
    if (kind == triad_count::pattern::stathyper_type2) {
      const std::int64_t x = cab;
      const std::int64_t y = da - x;
      const std::int64_t z = db - x;
      if (x >= 2 && y > 0 && z > 0) {
        add_checked(bins[0], static_cast<std::uint64_t>(x * (x - 1) / 2 * y * z));
      }
      continue;
    }
    // The triangles (a, b, k), k > b: the rows of a after b and of b after b.
    int x = p + 1;
    int y = adj.out[b];
    const int end_b = adj.loc[b] + adj.len[b];
    while (x < end_a && y < end_b) {
      const int ka = adj.flat[x];
      const int kb = adj.flat[y];
      if (ka == kb) {
        const int* row_c = h2v.flat + h2v.loc[ka];
        const int dc = h2v.len[ka];
        int cbc = 0;
        int cca = 0;
        int g = 0;
        venn3(row_a, da, row_b, db, row_c, dc, cbc, cca, g);
        if (kind == triad_count::pattern::h_motif) {
          const int bin = motif_bin(da, db, dc, cab, cbc, cca, g);
          if (bin < 0) {
            DYNG_FAIL("triad_count: the triangle (", a - 1, ", ", b - 1, ", ", ka - 1,
                      ") of the line graph has no h-motif class");
          }
          add_checked(bins[static_cast<std::size_t>(bin)], 1);
        } else {
          const std::int64_t u = cab - g;
          const std::int64_t v = cbc - g;
          const std::int64_t w = cca - g;
          if (u > 0 && v > 0 && w > 0) {
            add_checked(bins[0], static_cast<std::uint64_t>(u * v * w));
          }
        }
        ++x;
        ++y;
      } else if (ka < kb) {
        ++x;
      } else {
        ++y;
      }
    }
  }
}

}  // namespace dyng::detail::triad_host
