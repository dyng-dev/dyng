// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (forEachWedge, TriangleBody, PairBody,
// keyKernel: the full count's wedges as a loop)
/**
 * @file host_count.hpp
 * @brief The full count of the host backends (sequential.cpp, openmp.cpp): ESCHER-GPU's full count
 *        as a loop over the source keys of the internal row views, with the engine's operations
 *        (engine/count_ops.hpp, motif_ops.hpp, stathyper_ops.hpp: the Op form of c6).
 *
 * The same semantics as the device engine's full mode (engine/wedge.cuh): for the triangle
 * patterns (h_motif, stathyper_type3) every out-wedge (a, b), b > a, of the line graph is closed
 * into its triangles (a, b, k) by the merge of the rows of a after b and of b after b (k > b); for
 * stathyper_type2 every out-wedge is a pair; stathyper_type1 sums C(|e|, 3) over the h2v keys
 * 1..views.h2v.max_key. The rows are the store's encoding (h2v members + 1, h2h neighbour keys);
 * the h2h keys above the id bound have empty rows; the operations see 0-based ids. This is the
 * recount the host backends run in M9a step 3 (stats::fallback_used); step 4 adds the frontier
 * mode (the incremental host engine).
 */
#pragma once

#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/sets.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/error.hpp>
#include <dyng/triad_count.hpp>

#include <cstddef>
#include <cstdint>

namespace dyng::detail::triad_host {

/// bin += v, or capacity_error past 2^64 - 1.
inline void add_checked(std::uint64_t& bin, std::uint64_t v) {
  if (__builtin_add_overflow(bin, v, &bin)) {
    throw capacity_error("dyng: triad_count: a count exceeds 2^64 - 1");
  }
}

/// The bins handle of the host engine: a negative bin (a triad of no class) is an internal_error,
/// a count past 2^64 - 1 a capacity_error.
struct host_bins {
  triad_bins& bins;
  void add(int bin, unsigned long long value) const {
    if (bin < 0) {
      DYNG_FAIL("triad_count: a triangle of the line graph has a Venn pattern of no h-motif class");
    }
    add_checked(bins[static_cast<std::size_t>(bin)], value);
  }
};

/// The rows handle of a state's h2v view.
inline triad::hyperedge_rows rows_of(const hyper::row_views& views) {
  return triad::hyperedge_rows{views.h2v.flat, views.h2v.loc, views.h2v.len};
}

/// The number of source keys of the full count: h2v's keys for type 1, h2h's otherwise.
inline int num_sources(triad_count::pattern kind, const hyper::row_views& views) noexcept {
  if (kind == triad_count::pattern::stathyper_type1) {
    return views.h2v.max_key;
  }
  return views.has_h2h ? views.h2h.max_key : 0;
}

/// The triangles of the out-wedges of the source keys [first, last] (triangle_kernel's work).
template <typename op_t>
void count_triangles(const hyper::row_views& views, int first, int last, triad_bins& bins) {
  const escher::row_view& adj = views.h2h;
  const triad::hyperedge_rows rows = rows_of(views);
  host_bins handle{bins};
  op_t op{};
  for (int a = first; a <= last; ++a) {
    const int end_a = adj.loc[a] + adj.len[a];
    for (int p = adj.out[a]; p < end_a; ++p) {
      const int b = adj.flat[p];
      op.begin_wedge(rows, a - 1, b - 1);
      int x = p + 1;
      int y = adj.out[b];
      const int end_b = adj.loc[b] + adj.len[b];
      while (x < end_a && y < end_b) {
        const int ka = adj.flat[x];
        const int kb = adj.flat[y];
        if (ka == kb) {
          op.triangle(rows, ka - 1, handle);
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
  op.flush(handle);
}

/// The pairs of the out-wedges of the source keys [first, last] (pair_kernel's work).
template <typename op_t>
void count_pairs(const hyper::row_views& views, int first, int last, triad_bins& bins) {
  const escher::row_view& adj = views.h2h;
  const triad::hyperedge_rows rows = rows_of(views);
  host_bins handle{bins};
  op_t op{};
  for (int a = first; a <= last; ++a) {
    const int end_a = adj.loc[a] + adj.len[a];
    for (int p = adj.out[a]; p < end_a; ++p) {
      op.pair(rows, a - 1, adj.flat[p] - 1, handle);
    }
  }
  op.flush(handle);
}

/// The keys [first, last] (key_kernel's work).
template <typename op_t>
void count_keys(const hyper::row_views& views, int first, int last, triad_bins& bins) {
  const triad::hyperedge_rows rows = rows_of(views);
  host_bins handle{bins};
  op_t op{};
  for (int a = first; a <= last; ++a) {
    op.key(rows, a - 1, handle);
  }
  op.flush(handle);
}

/**
 * @brief The full count's contribution of the source keys [first, last] (1-based) to `bins`.
 * @param[in]     kind  The pattern.
 * @param[in]     views The row views (host memory).
 * @param[in]     first The first source key.
 * @param[in]     last  The last source key.
 * @param[in,out] bins  The bins.
 * @throws internal_error for a triad of no class; capacity_error past 2^64 - 1.
 */
inline void count_sources(triad_count::pattern kind, const hyper::row_views& views, int first,
                          int last, triad_bins& bins) {
  switch (kind) {
    case triad_count::pattern::h_motif:
      count_triangles<triad::motif_op>(views, first, last, bins);
      return;
    case triad_count::pattern::stathyper_type1:
      count_keys<triad::choose3_op>(views, first, last, bins);
      return;
    case triad_count::pattern::stathyper_type2:
      count_pairs<triad::type2_op>(views, first, last, bins);
      return;
    case triad_count::pattern::stathyper_type3:
      count_triangles<triad::type3_op>(views, first, last, bins);
      return;
  }
}

}  // namespace dyng::detail::triad_host
