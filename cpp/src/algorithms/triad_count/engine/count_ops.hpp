// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/count_ops.cuh
/**
 * @file count_ops.hpp
 * @brief The per-key counting operation of StatHyper's type 1: sum over the keys of C(|row|, 3)
 *        (ESCHER-GPU's Choose3Op).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 *
 * dynG (the Op form, c6): a host-and-device struct that reads the row sizes through the rows
 * handle and adds to the bins handle. dynG (ADR 0057): exact C(|e|, 3) and a checked sum.
 */
#pragma once

// Small per-thread counting operations for the kernels in wedge.cuh. Each
// keeps a register partial sum that flush() adds to the block bins.

#include "algorithms/triad_count/engine/exact.hpp"
#include "algorithms/triad_count/engine/sets.hpp"

namespace dyng::detail::triad {

// sum over keys of C(|row|, 3) (StatHyper type 1 / coarse inner triangles).
// dynG (ADR 0057): C(|row|, 3) exact (exact.hpp; ESCHER-GPU's signed n * (n - 1) * (n - 2)
// overflows from 2,097,152 members), the sum checked; flush() reports a count past 2^64 - 1.
struct choose3_op {
  static constexpr int num_bins = 1;
  static constexpr bool unit_counts = false;  // dynG: adds C(|row|, 3), not 1 (wedge.cuh)
  unsigned long long sum;
  bool over;  // dynG: a C(|row|, 3) or the sum passed 2^64 - 1

  template <class rows_t, class bins_t>
  DYNG_TRIAD_HD void key(const rows_t& rows, int a, bins_t&) {
    unsigned long long c;
    if (choose3_overflows(rows.size(a), c) || add_overflows(sum, c, sum)) over = true;
  }
  template <class bins_t>
  DYNG_TRIAD_HD void flush(bins_t& bins) {
    if (over) {
      bins.overflow();
    } else if (sum) {
      bins.add(0, sum);
    }
  }
};

}  // namespace dyng::detail::triad
