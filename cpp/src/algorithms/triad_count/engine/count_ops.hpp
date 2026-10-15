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
 * handle and adds to the bins handle.
 */
#pragma once

// Small per-thread counting operations for the kernels in wedge.cuh. Each
// keeps a register partial sum that flush() adds to the block bins.

#include "algorithms/triad_count/engine/sets.hpp"

namespace dyng::detail::triad {

// sum over keys of C(|row|, 3) (StatHyper type 1 / coarse inner triangles).
struct choose3_op {
  static constexpr int num_bins = 1;
  unsigned long long sum;

  template <class rows_t, class bins_t>
  DYNG_TRIAD_HD void key(const rows_t& rows, int a, bins_t&) {
    long long n = rows.size(a);
    if (n >= 3) sum += static_cast<unsigned long long>(n * (n - 1) * (n - 2) / 6);
  }
  template <class bins_t>
  DYNG_TRIAD_HD void flush(bins_t& bins) {
    if (sum) bins.add(0, sum);
  }
};

}  // namespace dyng::detail::triad
