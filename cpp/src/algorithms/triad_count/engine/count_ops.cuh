// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/count_ops.cuh
/**
 * @file count_ops.cuh
 * @brief The per-key counting operation of StatHyper's type 1: sum over the keys of C(|row|, 3)
 *        (ESCHER-GPU's Choose3Op).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

// Small per-thread counting operations for the kernels in wedge.cuh. Each
// keeps a register partial sum that flush() adds to the block bins.

#include "algorithms/triad_count/engine/wedge.cuh"

namespace dyng::detail::triad {

// sum over keys of C(|row|, 3) (StatHyper type 1 / coarse inner triangles).
struct choose3_op {
  static constexpr int num_bins = 1;
  escher::row_view rows;
  unsigned long long sum;

  __device__ void key(int a, unsigned long long*) {
    long long n = rows.len[a];
    if (n >= 3) sum += static_cast<unsigned long long>(n * (n - 1) * (n - 2) / 6);
  }
  __device__ void flush(unsigned long long* bins) {
    if (sum) atomicAdd(&bins[0], sum);
  }
};

inline choose3_op make_choose3_op(const escher::row_view& rows) {
  choose3_op op{};
  op.rows = rows;
  return op;
}

}  // namespace dyng::detail::triad
