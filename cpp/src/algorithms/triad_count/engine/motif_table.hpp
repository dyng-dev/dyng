// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/motif_table.hpp
/**
 * @file motif_table.hpp
 * @brief The class of a hyperedge triple by its 7-bit Venn-region pattern: the h-motif id 1..30, 0
 *        for no triad (ESCHER-GPU's ESCHER_MOTIF_ID_TO_INDEX).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

namespace dyng::detail::triad {

// Maps the 7-bit Venn-region pattern of a hyperedge triple to its h-motif
// id (1..30; 0 = not a valid triple). Bits: 0..2 = vertices only in a / b /
// c, 3 = only in a and b, 4 = only in b and c, 5 = only in c and a,
// 6 = in all three. The table is invariant under permutations of (a, b, c).
// Shared by the GPU kernels (kernel/motif_utils.cuh) and the CPU reference.
#define DYNG_TRIAD_MOTIF_ID_TO_INDEX                                                       \
  {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  \
   0,  0,  21, 23, 22, 24, 23, 25, 24, 26, 0,  0,  0,  0,  0,  0,  0,  0,  21, 22, 23, 24, \
   23, 24, 25, 26, 21, 23, 23, 25, 22, 24, 24, 26, 27, 28, 28, 29, 28, 29, 29, 30, 1,  2,  \
   2,  3,  2,  3,  3,  4,  5,  6,  6,  8,  7,  9,  9,  10, 5,  7,  6,  9,  6,  9,  8,  10, \
   11, 13, 12, 14, 13, 15, 14, 16, 5,  6,  7,  9,  6,  8,  9,  10, 11, 12, 13, 14, 13, 14, \
   15, 16, 11, 13, 13, 15, 12, 14, 14, 16, 17, 18, 18, 19, 18, 19, 19, 20}

}  // namespace dyng::detail::triad
