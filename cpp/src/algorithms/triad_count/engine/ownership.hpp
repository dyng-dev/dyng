// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (forEachWedge's and TriangleBody's frontier
// filters) and include/frontier.hpp (the exactly-once rule)
/**
 * @file ownership.hpp
 * @brief The exactly-once rule of the frontier phases (ownership::min_member, invariant I2) as
 *        the engine applies it: each triad (pair) through the frontier is counted once, at its
 *        smallest frontier member f; from f the engine skips every other frontier member smaller
 *        than f, as the second hyperedge b of a wedge (f, b) and as the closing k of a triangle
 *        (f, b, k). On keys (id + 1), on the host and the device.
 */
#pragma once

#include "algorithms/triad_count/engine/sets.hpp"

namespace dyng::detail::triad {

/// Whether the wedge (f, b) of a frontier phase belongs to a smaller frontier member: b is in the
/// frontier and b < f (ESCHER-GPU's forEachWedge). No flags (a full count): never.
DYNG_TRIAD_INLINE bool min_member_skips_wedge(const unsigned char* in_frontier, int b, int f) {
  return in_frontier != nullptr && in_frontier[b] && b < f;
}

/// Whether the triangle (f, b, k) closed by k belongs to a smaller frontier member: k is in the
/// frontier and k < f (ESCHER-GPU's TriangleBody). No flags (a full count): never.
DYNG_TRIAD_INLINE bool min_member_skips_closing(const unsigned char* in_frontier, int k, int f) {
  return in_frontier != nullptr && in_frontier[k] && k < f;
}

}  // namespace dyng::detail::triad
