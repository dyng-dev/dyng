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
 *        (f, b, k). On keys (id + 1), on the host and the device: the CUDA kernels and the host
 *        work of every pattern apply the same two predicates, so the recorded mutations
 *        DYNG_MUTATION_TRIAD_SKIP_FRONTIER_FILTER_B and _K (PLAN Section 8.4: a skipped frontier
 *        filter) act on both engines.
 *
 * Since M9b (ADR 0058) the predicates are the local-pattern engine's
 * (<dyng/experimental/detail/local_patterns_engine.hpp>, with their mutation hooks), which the
 * built-in patterns and the Ops of users share; the definitions moved there unchanged.
 */
#pragma once

#include "algorithms/triad_count/engine/sets.hpp"

#include <dyng/experimental/detail/local_patterns_engine.hpp>

namespace dyng::detail::triad {

using local_patterns::min_member_skips_closing;
using local_patterns::min_member_skips_wedge;

}  // namespace dyng::detail::triad
