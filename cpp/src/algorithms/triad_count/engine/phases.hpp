// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (countWedgeShape's early outs) and
// src/driver.cu (the subtract and add phases' signs)
/**
 * @file phases.hpp
 * @brief What the CUDA engine (counters.cu, cuda.cu) and the host engine (host_engine.hpp) of
 *        triad_count share besides the operations and the exactly-once rule (ownership.hpp): the
 *        outcome of one count (its early outs), the sign of each phase and the source keys of the
 *        full count. Host code (the CUDA engine reads it on the host).
 *
 * The recorded triad mutations of M9a (PLAN Section 8.4; parity/mutate.py, suite triad_count)
 * that act here act on both engines at once: DYNG_MUTATION_TRIAD_WRONG_BIN_SIGN (the subtract
 * phase adds) and DYNG_MUTATION_TRIAD_TYPE1_UNBOUNDED_KEYS (the full count of the keys shape,
 * type 1's, runs over the largest id bound the hypergraph has had, the line graph's keys, instead
 * of the h2v view's).
 */
#pragma once

#include "framework/policies.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/experimental/local_patterns.hpp>

namespace dyng::detail::triad {

namespace wedge {

/// What one count call did (the launch counters per mode and shape, M9a criterion 9): no
/// sources (the first early out), no wedges (the second), or a counting pass (a kernel on cuda).
enum class launch { no_sources, no_wedges, kernel };

}  // namespace wedge

/**
 * @brief The sign a phase counts with: count(-) on G_t subtracts, count(+) on G_{t+1} adds
 *        (ESCHER-GPU's subtract() and add()).
 * @param[in] s The phase's sign as the enactor passes it.
 * @return The sign the engines apply.
 */
inline framework::sign phase_sign(framework::sign s) noexcept {
#if defined(DYNG_MUTATION_TRIAD_WRONG_BIN_SIGN)
  // MUTATION (M9a criterion 11, triad_wrong_bin_sign): count(-) adds.
  (void)s;
  return framework::sign::plus;
#else
  return s;
#endif
}

/**
 * @brief The source keys of a full count: 1..views.h2v.max_key for the keys shape (every hyperedge
 *        id below the bound: type 1, and the Ops of shape keys; the h2v view covers the keys up to
 *        the id bound, which can go down), 1..views.h2h.max_key for the wedge shapes (ESCHER-GPU's
 *        numHyperedgeKeys, the largest bound so far).
 * @param[in] s     The pattern's shape.
 * @param[in] views The row views.
 * @return The number of source keys.
 */
inline int full_count_keys(experimental::shape s, const hyper::row_views& views) noexcept {
  if (s == experimental::shape::keys) {
#if defined(DYNG_MUTATION_TRIAD_TYPE1_UNBOUNDED_KEYS)
    // MUTATION (M9a criterion 11, triad_type1_unbounded_keys): the largest bound so far.
    return views.has_h2h && views.h2h.max_key > views.h2v.max_key ? views.h2h.max_key
                                                                  : views.h2v.max_key;
#else
    return views.h2v.max_key;
#endif
  }
  return views.has_h2h ? views.h2h.max_key : 0;
}

}  // namespace dyng::detail::triad
