// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (intersectSize, venn3, intersects3)
/**
 * @file sets.hpp
 * @brief The rows the counting operations read and the set primitives on them (ESCHER-GPU's
 *        intersectSize, venn3 and intersects3), on the host and the device: the names of the
 *        triad engine for the local-pattern engine's installed ones.
 *
 * The Op form of the engine (c6 of the port, ADR 0053): an operation sees 0-based hyperedge ids
 * and a rows handle (hyperedge_rows) that gives the members of an id as a row_ref; the store's
 * encoding (the row of id i at key i + 1, its members + 1, sorted) stays inside the handle and the
 * set primitives, which only compare members. Since M9b (ADR 0058) the handle, the span and the
 * primitives are the experimental extension point's (<dyng/experimental/local_patterns.hpp>:
 * experimental::hyperedge_rows, experimental::member_span, detail::local_patterns), so the
 * built-in operations and the Ops of users read the same rows; the definitions moved there
 * unchanged. Host-compilable: the host engine reads the same rows with the same operations.
 */
#pragma once

#include <dyng/experimental/local_patterns.hpp>

#include <climits>

/// A function of the engine inlined on the host and the device.
#define DYNG_TRIAD_INLINE DYNG_HD_INLINE
/// A member of a counting operation (host and device).
#define DYNG_TRIAD_HD DYNG_HD

namespace dyng::detail::triad {

/// The members of one hyperedge: size() sorted values from begin() (the store's encoding).
using row_ref = experimental::member_span;

/// The rows handle of the operations: the members of a hyperedge by its 0-based id (the h2v row
/// view: the row of id i is at key i + 1).
using hyperedge_rows = experimental::hyperedge_rows;

using local_patterns::intersect_size;
using local_patterns::intersects3;
using local_patterns::max_of;
using local_patterns::min_of;
using local_patterns::venn3;

/// The rows handle of a state's h2v view (flat, loc, len by key).
DYNG_TRIAD_INLINE hyperedge_rows make_rows(const int* flat, const int* loc, const int* len) {
  return local_patterns::rows_factory::make(flat, loc, len);
}

}  // namespace dyng::detail::triad
