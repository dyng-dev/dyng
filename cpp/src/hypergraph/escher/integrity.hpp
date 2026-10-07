// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/integrity.hpp
/**
 * @file integrity.hpp
 * @brief The CBST integrity checker (a test oracle of the store): rows by BST search, reachability,
 *        tail metadata, disjoint segments.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"

#include <iosfwd>
#include <vector>

namespace dyng::detail::escher {

// Data-structure integrity checker (test oracle). Copies a CBST to the host
// and compares it with the expected rows:
//  - every expected non-empty row `key` must be found by a BST search from
//    the root and hold exactly the expected values (in order, or as a sorted
//    multiset when `order_insensitive`), following overflow chains;
//  - keys with an empty expected row must be absent, deleted or empty;
//  - every live node must satisfy the tail-segment invariants used by fill
//    and unfill: `occupancy` equals the number of live entries in the tail
//    segment and does not exceed `tail_capacity`, and no live entry lies in
//    the free part of the tail segment;
//  - the payload segments of different rows must not overlap.
// Returns the number of violations; details of the first few go to `log`.
int check_tree_rows(const cbst_context& ctx, const std::vector<std::vector<int>>& expected,
                    bool order_insensitive, const char* name, std::ostream& log);

}  // namespace dyng::detail::escher
