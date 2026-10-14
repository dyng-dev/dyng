// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/integrity.hpp
// Derived from MOSP_ESCHER@4b86159:escher/include/integrity.hpp (checkSubtreeAvail)
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
// Returns the number of violations (a long long, MOSP_ESCHER's; ESCHER-GPU returned an int); details
// of the first few go to `log`. Order-insensitive rows are compared as sorted multisets on both
// sides (the expected rows need not be sorted).
long long check_tree_rows(const cbst_context& ctx, const std::vector<std::vector<int>>& expected,
                          bool order_insensitive, const char* name, std::ostream& log);

// MOSP_ESCHER's availability check (checkSubtreeAvail): subtree_avail[i] must equal avail[i] +
// subtree_avail[2i+1] + subtree_avail[2i+2] for every node, so that the root holds the number of
// reusable (deleted) slots. Returns the number of inconsistent nodes.
long long check_subtree_avail(const cbst_context& ctx, const char* name, std::ostream& log);

// dynG's checks of the index arrays, which neither original has (M8a step 4): every avail flag is
// 0 or 1; the keys array (by in-order rank) is strictly increasing and holds the key of the node
// at the heap position of each rank, and the row offsets array that node's first segment; the
// child and parent pointers are the heap layout. Returns the number of violations.
long long check_tree_index(const cbst_context& ctx, const char* name, std::ostream& log);

}  // namespace dyng::detail::escher
