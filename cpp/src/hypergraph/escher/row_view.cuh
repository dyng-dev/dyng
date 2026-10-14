// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:include/row_view.hpp
/**
 * @file row_view.cuh
 * @brief The key-indexed row view of a CBST (the locator): offset, length and the first entry above
 *        the key of every row.
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"
#include "hypergraph/row_views.hpp"

#include <cstddef>

namespace dyng::detail::escher {

// The row view itself (struct row_view) is in hypergraph/row_views.hpp (host-only: the host
// stores of M8b give the same view of their rows; ADR 0042).

// Owns the arrays behind a row_view.
class row_locator {
 public:
  row_locator() = default;
  ~row_locator() = default;
  row_locator(const row_locator&) = delete;
  row_locator& operator=(const row_locator&) = delete;

  // Rebuilds the view of `tree` for keys 1..max_key. Every live row must be a
  // single contiguous segment (H2V, H2H and V2V rows are). With with_out the
  // rows must be sorted and out[] is computed as well.
  void build(const cbst_context& tree, int max_key, bool with_out);
  // Builds the view over a sorted copy of the rows of `tree` (rows may be
  // chained and unsorted, as V2H rows after fill/unfill). The tree itself
  // is not modified.
  void build_sorted_copy(const cbst_context& tree, int max_key);
  const row_view& view() const {
    return view_;
  }
  size_t device_bytes() const {
    return static_cast<size_t>(capacity_) * 3 * sizeof(int) + copy_bytes_;
  }

 private:
  // The arrays (from the memory resource of the tree's resources) and their device pointers.
  buffer<int> loc_storage_, len_storage_, out_storage_, copy_storage_;
  int* loc_ = nullptr;
  int* len_ = nullptr;
  int* out_ = nullptr;
  int* copy_ = nullptr;  // payload of the sorted copy
  size_t copy_bytes_ = 0;
  int capacity_ = 0;
  row_view view_;
};

}  // namespace dyng::detail::escher
