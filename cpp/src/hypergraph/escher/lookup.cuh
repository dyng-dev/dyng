// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lookup.cuh
 * @brief The node an append (fill) or remove_values (unfill) item names, and the rule for items
 *        that name an absent or deleted key (dynG, M8a step 4, ADR 0039).
 *
 * Neither original defined the rule: an absent key was skipped silently; a deleted key (which
 * keeps its node) was written to or compacted like a live one, changing a row nobody can read.
 * dynG skips both, counts them in the store's control block and raises the device error word,
 * which the host turns into internal_error at the next read point (the container never names
 * such a key: it is a broken invariant of the caller). No trace of either original has such an
 * item.
 */
#pragma once

#include "hypergraph/escher/cbst_store.hpp"
#include "util/device_error_flags.cuh"

#include <cstdint>

namespace dyng::detail::escher {

/**
 * @brief The live node of `key`, or nullptr for an absent or deleted key, which is then counted and
 *        raised (once per item: only the thread with `report` does it).
 * @param nodes   The tree (searched from the root, as every kernel of the store does).
 * @param avail   The deleted flags by heap position.
 * @param key     The item's key.
 * @param control The store's control block.
 * @param report  This thread reports a rejected item (one per item).
 * @return The node, or nullptr.
 */
__device__ inline cbst_node* find_item_node(cbst_node* nodes, const int* avail, int key,
                                            std::uint32_t* control, bool report) {
  cbst_node* current = nodes;
  while (current != nullptr && current->index != key)
    current = current->index > key ? current->left : current->right;
  if (current != nullptr && avail[current - nodes] == 0) return current;
  if (report) {
    raise_device_error(control + control_errors, device_error::internal);
    atomicAdd(reinterpret_cast<unsigned int*>(control + control_rejected), 1U);
  }
  return nullptr;
}

}  // namespace dyng::detail::escher
