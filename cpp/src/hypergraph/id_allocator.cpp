// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file id_allocator.cpp
 * @brief The host best-fit id allocator (ADR 0041, section 5).
 *
 * Every rule below is the store's (cpp/src/hypergraph/escher/README.md, "The operations" and "The
 * policies"); the comments name the original each one comes from. The allocator only keeps what
 * decides ids and the payload accounting: which keys are in the tree, which are deleted slots,
 * the slot lengths, the value counts and tail segments of the rows, the bump pointer and the
 * capacity. It never holds a value of a row (the host store does).
 */
#include "hypergraph/id_allocator.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <numeric>
#include <string>
#include <utility>

namespace dyng::detail::hyper {

namespace {

// Padding rules 1 and 3 (ESCHER-GPU's flatten2DVector, utils/flatten.cpp@abcf9b2, and
// nextMultipleOf4, kernel/device_utils.cuh@abcf9b2): the values, zeros, one terminator slot.
std::int64_t padded4(std::int64_t count) noexcept {
  return ((count + 4) / 4) * 4;
}

// Padding rule 2 (a surplus row of an insert, structure/operations.cu@abcf9b2 insertCBST):
// ceil(n / alignment) * alignment slots for the values and padding (0 for n = 0).
std::int64_t aligned(std::int64_t count, std::int64_t alignment) noexcept {
  return count <= 0 ? 0 : (count + alignment - 1) / alignment * alignment;
}

// The degree bin of an item of `size` values (thread < 32, warp < 1024, block otherwise).
int degree_bin(std::int64_t size) noexcept {
  return size < 32 ? 0 : (size < 1024 ? 1 : 2);
}

// One path counter for every non-empty bin of an operation.
void record_paths(const std::vector<std::int64_t>& sizes, std::array<std::int64_t, 3>& paths) {
  bool used[3] = {false, false, false};
  for (const std::int64_t s : sizes) used[degree_bin(s)] = true;
  for (int b = 0; b < 3; ++b) paths[static_cast<std::size_t>(b)] += used[b] ? 1 : 0;
}

}  // namespace

id_allocator::id_allocator(escher::store_policy policy, std::int64_t capacity, std::string name)
    : policy_(std::move(policy)), name_(std::move(name)), slots_(1), capacity_(capacity) {}

id_allocator::slot& id_allocator::at(std::int64_t key) {
  if (static_cast<std::size_t>(key) >= slots_.size()) {
    slots_.resize(static_cast<std::size_t>(key) + 1);
  }
  return slots_[static_cast<std::size_t>(key)];
}

id_allocator::key_state id_allocator::state_of(std::int64_t key) const noexcept {
  if (key < 1 || static_cast<std::size_t>(key) >= slots_.size()) return key_state::absent;
  return slots_[static_cast<std::size_t>(key)].state;
}

std::int32_t id_allocator::length(std::int64_t key) const noexcept {
  return state_of(key) == key_state::absent ? 0 : slots_[static_cast<std::size_t>(key)].length;
}

std::int32_t id_allocator::count(std::int64_t key) const noexcept {
  return state_of(key) == key_state::live ? slots_[static_cast<std::size_t>(key)].count : 0;
}

void id_allocator::construct(const std::vector<std::int32_t>& counts) {
  std::int64_t total = 0;
  for (const std::int32_t c : counts) total += padded4(c);
  DYNG_EXPECTS(total <= capacity_, "CBST ", name_, ": payload capacity ", capacity_,
               " is below the payload size ", total);
  slots_.assign(counts.size() + 1, slot{});
  for (std::size_t i = 0; i < counts.size(); ++i) {
    slot& s = slots_[i + 1];
    s.state = key_state::live;
    s.length = static_cast<std::int32_t>(padded4(counts[i]));
    s.count = counts[i];
    s.tail_capacity = s.length - 1;  // ESCHER-GPU's initTailMetadata: one segment
    s.occupancy = counts[i];
  }
  max_key_ = static_cast<std::int64_t>(counts.size());
  live_ = max_key_;
  dead_ = 0;
  bump_ = total;
}

void id_allocator::restore(const allocator_state& state) {
  slots_.assign(1, slot{});
  live_ = 0;
  dead_ = 0;
  max_key_ = 0;
  for (std::size_t i = 0; i < state.keys.size(); ++i) {
    slot& s = at(state.keys[i]);
    s.state = state.dead[i] != 0 ? key_state::dead : key_state::live;
    s.length = state.lengths[i];
    s.count = state.counts[i];
    s.tail_capacity = state.tail_capacity[i];
    s.occupancy = state.occupancy[i];
    if (s.state == key_state::live) {
      ++live_;
    } else {
      ++dead_;
    }
    max_key_ = std::max<std::int64_t>(max_key_, state.keys[i]);
  }
  bump_ = state.bump;
  capacity_ = state.capacity;
  growths_ = state.growths;
  compactions_ = state.compactions;
}

allocator_state id_allocator::state() const {
  allocator_state out;
  for (std::int64_t key = 1; key <= max_key_; ++key) {
    const slot& s = slots_[static_cast<std::size_t>(key)];
    if (s.state == key_state::absent) continue;
    out.keys.push_back(static_cast<std::int32_t>(key));
    out.dead.push_back(s.state == key_state::dead ? 1 : 0);
    out.lengths.push_back(s.length);
    out.counts.push_back(s.state == key_state::live ? s.count : 0);
    out.tail_capacity.push_back(s.tail_capacity);
    out.occupancy.push_back(s.occupancy);
  }
  out.bump = bump_;
  out.capacity = capacity_;
  out.growths = growths_;
  out.compactions = compactions_;
  return out;
}

void id_allocator::ensure_capacity(std::int64_t needed) {
  if (needed <= capacity_) return;
  // The policy's growth (ESCHER-GPU's ensurePayloadCapacity; capacity_error without growth).
  capacity_ = policy_.grown_capacity(needed, capacity_, name_);
  ++growths_;
}

// The hysteresis of dyng_default() (ADR 0040): after an operation that compacted at a trigger
// point, grow while fewer than capacity / headroom entries are free.
void id_allocator::keep_headroom_after_compaction(std::int64_t compactions_before) {
  const std::int64_t headroom = policy_.compaction_headroom;
  if (headroom <= 0 || compactions_ == compactions_before ||
      policy_.growth == escher::store_policy::growth_rule::none) {
    return;
  }
  const std::int64_t slack = capacity_ / headroom;
  if (capacity_ - bump_ >= slack) return;
  const std::int64_t needed =
      std::min(bump_ + slack, std::min<std::int64_t>(policy_.max_payload, INT_MAX));
  ensure_capacity(needed);
}

std::int64_t id_allocator::compact() {
  // MOSP_ESCHER's compactCBST (escher/structure/operations.cu@4b86159:289-339): every row one
  // segment in key order; a live row of c values takes ((c + 4) / 4) * 4 slots, a deleted slot
  // keeps its first segment of `length` slots, emptied.
  if (live_ + dead_ == 0) return bump_;
  std::int64_t total = 0;
  for (std::int64_t key = 1; key <= max_key_; ++key) {
    const slot& s = slots_[static_cast<std::size_t>(key)];
    if (s.state == key_state::live) {
      total += padded4(s.count);
    } else if (s.state == key_state::dead) {
      total += s.length;
    }
  }
  if (total > INT_MAX) {
    throw capacity_error("dyng: CBST " + name_ + ": the compacted rows need " +
                         std::to_string(total) +
                         " payload slots, more than the int32 offsets of the CBST hold");
  }
  ensure_capacity(total);  // nothing has changed when this throws
  for (std::int64_t key = 1; key <= max_key_; ++key) {
    slot& s = slots_[static_cast<std::size_t>(key)];
    if (s.state == key_state::live) {
      s.length = static_cast<std::int32_t>(padded4(s.count));
      s.tail_capacity = s.length - 1;
      s.occupancy = s.count;
    } else if (s.state == key_state::dead) {
      s.tail_capacity = s.length - 1;
      s.occupancy = 0;
    }
  }
  bump_ = total;
  ++compactions_;
  return bump_;
}

void id_allocator::erase(const std::vector<std::int32_t>& keys) {
  // ESCHER-GPU's deleteCBST: a found node is marked available; its key and slot stay.
  for (const std::int32_t key : keys) {
    if (state_of(key) == key_state::live) {
      slots_[static_cast<std::size_t>(key)].state = key_state::dead;
      --live_;
      ++dead_;
    }
  }
}

std::vector<std::int32_t> id_allocator::insert(const std::vector<std::int32_t>& counts) {
  const std::size_t num_items = counts.size();
  std::vector<std::int32_t> keys(num_items, 0);
  if (num_items == 0) return keys;
  const std::int64_t compactions_before = compactions_;
  const std::int64_t num_dead = dead_;
  // Eligibility (ESCHER+ best fit, structure/operations.cu@abcf9b2 insertCBST): with D deleted
  // slots only the first min(K, D) items may reuse one.
  const std::size_t reuse_k = std::min<std::size_t>(num_items, static_cast<std::size_t>(num_dead));
  std::vector<char> matched(num_items, 0);
  if (reuse_k > 0) {
    // The deleted slots in key order (the order of the order-statistic walk) and their
    // capacities (a slot of `length` slots holds length - 1 values).
    std::vector<std::int32_t> dead_keys;
    dead_keys.reserve(static_cast<std::size_t>(num_dead));
    for (std::int64_t key = 1; key <= max_key_; ++key) {
      if (slots_[static_cast<std::size_t>(key)].state == key_state::dead) {
        dead_keys.push_back(static_cast<std::int32_t>(key));
      }
    }
    const std::size_t d = dead_keys.size();
    std::vector<std::int64_t> slot_capacity(d);
    for (std::size_t j = 0; j < d; ++j) {
      slot_capacity[j] = slots_[static_cast<std::size_t>(dead_keys[j])].length - 1;
    }
    // Both sorts stable: slots of equal capacity keep the key order, items of equal size the
    // input order (the tie rule).
    std::vector<std::size_t> slot_order(d);
    std::iota(slot_order.begin(), slot_order.end(), std::size_t{0});
    std::stable_sort(slot_order.begin(), slot_order.end(), [&](std::size_t a, std::size_t b) {
      return slot_capacity[a] < slot_capacity[b];
    });
    std::vector<std::int64_t> sorted_capacity(d);
    for (std::size_t j = 0; j < d; ++j) sorted_capacity[j] = slot_capacity[slot_order[j]];
    std::vector<std::size_t> item_order(reuse_k);
    std::iota(item_order.begin(), item_order.end(), std::size_t{0});
    std::stable_sort(item_order.begin(), item_order.end(),
                     [&](std::size_t a, std::size_t b) { return counts[a] < counts[b]; });
    // lo[i]: the first slot (in capacity order) that fits sorted item i;
    // assigned[i] = i + max_{j <= i}(lo[j] - j), strictly increasing; matched while < D.
    std::int64_t prefix_max = LLONG_MIN;
    std::vector<std::int64_t> matched_sizes;
    for (std::size_t i = 0; i < reuse_k; ++i) {
      const std::size_t item = item_order[i];
      const auto lo =
          static_cast<std::int64_t>(std::lower_bound(sorted_capacity.begin(), sorted_capacity.end(),
                                                     static_cast<std::int64_t>(counts[item])) -
                                    sorted_capacity.begin());
      prefix_max = std::max(prefix_max, lo - static_cast<std::int64_t>(i));
      const std::int64_t assigned = static_cast<std::int64_t>(i) + prefix_max;
      if (assigned >= static_cast<std::int64_t>(d)) break;  // so are the later ones
      const std::int32_t key = dead_keys[slot_order[static_cast<std::size_t>(assigned)]];
      slot& s = slots_[static_cast<std::size_t>(key)];
      // A reused slot keeps its key and length; its row is rewritten from the first slot.
      s.state = key_state::live;
      s.count = counts[item];
      s.tail_capacity = s.length - 1;
      s.occupancy = counts[item];
      --dead_;
      ++live_;
      keys[item] = key;
      matched[item] = 1;
      matched_sizes.push_back(counts[item]);
    }
    if (!matched_sizes.empty()) record_paths(matched_sizes, reuse_paths_);
  }
  // The surplus: unmatched eligible items and every item beyond min(K, D), in item order.
  std::vector<std::size_t> surplus;
  for (std::size_t i = 0; i < num_items; ++i) {
    if (matched[i] == 0) surplus.push_back(i);
  }
  if (!surplus.empty()) {
    const std::int64_t alignment = policy_.alignment;
    std::int64_t surplus_slots = 0;
    for (const std::size_t i : surplus) surplus_slots += aligned(counts[i], alignment) + 1;
    // MOSP_ESCHER's trigger point (escher/structure/operations.cu@4b86159:694-701): compact
    // after the reuse when the surplus rows do not fit behind the bump pointer; then a policy
    // that does not grow throws (MOSP_ESCHER's EscherError, :713-721).
    if (bump_ + surplus_slots > capacity_ &&
        policy_.compaction != escher::store_policy::compaction_rule::never) {
      (void)compact();
    }
    if (bump_ + surplus_slots > capacity_ &&
        policy_.growth == escher::store_policy::growth_rule::none) {
      throw capacity_error("dyng: CBST " + name_ + ": insert: " + std::to_string(surplus.size()) +
                           " surplus rows exceed the payload capacity (" +
                           std::to_string(bump_ + surplus_slots) + " > " +
                           std::to_string(capacity_) + "; policy " + policy_.name +
                           " does not grow the payload)");
    }
    ensure_capacity(bump_ + surplus_slots);
    bump_ += surplus_slots;
    // The rebuild: the deleted slots that were not reused are dropped; the surplus rows get the
    // keys after the last surviving (live) key, in item order.
    std::int64_t last_live = 0;
    for (std::int64_t key = 1; key <= max_key_; ++key) {
      slot& s = slots_[static_cast<std::size_t>(key)];
      if (s.state == key_state::dead) {
        s = slot{};
      } else if (s.state == key_state::live) {
        last_live = key;
      }
    }
    dead_ = 0;
    std::int64_t next = last_live + 1;
    for (const std::size_t i : surplus) {
      const std::int64_t a = aligned(counts[i], alignment);
      slot& s = at(next);
      s.state = key_state::live;
      s.length = static_cast<std::int32_t>(a + 1);
      s.count = counts[i];
      s.tail_capacity = static_cast<std::int32_t>(a);
      s.occupancy = counts[i];
      ++live_;
      keys[i] = static_cast<std::int32_t>(next);
      ++next;
    }
    // Keys above the new last key that are still in the vector are absent.
    for (std::int64_t key = next; key <= max_key_; ++key) {
      slots_[static_cast<std::size_t>(key)] = slot{};
    }
    max_key_ = next - 1;
  }
  keep_headroom_after_compaction(compactions_before);
  return keys;
}

void id_allocator::fill_impl(const std::vector<std::int32_t>& keys,
                             const std::vector<std::int32_t>& counts, bool allow_compact) {
  if (keys.empty()) return;
  std::vector<std::int64_t> sizes(counts.begin(), counts.end());
  record_paths(sizes, fill_paths_);
  // ESCHER-GPU's insertNode kernels (kernel/payload.cu@abcf9b2) with MOSP_ESCHER's clamp: the
  // values that fit the tail segment are written there; the rest of an item goes to an overflow
  // segment of ((k + 4) / 4) * 4 slots at the bump pointer, which becomes the row's tail.
  std::vector<std::int64_t> taken(keys.size(), 0), segment(keys.size(), 0);
  std::int64_t total = 0;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    slot& s = slots_[static_cast<std::size_t>(keys[i])];
    const std::int32_t occupancy = std::min(std::max(s.occupancy, 0), s.tail_capacity);
    const std::int64_t room = s.tail_capacity - occupancy;
    if (counts[i] <= room) {
      s.occupancy = occupancy + counts[i];
      s.count += counts[i];
      taken[i] = counts[i];
    } else {
      s.occupancy = s.tail_capacity;
      s.count += static_cast<std::int32_t>(room);
      taken[i] = room;
      segment[i] = padded4(counts[i] - room);
      total += segment[i];
    }
  }
  const std::int64_t needed = bump_ + total;
  if (total > 0 && needed > capacity_ && allow_compact &&
      policy_.compaction != escher::store_policy::compaction_rule::never) {
    // MOSP_ESCHER's fillImpl (escher/structure/operations.cu@4b86159:409-444): compact, then
    // append the values that did not fit their tails once more, without compacting.
    std::vector<std::int32_t> rest_keys, rest_counts;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (segment[i] != 0) {
        rest_keys.push_back(keys[i]);
        rest_counts.push_back(static_cast<std::int32_t>(counts[i] - taken[i]));
      }
    }
    (void)compact();
    fill_impl(rest_keys, rest_counts, false);
    return;
  }
  if (total > 0) {
    ensure_capacity(needed);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (segment[i] == 0) continue;
      slot& s = slots_[static_cast<std::size_t>(keys[i])];
      s.count += static_cast<std::int32_t>(counts[i] - taken[i]);
      s.tail_capacity = static_cast<std::int32_t>(segment[i] - 1);
      s.occupancy = static_cast<std::int32_t>(counts[i] - taken[i]);
    }
    bump_ = needed;
  }
}

void id_allocator::append(const std::vector<std::int32_t>& keys,
                          const std::vector<std::int32_t>& counts) {
  const std::int64_t compactions_before = compactions_;
  fill_impl(keys, counts, true);
  keep_headroom_after_compaction(compactions_before);
}

std::vector<std::string> id_allocator::check() const {
  std::vector<std::string> problems;
  std::int64_t live = 0, dead = 0, lengths = 0, top = 0;
  for (std::size_t key = 1; key < slots_.size(); ++key) {
    const slot& s = slots_[key];
    if (s.state == key_state::absent) continue;
    top = static_cast<std::int64_t>(key);
    lengths += s.length;
    if (s.length < 1) problems.push_back("key " + std::to_string(key) + " has no slot");
    if (s.state == key_state::live) {
      ++live;
      if (s.count < 0 || s.occupancy < 0 || s.occupancy > s.tail_capacity) {
        problems.push_back("key " + std::to_string(key) + " has inconsistent tail metadata");
      }
    } else {
      ++dead;
    }
  }
  if (live != live_ || dead != dead_) problems.push_back("the live or dead counters are wrong");
  if (top != max_key_) problems.push_back("the largest key is wrong");
  if (lengths > bump_) problems.push_back("the slots exceed the bump pointer");
  if (bump_ > capacity_) problems.push_back("the bump pointer exceeds the capacity");
  return problems;
}

}  // namespace dyng::detail::hyper
