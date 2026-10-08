// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/operations.cu (constructCBST, rebuildCBSTFromDevice,
// ensurePayloadCapacity, fillCBST, deleteCBST, insertCBST, unfillCBST, reserveScratch,
// deviceBytes) and kernel/{build_tree,insert_reuse,payload,unfill,delete_avail}.cu,
// kernel/device_utils.cuh (cbstRankOfPosition), tests/cbst_test.cu (the layout checks)
// Derived from MOSP_ESCHER@4b86159:escher/structure/operations.cu (constructCBST validation,
// compactCBST, fillImpl, insertCBST, ensureScratchCapacity) and escher/kernel/payload.cu (the
// fill clamps, compactSizesKernel, compactCopyKernel)
/**
 * @file cbst_host_model.cpp
 * @brief dyng::testing::cbst_host_model: the CBST store of both ESCHER cores, sequentially on the
 *        host (the oracle of the M8a store port; ADR 0037).
 *
 * Not a port: the originals' kernels are restated as sequential loops (one item after the other;
 * the originals' items touch disjoint rows, so the order does not matter, and the model reports
 * cbst_op_status::undefined where they would not: two items naming one row, a read or write
 * outside the payload, an endless chain). The best-fit matching is the greedy formulation
 * (the smallest unused fitting slot in ascending item size order) of the originals' prefix-max
 * pipeline, which assigns the same slots. Nothing here includes or calls dynG store code.
 */
#include <dyng/testing/cbst_host_model.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

namespace dyng::testing {

namespace {

constexpr std::int32_t terminator = INT_MIN;
// sizeof(CBSTNode) in both originals (7 ints, padding, 3 pointers) and the four int arrays per
// record that ESCHER-GPU's deviceBytes() counts.
constexpr std::int64_t node_bytes = 56;
constexpr std::int64_t per_record_bytes = node_bytes + 4 * 4;

int floor_log2(std::int64_t x) {
  int log = 0;
  while (x >>= 1) ++log;
  return log;
}

// The degree-bin level of an item of `size` values: thread (< 32), warp (< 1024), block.
int level_of(std::int64_t size) {
  return size < 32 ? 0 : (size < 1024 ? 1 : 2);
}

// The overflow segment of c values: ((c + 4) / 4) * 4 slots (the terminator included), 0 for none.
std::int64_t padded_overflow(std::int64_t c) {
  return c == 0 ? 0 : ((c + 4) / 4) * 4;
}

// A surplus row of n values: n rounded up to the alignment (0 for an empty row).
std::int64_t aligned_size(std::int64_t n, std::int64_t alignment) {
  if (n <= 0) return 0;
  return ((n + alignment - 1) / alignment) * alignment;
}

struct node {
  std::int32_t index = 0;
  std::int32_t value = 0;
  std::int32_t length = 0;
  std::int32_t size = 0;
  std::int32_t occupancy = 0;
  std::int32_t tail_base = 0;
  std::int32_t tail_capacity = 0;
};

/// Thrown inside an operation when the original's result is not defined.
struct undefined_behaviour {
  std::string what;
};

}  // namespace

std::int64_t cbst_rank_of_position(std::int64_t position, std::int64_t num_nodes) noexcept {
  if (num_nodes < 1 || position < 0 || position >= num_nodes) return -1;
  const int k = floor_log2(position + 1);
  const int levels = floor_log2(num_nodes);
  const std::int64_t index = (2 * (position + 1 - (std::int64_t{1} << k)) + 1) << (levels - k);
  const std::int64_t index2 =
      std::min(index, index - index / 2 + (num_nodes + 1 - (std::int64_t{1} << levels)));
  return index2 - 1;
}

cbst_layout_check check_cbst_layout(std::int64_t num_nodes, bool search_every_key) {
  cbst_layout_check out;
  out.num_nodes = num_nodes;
  if (num_nodes < 1 || num_nodes > (std::int64_t{1} << 30)) {
    out.not_a_permutation = 1;
    return out;
  }
  const auto n = static_cast<std::size_t>(num_nodes);
  // Keys 1..n at their heap positions; row k at offset 4(k - 1) (cbst_test's construct).
  std::vector<std::int64_t> key_at(n, 0);
  std::vector<char> seen(n, 0);
  for (std::size_t t = 0; t < n; ++t) {
    const std::int64_t rank = cbst_rank_of_position(static_cast<std::int64_t>(t), num_nodes);
    if (rank < 0 || rank >= num_nodes || seen[static_cast<std::size_t>(rank)]) {
      ++out.not_a_permutation;
      continue;
    }
    seen[static_cast<std::size_t>(rank)] = 1;
    key_at[t] = rank + 1;
  }
  // The in-order traversal of the heap must visit the ranks 0, 1, ..., n - 1.
  {
    std::vector<std::size_t> stack;
    std::size_t t = 0;
    bool go_left = true;
    std::int64_t expected = 1;
    while (true) {
      if (go_left && t < n) {
        stack.push_back(t);
        t = 2 * t + 1;
        continue;
      }
      if (stack.empty()) break;
      t = stack.back();
      stack.pop_back();
      if (key_at[t] != expected) ++out.not_in_order;
      ++expected;
      t = 2 * t + 2;
      go_left = true;
    }
  }
  // A BST search from the root for every key (or a deterministic sample).
  const std::int64_t step = search_every_key ? 1 : std::max<std::int64_t>(1, num_nodes / 65536);
  for (std::int64_t k = 1; k <= num_nodes; k += step) {
    std::size_t t = 0;
    while (t < n && key_at[t] != k) t = key_at[t] > k ? 2 * t + 1 : 2 * t + 2;
    if (t >= n) {
      ++out.not_found;
      continue;
    }
    if (4 * (key_at[t] - 1) != 4 * (k - 1)) ++out.wrong_offset;
  }
  // The inverse layout: position -> rank -> the sorted key array must give the stored key.
  for (std::size_t t = 0; t < n; ++t) {
    const std::int64_t rank = cbst_rank_of_position(static_cast<std::int64_t>(t), num_nodes);
    if (rank < 0 || rank + 1 != key_at[t]) ++out.inverse_mismatches;
  }
  return out;
}

// --- Policies ---------------------------------------------------------------------------------

cbst_host_policy cbst_host_policy::escher_gpu() {
  cbst_host_policy p;
  p.name = "escher_gpu";
  p.growth = growth_rule::escher_gpu;
  p.compaction = compaction_rule::none;
  p.failure = failure_rule::process_exit;
  p.occupancy = occupancy_rule::scan_payload;
  p.scratch = scratch_rule::exact;
  p.validate_construct = false;
  p.clamp_fill_tail = false;
  p.has_rebuild = true;
  p.has_compact = false;
  p.counts_bin_launches = true;
  p.error_type.clear();
  return p;
}

cbst_host_policy cbst_host_policy::mosp_escher() {
  cbst_host_policy p;
  p.name = "mosp_escher";
  p.growth = growth_rule::none;
  p.compaction = compaction_rule::mosp_escher;
  p.failure = failure_rule::exception;
  p.occupancy = occupancy_rule::caller_counts_or_zero;
  p.scratch = scratch_rule::geometric;
  p.validate_construct = true;
  p.clamp_fill_tail = true;
  p.has_rebuild = false;
  p.has_compact = true;
  p.counts_bin_launches = false;
  p.error_type = "escher::EscherError";
  return p;
}

cbst_host_policy cbst_host_policy::dyng_default() {
  cbst_host_policy p = mosp_escher();
  p.name = "dyng_default";
  p.growth = growth_rule::escher_gpu;
  p.occupancy = occupancy_rule::caller_counts_or_scan;
  p.has_rebuild = true;
  p.counts_bin_launches = true;
  p.error_type = "dyng::capacity_error";
  return p;
}

bool cbst_host_policy::from_name(const std::string& name, cbst_host_policy& out) {
  if (name == "escher_gpu") {
    out = escher_gpu();
  } else if (name == "mosp_escher") {
    out = mosp_escher();
  } else if (name == "dyng_default") {
    out = dyng_default();
  } else {
    return false;
  }
  return true;
}

// --- The model --------------------------------------------------------------------------------

struct cbst_host_model::impl {
  cbst_host_policy policy;
  std::string name;
  std::int32_t alignment = 4;
  std::int64_t capacity = 0;     // fixedSize
  std::int64_t bump = 0;         // initialPayloadSize
  std::int64_t num_records = 0;  // numRecords (may exceed nodes.size(): no tree was built)
  std::int64_t scratch_items = 0;
  std::int64_t scratch_payload = 0;
  std::vector<std::int32_t> payload;      // [0, payload.size()); the rest of the capacity is 0
  std::vector<node> nodes;                // by heap position
  std::vector<std::int32_t> avail;        // by heap position
  std::vector<std::int32_t> subtree;      // by heap position
  std::vector<std::int32_t> sorted_keys;  // by rank (the originals' d_keys)
  std::vector<std::int32_t> rank_of;      // heap position -> rank
  std::vector<std::int32_t> position_of;  // rank -> heap position
  std::vector<cbst_model_event> events;
  std::int64_t operations = 0;
  bool exited = false;

  // The current operation's outcome.
  cbst_op_result* result = nullptr;

  [[nodiscard]] std::int64_t n() const {
    return static_cast<std::int64_t>(nodes.size());
  }

  /// Runs one operation: counts it and turns what the original leaves undefined into
  /// cbst_op_status::undefined.
  template <class body_t>
  cbst_op_result run(body_t&& body) {
    cbst_op_result out;
    if (exited) {
      out.status = cbst_op_status::undefined;
      out.message = "the original's process ended at an earlier operation";
      return out;
    }
    result = &out;
    try {
      body();
    } catch (const undefined_behaviour& u) {
      out.status = cbst_op_status::undefined;
      out.message = u.what;
    }
    result = nullptr;
    ++operations;
    return out;
  }

  void clear_tree() {
    nodes.clear();
    avail.clear();
    subtree.clear();
    rank_of.clear();
    position_of.clear();
    sorted_keys.clear();
  }

  void set_layout(std::size_t count) {
    nodes.assign(count, node{});
    avail.assign(count, 0);
    subtree.assign(count, 0);
    rank_of.assign(count, 0);
    position_of.assign(count, 0);
    for (std::size_t t = 0; t < count; ++t) {
      const auto r = static_cast<std::int32_t>(
          cbst_rank_of_position(static_cast<std::int64_t>(t), static_cast<std::int64_t>(count)));
      rank_of[t] = r;
      position_of[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(t);
    }
  }

  // --- payload access (the device buffer: capacity values, zero beyond what was written) -----
  [[nodiscard]] std::int32_t read(std::int64_t p) const {
    if (p < 0 || p >= capacity)
      throw undefined_behaviour{"a read at payload offset " + std::to_string(p) +
                                " outside the capacity " + std::to_string(capacity)};
    return p < static_cast<std::int64_t>(payload.size()) ? payload[static_cast<std::size_t>(p)] : 0;
  }
  void write(std::int64_t p, std::int32_t v) {
    if (p < 0 || p >= capacity)
      throw undefined_behaviour{"a write at payload offset " + std::to_string(p) +
                                " outside the capacity " + std::to_string(capacity)};
    if (p >= static_cast<std::int64_t>(payload.size())) {
      if (v == 0) return;
      payload.resize(static_cast<std::size_t>(p) + 1, 0);
    }
    payload[static_cast<std::size_t>(p)] = v;
  }

  // --- tree access ------------------------------------------------------------------------------
  /// The heap position of `key` by a BST search from the root, or -1 (bstFind).
  [[nodiscard]] std::int64_t find(std::int32_t key) const {
    std::int64_t t = 0;
    const std::int64_t count = n();
    while (t < count) {
      const std::int32_t idx = nodes[static_cast<std::size_t>(t)].index;
      if (idx == key) return t;
      t = idx > key ? 2 * t + 1 : 2 * t + 2;
    }
    return -1;
  }

  /// subtreeAvail bottom-up (recomputeSubtreeAvail): an empty tree is a zero-block launch.
  bool recompute_subtree() {
    if (num_records <= 0 || nodes.empty()) {
      if (num_records > 0) throw undefined_behaviour{"recomputeSubtreeAvail without a tree"};
      device_error("recomputeSubtreeAvail launches a kernel of 0 blocks (an empty tree)");
      return false;
    }
    for (std::int64_t t = n() - 1; t >= 0; --t) {
      const auto u = static_cast<std::size_t>(t);
      std::int32_t s = avail[u];
      if (2 * t + 1 < n()) s += subtree[static_cast<std::size_t>(2 * t + 1)];
      if (2 * t + 2 < n()) s += subtree[static_cast<std::size_t>(2 * t + 2)];
      subtree[u] = s;
    }
    return true;
  }

  // --- failures -----------------------------------------------------------------------------------
  void fail(const std::string& message) {
    if (policy.failure == cbst_host_policy::failure_rule::process_exit) {
      result->status = cbst_op_status::process_exit;
      exited = true;
    } else {
      result->status = cbst_op_status::error;
      result->error_type = policy.error_type;
    }
    result->message = message;
  }
  // A CUDA error of the original (a null device pointer, a zero-block launch): ESCHER-GPU's
  // checkCuda exits, MOSP_ESCHER's throws escher::EscherError.
  void device_error(const std::string& message) {
    fail("CUDA error: " + message);
  }

  /// Room for `needed` payload values: true if it fits or grew; false after a failure.
  bool make_room(std::int64_t needed, const char* what) {
    if (needed <= capacity) return true;
    if (policy.growth == cbst_host_policy::growth_rule::escher_gpu) {
      std::int64_t grown = std::max(needed + needed / 4, 3 * capacity / 2);
      grown = std::min<std::int64_t>(grown, std::min<std::int64_t>(policy.max_payload, INT_MAX));
      if (needed > grown) {
        fail(std::string(what) + ": " + std::to_string(needed) +
             " payload entries exceed the largest payload (" +
             std::to_string(std::min<std::int64_t>(policy.max_payload, INT_MAX)) +
             "; the int32 offsets of the CBST at most)");
        return false;
      }
      // The used prefix is copied, the rest of the new buffer is zero.
      if (static_cast<std::int64_t>(payload.size()) > bump)
        payload.resize(static_cast<std::size_t>(bump));
      events.push_back({cbst_model_event::kind::growth, operations, capacity, grown});
      capacity = grown;
      result->grew = true;
      return true;
    }
    fail(std::string(what) + ": payload overflow (" + std::to_string(needed) + " > " +
         std::to_string(capacity) + ")");
    return false;
  }

  void reserve_scratch(std::int64_t items, std::int64_t values) {
    if (policy.scratch == cbst_host_policy::scratch_rule::exact) {
      scratch_items = std::max(scratch_items, items);
      scratch_payload = std::max(scratch_payload, values);
      return;
    }
    if (items <= scratch_items && values <= scratch_payload) return;
    std::int64_t keys_cap = scratch_items > 0 ? scratch_items : 1;
    while (keys_cap < items) keys_cap += keys_cap / 2 + 1;
    std::int64_t values_cap = scratch_payload > 0 ? scratch_payload : 4;
    while (values_cap < values) values_cap += values_cap / 2 + 1;
    scratch_items = keys_cap;
    scratch_payload = values_cap;
  }

  void count_bins(const std::vector<std::int64_t>& sizes, int which) {
    if (!policy.counts_bin_launches) return;
    bool used[3] = {false, false, false};
    for (std::int64_t s : sizes) used[level_of(s)] = true;
    for (int b = 0; b < 3; ++b)
      result->bin_launches[static_cast<std::size_t>(3 * which + b)] += used[b];
  }

  /// The item sizes of an inclusive prefix-sum layout, checked against the value count.
  static std::vector<std::int64_t> item_sizes(const std::vector<std::int32_t>& prefix,
                                              std::size_t num_items, std::size_t num_values) {
    if (prefix.size() != num_items)
      throw undefined_behaviour{"the prefix sums have " + std::to_string(prefix.size()) +
                                " entries for " + std::to_string(num_items) + " items"};
    std::vector<std::int64_t> sizes(num_items);
    std::int64_t previous = 0;
    for (std::size_t i = 0; i < num_items; ++i) {
      sizes[i] = static_cast<std::int64_t>(prefix[i]) - previous;
      if (sizes[i] < 0 || prefix[i] < 0 || static_cast<std::size_t>(prefix[i]) > num_values)
        throw undefined_behaviour{"the prefix sums are not an inclusive scan of the values"};
      previous = prefix[i];
    }
    return sizes;
  }

  /// The tail occupancy of a constructed row: the values before its terminator.
  [[nodiscard]] std::int32_t scanned_occupancy(const node& nd) const {
    std::int32_t count = 0;
    while (count < nd.tail_capacity) {
      const std::int32_t v = read(static_cast<std::int64_t>(nd.value) + count);
      if (v == 0 || v == terminator) break;
      ++count;
    }
    return count;
  }

  /// Places the rows of a flattened layout (storeItemsIntoNodes).
  void place_rows(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& starts,
                  std::int64_t flat_size) {
    const std::size_t count = keys.size();
    set_layout(count);
    sorted_keys = keys;
    for (std::size_t t = 0; t < count; ++t) {
      const auto r = static_cast<std::size_t>(rank_of[t]);
      node& nd = nodes[t];
      nd.size = static_cast<std::int32_t>(flat_size);
      nd.index = keys[r];
      nd.value = starts[r];
      const std::int64_t length = r + 1 < count
                                      ? static_cast<std::int64_t>(starts[r + 1]) - starts[r]
                                      : flat_size - starts[r];
      nd.length = static_cast<std::int32_t>(length);
      nd.tail_base = starts[r];
      nd.tail_capacity = length > 0 ? static_cast<std::int32_t>(length - 1) : 0;
      nd.occupancy = 0;
    }
  }

  // --- the operations ---------------------------------------------------------------------------

  void construct(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& starts,
                 const std::vector<std::int32_t>& flat, const std::vector<std::int32_t>* counts) {
    const auto flat_size = static_cast<std::int64_t>(flat.size());
    if (starts.size() != keys.size())
      throw undefined_behaviour{"construct: keys and start offsets differ in size"};
    if (policy.validate_construct && capacity < flat_size) {
      result->status = cbst_op_status::error;
      result->error_type = policy.error_type;
      result->message = "constructCBST: payloadCapacity < flatPayloadSize";
      return;
    }
    num_records = static_cast<std::int64_t>(keys.size());
    bump = flat_size;
    if (keys.empty()) {
      clear_tree();  // the recorder reads no nodes of a tree of 0 records
      return;
    }
    if (capacity < flat_size) {
      // ESCHER-GPU prints "Overflow" and returns before allocating the tree.
      clear_tree();
      payload.clear();
      return;
    }
    payload.assign(flat.begin(), flat.end());
    place_rows(keys, starts, flat_size);
    using occ = cbst_host_policy::occupancy_rule;
    const bool caller = counts != nullptr && policy.occupancy != occ::scan_payload;
    if (caller && counts->size() != keys.size())
      throw undefined_behaviour{"construct: the row counts differ in size from the keys"};
    for (std::size_t t = 0; t < nodes.size(); ++t) {
      node& nd = nodes[t];
      if (caller)
        nd.occupancy = (*counts)[static_cast<std::size_t>(rank_of[t])];
      else if (policy.occupancy == occ::caller_counts_or_zero)
        nd.occupancy = 0;
      else
        nd.occupancy = scanned_occupancy(nd);
    }
    scratch_items = num_records;
    scratch_payload = 3 * num_records;
  }

  void rebuild(const std::vector<std::int32_t>& starts, const std::vector<std::int32_t>& flat,
               std::int64_t flat_size, std::int64_t new_capacity) {
    if (!policy.has_rebuild)
      throw undefined_behaviour{"the policy has no rebuild (MOSP_ESCHER has no rebuildFromDevice)"};
    if (static_cast<std::int64_t>(flat.size()) < flat_size)
      throw undefined_behaviour{"rebuild: the payload is shorter than its size"};
    capacity = new_capacity;
    bump = flat_size;
    num_records = static_cast<std::int64_t>(starts.size());
    payload.assign(flat.begin(), flat.begin() + static_cast<std::ptrdiff_t>(flat_size));
    if (starts.empty()) {
      clear_tree();
      device_error("rebuildCBSTFromDevice launches kernels of 0 blocks (no records)");
      return;
    }
    std::vector<std::int32_t> keys(starts.size());
    std::iota(keys.begin(), keys.end(), 1);
    place_rows(keys, starts, flat_size);
    for (node& nd : nodes) nd.occupancy = scanned_occupancy(nd);
  }

  void erase(const std::vector<std::int32_t>& keys) {
    if (keys.empty()) return;
    for (std::int32_t key : keys) {
      const std::int64_t t = find(key);
      if (t < 0) continue;
      node& nd = nodes[static_cast<std::size_t>(t)];
      avail[static_cast<std::size_t>(t)] = 1;
      nd.occupancy = 0;
      nd.tail_base = nd.value;
      nd.tail_capacity = nd.length > 0 ? nd.length - 1 : 0;
    }
    recompute_subtree();
  }

  /// MOSP_ESCHER's row walk (warpWalkRow): the values of the row from `base`; reads at or past
  /// `limit` end it.
  void walk(std::int64_t base, std::int64_t limit, std::vector<std::int32_t>* out,
            std::int64_t& count) const {
    count = 0;
    std::int64_t p = base;
    for (std::int64_t steps = 0;; ++steps) {
      if (steps > limit + 1) throw undefined_behaviour{"compaction: an endless row chain"};
      const std::int32_t v = (p >= 0 && p < limit) ? read(p) : 0;
      if (v > 0) {
        if (out) out->push_back(v);
        ++count;
        ++p;
        continue;
      }
      if (v < 0 && v != terminator) {
        p = -static_cast<std::int64_t>(v);
        continue;
      }
      return;
    }
  }

  /// MOSP_ESCHER's compactCBST: every row as one segment in key order. Returns false after a
  /// failure.
  bool compact_payload() {
    if (n() <= 0) {
      result->compact_return = bump;
      return true;
    }
    const std::int64_t limit = bump;
    const auto count = static_cast<std::size_t>(n());
    std::vector<std::int64_t> live_count(count, 0), slots_by_rank(count + 1, 0);
    for (std::size_t t = 0; t < count; ++t) {
      const bool live = avail[t] == 0;
      std::int64_t c = 0;
      if (live) walk(nodes[t].value, limit, nullptr, c);
      live_count[t] = c;
      slots_by_rank[static_cast<std::size_t>(rank_of[t])] =
          live ? (c + 4) / 4 * 4 : nodes[t].length;
    }
    std::vector<std::int64_t> offset(count + 1, 0);
    for (std::size_t r = 0; r < count; ++r) offset[r + 1] = offset[r] + slots_by_rank[r];
    const std::int64_t total = offset[count];
    if (total > capacity) {
      if (policy.growth != cbst_host_policy::growth_rule::escher_gpu) {
        fail("compactCBST [" + name + "]: the compacted rows need " + std::to_string(total) +
             " payload slots, more than the capacity " + std::to_string(capacity));
        return false;
      }
      if (!make_room(total, "compactCBST")) return false;
    }
    std::vector<std::int32_t> dst(static_cast<std::size_t>(total), 0);
    std::vector<std::int32_t> row;
    bool changed = total < limit;
    for (std::size_t t = 0; t < count; ++t) {
      node& nd = nodes[t];
      const auto r = static_cast<std::size_t>(rank_of[t]);
      const std::int64_t base = offset[r];
      const std::int64_t slots = offset[r + 1] - base;
      const std::int64_t c = live_count[t];
      if (slots <= c)
        throw undefined_behaviour{"compaction: a deleted row of length " + std::to_string(slots) +
                                  " has no slot for its terminator"};
      if (avail[t] == 0) {
        row.clear();
        std::int64_t again = 0;
        walk(nd.value, limit, &row, again);
        std::copy(row.begin(), row.end(), dst.begin() + base);
      }
      dst[static_cast<std::size_t>(base + c)] = terminator;
      if (nd.value != base || nd.length != slots || nd.tail_base != base ||
          nd.tail_capacity != slots - 1 || nd.occupancy != c)
        changed = true;
      nd.value = static_cast<std::int32_t>(base);
      nd.length = static_cast<std::int32_t>(slots);
      nd.tail_base = static_cast<std::int32_t>(base);
      nd.tail_capacity = static_cast<std::int32_t>(slots - 1);
      nd.occupancy = static_cast<std::int32_t>(c);
    }
    // The compacted rows go to the front; the rest of the old used prefix is cleared.
    if (static_cast<std::int64_t>(payload.size()) <= limit) {
      payload = std::move(dst);
    } else {
      std::copy(dst.begin(), dst.end(), payload.begin());
      std::fill(payload.begin() + total, payload.begin() + limit, 0);
    }
    events.push_back({cbst_model_event::kind::compaction, operations, limit, total});
    bump = total;
    result->compact_return = bump;
    ++result->compactions;
    result->compaction_changed = result->compaction_changed || changed;
    return true;
  }

  void check_unique_rows(const std::vector<std::int32_t>& keys, const char* what) const {
    std::unordered_set<std::int64_t> rows;
    rows.reserve(keys.size() * 2);
    for (std::int32_t key : keys) {
      const std::int64_t t = find(key);
      if (t >= 0 && !rows.insert(t).second)
        throw undefined_behaviour{std::string(what) + ": two items name key " +
                                  std::to_string(key) + " (the original's threads race)"};
    }
  }

  void append(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& values,
              const std::vector<std::int32_t>& prefix, bool allow_compact) {
    if (keys.empty()) return;
    const std::size_t count = keys.size();
    const std::vector<std::int64_t> sizes = item_sizes(prefix, count, values.size());
    reserve_scratch(static_cast<std::int64_t>(count), static_cast<std::int64_t>(values.size()));
    check_unique_rows(keys, "fill");
    count_bins(sizes, 0);
    // Pass 1 (insertNode_{thread,warp,block}): the values that fit into each tail; the overflow
    // plan: where the chain pointer goes, how many values fit, how many overflow.
    std::vector<std::int64_t> at(count, 0), fit(count, 0), over(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
      const std::int64_t t = find(keys[i]);
      if (t < 0) continue;
      node& nd = nodes[static_cast<std::size_t>(t)];
      const std::int64_t start = i == 0 ? 0 : prefix[i - 1];
      const std::int64_t num = sizes[i];
      const std::int64_t write_pos = static_cast<std::int64_t>(nd.tail_base) + nd.occupancy;
      std::int64_t remaining = static_cast<std::int64_t>(nd.tail_capacity) - nd.occupancy;
      if (policy.clamp_fill_tail) remaining = std::max<std::int64_t>(0, remaining);
      if (remaining < 0)
        throw undefined_behaviour{"fill: the tail of key " + std::to_string(keys[i]) +
                                  " holds more values than its capacity (unclamped)"};
      if (num <= remaining) {
        for (std::int64_t j = 0; j < num; ++j)
          write(write_pos + j, values[static_cast<std::size_t>(start + j)]);
        write(write_pos + num, terminator);
        nd.occupancy = static_cast<std::int32_t>(nd.occupancy + num);
      } else {
        for (std::int64_t j = 0; j < remaining; ++j)
          write(write_pos + j, values[static_cast<std::size_t>(start + j)]);
        at[i] = write_pos + remaining;
        fit[i] = remaining;
        over[i] = num - remaining;
        nd.occupancy = nd.tail_capacity;
      }
    }
    // The padded overflow segments and their cumulative offsets.
    std::vector<std::int64_t> cumulative(count, 0);
    std::int64_t total = 0;
    for (std::size_t i = 0; i < count; ++i) {
      total += padded_overflow(over[i]);
      cumulative[i] = total;
    }
    if (total > INT_MAX) throw undefined_behaviour{"fill: the overflow exceeds int32"};
    if (bump + total > capacity) {
      if (policy.compaction == cbst_host_policy::compaction_rule::mosp_escher && allow_compact) {
        // Compact, then fill the values that did not fit into their tails once more.
        std::vector<std::int32_t> rest_keys, rest_values, rest_prefix;
        for (std::size_t i = 0; i < count; ++i) {
          const std::int64_t before = i == 0 ? 0 : cumulative[i - 1];
          if (cumulative[i] == before) continue;
          const std::int64_t start = i == 0 ? 0 : prefix[i - 1];
          rest_keys.push_back(keys[i]);
          rest_values.insert(rest_values.end(),
                             values.begin() + static_cast<std::ptrdiff_t>(start + fit[i]),
                             values.begin() + prefix[i]);
          rest_prefix.push_back(static_cast<std::int32_t>(rest_values.size()));
        }
        if (!compact_payload()) return;
        append(rest_keys, rest_values, rest_prefix, false);
        return;
      }
      if (!make_room(bump + total, "fillCBST")) return;
    }
    if (total == 0) return;
    // Pass 2 (allocateSpace, fixupOverflowMetadata): the overflow segments behind the bump
    // pointer and the chain pointers to them.
    for (std::size_t i = 0; i < count; ++i) {
      const std::int64_t before = i == 0 ? 0 : cumulative[i - 1];
      const std::int64_t padded = cumulative[i] - before;
      if (padded == 0) continue;
      const std::int64_t base = bump + before;
      const std::int64_t start = i == 0 ? 0 : prefix[i - 1];
      const std::int64_t num = sizes[i];
      std::int64_t p = base;
      for (std::int64_t j = fit[i]; j < num; ++j, ++p)
        write(p, values[static_cast<std::size_t>(start + j)]);
      const std::int64_t written = num - fit[i];
      for (std::int64_t z = written; z < padded - 1; ++z) write(base + z, 0);
      write(base + padded - 1, terminator);
      write(at[i], static_cast<std::int32_t>(-base));
      const std::int64_t t = find(keys[i]);
      if (t >= 0) {
        node& nd = nodes[static_cast<std::size_t>(t)];
        nd.tail_base = static_cast<std::int32_t>(base);
        nd.tail_capacity = static_cast<std::int32_t>(padded - 1);
        nd.occupancy = static_cast<std::int32_t>(written);
      }
    }
    bump += total;
  }

  /// unfillRowSerial: removes the values from every segment of the row, keeping the chain.
  std::int32_t unfill_serial(const node& nd, const std::vector<std::int32_t>& remove) {
    std::int64_t seg = nd.value;
    for (std::int64_t hops = 0;; ++hops) {
      if (hops > capacity) throw undefined_behaviour{"unfill: an endless row chain"};
      std::int64_t w = 0, i = 0;
      for (;; ++i) {
        const std::int32_t v = read(seg + i);
        if (v == terminator || v == 0 || v < 0) break;
        if (!std::binary_search(remove.begin(), remove.end(), v)) {
          write(seg + w, v);
          ++w;
        }
      }
      const std::int32_t end = read(seg + i);
      const bool chained = end < 0 && end != terminator;
      write(seg + w, chained ? end : terminator);
      for (std::int64_t z = w + 1; z <= i; ++z) write(seg + z, 0);
      if (!chained) return static_cast<std::int32_t>(w);
      seg = -static_cast<std::int64_t>(end);
    }
  }

  /// unfill_warp / unfill_block on a single-segment row: the first `occupancy` slots.
  std::int32_t unfill_segment(const node& nd, const std::vector<std::int32_t>& remove) {
    const std::int64_t seg = nd.tail_base;
    const std::int64_t occupancy = nd.occupancy;
    std::int64_t w = 0;
    for (std::int64_t j = 0; j < occupancy; ++j) {
      const std::int32_t v = read(seg + j);
      if (v > 0 && !std::binary_search(remove.begin(), remove.end(), v)) {
        write(seg + w, v);
        ++w;
      }
    }
    write(seg + w, terminator);
    for (std::int64_t z = w + 1; z <= occupancy; ++z) write(seg + z, 0);
    return static_cast<std::int32_t>(w);
  }

  void remove_values(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& values,
                     const std::vector<std::int32_t>& prefix) {
    if (keys.empty()) return;
    const std::size_t count = keys.size();
    const std::vector<std::int64_t> sizes = item_sizes(prefix, count, values.size());
    reserve_scratch(static_cast<std::int64_t>(count), static_cast<std::int64_t>(values.size()));
    check_unique_rows(keys, "unfill");
    count_bins(sizes, 2);
    std::vector<std::int32_t> remove;
    for (std::size_t i = 0; i < count; ++i) {
      const std::int64_t t = find(keys[i]);
      if (t < 0) continue;
      node& nd = nodes[static_cast<std::size_t>(t)];
      const std::int64_t start = i == 0 ? 0 : prefix[i - 1];
      remove.assign(values.begin() + start, values.begin() + prefix[i]);
      std::sort(remove.begin(), remove.end());
      if (level_of(sizes[i]) == 0 || nd.tail_base != nd.value)
        nd.occupancy = unfill_serial(nd, remove);
      else
        nd.occupancy = unfill_segment(nd, remove);
    }
  }

  /// locateReusableSlots: the heap position of the k-th deleted slot (in key order), or -1.
  [[nodiscard]] std::int64_t kth_deleted(std::int64_t k) const {
    std::int64_t t = 0;
    const std::int64_t count = n();
    while (t < count) {
      const std::int64_t left = 2 * t + 1;
      const std::int64_t left_count = left < count ? subtree[static_cast<std::size_t>(left)] : 0;
      const std::int64_t self = avail[static_cast<std::size_t>(t)];
      if (k <= left_count) {
        t = left;
        continue;
      }
      if (self == 1 && k == left_count + 1) break;
      k -= left_count + self;
      t = 2 * t + 2;
    }
    return t < count ? t : -1;
  }

  void insert(const std::vector<std::int32_t>& keys, const std::vector<std::int32_t>& values,
              const std::vector<std::int32_t>& prefix) {
    const std::size_t count = keys.size();
    result->mapping.assign(count, 0);
    if (count == 0) return;
    const std::vector<std::int64_t> sizes = item_sizes(prefix, count, values.size());
    reserve_scratch(static_cast<std::int64_t>(count), static_cast<std::int64_t>(values.size()));
    if (nodes.empty()) {
      if (num_records > 0) throw undefined_behaviour{"insert on a tree that was not built"};
      device_error("insertCBST reads D from the subtree counts of an empty tree");
      return;
    }
    const std::int64_t d = subtree[0];
    result->reusable_slots = d;
    const std::int64_t reuse_k = std::min<std::int64_t>(static_cast<std::int64_t>(count), d);
    std::vector<char> matched(count, 0);
    if (reuse_k > 0) {
      // The deleted slots in key order, their capacities (length - 1) and keys.
      const auto num_slots = static_cast<std::size_t>(d);
      std::vector<std::int64_t> position(num_slots), slot_capacity(num_slots);
      std::vector<std::int32_t> slot_key(num_slots);
      for (std::size_t s = 0; s < num_slots; ++s) {
        position[s] = kth_deleted(static_cast<std::int64_t>(s) + 1);
        if (position[s] >= 0) {
          slot_capacity[s] =
              static_cast<std::int64_t>(nodes[static_cast<std::size_t>(position[s])].length) - 1;
          slot_key[s] =
              sorted_keys[static_cast<std::size_t>(rank_of[static_cast<std::size_t>(position[s])])];
        } else {
          slot_capacity[s] = 0;
          slot_key[s] = 0;
        }
      }
      // Best fit: slots by capacity, the first reuse_k items by size (both stable); in
      // ascending size, each item takes the smallest unused slot that fits it.
      std::vector<std::size_t> slot_order(num_slots), item_order(static_cast<std::size_t>(reuse_k));
      std::iota(slot_order.begin(), slot_order.end(), std::size_t{0});
      std::iota(item_order.begin(), item_order.end(), std::size_t{0});
      std::stable_sort(slot_order.begin(), slot_order.end(), [&](std::size_t a, std::size_t b) {
        return slot_capacity[a] < slot_capacity[b];
      });
      std::stable_sort(item_order.begin(), item_order.end(),
                       [&](std::size_t a, std::size_t b) { return sizes[a] < sizes[b]; });
      std::vector<std::int64_t> sorted_capacity(num_slots);
      for (std::size_t s = 0; s < num_slots; ++s) sorted_capacity[s] = slot_capacity[slot_order[s]];
      std::vector<std::pair<std::size_t, std::size_t>> matches;  // (item, slot)
      std::int64_t previous = -1;
      for (std::size_t item : item_order) {
        const auto lo = static_cast<std::int64_t>(
            std::lower_bound(sorted_capacity.begin(), sorted_capacity.end(), sizes[item]) -
            sorted_capacity.begin());
        const std::int64_t assigned = std::max(lo, previous + 1);
        if (assigned >= d) break;
        matches.emplace_back(item, slot_order[static_cast<std::size_t>(assigned)]);
        previous = assigned;
      }
      std::vector<std::int64_t> matched_sizes;
      for (const auto& m : matches) {
        result->mapping[m.first] = slot_key[m.second];
        matched[m.first] = 1;
        matched_sizes.push_back(sizes[m.first]);
      }
      if (!matches.empty()) count_bins(matched_sizes, 1);
      // applyReuse: the row, its terminator, then zeros up to the end of the first segment.
      for (const auto& m : matches) {
        const std::int64_t t = position[m.second];
        if (t < 0) continue;
        node& nd = nodes[static_cast<std::size_t>(t)];
        const std::int64_t start = m.first == 0 ? 0 : prefix[m.first - 1];
        const std::int64_t len = sizes[m.first];
        const std::int64_t base = nd.value;
        const std::int64_t slot_capacity_now = static_cast<std::int64_t>(nd.length) - 1;
        for (std::int64_t i = 0; i <= slot_capacity_now; ++i)
          write(base + i, i < len ? values[static_cast<std::size_t>(start + i)]
                                  : (i == len ? terminator : 0));
        nd.occupancy = static_cast<std::int32_t>(len);
        nd.index = slot_key[m.second];
        nd.tail_base = static_cast<std::int32_t>(base);
        nd.tail_capacity = static_cast<std::int32_t>(slot_capacity_now);
        avail[static_cast<std::size_t>(t)] = 0;
      }
    }
    // Surplus items (unmatched, in input order): new rows behind the bump pointer, new keys
    // after the last live key, and a rebuild from the live nodes.
    std::vector<std::size_t> surplus;
    for (std::size_t i = 0; i < count; ++i)
      if (!matched[i]) surplus.push_back(i);
    if (!surplus.empty()) {
      std::int64_t slots = 0;
      for (std::size_t i : surplus) slots += aligned_size(sizes[i], alignment) + 1;
      if (bump + slots > capacity &&
          policy.compaction == cbst_host_policy::compaction_rule::mosp_escher) {
        if (!compact_payload()) return;
      }
      if (bump + slots > capacity && !make_room(bump + slots, "insertCBST")) return;
      if (bump + slots > INT_MAX) throw undefined_behaviour{"insert: the payload exceeds int32"};
      std::vector<node> records;
      std::int64_t cursor = bump;
      std::vector<node> new_rows(surplus.size());
      for (std::size_t s = 0; s < surplus.size(); ++s) {
        const std::size_t i = surplus[s];
        const std::int64_t start = i == 0 ? 0 : prefix[i - 1];
        const std::int64_t len = sizes[i];
        const std::int64_t aligned = aligned_size(len, alignment);
        node& r = new_rows[s];
        r.value = static_cast<std::int32_t>(cursor);
        r.length = static_cast<std::int32_t>(aligned + 1);
        r.occupancy = static_cast<std::int32_t>(len);
        r.tail_base = r.value;
        r.tail_capacity = static_cast<std::int32_t>(aligned);
        for (std::int64_t j = 0; j < len; ++j)
          write(cursor + j, values[static_cast<std::size_t>(start + j)]);
        for (std::int64_t j = len; j < aligned; ++j) write(cursor + j, 0);
        write(cursor + aligned, terminator);
        cursor += aligned + 1;
      }
      bump = cursor;
      // The live nodes in key (rank) order keep their records.
      records.reserve(nodes.size() + surplus.size());
      for (std::size_t r = 0; r < nodes.size(); ++r) {
        const auto t = static_cast<std::size_t>(position_of[r]);
        if (avail[t] == 0) records.push_back(nodes[t]);
      }
      std::int32_t next_key = records.empty() ? 1 : records.back().index + 1;
      for (std::size_t s = 0; s < surplus.size(); ++s) {
        new_rows[s].index = next_key;
        new_rows[s].size = static_cast<std::int32_t>(bump);
        result->mapping[surplus[s]] = next_key;
        ++next_key;
        records.push_back(new_rows[s]);
      }
      const std::size_t new_count = records.size();
      num_records = static_cast<std::int64_t>(new_count);
      set_layout(new_count);
      sorted_keys.resize(new_count);
      for (std::size_t r = 0; r < new_count; ++r) sorted_keys[r] = records[r].index;
      for (std::size_t t = 0; t < new_count; ++t)
        nodes[t] = records[static_cast<std::size_t>(rank_of[t])];
    }
    recompute_subtree();
  }
};

cbst_host_model::cbst_host_model(cbst_host_policy policy, std::int64_t payload_capacity,
                                 std::int32_t alignment, std::string name)
    : impl_(std::make_unique<impl>()) {
  impl_->policy = std::move(policy);
  impl_->capacity = payload_capacity;
  impl_->alignment = alignment > 0 ? alignment : 4;
  impl_->name = std::move(name);
}

cbst_host_model::~cbst_host_model() = default;
cbst_host_model::cbst_host_model(const cbst_host_model& other)
    : impl_(std::make_unique<impl>(*other.impl_)) {}
cbst_host_model& cbst_host_model::operator=(const cbst_host_model& other) {
  if (this != &other) impl_ = std::make_unique<impl>(*other.impl_);
  return *this;
}
cbst_host_model::cbst_host_model(cbst_host_model&& other) noexcept = default;
cbst_host_model& cbst_host_model::operator=(cbst_host_model&& other) noexcept = default;

cbst_op_result cbst_host_model::construct(const std::vector<std::int32_t>& keys,
                                          const std::vector<std::int32_t>& starts,
                                          const std::vector<std::int32_t>& flat,
                                          const std::vector<std::int32_t>* row_counts) {
  return impl_->run([&] { impl_->construct(keys, starts, flat, row_counts); });
}

cbst_op_result cbst_host_model::rebuild(const std::vector<std::int32_t>& starts,
                                        const std::vector<std::int32_t>& flat,
                                        std::int64_t flat_size, std::int64_t payload_capacity) {
  return impl_->run([&] { impl_->rebuild(starts, flat, flat_size, payload_capacity); });
}

cbst_op_result cbst_host_model::insert(const std::vector<std::int32_t>& keys,
                                       const std::vector<std::int32_t>& payload,
                                       const std::vector<std::int32_t>& prefix) {
  return impl_->run([&] { impl_->insert(keys, payload, prefix); });
}

cbst_op_result cbst_host_model::erase(const std::vector<std::int32_t>& keys) {
  return impl_->run([&] { impl_->erase(keys); });
}

cbst_op_result cbst_host_model::append(const std::vector<std::int32_t>& keys,
                                       const std::vector<std::int32_t>& payload,
                                       const std::vector<std::int32_t>& prefix) {
  return impl_->run([&] { impl_->append(keys, payload, prefix, true); });
}

cbst_op_result cbst_host_model::remove_values(const std::vector<std::int32_t>& keys,
                                              const std::vector<std::int32_t>& values,
                                              const std::vector<std::int32_t>& prefix) {
  return impl_->run([&] { impl_->remove_values(keys, values, prefix); });
}

cbst_op_result cbst_host_model::compact() {
  return impl_->run([&] {
    if (!impl_->policy.has_compact)
      throw undefined_behaviour{"the policy has no compact() (ESCHER-GPU has none)"};
    impl_->compact_payload();
  });
}

const cbst_host_policy& cbst_host_model::policy() const noexcept {
  return impl_->policy;
}
const std::string& cbst_host_model::name() const noexcept {
  return impl_->name;
}
std::int64_t cbst_host_model::num_records() const noexcept {
  return impl_->num_records;
}
std::int64_t cbst_host_model::num_nodes() const noexcept {
  return impl_->n();
}
std::int64_t cbst_host_model::payload_capacity() const noexcept {
  return impl_->capacity;
}
std::int64_t cbst_host_model::bump() const noexcept {
  return impl_->bump;
}
std::int32_t cbst_host_model::alignment() const noexcept {
  return impl_->alignment;
}
std::int64_t cbst_host_model::scratch_items() const noexcept {
  return impl_->scratch_items;
}
std::int64_t cbst_host_model::scratch_payload() const noexcept {
  return impl_->scratch_payload;
}
std::int64_t cbst_host_model::device_bytes() const noexcept {
  return impl_->num_records * per_record_bytes + impl_->capacity * 4 + impl_->scratch_items * 20 +
         impl_->scratch_payload * 4;
}
std::int64_t cbst_host_model::reusable_slots() const noexcept {
  return impl_->subtree.empty() ? -1 : impl_->subtree[0];
}
const std::vector<cbst_model_event>& cbst_host_model::events() const noexcept {
  return impl_->events;
}
std::int64_t cbst_host_model::num_operations() const noexcept {
  return impl_->operations;
}
bool cbst_host_model::exited() const noexcept {
  return impl_->exited;
}

std::vector<std::int32_t> cbst_host_model::keys() const {
  std::vector<std::int32_t> out;
  out.reserve(impl_->nodes.size());
  for (std::int32_t t : impl_->position_of)
    out.push_back(impl_->nodes[static_cast<std::size_t>(t)].index);
  return out;
}

std::vector<std::int32_t> cbst_host_model::subtree_avail() const {
  return impl_->subtree;
}

namespace {

/// The row as the recorder reads it: positive values from `value`, a negative value other than
/// INT_MIN continues at its negation, 0 or INT_MIN ends it; reads stay below `limit`.
void read_row(const std::vector<std::int32_t>& payload, std::int64_t limit, std::int64_t value,
              cbst_key_record& out) {
  out.row.clear();
  std::int64_t loc = value;
  std::int32_t segments = 1;
  bool terminated = false;
  for (std::int64_t steps = 0; steps <= limit; ++steps) {
    if (loc < 0 || loc >= limit) break;
    const std::int32_t x = loc < static_cast<std::int64_t>(payload.size())
                               ? payload[static_cast<std::size_t>(loc)]
                               : 0;
    if (x < 0 && x != INT_MIN) {
      loc = -static_cast<std::int64_t>(x);
      ++segments;
      continue;
    }
    if (x == 0 || x == INT_MIN) {
      terminated = true;
      break;
    }
    out.row.push_back(x);
    ++loc;
  }
  out.segments = segments;
  out.terminated = terminated;
  out.row_len = static_cast<std::int32_t>(out.row.size());
}

}  // namespace

std::vector<cbst_key_record> cbst_host_model::records() const {
  std::vector<cbst_key_record> out(impl_->nodes.size());
  const std::int64_t limit = std::min(impl_->bump, impl_->capacity);
  for (std::size_t r = 0; r < out.size(); ++r) {
    const auto t = static_cast<std::size_t>(impl_->position_of[r]);
    const auto& nd = impl_->nodes[t];
    cbst_key_record& k = out[r];
    k.key = nd.index;
    k.avail = impl_->avail[t];
    k.length = nd.length;
    k.size = nd.size;
    k.occupancy = nd.occupancy;
    k.tail_capacity = nd.tail_capacity;
    k.value = nd.value;
    k.tail_base = nd.tail_base;
    read_row(impl_->payload, limit, nd.value, k);
  }
  return out;
}

bool cbst_host_model::record(std::int32_t key, cbst_key_record& out) const {
  const std::int64_t t = impl_->find(key);
  if (t < 0) return false;
  const auto& nd = impl_->nodes[static_cast<std::size_t>(t)];
  out = cbst_key_record{};
  out.key = nd.index;
  out.avail = impl_->avail[static_cast<std::size_t>(t)];
  out.length = nd.length;
  out.size = nd.size;
  out.occupancy = nd.occupancy;
  out.tail_capacity = nd.tail_capacity;
  out.value = nd.value;
  out.tail_base = nd.tail_base;
  read_row(impl_->payload, std::min(impl_->bump, impl_->capacity), nd.value, out);
  return true;
}

std::vector<std::string> cbst_host_model::check() const {
  std::vector<std::string> problems;
  const auto& m = *impl_;
  const std::int64_t count = m.n();
  for (std::int64_t t = 0; t < count; ++t) {
    std::int64_t want = m.avail[static_cast<std::size_t>(t)];
    if (2 * t + 1 < count) want += m.subtree[static_cast<std::size_t>(2 * t + 1)];
    if (2 * t + 2 < count) want += m.subtree[static_cast<std::size_t>(2 * t + 2)];
    if (want != m.subtree[static_cast<std::size_t>(t)]) {
      problems.push_back("subtree_avail at heap position " + std::to_string(t) + " is " +
                         std::to_string(m.subtree[static_cast<std::size_t>(t)]) + ", expected " +
                         std::to_string(want));
      break;
    }
  }
  const std::vector<std::int32_t> k = keys();
  for (std::size_t i = 1; i < k.size(); ++i) {
    if (k[i - 1] >= k[i]) {
      problems.push_back("keys out of BST order at rank " + std::to_string(i));
      break;
    }
  }
  for (const cbst_key_record& r : records()) {
    if (!r.terminated) {
      problems.push_back("the row of key " + std::to_string(r.key) +
                         " does not end in the payload");
      break;
    }
  }
  return problems;
}

}  // namespace dyng::testing
