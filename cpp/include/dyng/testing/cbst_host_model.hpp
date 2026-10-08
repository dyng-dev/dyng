// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_host_model.hpp
 * @brief dyng::testing::cbst_host_model: a sequential host model of the ESCHER CBST store of
 *        ESCHER-GPU\@abcf9b2 and MOSP_ESCHER\@4b86159, the oracle of the store port (M8a).
 * @ingroup testing
 *
 * The CBST ("complete binary search tree") store keeps one row of positive 32-bit values per
 * key in a flat payload array: a complete binary search tree of nodes in heap order (the node at
 * heap position t holds the key of in-order rank cbst_rank_of_position(t, n)), and per node the
 * offset and length of the row's first segment, the tail segment the next append writes to, and
 * an availability bit (a deleted key whose slot an insert may reuse). Rows end with 0 or INT_MIN;
 * a negative value other than INT_MIN continues the row at its negation (an overflow segment).
 *
 * The model replays the store's operations one by one on the host and predicts everything the
 * CBST trace of either original records (ADR 0036): the best-fit InsertMapping, every row in
 * read order, avail, subtree_avail, the slot lengths and offsets, the bump pointer, the payload
 * capacity, every payload growth and compaction, and the exception of a failing operation. It
 * keeps the flat payload itself, because the originals' observable rows depend on it (erase
 * keeps a dead row's values, a reused slot is cleared only up to its first segment, the thread-
 * and warp-level unfill kernels treat a row differently). Growth and compaction are PREDICTED
 * from the policy (cbst_host_policy); the model never reads them from a trace.
 *
 * It is an oracle (PLAN Section 5.8): it shares no code with the dynG store (cpp/src/hypergraph)
 * and includes nothing of it; an include-graph test checks this. It follows the originals'
 * documented algorithms (best-fit matching, the surplus rebuild, the fill overflow plan, the
 * unfill compaction, MOSP_ESCHER's compaction) in plain sequential C++. ADR 0037 describes its
 * scope and the comparison rules of the replay.
 *
 * This header is the documented exception to the rule that no public header exposes CBST keys,
 * offsets or sentinels (the subject of this oracle is the CBST layout itself); the hypergraph
 * API hides them.
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dyng::testing {

/**
 * @brief The in-order rank of the node at heap position `position` of a complete binary search
 *        tree with `num_nodes` nodes (the CBST layout of both originals).
 *
 * With k = floor(log2(position + 1)) and L = floor(log2(num_nodes)), the rank is
 * min(i, i - i/2 + (num_nodes + 1 - 2^L)) - 1 for i = (2(position + 1 - 2^k) + 1) * 2^(L - k),
 * all in 64 bits.
 * @param[in] position  The heap position, 0 <= position < num_nodes.
 * @param[in] num_nodes The number of nodes, at least 1.
 * @return The rank, in [0, num_nodes), or -1 when the arguments are out of range.
 * @ingroup testing
 */
[[nodiscard]] std::int64_t cbst_rank_of_position(std::int64_t position,
                                                 std::int64_t num_nodes) noexcept;

/**
 * @brief The findings of check_cbst_layout().
 * @ingroup testing
 */
struct cbst_layout_check {
  std::int64_t num_nodes = 0;           ///< the tree size checked
  std::int64_t not_a_permutation = 0;   ///< ranks out of range or repeated
  std::int64_t not_in_order = 0;        ///< in-order traversal positions with a wrong rank
  std::int64_t not_found = 0;           ///< keys a search from the root does not reach
  std::int64_t wrong_offset = 0;        ///< keys found at a node that holds another row
  std::int64_t inverse_mismatches = 0;  ///< positions whose recovered key differs
  /**
   * @brief Whether the layout passed.
   * @return true if every counter is 0.
   */
  [[nodiscard]] bool ok() const noexcept {
    return not_a_permutation == 0 && not_in_order == 0 && not_found == 0 && wrong_offset == 0 &&
           inverse_mismatches == 0;
  }
};

/**
 * @brief The layout checks of ESCHER-GPU's tests/cbst_test.cu on the model's layout function.
 *
 * Keys 1..n are placed at their heap positions with row k at offset 4(k - 1); the check counts
 * (1) ranks that do not form a permutation, (2) heap positions whose in-order traversal rank is
 * wrong, (3) keys a BST search from the root does not find (searched for every key when
 * `search_every_key`, else for a deterministic sample of at most 65,536 keys), (4) found nodes
 * whose offset is not that of the key's row and (5) positions whose key, recovered by the layout
 * formula, differs from the key stored there.
 * @param[in] num_nodes        The tree size, 1 <= num_nodes <= 2^30.
 * @param[in] search_every_key Search every key (cost n log n) or a sample.
 * @return The findings.
 * @ingroup testing
 */
[[nodiscard]] cbst_layout_check check_cbst_layout(std::int64_t num_nodes,
                                                  bool search_every_key = true);

/**
 * @brief How a cbst_host_model behaves where the originals differ (and where the dynG presets
 *        will choose): payload growth, compaction, failures, the construct, scratch buffers.
 *
 * The presets reproduce each original: escher_gpu() is ESCHER-GPU\@abcf9b2, mosp_escher() is
 * MOSP_ESCHER\@4b86159. dyng_default() is the target policy of PLAN 6.4.5 ("compact first, then
 * grow, fail only at the int32 offset limit"), provisional until the dynG store's presets are
 * merged (M8a step 4). Every field can be set on its own.
 * @ingroup testing
 */
struct cbst_host_policy {
  /// What happens when an operation needs more payload than the capacity.
  enum class growth_rule {
    none,       ///< fail (MOSP_ESCHER: a fixed capacity)
    escher_gpu  ///< grow to max(needed + needed/4, 3 * capacity / 2), at most INT_MAX (ESCHER-GPU)
  };
  /// Whether and where the payload is compacted.
  enum class compaction_rule {
    none,        ///< never (ESCHER-GPU)
    mosp_escher  ///< MOSP_ESCHER's trigger points: a fill whose overflow segments do not fit
                 ///< (compact, then refill the rest once), an insert whose surplus rows do not
                 ///< fit (compact before appending them), and compact()
  };
  /// How a failure is reported.
  enum class failure_rule {
    process_exit,  ///< the original ends the process (ESCHER-GPU: exit(2) / checkCuda's exit)
    exception      ///< the original throws (MOSP_ESCHER: escher::EscherError)
  };
  /// The tail occupancy of a constructed row.
  enum class occupancy_rule {
    scan_payload,           ///< the values before the row's terminator, whatever the caller
                            ///< passes (ESCHER-GPU's initTailMetadata)
    caller_counts_or_zero,  ///< the caller's row counts, or 0 without them (MOSP_ESCHER)
    caller_counts_or_scan   ///< the caller's row counts, or the scan without them
  };
  /// How the per-operation input buffers grow.
  enum class scratch_rule {
    exact,     ///< to the operation's size (ESCHER-GPU's reserveScratch)
    geometric  ///< by +n/2+1 steps from the current size (MOSP_ESCHER's ensureScratchCapacity)
  };

  std::string name;                                         ///< the preset name (reports)
  growth_rule growth = growth_rule::escher_gpu;             ///< payload growth
  compaction_rule compaction = compaction_rule::none;       ///< payload compaction
  failure_rule failure = failure_rule::process_exit;        ///< how failures are reported
  occupancy_rule occupancy = occupancy_rule::scan_payload;  ///< construct occupancy
  scratch_rule scratch = scratch_rule::exact;               ///< scratch growth
  bool validate_construct = false;  ///< reject negative sizes and a payload above the capacity
                                    ///< with an error (MOSP_ESCHER); ESCHER-GPU returns without
                                    ///< building the tree when the payload does not fit
  bool clamp_fill_tail = false;     ///< a tail with occupancy above its capacity has no room
                                    ///< (MOSP_ESCHER's max(0, ...)); unclamped it is undefined
  bool has_rebuild = true;          ///< rebuild() exists (ESCHER-GPU's rebuildFromDevice)
  bool has_compact = false;         ///< compact() exists (MOSP_ESCHER)
  bool counts_bin_launches = true;  ///< degree-bin launches are counted (ESCHER-GPU)
  std::string error_type;           ///< the exception type a failure reports
                                    ///< ("escher::EscherError" for MOSP_ESCHER)
  std::int64_t max_payload = 2147483647;  ///< the largest payload capacity a growth reaches
                                          ///< (2^31 - 1: the int32 offsets; tests lower it)

  /**
   * @brief ESCHER-GPU\@abcf9b2: growth, no compaction, process exit on failure, scanned
   *        occupancy, exact scratch, rebuild and launch counters.
   * @return The preset.
   */
  [[nodiscard]] static cbst_host_policy escher_gpu();
  /**
   * @brief MOSP_ESCHER\@4b86159: a fixed capacity, compaction at its trigger points, then
   *        escher::EscherError; validated construct with the caller's row counts, the fill
   *        clamps, geometric scratch, compact().
   * @return The preset.
   */
  [[nodiscard]] static cbst_host_policy mosp_escher();
  /**
   * @brief dynG's default, the target policy of PLAN 6.4.5 (M8a step 5): MOSP_ESCHER's
   *        compaction at its trigger points first, then growth by ESCHER-GPU's rule where
   *        MOSP_ESCHER throws, a failure only beyond max_payload (dyng::capacity_error); scanned
   *        occupancy when the caller gives no counts; the construct validation and the fill
   *        clamps; geometric scratch; rebuild, compact() and the launch counters.
   * @return The preset.
   */
  [[nodiscard]] static cbst_host_policy dyng_default();
  /**
   * @brief The preset of a name: "escher_gpu", "mosp_escher" or "dyng_default".
   * @param[in] name The name.
   * @param[out] out The preset.
   * @return false if the name is unknown.
   */
  [[nodiscard]] static bool from_name(const std::string& name, cbst_host_policy& out);
};

/**
 * @brief How an operation of the model ended.
 * @ingroup testing
 */
enum class cbst_op_status {
  ok,            ///< it completed
  error,         ///< it failed and the original throws (cbst_op_result::error_type)
  process_exit,  ///< it failed and the original ends the process
  undefined      ///< the original's result is not defined (a race, an out-of-bounds access,
                 ///< an endless chain); the model cannot predict it (cbst_op_result::message)
};

/**
 * @brief The outcome of one operation of the model.
 * @ingroup testing
 */
struct cbst_op_result {
  cbst_op_status status = cbst_op_status::ok;  ///< how it ended
  std::string error_type;                      ///< the original's exception type (status error)
  std::string message;                         ///< what failed (status other than ok)
  std::vector<std::int32_t> mapping;           ///< insert: the key of every input item
  std::int64_t compact_return = 0;             ///< compact(): the new bump pointer
  std::int64_t reusable_slots = -1;            ///< insert: D, the deleted slots at entry
  bool grew = false;                           ///< the payload capacity grew
  std::int64_t compactions = 0;                ///< payload compactions run
  bool compaction_changed = false;             ///< a compaction moved a row, changed a length or
                                               ///< lowered the bump pointer
  std::array<std::int64_t, 9> bin_launches{};  ///< degree-bin launches, (fill, reuse, unfill)
                                               ///< x (thread, warp, block): one per non-empty bin
};

/**
 * @brief One key of a model tree, as the originals' state digests describe it.
 * @ingroup testing
 */
struct cbst_key_record {
  std::int32_t key = 0;            ///< the key (1-based in both originals)
  std::int32_t avail = 0;          ///< 1: a deleted key whose slot an insert may reuse
  std::int32_t length = 0;         ///< slots of the first segment, the terminator included
  std::int32_t size = 0;           ///< the node's size field (the payload size it was built at)
  std::int32_t occupancy = 0;      ///< values in the tail segment
  std::int32_t tail_capacity = 0;  ///< usable slots of the tail segment
  std::int32_t value = 0;          ///< offset of the first segment
  std::int32_t tail_base = 0;      ///< offset of the tail segment
  std::int32_t segments = 1;       ///< segments the row walk visited
  std::int32_t row_len = 0;        ///< values of the row
  bool terminated = true;          ///< the row walk ended at a terminator inside the payload
  std::vector<std::int32_t> row;   ///< the row in read order
};

/**
 * @brief A payload growth or compaction of a model tree.
 * @ingroup testing
 */
struct cbst_model_event {
  /// What happened.
  enum class kind {
    growth,     ///< the payload capacity grew
    compaction  ///< the payload was compacted
  };
  kind what = kind::growth;   ///< growth or compaction
  std::int64_t op_index = 0;  ///< the tree's operation (0-based count of operations)
  std::int64_t before = 0;    ///< capacity (growth) or bump pointer (compaction) before
  std::int64_t after = 0;     ///< capacity or bump pointer after
};

/**
 * @brief A sequential host model of one CBST of ESCHER-GPU\@abcf9b2 / MOSP_ESCHER\@4b86159.
 *
 * The operations take the originals' inputs: keys, the flat values of every item and their
 * inclusive prefix sums (item i owns values [prefix[i-1], prefix[i])). Keys are those of the
 * originals (1-based; requested insert keys are ignored, as by both originals). The model is
 * not thread-safe; copies are independent.
 * @ingroup testing
 */
class cbst_host_model {
 public:
  /**
   * @brief An empty tree (the originals' CBSTOperations constructor).
   * @param[in] policy           The behaviour.
   * @param[in] payload_capacity The payload capacity in values.
   * @param[in] alignment        The row alignment of surplus inserts.
   * @param[in] name             The dataset name (reports).
   */
  cbst_host_model(cbst_host_policy policy, std::int64_t payload_capacity,
                  std::int32_t alignment = 4, std::string name = {});
  /// @brief Releases the model.
  ~cbst_host_model();
  /// @brief A deep copy. @param[in] other The model.
  cbst_host_model(const cbst_host_model& other);
  /// @brief A deep copy. @param[in] other The model. @return This model.
  cbst_host_model& operator=(const cbst_host_model& other);
  /// @brief Moves. @param[in,out] other The model.
  cbst_host_model(cbst_host_model&& other) noexcept;
  /// @brief Moves. @param[in,out] other The model. @return This model.
  cbst_host_model& operator=(cbst_host_model&& other) noexcept;

  /**
   * @brief Builds the tree over a flattened payload (constructCBST).
   * @param[in] keys        The keys in sorted order, one per row.
   * @param[in] starts      The offset of every row in `flat`.
   * @param[in] flat        The flattened rows (`flat.size()` values are used).
   * @param[in] row_counts  The caller's row counts (MOSP_ESCHER's rowOccupancy), or nullptr.
   * @return The outcome.
   */
  cbst_op_result construct(const std::vector<std::int32_t>& keys,
                           const std::vector<std::int32_t>& starts,
                           const std::vector<std::int32_t>& flat,
                           const std::vector<std::int32_t>* row_counts = nullptr);
  /**
   * @brief Rebuilds the tree over a new payload with keys 1..n (rebuildCBSTFromDevice).
   * @param[in] starts           The offset of every row.
   * @param[in] flat             The payload (its used prefix).
   * @param[in] flat_size        The used payload size.
   * @param[in] payload_capacity The new capacity.
   * @return The outcome.
   */
  cbst_op_result rebuild(const std::vector<std::int32_t>& starts,
                         const std::vector<std::int32_t>& flat, std::int64_t flat_size,
                         std::int64_t payload_capacity);
  /**
   * @brief Inserts rows: best-fit reuse of deleted slots, then surplus keys (insertCBST).
   * @param[in] keys    The requested keys (ignored, as by both originals; only their number).
   * @param[in] payload The values of every item.
   * @param[in] prefix  The inclusive prefix sums of the item sizes.
   * @return The outcome; `mapping` holds the key of every item.
   */
  cbst_op_result insert(const std::vector<std::int32_t>& keys,
                        const std::vector<std::int32_t>& payload,
                        const std::vector<std::int32_t>& prefix);
  /**
   * @brief Marks keys as deleted (deleteCBST); the rows keep their values.
   * @param[in] keys The keys; missing keys are ignored.
   * @return The outcome.
   */
  cbst_op_result erase(const std::vector<std::int32_t>& keys);
  /**
   * @brief Appends values to rows (fillCBST): into the tail, then overflow segments.
   * @param[in] keys    The keys, one per item.
   * @param[in] payload The values of every item.
   * @param[in] prefix  The inclusive prefix sums of the item sizes.
   * @return The outcome.
   */
  cbst_op_result append(const std::vector<std::int32_t>& keys,
                        const std::vector<std::int32_t>& payload,
                        const std::vector<std::int32_t>& prefix);
  /**
   * @brief Removes every occurrence of values from rows (unfillCBST).
   * @param[in] keys   The keys, one per item.
   * @param[in] values The values to remove of every item.
   * @param[in] prefix The inclusive prefix sums of the removal counts.
   * @return The outcome.
   */
  cbst_op_result remove_values(const std::vector<std::int32_t>& keys,
                               const std::vector<std::int32_t>& values,
                               const std::vector<std::int32_t>& prefix);
  /**
   * @brief Compacts the payload (MOSP_ESCHER's compactCBST).
   * @return The outcome; `compact_return` is the new bump pointer.
   */
  cbst_op_result compact();

  /// @brief The policy. @return The policy.
  [[nodiscard]] const cbst_host_policy& policy() const noexcept;
  /// @brief The dataset name. @return The name.
  [[nodiscard]] const std::string& name() const noexcept;
  /// @brief The record count of the tree (the originals' numRecords). @return The count.
  [[nodiscard]] std::int64_t num_records() const noexcept;
  /// @brief The nodes that exist (0 when the original allocated none). @return The count.
  [[nodiscard]] std::int64_t num_nodes() const noexcept;
  /// @brief The payload capacity. @return The capacity in values.
  [[nodiscard]] std::int64_t payload_capacity() const noexcept;
  /// @brief The bump pointer (the used payload prefix). @return The offset.
  [[nodiscard]] std::int64_t bump() const noexcept;
  /// @brief The row alignment of surplus inserts. @return The alignment.
  [[nodiscard]] std::int32_t alignment() const noexcept;
  /// @brief The item capacity of the scratch buffers. @return The capacity.
  [[nodiscard]] std::int64_t scratch_items() const noexcept;
  /// @brief The value capacity of the scratch buffers. @return The capacity.
  [[nodiscard]] std::int64_t scratch_payload() const noexcept;
  /**
   * @brief ESCHER-GPU's DynamicHypergraph::deviceBytes() formula on the model's sizes.
   * @return 72 bytes per record, 4 per payload value, 20 per scratch item, 4 per scratch value.
   */
  [[nodiscard]] std::int64_t device_bytes() const noexcept;
  /// @brief The deleted slots (the root's subtree_avail), or -1 without nodes. @return D.
  [[nodiscard]] std::int64_t reusable_slots() const noexcept;
  /// @brief The keys in BST (in-order) order. @return The keys.
  [[nodiscard]] std::vector<std::int32_t> keys() const;
  /// @brief The deleted slots of every subtree, by heap position. @return The counts.
  [[nodiscard]] std::vector<std::int32_t> subtree_avail() const;
  /// @brief Every key's record and row, in BST order. @return The records.
  [[nodiscard]] std::vector<cbst_key_record> records() const;
  /**
   * @brief One key's record and row.
   * @param[in] key The key.
   * @param[out] out The record.
   * @return false if no node holds the key.
   */
  [[nodiscard]] bool record(std::int32_t key, cbst_key_record& out) const;
  /// @brief The growths and compactions so far. @return The events.
  [[nodiscard]] const std::vector<cbst_model_event>& events() const noexcept;
  /// @brief The operations applied so far. @return The count.
  [[nodiscard]] std::int64_t num_operations() const noexcept;
  /**
   * @brief Whether a failed operation ended the original's process (the tree takes no more
   *        operations: they return cbst_op_status::undefined).
   * @return true after a process_exit.
   */
  [[nodiscard]] bool exited() const noexcept;
  /**
   * @brief The model's own consistency check: subtree_avail sums (MOSP_ESCHER's
   *        checkSubtreeAvail), BST order of the keys, rows that end inside the payload.
   * @return The problems found (empty: none).
   */
  [[nodiscard]] std::vector<std::string> check() const;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace dyng::testing
