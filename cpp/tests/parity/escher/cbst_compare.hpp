// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_compare.hpp
 * @brief The CBST trace replay comparator (M8a, ADR 0037): drives a store through the operations
 *        of a trace of ESCHER-GPU@abcf9b2 or MOSP_ESCHER@4b86159 and compares its state after
 *        every operation with the state the original recorded (trace format 1, ADR 0036).
 *
 * A store is reached through a small adapter (store_adapter): construct, rebuild, insert (the
 * InsertMapping), erase, append (fill), remove_values (unfill), compact, and a digest of a tree
 * (the scalars, the keys, subtree_avail and every key's record and row), under a named policy.
 * host_model_adapter drives dyng::testing::cbst_host_model; the dynG store's adapter follows in
 * M8a step 3.
 *
 * Compared fields (ADR 0037):
 *  - field_set::semantic (the M8a acceptance criterion 8): the exception of every operation (the
 *    original's type, or the end of its process), every InsertMapping, the keys, subtree_avail,
 *    per key avail, length and the row (in order for H2V, H2H and V2V, as a multiset otherwise),
 *    the bump pointer and the payload capacity, and the operations at which the payload grew or
 *    was compacted (inferred from the states on both sides by the same rule);
 *  - field_set::offsets: also every row's offset and tail base;
 *  - field_set::exact: also the size, occupancy, tail capacity, segment count, row length and
 *    termination of every key, every row in order, the scratch capacities, the record count,
 *    ESCHER-GPU's deviceBytes() and degree-bin launch counts where the trace has them, and the
 *    state before every operation.
 * A row recorded as a hash only is compared by its hash (in order). An operation the store
 * cannot run is a mismatch; an operation whose inputs the trace holds as hashes only cannot be
 * replayed: its tree is reported as unreplayable from there on, never silently skipped.
 */
#pragma once

#include "parity/escher/cbst_trace.hpp"

#include <dyng/testing/cbst_host_model.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dyng::test::cbst_compare {

/// The fields a replay compares.
enum class field_set { semantic, offsets, exact };

/// The name of a field set ("semantic", "offsets", "exact").
const char* to_string(field_set f);
/// Parses a field set name; false if unknown.
bool parse_field_set(const std::string& name, field_set& out);

/// One operation of a trace, decoded.
struct op_request {
  std::string op;  ///< construct, rebuild, insert, fill, erase, unfill, compact
  std::vector<std::int32_t> keys, payload, prefix;  ///< insert, fill, erase, unfill
  std::vector<std::int32_t> starts, flat;           ///< construct, rebuild
  std::vector<std::int32_t> row_counts;             ///< construct (MOSP_ESCHER's rowOccupancy)
  bool has_row_counts = false;                      ///< the caller passed row counts (not nullptr)
  std::int64_t num_records = 0;                     ///< construct, rebuild
  std::int64_t flat_size = 0;                       ///< construct, rebuild
  std::int64_t capacity = 0;   ///< construct (the tree's capacity), rebuild (the new one)
  std::int64_t alignment = 4;  ///< construct
};

/// How a store's operation ended.
enum class op_status { ok, error, process_exit, undefined, unsupported };

/// The outcome of one operation of a store.
struct op_outcome {
  op_status status = op_status::ok;
  std::string error_type;              ///< the exception type (status error)
  std::string message;                 ///< what failed
  std::vector<std::int32_t> mapping;   ///< insert
  std::int64_t compact_return = 0;     ///< compact
  bool has_bins = false;               ///< bins holds degree-bin launch counts
  std::array<std::int64_t, 9> bins{};  ///< (fill, reuse, unfill) x (thread, warp, block)
};

/// One key of a tree digest.
struct key_digest {
  std::int32_t key = 0, avail = 0, length = 0, size = 0, occupancy = 0, tail_capacity = 0;
  std::int32_t value = 0, tail_base = 0, segments = 1, row_len = 0, terminated = 1;
  std::uint64_t row_hash = 0;     ///< FNV-1a 64 of the row (the trace's hash)
  std::vector<std::int32_t> row;  ///< the row in read order
};

/// The state of one tree.
struct tree_digest {
  std::int64_t num_records = 0, fixed_size = 0, bump = 0, alignment = 0;
  std::int64_t scratch_items = -1, scratch_payload = -1, device_bytes = -1, d = -1;
  std::vector<std::int32_t> keys;           ///< BST order
  std::vector<std::int32_t> subtree_avail;  ///< by heap position
  std::vector<key_digest> records;          ///< BST order (digest(), not scalars())
};

/// A store driven by the comparator.
class store_adapter {
 public:
  virtual ~store_adapter() = default;
  /// The store and its policy, for reports.
  [[nodiscard]] virtual std::string name() const = 0;
  /// A new tree object (the originals' CBSTOperations constructor).
  virtual void create_tree(std::uint32_t tree, const std::string& dataset, std::int64_t capacity,
                           std::int64_t alignment) = 0;
  /// The tree object is gone.
  virtual void destroy_tree(std::uint32_t tree) = 0;
  /// Runs one operation on a tree.
  virtual op_outcome run(std::uint32_t tree, const op_request& op) = 0;
  /// The scalars of a tree (keys, subtree_avail and records may be left empty).
  [[nodiscard]] virtual tree_digest scalars(std::uint32_t tree) = 0;
  /// The full state of a tree.
  [[nodiscard]] virtual tree_digest digest(std::uint32_t tree) = 0;
  /// The growth and compaction events the store predicted for a tree: (op index, kind) with
  /// kind 'g' or 'c' (empty when the store does not report them).
  [[nodiscard]] virtual std::vector<std::pair<std::int64_t, char>> events(std::uint32_t tree) {
    (void)tree;
    return {};
  }
};

/// The host model as a store.
class host_model_adapter final : public store_adapter {
 public:
  /// Every tree runs under `policy`.
  explicit host_model_adapter(dyng::testing::cbst_host_policy policy);
  ~host_model_adapter() override;
  [[nodiscard]] std::string name() const override;
  void create_tree(std::uint32_t tree, const std::string& dataset, std::int64_t capacity,
                   std::int64_t alignment) override;
  void destroy_tree(std::uint32_t tree) override;
  op_outcome run(std::uint32_t tree, const op_request& op) override;
  [[nodiscard]] tree_digest scalars(std::uint32_t tree) override;
  [[nodiscard]] tree_digest digest(std::uint32_t tree) override;
  [[nodiscard]] std::vector<std::pair<std::int64_t, char>> events(std::uint32_t tree) override;
  /// The model of a tree (nullptr if there is none).
  [[nodiscard]] const dyng::testing::cbst_host_model* model(std::uint32_t tree) const;

 private:
  dyng::testing::cbst_host_policy policy_;
  std::map<std::uint32_t, std::unique_ptr<dyng::testing::cbst_host_model>> trees_;
};

/// Options of replay().
struct options {
  field_set fields = field_set::semantic;  ///< what is compared
  std::size_t max_mismatches = 20;         ///< mismatches recorded in full (all are counted)
};

/// One difference between the store and the trace.
struct mismatch {
  std::int64_t seq = -1;       ///< the trace record
  std::int64_t op_index = -1;  ///< the operation of the trace (0-based)
  std::int64_t tree_op = -1;   ///< the operation of the tree (0-based)
  std::uint32_t tree = 0;
  std::string dataset, op, field;
  std::int64_t key = -1;         ///< the key, or -1
  std::string expected, actual;  ///< the trace's and the store's value
};

/// What happened to one tree.
struct tree_summary {
  std::uint32_t id = 0;
  std::string dataset;
  std::int64_t ops = 0, compared = 0, mismatches = 0, unreplayable = 0;
  std::int64_t growths = 0, compactions = 0, exceptions = 0;  ///< as seen in the trace
  std::int64_t hash_only_rows = 0;                            ///< rows compared by their hash
  std::string unreplayable_reason;  ///< why the tree stopped being replayed
  bool order_sensitive = false;     ///< rows compared in order (H2V, H2H, V2V)
};

/// The result of one trace replay.
struct replay_report {
  std::string trace, repo, store, fields;
  std::int64_t ops = 0, compared = 0, unreplayable = 0, keys_compared = 0;
  std::int64_t hash_only_rows = 0, mappings = 0, exceptions = 0, growths = 0, compactions = 0;
  std::int64_t mismatch_count = 0;
  std::string exit;                  ///< the trace's end record ("normal", ...)
  std::vector<mismatch> mismatches;  ///< the first options::max_mismatches
  std::map<std::uint32_t, tree_summary> trees;
  std::vector<std::string> notes;
  /// No mismatch (unreplayable operations are reported separately).
  [[nodiscard]] bool ok() const {
    return mismatch_count == 0;
  }
  /// The report as a JSON object.
  [[nodiscard]] std::string to_json() const;
};

/**
 * @brief Replays a trace through a store and compares every operation.
 * @param[in,out] in    The trace (its side file set when it has one).
 * @param[in,out] store The store.
 * @param[in] opt       What to compare.
 * @param[in] name      The trace's name for the report.
 * @return The report.
 */
replay_report replay(cbst_trace::reader& in, store_adapter& store, const options& opt,
                     const std::string& name);

/// Makes the store of a trace, given its original ("ESCHER-GPU", "MOSP_ESCHER").
using store_factory = std::function<std::unique_ptr<store_adapter>(const std::string& repo)>;

/**
 * @brief Replays a trace through the store `make_store` returns for the trace's original (read
 *        from its header record).
 * @param[in,out] in     The trace.
 * @param[in] make_store Makes the store.
 * @param[in] opt        What to compare.
 * @param[in] name       The trace's name for the report.
 * @return The report.
 */
replay_report replay(cbst_trace::reader& in, const store_factory& make_store, const options& opt,
                     const std::string& name);

/**
 * @brief A trace opened for reading: a `.cbt` file (with its `<name>.blob` side file) or a
 *        `.cbt.zst` file (read through `zstd -dc --long=31`; its `trace.blob.zst` side file is
 *        decompressed into a temporary file under `tmp_dir`, removed by the destructor).
 */
class trace_source {
 public:
  /// Opens `path`; throws cbst_trace::trace_error when it cannot.
  trace_source(const std::string& path, const std::string& tmp_dir);
  ~trace_source();
  trace_source(const trace_source&) = delete;
  trace_source& operator=(const trace_source&) = delete;
  /// The reader.
  cbst_trace::reader& reader() {
    return *reader_;
  }

 private:
  struct pipe;
  std::unique_ptr<pipe> pipe_;
  std::unique_ptr<std::istream> stream_;
  std::unique_ptr<cbst_trace::reader> reader_;
  std::string blob_tmp_;
};

/// The host model under `policy`, or under the preset of the trace's original when `policy` is
/// empty (an unknown original then throws cbst_trace::trace_error).
store_factory host_model_factory(const std::string& policy);

/// Whether the rows of a dataset are compared in order (H2V, H2H, V2V; any case).
bool order_sensitive(const std::string& dataset);

/// The default policy of an original ("ESCHER-GPU" -> escher_gpu, "MOSP_ESCHER" -> mosp_escher).
std::string default_policy(const std::string& repo);

}  // namespace dyng::test::cbst_compare
