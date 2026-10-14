// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_trace.hpp
 * @brief Host-only reader of the CBST traces of ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159
 *        (trace format 1, ADR 0036).
 *
 * The traces are recorded by the recording twins of `parity/exporters/escher` (the originals'
 * unchanged objects relinked with `-Xlinker --wrap`; PLAN 6.4.5 and 8.3). A trace is a sequence
 * of records (header, tree, op, event, batch, note, end), each a list of named fields: 64-bit
 * integers, strings, string lists and arrays of 32- or 64-bit integers. An array is stored
 * inline, in the trace's side file (`<trace>.blob`, large inputs in digest mode), as a hash only
 * (labelled; DYNG_CBST_TRACE_HASH_ONLY) or as null.
 *
 * reader parses a trace from any `std::istream` (a large trace is decompressed by the caller,
 * e.g. `zstd -dc trace.cbt.zst |`); state_tracker rebuilds the state of every tree after every
 * operation from the state digests (full digests, and delta digests that write only the keys
 * that changed) and recomputes every recorded hash. The comparator of M8a step 2 and the store
 * replays build on these two classes. Host-only: no CUDA, no dynG library code.
 */
#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::cbst_trace {

/// The trace format this reader reads.
inline constexpr std::uint32_t format_version = 1;

/// A malformed or unsupported trace.
class trace_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// The kinds of records.
enum class record_kind { header = 1, tree = 2, op = 3, event = 4, batch = 5, note = 6, end = 7 };

/// The name of a record kind ("header", "op", ...).
const char* to_string(record_kind kind);

/// An integer array field.
struct array_field {
  /// Where the values are.
  enum class storage {
    inline_data,  ///< in the trace
    side_file,    ///< in the side file, at `offset`
    hash_only,    ///< only their FNV-1a 64 hash was recorded
    null          ///< no array (a null pointer in the original's call)
  };
  storage kind = storage::null;
  int elem_bytes = 4;             ///< 4 (int32) or 8 (int64)
  std::uint64_t count = 0;        ///< number of values
  std::uint64_t offset = 0;       ///< byte offset in the side file (side_file)
  std::uint64_t hash = 0;         ///< FNV-1a 64 of the values (side_file, hash_only)
  std::vector<std::int32_t> i32;  ///< the values (inline, elem_bytes == 4)
  std::vector<std::int64_t> i64;  ///< the values (inline, elem_bytes == 8)
};

/// One named field of a record.
struct field {
  /// The field's type.
  enum class type { integer, text, array, texts };
  type kind = type::integer;
  std::int64_t integer = 0;        ///< type::integer
  std::string text;                ///< type::text
  std::vector<std::string> texts;  ///< type::texts
  array_field array;               ///< type::array
};

/// One record: its kind and its fields in file order.
struct record {
  record_kind kind = record_kind::note;
  std::vector<std::pair<std::string, field>> fields;

  /// The field `name`, or nullptr.
  [[nodiscard]] const field* find(const std::string& name) const;
  /// Whether the record has the field `name`.
  [[nodiscard]] bool has(const std::string& name) const {
    return find(name) != nullptr;
  }
  /// The integer field `name`, or `fallback` when it is absent.
  [[nodiscard]] std::int64_t integer(const std::string& name, std::int64_t fallback = 0) const;
  /// The text field `name` ("" when absent).
  [[nodiscard]] std::string text(const std::string& name) const;
  /// The array field `name`; throws trace_error when it is absent or not an array.
  [[nodiscard]] const array_field& array(const std::string& name) const;
};

/// Reads the records of one trace.
class reader {
 public:
  /// Reads from `in` (the trace bytes, not compressed). Checks the magic and the version.
  explicit reader(std::istream& in);
  ~reader();
  reader(const reader&) = delete;
  reader& operator=(const reader&) = delete;

  /// Opens the trace file `path` and its side file `<trace>.blob` if it exists.
  static std::unique_ptr<reader> open(const std::string& path);

  /// Uses `path` as the side file for side-file arrays.
  void set_side_file(const std::string& path);

  /// The next record; false at the end of the trace. Throws trace_error on malformed data.
  bool next(record& out);

  /// The int32 values of an array field: inline, or read from the side file; throws
  /// trace_error for a hash-only array or a missing side file.
  std::vector<std::int32_t> ints(const array_field& a);
  /// The int64 values of an array field (as ints()).
  std::vector<std::int64_t> longs(const array_field& a);

 private:
  struct impl;
  std::unique_ptr<std::istream> owned_;
  std::istream* in_;
  std::unique_ptr<impl> impl_;
};

/// FNV-1a 64 of `n` bytes, continuing from `h` (the hash of the trace format).
std::uint64_t fnv1a64(const void* data, std::size_t n, std::uint64_t h = 0xcbf29ce484222325ULL);

/// One key of a tree as the digests describe it.
struct key_state {
  std::int32_t key = 0;
  std::int32_t avail = 0;          ///< 1: a deleted (reusable) slot
  std::int32_t length = 0;         ///< slots of the first segment, sentinel included
  std::int32_t size = 0;           ///< the node's `size` field
  std::int32_t occupancy = 0;      ///< values in the tail segment
  std::int32_t tail_capacity = 0;  ///< usable slots of the tail segment
  std::int32_t value = 0;          ///< offset of the first segment
  std::int32_t tail_base = 0;      ///< offset of the tail segment
  std::int32_t segments = 1;       ///< segments of the row (chain hops + 1)
  std::int32_t row_len = 0;        ///< values of the row
  std::int32_t terminated = 1;     ///< 0: the row walk did not end
  std::uint64_t row_hash = 0;      ///< FNV-1a 64 of the row
  bool row_known = false;          ///< false: the row was recorded as a hash only
  std::vector<std::int32_t> row;   ///< the row in read order (when row_known)
};

/// One tree after an operation.
struct tree_state {
  std::uint32_t id = 0;
  std::string name;                   ///< the dataset name of the original
  std::int64_t payload_capacity = 0;  ///< from the tree record
  std::int64_t fixed_size = 0;        ///< payload capacity now
  std::int64_t num_records = 0;
  std::int64_t initial_payload_size = 0;  ///< the bump pointer
  std::int64_t alignment = 0;
  std::vector<std::int32_t> keys;           ///< BST (in-order) order
  std::vector<std::int32_t> subtree_avail;  ///< by heap position
  std::map<std::int32_t, key_state> by_key;
};

/// Rebuilds the trees of a trace from its records.
class state_tracker {
 public:
  /// Applies a record: a tree record registers a tree, an op record's digest updates its tree.
  /// Returns the problems found (a hash that does not match the rebuilt state; empty: none).
  std::vector<std::string> apply(const record& r, reader& source);
  /// The tree `id`, or nullptr.
  [[nodiscard]] const tree_state* tree(std::uint32_t id) const;
  /// All trees by id.
  [[nodiscard]] const std::map<std::uint32_t, tree_state>& trees() const {
    return trees_;
  }

 private:
  std::map<std::uint32_t, tree_state> trees_;
};

}  // namespace dyng::test::cbst_trace
