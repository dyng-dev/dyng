// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_dump.hpp
 * @brief Host-only reader of the H-SOSP dumps of MOSP_ESCHER@4b86159 (dump format 1, ADR 0081),
 *        and the oracle check of a dump against dyng::testing's hyper_sssp oracle (the go/no-go
 *        of M11a criterion 5).
 *
 * The dumps are recorded by the recording twins of `parity/exporters/hsosp` (the original's
 * unchanged objects relinked with `-Xlinker --wrap`; DYNG_HSOSP_DUMP_DIR) and stored in the
 * golden set `hyper_sssp_mosp_escher` (parity/hsosp_goldens.py). A dump is a sequence of records
 * (header, session, load, device_build, begin_batch, apply_batch, sssp, compare, end), each a list
 * of named fields: 64-bit integers, strings, string lists and inline arrays of 32- or 64-bit
 * integers. Host-only: no CUDA and no dynG library code (the oracle check links dyng::testing).
 */
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dyng::test::hsosp_dump {

/// The dump format this reader reads.
inline constexpr std::uint32_t format_version = 1;

/// A malformed or unsupported dump.
class dump_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// One field of a record.
struct field {
  enum class kind { i64, str, strings, array32, array64 };
  kind type = kind::i64;
  std::int64_t value = 0;
  std::string text;
  std::vector<std::string> list;
  std::vector<std::int32_t> ints;
  std::vector<std::int64_t> longs;
};

/// One record: its tag name and its fields (by name).
struct record {
  std::string tag;
  std::map<std::string, field> fields;

  [[nodiscard]] bool has(const std::string& name) const {
    return fields.count(name) != 0;
  }
  /// An int64 field (throws dump_error when it is missing or of another type).
  [[nodiscard]] std::int64_t i64(const std::string& name) const;
  /// A string field.
  [[nodiscard]] const std::string& str(const std::string& name) const;
  /// An int32 array field.
  [[nodiscard]] const std::vector<std::int32_t>& ints(const std::string& name) const;
  /// An int64 array field.
  [[nodiscard]] const std::vector<std::int64_t>& longs(const std::string& name) const;
};

/// Reads a dump (`.hsd`, or `.hsd.zst` through `zstd -dc`).
std::vector<record> read_dump(const std::string& path);
/// Parses a dump from its bytes.
std::vector<record> parse_dump(const std::string& bytes);

/// What the oracle check of a dump found.
struct oracle_report {
  std::int64_t sessions = 0;
  std::int64_t sssp_checked = 0;         ///< sssp records compared with the oracle
  std::int64_t distance_mismatches = 0;  ///< over all compared records and ids
  std::int64_t parent_mismatches = 0;    ///< over all compared records and ids
  std::int64_t batches_checked = 0;      ///< apply_batch records whose pairs were compared
  std::int64_t pair_mismatches = 0;      ///< batches whose deleted or inserted pairs differ
  std::int64_t not_checked = 0;          ///< sssp records on a state the dump does not give
  std::vector<std::string> problems;     ///< the first differences, as text

  [[nodiscard]] bool ok() const {
    return distance_mismatches == 0 && parent_mismatches == 0 && pair_mismatches == 0 &&
           problems.empty();
  }
};

/**
 * @brief The go/no-go of M11a criterion 5 on one dump: every session's hypergraph is rebuilt from
 *        its records (the loaded rows, then each batch's post-batch rows), and on every state
 *        dyng::testing::hyper_sssp_reference (source = the session's source row, B's weights:
 *        the virtual rows weigh 0) must reproduce the distances and parents of every recorded
 *        hsospUpdate and hsospRecompute by id, and dyng::testing::line_graph_delta(pre, post)
 *        the deleted and inserted pairs of every batch.
 * @param records The dump.
 * @param max_problems How many differences to describe.
 * @return The findings.
 */
oracle_report check_with_oracle(const std::vector<record>& records, std::size_t max_problems = 20);

}  // namespace dyng::test::hsosp_dump
