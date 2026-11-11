// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file trace.hpp
 * @brief label_propagation's stage trace for the parity harness (ADR 0073): the records of
 *        LabelPropagation-CUDA@a276a3a's recording twin (ADR 0065), written by the library
 *        itself when the environment variable DYNG_DYNLP_TRACE_DIR is set.
 *
 * The format is the twin's (parity/exporters/dynlp/dynlp_trace.hpp; the reader
 * parity/dynlp_trace.py): the file <DYNG_DYNLP_TRACE_DIR>/<program>.dlt, never overwritten;
 * with DYNG_DYNLP_TRACE_DIGEST=1 every array is stored as its element count and SHA-256. The
 * records of one update are the twin's of one batch (apply_vertex_changes, append_edges,
 * sparsify, connected_components, label_init, propagation_begin, propagation_end), with the
 * statuses in the original's codes by name (ADR 0069) and the batch number the result's update
 * count; compute() writes an irlp_snapshot record. dyng-compat-dynlp --trace DIR sets the variable
 * (it includes public headers only). Without the variable nothing is written and the cost is one
 * test of a pointer per stage.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dyng::detail::lp_trace {

/// The field types of the format.
enum dtype : std::uint8_t { i32 = 1, i64 = 2, u64 = 3, f32 = 4, u8 = 5, f64 = 6, text = 7 };

/// One record under construction.
struct record {
  std::string tag;                  ///< the stage
  std::int32_t batch = -1;          ///< the batch (-1 outside the batch loop)
  std::vector<std::uint8_t> bytes;  ///< the encoded fields
  std::uint32_t fields = 0;         ///< their number
};

/// The process's trace (one file per process).
class writer {
 public:
  /**
   * @brief The trace, opened on first use when DYNG_DYNLP_TRACE_DIR is set.
   * @return The writer, or nullptr when tracing is off.
   * @throws io_error if the trace cannot be created (or exists already).
   */
  static writer* get();

  writer(const writer&) = delete;             ///< not copyable
  writer& operator=(const writer&) = delete;  ///< not copyable
  ~writer();                                  ///< closes the trace (its end record)

  /// A new record.
  [[nodiscard]] record begin(const char* tag, std::int32_t batch) const;
  /// A scalar or raw field.
  void add(record& r, const char* name, dtype type, const void* data, std::uint64_t count) const;
  /// An int32 scalar.
  void add_i32(record& r, const char* name, std::int32_t v) const {
    add(r, name, i32, &v, 1);
  }
  /// An int64 scalar.
  void add_i64(record& r, const char* name, std::int64_t v) const {
    add(r, name, i64, &v, 1);
  }
  /// A float scalar (its bits).
  void add_f32(record& r, const char* name, float v) const {
    add(r, name, f32, &v, 1);
  }
  /// An array field: raw, or its SHA-256 in the digest mode.
  void add_array(record& r, const char* name, dtype type, const void* data,
                 std::uint64_t count) const;
  /// Append the record to the file.
  void end_record(const record& r);
  /// Close the trace with its end record (also at exit).
  void close();

 private:
  writer(const std::string& path, bool digest);
  std::FILE* f_ = nullptr;
  bool digest_ = false;
  std::uint64_t records_ = 0;
  std::vector<std::pair<std::string, std::int32_t>> sequence_;
};

/// The original's VertexStatus codes (headers/graphTypes.h), mapped by name (ADR 0069).
enum original_status : std::int32_t {
  original_active = 0,
  original_deleted = 1,
  original_ground_truth = 2,
  original_inactive = 3
};

}  // namespace dyng::detail::lp_trace
