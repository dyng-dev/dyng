// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The H-SOSP dump writer of the parity harness (PLAN Sections 6.4.8 and 8.3; ADR 0081).
//
// parity/export_patches/MOSP_ESCHER/build.sh compiles this writer and its adapter
// (recorder_mosp_escher_hsosp.cpp) against the PATCHED scratch copy's unchanged headers and
// relinks the copy's unchanged objects with them through `-Xlinker --wrap=<symbol>`
// (parity/exporters/escher/relink_twins.sh, its --extra group): every call of a wrapped H-SOSP
// entry point from another object reaches the adapter's __wrap_ function, which calls __real_
// (the original, unchanged) and records its inputs and outputs. No original file is changed and
// no original code is copied: the adapter reads the documented fields of the original's structs
// (HostHypergraph, DeviceH2H, HsospState, BatchTimes, UpdateConfig, UpdateStats) and calls the
// original's own download functions.
//
// Recording is on only when DYNG_HSOSP_DUMP_DIR is set; otherwise every __wrap_ function calls
// __real_ and returns (one getenv at the first call). It is independent of the CBST trace
// recorder (DYNG_CBST_TRACE_DIR): either, both or neither can be on.
//
// The dump format (version 1; readers: parity/hsosp_dump.py): one file per process,
// <DYNG_HSOSP_DUMP_DIR>/NNNNNN.hsd (numbered in the order of first use), "DYNGHSD1", u32 format
// version, u32 0, then records: u32 tag, u64 payload length, payload. The payload is the field
// list of the CBST trace format (ADR 0036 section 2: u32 count, then per field u8 type, u16 name
// length, name, value; types 1 int64, 2 string, 3 array (u8 storage 0 = inline, u8 element size
// 4 or 8, u64 count, the values), 4 string list). Little-endian.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dyng_parity::hsosp_dump {

/// Version of this writer (written into every dump header).
inline constexpr const char* writer_version = "1.0.0";
/// Version of the dump format.
inline constexpr std::uint32_t format_version = 1;

/// Record tags.
enum tag : std::uint32_t {
  tag_header = 1,        ///< the process: argv, inputs, environment
  tag_session = 2,       ///< a DynamicHypergraph(numVertices, caps)
  tag_load = 3,          ///< DynamicHypergraph::bulkLoad (phase enter / exit)
  tag_device_build = 4,  ///< hsosp::buildDeviceH2H
  tag_begin_batch = 5,   ///< DynamicHypergraph::beginBatch (phase enter / exit)
  tag_apply_batch = 6,   ///< hsosp::applyBatch (exit: the delta, the counters, the rows)
  tag_sssp = 7,          ///< hsosp::hsospUpdate / hsospRecompute (exit: distances, parents)
  tag_compare = 8,       ///< hsosp::compareDistances
  tag_end = 9,           ///< the process exited normally
};

/// One record's field list.
class fields {
 public:
  fields();
  void i64(const char* name, long long v);
  void str(const char* name, const std::string& s);
  void strings(const char* name, const std::vector<std::string>& v);
  void ints(const char* name, const int* data, std::size_t count);
  void ints(const char* name, const std::vector<int>& v) { ints(name, v.data(), v.size()); }
  void longs(const char* name, const long long* data, std::size_t count);
  void longs(const char* name, const std::vector<long long>& v) {
    longs(name, v.data(), v.size());
  }
  /// Rows as `<name>.offsets` (int64, rows + 1, from 0) and `<name>.values` (int32).
  void rows(const char* name, const std::vector<std::vector<int>>& rows);
  /// A double as its shortest round-trip text ("%.17g").
  void real(const char* name, double v);
  const std::vector<unsigned char>& bytes();

 private:
  void head(std::uint8_t type, const char* name);
  void array(const char* name, std::uint8_t elem, const void* data, std::uint64_t count);
  void put_u32(std::uint32_t v);
  void put_u64(std::uint64_t v);
  std::vector<unsigned char> buf_;
  std::uint32_t count_ = 0;
};

/// The process-wide dump: nullptr when DYNG_HSOSP_DUMP_DIR is not set (or the file cannot be
/// opened). Opened at the first call; finished at exit.
class dump;
dump* get();

/// Writes one record (thread-safe); `seq` is added, the global record order.
void write(dump* d, tag t, fields& f);

/// Whether the post-batch rows of every hyperedge are recorded (DYNG_HSOSP_DUMP_ROWS=all), or
/// only those of the batch's touched hyperedges (the default, "touched").
bool all_rows(dump* d);

/// The error text of the exception being handled (called inside a catch block).
std::string current_exception_text();

}  // namespace dyng_parity::hsosp_dump
