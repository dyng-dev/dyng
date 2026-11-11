// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The DynLP stage-trace writer of the recording twins of LabelPropagation-CUDA@a276a3a (PLAN 6.3
// step 1, 8.3; ADR 0065). The twins are O's UNCHANGED objects relinked with the wrappers of
// recorder_dynlp.cpp / recorder_dynlp_hub.cpp through -Xlinker --wrap (no source patch;
// parity/export_patches/LabelPropagation-CUDA/build.sh). This file is compiled with the host
// compiler and knows nothing of O's types.
//
// Recording is on only when DYNG_DYNLP_TRACE_DIR is set; the trace is the file
// $DYNG_DYNLP_TRACE_DIR/<program>.dlt (an existing file is never overwritten: the twin stops).
// With DYNG_DYNLP_TRACE_DIGEST=1 every array is stored as its element count and the SHA-256 of
// its little-endian bytes instead of the bytes (the 5M cases; scalars stay as they are). With
// DYNG_DYNLP_TRACE_TAGS=<tag>,<tag>,... only the records of those tags (and the end record) are
// written (M10a step 2: the labels and the --check records of O's oracle runs at 5M without the
// graph's arrays).
//
// File format (little-endian; parity/label_propagation_goldens.py reads it):
//   header   8 bytes "DYNGDLT1", u32 format version (1), u32 flags (bit 0: digest mode)
//   record   u32 magic 0x5250444c ("LDPR"), u16 tag length, tag bytes (ASCII),
//            i32 batch (-1 outside the batch loop), i32 sequence (per tag, from 0),
//            u32 number of fields, the fields
//   field    u16 name length, name bytes, u8 dtype, u8 storage, u64 element count, then
//            storage 0: count * size(dtype) bytes; storage 1: 32 bytes SHA-256 of them
//   dtype    1 i32, 2 i64, 3 u64, 4 f32 (IEEE-754 bits), 5 u8, 6 f64 (bits), 7 UTF-8 text
//   end      a record with tag "end" (its field "records" = the records before it); a trace
//            without it was not closed (the process died) and is refused by the reader.
// Floats are stored as their bits, never formatted. A stage that is not called in a batch has
// no record in that batch (the reader reports it as absent).
#ifndef DYNG_PARITY_DYNLP_TRACE_HPP
#define DYNG_PARITY_DYNLP_TRACE_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dyng_parity::dynlp_rec {

/// SHA-256 of a byte range (FIPS 180-4), for the digest mode.
class sha256 {
 public:
  sha256();
  void update(const void* data, std::size_t size);
  /// The 32-byte digest (the object cannot be updated afterwards).
  std::vector<std::uint8_t> finish();

 private:
  void block(const std::uint8_t* p);
  std::uint32_t h_[8];
  std::uint8_t buf_[64];
  std::size_t used_ = 0;
  std::uint64_t bytes_ = 0;
};

enum dtype : std::uint8_t { i32 = 1, i64 = 2, u64 = 3, f32 = 4, u8 = 5, f64 = 6, text = 7 };

/// One record under construction; written by trace::end_record.
struct record {
  std::string tag;
  int batch = -1;
  std::vector<std::uint8_t> bytes;  // the encoded fields
  std::uint32_t fields = 0;
};

/// The process's trace (a singleton; nullptr when DYNG_DYNLP_TRACE_DIR is not set).
class trace {
 public:
  static trace* get();
  bool digest() const {
    return digest_;
  }

  record begin(const char* tag, int batch) const;
  void add(record& r, const char* name, dtype type, const void* data, std::uint64_t count) const;
  void add_text(record& r, const char* name, const std::string& value) const;
  void add_i32(record& r, const char* name, std::int32_t v) const {
    add(r, name, i32, &v, 1);
  }
  void add_i64(record& r, const char* name, std::int64_t v) const {
    add(r, name, i64, &v, 1);
  }
  void add_f32(record& r, const char* name, float v) const {
    add(r, name, f32, &v, 1);
  }
  /// An array field: stored raw, or as its SHA-256 in the digest mode.
  void add_array(record& r, const char* name, dtype type, const void* data,
                 std::uint64_t count) const;
  void end_record(const record& r);
  /// Closes the trace with its "end" record (also run at exit).
  void close();
  /// Whether records of `tag` are written (DYNG_DYNLP_TRACE_TAGS; all when it is not set).
  bool wants(const std::string& tag) const;

 private:
  trace(const std::string& path, bool digest);
  std::FILE* f_ = nullptr;
  bool digest_ = false;
  std::uint64_t records_ = 0;
  std::vector<std::pair<std::string, int>> sequence_;
  std::vector<std::string> tags_;  // DYNG_DYNLP_TRACE_TAGS (empty: every tag)
};

/// Element size of a dtype in bytes.
std::size_t dtype_size(dtype type);

/// Stops the process with a message (a recording failure is never ignored).
[[noreturn]] void fatal(const std::string& what);

}  // namespace dyng_parity::dynlp_rec

#endif  // DYNG_PARITY_DYNLP_TRACE_HPP
