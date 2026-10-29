// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file dlt_reader.hpp
 * @brief A reader of the DynLP stage traces (.dlt) of the LabelPropagation-CUDA@a276a3a recording
 *        twins, for the C++ replays (the format: parity/exporters/dynlp/dynlp_trace.hpp; the
 *        Python reader: parity/dynlp_trace.py; ADR 0065).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::dlt {

/// Element types of a field.
enum class dtype : std::uint8_t { i32 = 1, i64 = 2, u64 = 3, f32 = 4, u8 = 5, f64 = 6, text = 7 };

/// Element size of a dtype in bytes.
inline std::size_t dtype_size(dtype t) {
  switch (t) {
    case dtype::i32:
    case dtype::f32:
      return 4;
    case dtype::i64:
    case dtype::u64:
    case dtype::f64:
      return 8;
    case dtype::u8:
    case dtype::text:
      return 1;
  }
  throw std::runtime_error("dlt: unknown dtype");
}

/// One field: raw little-endian bytes, or (digest mode) the SHA-256 of them.
struct field {
  std::string name;
  dtype type = dtype::i32;
  bool digest = false;
  std::uint64_t count = 0;
  std::vector<std::uint8_t> raw;  ///< count * size bytes (empty in the digest mode)
  std::string sha256;             ///< hex digest (digest mode only)

  /// The elements as a vector of `value_t` (its size must be the dtype's).
  template <typename value_t>
  [[nodiscard]] std::vector<value_t> as() const {
    if (digest) {
      throw std::runtime_error("dlt: field '" + name + "' is stored as a digest");
    }
    if (sizeof(value_t) != dtype_size(type)) {
      throw std::runtime_error("dlt: field '" + name + "' read with the wrong element size");
    }
    std::vector<value_t> out(count);
    if (count > 0) {
      std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
  }

  /// A scalar field's value as int64 (integer dtypes).
  [[nodiscard]] std::int64_t integer() const {
    if (count != 1 || digest) {
      throw std::runtime_error("dlt: field '" + name + "' is not a scalar");
    }
    switch (type) {
      case dtype::i32:
        return as<std::int32_t>()[0];
      case dtype::i64:
        return as<std::int64_t>()[0];
      case dtype::u64:
        return static_cast<std::int64_t>(as<std::uint64_t>()[0]);
      case dtype::u8:
        return as<std::uint8_t>()[0];
      default:
        throw std::runtime_error("dlt: field '" + name + "' is not an integer");
    }
  }

  /// A scalar f32 field's bits.
  [[nodiscard]] std::uint32_t f32_bits() const {
    if (count != 1 || digest || type != dtype::f32) {
      throw std::runtime_error("dlt: field '" + name + "' is not a scalar f32");
    }
    return as<std::uint32_t>()[0];
  }
};

/// One record: a tag, the batch (-1 outside the batch loop), the per-tag sequence, its fields.
struct record {
  std::string tag;
  std::int32_t batch = -1;
  std::int32_t sequence = 0;
  std::map<std::string, field> fields;

  [[nodiscard]] const field& operator[](const std::string& name) const {
    const auto it = fields.find(name);
    if (it == fields.end()) {
      throw std::runtime_error("dlt: record '" + tag + "' has no field '" + name + "'");
    }
    return it->second;
  }
  [[nodiscard]] bool has(const std::string& name) const {
    return fields.count(name) != 0;
  }
};

/// A whole trace.
struct trace {
  bool digest_mode = false;
  std::vector<record> records;

  /// The records of a tag, in order.
  [[nodiscard]] std::vector<const record*> all(const std::string& tag) const {
    std::vector<const record*> out;
    for (const record& r : records) {
      if (r.tag == tag) {
        out.push_back(&r);
      }
    }
    return out;
  }
  /// The only record of a tag.
  [[nodiscard]] const record& one(const std::string& tag) const {
    const auto rs = all(tag);
    if (rs.size() != 1) {
      throw std::runtime_error("dlt: " + std::to_string(rs.size()) + " records '" + tag +
                               "', expected one");
    }
    return *rs.front();
  }
  /// The record of a tag in a batch, or nullptr (the stage did not run there).
  [[nodiscard]] const record* in_batch(const std::string& tag, std::int32_t batch) const {
    const record* found = nullptr;
    for (const record& r : records) {
      if (r.tag == tag && r.batch == batch) {
        if (found != nullptr) {
          throw std::runtime_error("dlt: two records '" + tag + "' in batch " +
                                   std::to_string(batch));
        }
        found = &r;
      }
    }
    return found;
  }
};

namespace detail {
template <typename value_t>
value_t take(const std::vector<std::uint8_t>& buf, std::size_t& pos) {
  if (pos + sizeof(value_t) > buf.size()) {
    throw std::runtime_error("dlt: truncated trace");
  }
  value_t v;
  std::memcpy(&v, buf.data() + pos, sizeof(value_t));
  pos += sizeof(value_t);
  return v;
}
inline std::string take_string(const std::vector<std::uint8_t>& buf, std::size_t& pos,
                               std::size_t n) {
  if (pos + n > buf.size()) {
    throw std::runtime_error("dlt: truncated trace");
  }
  std::string s(reinterpret_cast<const char*>(buf.data() + pos), n);
  pos += n;
  return s;
}
}  // namespace detail

/// Parse a trace from its bytes (refuses one without its end record).
inline trace parse(const std::vector<std::uint8_t>& buf) {
  using detail::take;
  if (buf.size() < 16 || std::memcmp(buf.data(), "DYNGDLT1", 8) != 0) {
    throw std::runtime_error("dlt: not a DynLP trace");
  }
  std::size_t pos = 8;
  const auto version = take<std::uint32_t>(buf, pos);
  const auto flags = take<std::uint32_t>(buf, pos);
  if (version != 1) {
    throw std::runtime_error("dlt: format version " + std::to_string(version));
  }
  trace t;
  t.digest_mode = (flags & 1U) != 0;
  bool end_seen = false;
  while (pos < buf.size()) {
    if (end_seen) {
      throw std::runtime_error("dlt: data after the end record");
    }
    if (take<std::uint32_t>(buf, pos) != 0x5250444CU) {
      throw std::runtime_error("dlt: bad record magic");
    }
    record r;
    r.tag = detail::take_string(buf, pos, take<std::uint16_t>(buf, pos));
    r.batch = take<std::int32_t>(buf, pos);
    r.sequence = take<std::int32_t>(buf, pos);
    const auto nfields = take<std::uint32_t>(buf, pos);
    for (std::uint32_t k = 0; k < nfields; ++k) {
      field f;
      f.name = detail::take_string(buf, pos, take<std::uint16_t>(buf, pos));
      f.type = static_cast<dtype>(take<std::uint8_t>(buf, pos));
      const auto storage = take<std::uint8_t>(buf, pos);
      f.count = take<std::uint64_t>(buf, pos);
      if (storage == 1) {
        f.digest = true;
        static const char digits[] = "0123456789abcdef";
        for (int i = 0; i < 32; ++i) {
          const auto b = take<std::uint8_t>(buf, pos);
          f.sha256.push_back(digits[b >> 4U]);
          f.sha256.push_back(digits[b & 15U]);
        }
      } else if (storage == 0) {
        const std::size_t bytes = static_cast<std::size_t>(f.count) * dtype_size(f.type);
        if (pos + bytes > buf.size()) {
          throw std::runtime_error("dlt: truncated field " + r.tag + "." + f.name);
        }
        f.raw.assign(buf.begin() + static_cast<std::ptrdiff_t>(pos),
                     buf.begin() + static_cast<std::ptrdiff_t>(pos + bytes));
        pos += bytes;
      } else {
        throw std::runtime_error("dlt: unknown storage in " + r.tag + "." + f.name);
      }
      std::string name = f.name;
      r.fields.emplace(std::move(name), std::move(f));
    }
    if (r.tag == "end") {
      if (r["records"].integer() != static_cast<std::int64_t>(t.records.size())) {
        throw std::runtime_error("dlt: the end record's count differs");
      }
      end_seen = true;
    } else {
      t.records.push_back(std::move(r));
    }
  }
  if (!end_seen) {
    throw std::runtime_error("dlt: no end record");
  }
  return t;
}

/// Read and parse a (decompressed) trace file.
inline trace read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("dlt: cannot open " + path);
  }
  const std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
  return parse(buf);
}

}  // namespace dyng::test::dlt
