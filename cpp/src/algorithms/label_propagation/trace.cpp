// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file trace.cpp
 * @brief label_propagation's stage trace (see trace.hpp; the format of ADR 0065's recorder).
 */
#include "algorithms/label_propagation/trace.hpp"

#include <dyng/core/error.hpp>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail::lp_trace {

namespace {

constexpr std::uint32_t k_init[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                     0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
constexpr std::uint32_t k_round[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

/// SHA-256 (FIPS 180-4) of the digest mode.
class sha256 {
 public:
  sha256() {
    std::memcpy(h_, k_init, sizeof(h_));
  }

  void update(const void* data, std::size_t size) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    bytes_ += size;
    while (size > 0) {
      if (used_ == 0 && size >= 64) {
        block(p);
        p += 64;
        size -= 64;
        continue;
      }
      const std::size_t take = (64 - used_) < size ? (64 - used_) : size;
      std::memcpy(buf_ + used_, p, take);
      used_ += take;
      p += take;
      size -= take;
      if (used_ == 64) {
        block(buf_);
        used_ = 0;
      }
    }
  }

  std::vector<std::uint8_t> finish() {
    const std::uint64_t bits = bytes_ * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (used_ != 56) {
      update(&zero, 1);
    }
    std::uint8_t len[8];
    for (int i = 0; i < 8; ++i) {
      len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    }
    update(len, 8);
    std::vector<std::uint8_t> out(32);
    for (int i = 0; i < 8; ++i) {
      out[static_cast<std::size_t>(4 * i)] = static_cast<std::uint8_t>(h_[i] >> 24);
      out[static_cast<std::size_t>(4 * i + 1)] = static_cast<std::uint8_t>(h_[i] >> 16);
      out[static_cast<std::size_t>(4 * i + 2)] = static_cast<std::uint8_t>(h_[i] >> 8);
      out[static_cast<std::size_t>(4 * i + 3)] = static_cast<std::uint8_t>(h_[i]);
    }
    return out;
  }

 private:
  static std::uint32_t rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void block(const std::uint8_t* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
             (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0];
    std::uint32_t b = h_[1];
    std::uint32_t c = h_[2];
    std::uint32_t d = h_[3];
    std::uint32_t e = h_[4];
    std::uint32_t f = h_[5];
    std::uint32_t g = h_[6];
    std::uint32_t h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + s1 + ch + k_round[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
  }

  std::uint32_t h_[8];
  std::uint8_t buf_[64] = {};
  std::size_t used_ = 0;
  std::uint64_t bytes_ = 0;
};

void put(std::vector<std::uint8_t>& out, const void* p, std::size_t n) {
  const auto* b = static_cast<const std::uint8_t*>(p);
  out.insert(out.end(), b, b + n);
}

template <typename value_t>
void put_value(std::vector<std::uint8_t>& out, value_t v) {
  put(out, &v, sizeof(value_t));  // little-endian hosts (x86-64, aarch64)
}

std::size_t dtype_size(dtype type) {
  switch (type) {
    case i32:
    case f32:
      return 4;
    case i64:
    case u64:
    case f64:
      return 8;
    case u8:
    case text:
      return 1;
  }
  return 1;
}

std::string program_name() {
  char buf[4096] = {};
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) {
    return "label_propagation";
  }
  const std::string path(buf, static_cast<std::size_t>(n));
  const std::size_t slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::mutex g_mutex;
std::unique_ptr<writer>* g_writer = nullptr;  // leaked on purpose: closed by the atexit handler
bool g_checked = false;

void close_at_exit() {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (g_writer != nullptr && *g_writer) {
    (*g_writer)->close();
  }
}

}  // namespace

writer* writer::get() {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_checked) {
    g_checked = true;
    const char* dir = std::getenv("DYNG_DYNLP_TRACE_DIR");
    if (dir != nullptr && dir[0] != '\0') {
      const char* d = std::getenv("DYNG_DYNLP_TRACE_DIGEST");
      const bool digest = d != nullptr && std::strcmp(d, "1") == 0;
      ::mkdir(dir, 0755);  // EEXIST is fine
      g_writer = new std::unique_ptr<writer>(
          new writer(std::string(dir) + "/" + program_name() + ".dlt", digest));
      std::atexit(close_at_exit);
    }
  }
  return g_writer != nullptr ? g_writer->get() : nullptr;
}

writer::writer(const std::string& path, bool digest) : digest_(digest) {
  struct stat st{};
  if (::stat(path.c_str(), &st) == 0) {
    throw io_error("dyng: the label_propagation trace " + path +
                       " exists already (a trace is never overwritten)",
                   path);
  }
  f_ = std::fopen(path.c_str(), "wb");
  if (f_ == nullptr) {
    throw io_error(
        "dyng: cannot create the label_propagation trace " + path + ": " + std::strerror(errno),
        path);
  }
  std::vector<std::uint8_t> head;
  put(head, "DYNGDLT1", 8);
  put_value<std::uint32_t>(head, 1);
  put_value<std::uint32_t>(head, digest ? 1U : 0U);
  if (std::fwrite(head.data(), 1, head.size(), f_) != head.size()) {
    throw io_error("dyng: cannot write the label_propagation trace " + path, path);
  }
}

writer::~writer() {
  if (f_ != nullptr) {
    std::fclose(f_);
  }
}

record writer::begin(const char* tag, std::int32_t batch) const {
  record r;
  r.tag = tag;
  r.batch = batch;
  return r;
}

void writer::add(record& r, const char* name, dtype type, const void* data,
                 std::uint64_t count) const {
  const std::size_t len = std::strlen(name);
  put_value<std::uint16_t>(r.bytes, static_cast<std::uint16_t>(len));
  put(r.bytes, name, len);
  put_value<std::uint8_t>(r.bytes, type);
  put_value<std::uint8_t>(r.bytes, 0);
  put_value<std::uint64_t>(r.bytes, count);
  if (count > 0) {
    put(r.bytes, data, static_cast<std::size_t>(count) * dtype_size(type));
  }
  ++r.fields;
}

void writer::add_array(record& r, const char* name, dtype type, const void* data,
                       std::uint64_t count) const {
  if (!digest_) {
    add(r, name, type, data, count);
    return;
  }
  sha256 h;
  if (count > 0) {
    h.update(data, static_cast<std::size_t>(count) * dtype_size(type));
  }
  const std::vector<std::uint8_t> digest = h.finish();
  const std::size_t len = std::strlen(name);
  put_value<std::uint16_t>(r.bytes, static_cast<std::uint16_t>(len));
  put(r.bytes, name, len);
  put_value<std::uint8_t>(r.bytes, type);
  put_value<std::uint8_t>(r.bytes, 1);
  put_value<std::uint64_t>(r.bytes, count);
  put(r.bytes, digest.data(), digest.size());
  ++r.fields;
}

void writer::end_record(const record& r) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  if (f_ == nullptr) {
    throw io_error("dyng: a label_propagation trace record after the end of the trace");
  }
  std::int32_t seq = 0;
  bool found = false;
  for (auto& [tag, next] : sequence_) {
    if (tag == r.tag) {
      seq = next++;
      found = true;
      break;
    }
  }
  if (!found) {
    sequence_.emplace_back(r.tag, 1);
  }
  std::vector<std::uint8_t> head;
  put_value<std::uint32_t>(head, 0x5250444cU);
  put_value<std::uint16_t>(head, static_cast<std::uint16_t>(r.tag.size()));
  put(head, r.tag.data(), r.tag.size());
  put_value<std::int32_t>(head, r.batch);
  put_value<std::int32_t>(head, seq);
  put_value<std::uint32_t>(head, r.fields);
  if (std::fwrite(head.data(), 1, head.size(), f_) != head.size() ||
      (!r.bytes.empty() && std::fwrite(r.bytes.data(), 1, r.bytes.size(), f_) != r.bytes.size())) {
    throw io_error("dyng: cannot write a label_propagation trace record");
  }
  ++records_;
}

void writer::close() {
  if (f_ == nullptr) {
    return;
  }
  // The end record (written without end_record's lock: close() runs under it at exit).
  record r = begin("end", -1);
  const std::uint64_t n = records_;
  add(r, "records", u64, &n, 1);
  std::vector<std::uint8_t> head;
  put_value<std::uint32_t>(head, 0x5250444cU);
  put_value<std::uint16_t>(head, static_cast<std::uint16_t>(r.tag.size()));
  put(head, r.tag.data(), r.tag.size());
  put_value<std::int32_t>(head, r.batch);
  put_value<std::int32_t>(head, 0);
  put_value<std::uint32_t>(head, r.fields);
  std::fwrite(head.data(), 1, head.size(), f_);
  std::fwrite(r.bytes.data(), 1, r.bytes.size(), f_);
  std::fclose(f_);
  f_ = nullptr;
}

}  // namespace dyng::detail::lp_trace
