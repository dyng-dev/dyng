// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The DynLP stage-trace writer (see dynlp_trace.hpp for the format; ADR 0065).
#include "dynlp_trace.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The marker the export patch looks for to tell a twin from the original executable.
extern "C" {
// NOLINTNEXTLINE
__attribute__((used)) const char dyng_dynlp_recorder_twin_marker[] =
    "dyng-dynlp-recorder-twin format 1";
}

namespace dyng_parity::dynlp_rec {

namespace {

constexpr std::uint32_t k_init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
constexpr std::uint32_t k_round[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

std::uint32_t rotr(std::uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

void put(std::vector<std::uint8_t>& out, const void* p, std::size_t n) {
  const auto* b = static_cast<const std::uint8_t*>(p);
  out.insert(out.end(), b, b + n);
}

template <typename T>
void put_value(std::vector<std::uint8_t>& out, T v) {
  put(out, &v, sizeof(T));  // the twins run on x86-64: little-endian
}

trace* g_trace = nullptr;
bool g_checked = false;

void close_at_exit() {
  if (g_trace != nullptr) {
    g_trace->close();
  }
}

std::string program_name() {
  // The executable's file name (/proc/self/comm would cut it to 15 characters).
  char buf[4096] = {};
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) {
    return "trace";
  }
  std::string path(buf, static_cast<std::size_t>(n));
  const std::size_t slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

void fatal(const std::string& what) {
  std::fprintf(stderr, "dynlp recorder: %s\n", what.c_str());
  std::fflush(stderr);
  std::_Exit(3);
}

sha256::sha256() {
  std::memcpy(h_, k_init, sizeof(h_));
}

void sha256::block(const std::uint8_t* p) {
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
  std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6],
                h = h_[7];
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

void sha256::update(const void* data, std::size_t size) {
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

std::vector<std::uint8_t> sha256::finish() {
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
    out[4 * i] = static_cast<std::uint8_t>(h_[i] >> 24);
    out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
    out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
    out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
  }
  return out;
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
  fatal("unknown dtype");
}

trace* trace::get() {
  if (!g_checked) {
    g_checked = true;
    const char* dir = std::getenv("DYNG_DYNLP_TRACE_DIR");
    if (dir != nullptr && dir[0] != '\0') {
      const char* d = std::getenv("DYNG_DYNLP_TRACE_DIGEST");
      const bool digest = d != nullptr && std::strcmp(d, "1") == 0;
      ::mkdir(dir, 0755);  // EEXIST is fine
      g_trace = new trace(std::string(dir) + "/" + program_name() + ".dlt", digest);
      std::atexit(close_at_exit);
    }
  }
  return g_trace;
}

trace::trace(const std::string& path, bool digest) : digest_(digest) {
  struct stat st{};
  if (::stat(path.c_str(), &st) == 0) {
    fatal("the trace " + path + " exists already (a trace is never overwritten)");
  }
  f_ = std::fopen(path.c_str(), "wb");
  if (f_ == nullptr) {
    fatal("cannot create " + path + ": " + std::strerror(errno));
  }
  std::vector<std::uint8_t> head;
  put(head, "DYNGDLT1", 8);
  put_value<std::uint32_t>(head, 1);
  put_value<std::uint32_t>(head, digest ? 1u : 0u);
  if (std::fwrite(head.data(), 1, head.size(), f_) != head.size()) {
    fatal("cannot write " + path);
  }
}

record trace::begin(const char* tag, int batch) const {
  record r;
  r.tag = tag;
  r.batch = batch;
  return r;
}

void trace::add(record& r, const char* name, dtype type, const void* data,
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

void trace::add_text(record& r, const char* name, const std::string& value) const {
  add(r, name, text, value.data(), value.size());
}

void trace::add_array(record& r, const char* name, dtype type, const void* data,
                      std::uint64_t count) const {
  if (!digest_) {
    add(r, name, type, data, count);
    return;
  }
  sha256 h;
  if (count > 0) {
    h.update(data, static_cast<std::size_t>(count) * dtype_size(type));
  }
  const std::vector<std::uint8_t> d = h.finish();
  const std::size_t len = std::strlen(name);
  put_value<std::uint16_t>(r.bytes, static_cast<std::uint16_t>(len));
  put(r.bytes, name, len);
  put_value<std::uint8_t>(r.bytes, type);
  put_value<std::uint8_t>(r.bytes, 1);
  put_value<std::uint64_t>(r.bytes, count);
  put(r.bytes, d.data(), d.size());
  ++r.fields;
}

void trace::end_record(const record& r) {
  if (f_ == nullptr) {
    fatal("a record after the end of the trace");
  }
  int seq = 0;
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
  put_value<std::uint32_t>(head, 0x5250444cu);
  put_value<std::uint16_t>(head, static_cast<std::uint16_t>(r.tag.size()));
  put(head, r.tag.data(), r.tag.size());
  put_value<std::int32_t>(head, r.batch);
  put_value<std::int32_t>(head, seq);
  put_value<std::uint32_t>(head, r.fields);
  if (std::fwrite(head.data(), 1, head.size(), f_) != head.size() ||
      (!r.bytes.empty() && std::fwrite(r.bytes.data(), 1, r.bytes.size(), f_) != r.bytes.size())) {
    fatal("cannot write a trace record");
  }
  ++records_;
}

void trace::close() {
  if (f_ == nullptr) {
    return;
  }
  record r = begin("end", -1);
  const std::uint64_t n = records_;
  add(r, "records", u64, &n, 1);
  end_record(r);
  if (std::fclose(f_) != 0) {
    f_ = nullptr;
    fatal("cannot close the trace");
  }
  f_ = nullptr;
}

}  // namespace dyng_parity::dynlp_rec
