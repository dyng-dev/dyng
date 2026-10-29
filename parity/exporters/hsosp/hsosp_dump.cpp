// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// The H-SOSP dump writer (see hsosp_dump.hpp; ADR 0081).
#include "hsosp_dump.hpp"

#include <cxxabi.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace dyng_parity::hsosp_dump {

namespace {

constexpr std::uint8_t type_i64 = 1;
constexpr std::uint8_t type_str = 2;
constexpr std::uint8_t type_array = 3;
constexpr std::uint8_t type_strings = 4;
constexpr std::uint8_t array_inline = 0;

constexpr std::uint64_t fnv_offset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t fnv_prime = 0x100000001b3ULL;

std::uint64_t fnv_bytes(std::uint64_t h, const void* data, std::size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= fnv_prime;
  }
  return h;
}

std::string env_or(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v ? v : fallback;
}

// An absolute path is written as "<abs>/<last component>" (no personal path in a dump).
std::string normalize_arg(const std::string& a) {
  if (a.empty() || a[0] != '/') return a;
  const auto slash = a.find_last_of('/');
  return "<abs>/" + a.substr(slash + 1);
}

// The value of these options is an output, never an input (a CSV that an earlier run appended
// to is not an input of this one).
bool is_output_option(const std::string& a) {
  return a == "--out" || a == "--timing" || a == "-o" || a == "--output";
}

std::vector<std::string> read_argv() {
  std::vector<std::string> args;
  std::ifstream in("/proc/self/cmdline", std::ios::binary);
  std::string a;
  while (std::getline(in, a, '\0')) args.push_back(a);
  return args;
}

}  // namespace

// --- fields -----------------------------------------------------------------------------------

fields::fields() {
  put_u32(0);
}

void fields::head(std::uint8_t type, const char* name) {
  ++count_;
  buf_.push_back(type);
  const auto len = static_cast<std::uint16_t>(std::strlen(name));
  buf_.push_back(static_cast<unsigned char>(len & 0xff));
  buf_.push_back(static_cast<unsigned char>(len >> 8));
  buf_.insert(buf_.end(), name, name + len);
}

void fields::put_u32(std::uint32_t v) {
  for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

void fields::put_u64(std::uint64_t v) {
  for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

void fields::i64(const char* name, long long v) {
  head(type_i64, name);
  put_u64(static_cast<std::uint64_t>(v));
}

void fields::str(const char* name, const std::string& s) {
  head(type_str, name);
  put_u32(static_cast<std::uint32_t>(s.size()));
  buf_.insert(buf_.end(), s.begin(), s.end());
}

void fields::strings(const char* name, const std::vector<std::string>& v) {
  head(type_strings, name);
  put_u32(static_cast<std::uint32_t>(v.size()));
  for (const auto& s : v) {
    put_u32(static_cast<std::uint32_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
}

void fields::array(const char* name, std::uint8_t elem, const void* data, std::uint64_t count) {
  head(type_array, name);
  buf_.push_back(array_inline);
  buf_.push_back(elem);
  put_u64(count);
  const auto* p = static_cast<const unsigned char*>(data);
  if (count > 0) buf_.insert(buf_.end(), p, p + count * elem);
}

void fields::ints(const char* name, const int* data, std::size_t count) {
  static_assert(sizeof(int) == 4, "the format stores int32");
  array(name, 4, data, count);
}

void fields::longs(const char* name, const long long* data, std::size_t count) {
  static_assert(sizeof(long long) == 8, "the format stores int64");
  array(name, 8, data, count);
}

void fields::rows(const char* name, const std::vector<std::vector<int>>& rows) {
  std::vector<long long> offsets;
  offsets.reserve(rows.size() + 1);
  offsets.push_back(0);
  std::size_t total = 0;
  for (const auto& r : rows) {
    total += r.size();
    offsets.push_back(static_cast<long long>(total));
  }
  std::vector<int> values;
  values.reserve(total);
  for (const auto& r : rows) values.insert(values.end(), r.begin(), r.end());
  const std::string base(name);
  longs((base + ".offsets").c_str(), offsets);
  ints((base + ".values").c_str(), values);
}

void fields::real(const char* name, double v) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  str(name, buf);
}

const std::vector<unsigned char>& fields::bytes() {
  std::memcpy(buf_.data(), &count_, 4);
  return buf_;
}

// --- the dump -----------------------------------------------------------------------------------

class dump {
 public:
  explicit dump(std::string dir) : dir_(std::move(dir)) {}
  bool open();
  void write(tag t, fields& f);
  void finish();

  std::recursive_mutex mutex_;
  bool all_rows_ = false;

 private:
  std::string dir_;
  FILE* out_ = nullptr;
  std::uint64_t seq_ = 0;
  std::uint64_t records_ = 0;
  bool finished_ = false;
};

namespace {

dump* g_dump = nullptr;
std::once_flag g_once;

void at_exit_handler() {
  if (g_dump) g_dump->finish();
}

}  // namespace

bool dump::open() {
  {
    const unsigned one = 1;
    unsigned char b;
    std::memcpy(&b, &one, 1);
    if (b != 1) return false;  // the format is little-endian
  }
  const std::string rows = env_or("DYNG_HSOSP_DUMP_ROWS", "touched");
  all_rows_ = rows == "all";
  ::mkdir(dir_.c_str(), 0777);
  int fd = -1;
  std::string path;
  for (int i = 1; i < 1000000 && fd < 0; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "%06d.hsd", i);
    path = dir_ + "/" + name;
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0 && errno != EEXIST) return false;
  }
  if (fd < 0) return false;
  out_ = ::fdopen(fd, "wb");
  if (!out_) return false;
  const char magic[8] = {'D', 'Y', 'N', 'G', 'H', 'S', 'D', '1'};
  std::fwrite(magic, 1, 8, out_);
  const std::uint32_t head[2] = {format_version, 0};
  std::fwrite(head, 4, 2, out_);

  fields f;
  f.i64("format_version", format_version);
  f.str("writer_version", writer_version);
  f.str("repo", "MOSP_ESCHER");
  f.str("commit", "4b86159ec0b3bfe6e18c718ff4bef1494d7276a5");
  f.str("rows", all_rows_ ? "all" : "touched");
  const std::vector<std::string> raw = read_argv();
  std::vector<std::string> args;
  for (const auto& a : raw) args.push_back(normalize_arg(a));
  f.strings("argv", args);
  // The input files named on the command line (not an output option's value): FNV-1a 64 and
  // size; files up to 16 MiB are copied into <dir>/inputs/<name>.<hash> (the awk-generated
  // random.hg of the original's test suite, which it deletes at its end).
  std::vector<std::string> inputs;
  for (std::size_t i = 1; i < raw.size(); ++i) {
    struct stat st{};
    if (is_output_option(raw[i - 1])) continue;
    if (::stat(raw[i].c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    std::string stored = "-";
    char hex[17] = "-";
    if (st.st_size <= (16 << 20)) {
      std::ifstream in(raw[i], std::ios::binary);
      std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      std::snprintf(hex, sizeof hex, "%016llx",
                    static_cast<unsigned long long>(fnv_bytes(fnv_offset, data.data(), data.size())));
      std::string leaf = normalize_arg(raw[i]);
      if (leaf.rfind("<abs>/", 0) == 0) leaf = leaf.substr(6);
      std::replace(leaf.begin(), leaf.end(), '/', '_');
      const std::string sub = dir_ + "/inputs";
      ::mkdir(sub.c_str(), 0777);
      stored = "inputs/" + leaf + "." + hex;
      const std::string dst = dir_ + "/" + stored;
      struct stat ds{};
      if (::stat(dst.c_str(), &ds) != 0) {
        std::ofstream o(dst, std::ios::binary);
        o.write(data.data(), static_cast<std::streamsize>(data.size()));
      }
    }
    inputs.push_back(std::to_string(i) + " " + normalize_arg(raw[i]) + " " + hex + " " +
                     std::to_string(static_cast<long long>(st.st_size)) + " " + stored);
  }
  f.strings("inputs", inputs);
  std::vector<std::string> env;
  for (const char* k : {"DYNG_HSOSP_DUMP_ROWS", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
                        "OMP_NUM_THREADS"}) {
    const char* v = std::getenv(k);
    if (v) env.push_back(std::string(k) + "=" + v);
  }
  f.strings("env", env);
  write(tag_header, f);
  std::atexit(at_exit_handler);
  return true;
}

void dump::write(tag t, fields& f) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!out_ || finished_) return;
  // seq goes last so that a record's own fields keep their order.
  f.i64("seq", static_cast<long long>(seq_++));
  const auto& b = f.bytes();
  const std::uint32_t tg = t;
  const std::uint64_t len = b.size();
  std::fwrite(&tg, 4, 1, out_);
  std::fwrite(&len, 8, 1, out_);
  std::fwrite(b.data(), 1, b.size(), out_);
  std::fflush(out_);
  ++records_;
}

void dump::finish() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (finished_ || !out_) return;
  fields f;
  f.str("exit", "normal");
  f.i64("records", static_cast<long long>(records_));
  write(tag_end, f);
  finished_ = true;
  std::fclose(out_);
  out_ = nullptr;
}

dump* get() {
  std::call_once(g_once, [] {
    const char* dir = std::getenv("DYNG_HSOSP_DUMP_DIR");
    if (dir == nullptr || *dir == '\0') return;
    auto* d = new dump(dir);
    if (d->open()) {
      g_dump = d;
    } else {
      delete d;
    }
  });
  return g_dump;
}

void write(dump* d, tag t, fields& f) {
  if (d) d->write(t, f);
}

bool all_rows(dump* d) {
  return d && d->all_rows_;
}

std::string current_exception_text() {
  std::string type = "unknown";
  if (const std::type_info* ti = abi::__cxa_current_exception_type()) {
    int status = 0;
    char* name = abi::__cxa_demangle(ti->name(), nullptr, nullptr, &status);
    type = status == 0 && name ? name : ti->name();
    std::free(name);
  }
  try {
    throw;
  } catch (const std::exception& e) {
    return type + ": " + e.what();
  } catch (...) {
    return type;
  }
}

}  // namespace dyng_parity::hsosp_dump
